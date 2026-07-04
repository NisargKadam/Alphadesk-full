"""Session 02: the raw model call becomes a LangGraph StateGraph.

Same product, real architecture — typed state, a router with conditional
edges, checkpointed conversation threads, and one honest human-in-the-loop
interrupt. (Old tutorials build agents with `langgraph.prebuilt
.create_react_agent`; that is deprecated in 1.x in favor of
`langchain.agents.create_agent`. We wire the graph by hand because seeing
the nodes and edges is the lesson.)
"""

import json
import os
from typing import Annotated, Literal, Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

# Old tutorials import MemorySaver — renamed InMemorySaver in 1.x, same class.
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .llm import SYSTEM_PROMPT  # the Session 01 prompt survives the rearchitecture
from .tools import MARKET_TOOLS, search_filings


class DeskState(TypedDict):
    # `add_messages` is a reducer: what a node returns is *appended* to the
    # history instead of replacing it. Checkpointed per thread, this list is
    # exactly the memory Session 01 lacked.
    messages: Annotated[list, add_messages]
    route: str
    clarifying_question: Optional[str]
    # Plain fields (no reducer) are *replaced* on write — but they still
    # persist across turns in the checkpoint, so the router clears this one.
    sources: list


class RouteDecision(BaseModel):
    """Structured output for the router: the model must pick a label, not prose."""

    route: Literal["general", "market", "filings", "unclear"] = Field(
        description="Where the latest user message should go."
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description="Only when route is 'unclear': the ONE question to ask the user.",
    )


llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"])

ROUTER_PROMPT = (
    "You route requests arriving at an equity research desk. Classify the "
    "user's LATEST message given the conversation so far:\n"
    "- 'filings': anything a company's annual report or 10-K would answer — "
    "reported financials and revenue, segments, dividends, risk factors, "
    "strategy, policies. When torn between filings and general for a "
    "company-specific fact, pick filings: the retrieval layer says honestly "
    "whether the documents cover it.\n"
    "- 'market': live prices, quotes, today's moves, recent news.\n"
    "- 'general': definitions, concepts, anything answerable without "
    "documents or live data.\n"
    "- 'unclear': only when the message cannot be acted on at all without one "
    "clarifying question (e.g. an ambiguous reference like 'the other one')."
)


def announce(name: str) -> None:
    # Nodes announce themselves on LangGraph's custom stream channel; the
    # frontend renders the NodeTrail from these events. This is the visible
    # pipeline — the UI grows a layer as the architecture does.
    get_stream_writer()({"type": "node", "name": name})


def router(state: DeskState) -> dict:
    announce("router")
    decision = llm.with_structured_output(RouteDecision).invoke(
        [("system", ROUTER_PROMPT), *state["messages"]]
    )
    # sources=[] : last turn's evidence must not leak into this turn.
    return {
        "route": decision.route,
        "clarifying_question": decision.clarifying_question,
        "sources": [],
    }


def clarify(state: DeskState) -> dict:
    announce("clarify")
    # interrupt() pauses the graph mid-run and persists the thread in the
    # checkpointer; nothing resumes until /api/chat/resume feeds the human's
    # answer back via Command(resume=...). HITL as a graph primitive, not an
    # if-statement. NB: on resume this whole node re-executes from the top —
    # keep everything before interrupt() cheap and deterministic.
    answer = interrupt({"question": state["clarifying_question"] or "Could you clarify?"})
    return {"messages": [HumanMessage(content=answer)]}


class Ranking(BaseModel):
    """Structured output for the re-rank pass: an ordering, not prose."""

    indices: list[int] = Field(
        description="0-based positions of the most relevant chunks, best first, at most 4."
    )


RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"


def rerank(query: str, chunks: list[dict]) -> list[dict]:
    # Listwise re-rank with one cheap structured model call: the retriever's
    # top 10 (fast, approximate) get re-ordered down to the 4 the answer will
    # actually cite. A trained cross-encoder does this job properly — that
    # arrives in Session 05; the *shape* of the pipeline is what matters here.
    numbered = "\n\n".join(f"[{i}] {c['text'][:400]}" for i, c in enumerate(chunks))
    ranking = llm.with_structured_output(Ranking).invoke(
        [
            ("system", "Rank the excerpts by how well they answer the question."),
            ("user", f"Question: {query}\n\nExcerpts:\n{numbered}"),
        ]
    )
    kept = [chunks[i] for i in ranking.indices if 0 <= i < len(chunks)][:4]
    return kept or chunks[:4]


def retrieve(state: DeskState) -> dict:
    announce("retrieve")
    query = state["messages"][-1].content
    hits = search_filings(query, k=10)
    sources = rerank(query, hits) if RERANK_ENABLED and hits else hits[:4]
    if sources:
        # The frontend numbers its citation chips from this event; respond()
        # below is instructed to cite with the same numbers. One vocabulary.
        get_stream_writer()(
            {
                "type": "citations",
                "items": [
                    {"id": i + 1, "source": s["source"], "page": s["page"], "snippet": s["text"][:500]}
                    for i, s in enumerate(sources)
                ],
            }
        )
    return {"sources": sources}


GROUNDED_RULES = (
    "Answer ONLY from the numbered sources below — not from memory. Every "
    "sentence that states a fact must end with the marker(s) of the sources "
    "backing it, like: The payout ratio was 44%. [1] Buybacks may supplement "
    "dividends. [2][3] If the sources do not cover the question, say so "
    "plainly instead of guessing."
)

NO_SOURCES_NOTE = (
    "The filings index returned nothing (it may be empty — the ingest CLI "
    "populates it). Say you have no filings to cite yet and suggest running "
    "the ingest; answer only what you can without inventing specifics."
)


def respond(state: DeskState) -> dict:
    announce("respond")
    model = llm
    if state["route"] == "market":
        # bind_tools attaches the tools' JSON schemas to the request. The
        # model can then *ask* for calls — executing them is our job (the
        # tools node below). Function calling is a protocol, not magic.
        model = llm.bind_tools(MARKET_TOOLS)
    system = SYSTEM_PROMPT
    if state["route"] == "filings":
        if state.get("sources"):
            numbered = "\n\n".join(
                f"[{i + 1}] ({s['source']}, p.{s['page']}) {s['text']}"
                for i, s in enumerate(state["sources"])
            )
            system = f"{SYSTEM_PROMPT}\n\n{GROUNDED_RULES}\n\nSOURCES:\n{numbered}"
        else:
            system = f"{SYSTEM_PROMPT}\n\n{NO_SOURCES_NOTE}"
    reply = model.invoke([("system", system), *state["messages"]])
    return {"messages": [reply]}


TOOL_REGISTRY = {t.name: t for t in MARKET_TOOLS}


def tools(state: DeskState) -> dict:
    # The other half of the function-calling protocol: run each call the
    # model requested and hand back ToolMessages keyed by tool_call_id so it
    # can match answers to requests. Hand-rolled (~10 lines) because seeing
    # the loop is the lesson — langgraph.prebuilt.ToolNode does the same job.
    announce("tools")
    writer = get_stream_writer()
    results = []
    for call in state["messages"][-1].tool_calls:
        writer({"type": "tool_call", "name": call["name"], "args": call["args"]})
        payload = TOOL_REGISTRY[call["name"]].invoke(call["args"])
        writer({"type": "tool_result", "name": call["name"], "payload": payload})
        results.append(ToolMessage(content=json.dumps(payload), tool_call_id=call["id"]))
    return {"messages": results}


builder = StateGraph(DeskState)
builder.add_node("router", router)
builder.add_node("clarify", clarify)
builder.add_node("retrieve", retrieve)
builder.add_node("respond", respond)
builder.add_node("tools", tools)
builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "general": "respond",
        "filings": "retrieve",  # Session 03: the retrieval path
        "market": "respond",  # Session 04: respond arrives tool-equipped
        "unclear": "clarify",
    },
)
builder.add_edge("clarify", "respond")
builder.add_edge("retrieve", "respond")
# The tool loop: if respond's reply requests tool calls, execute them and go
# around again (respond → tools → respond …) until it answers in plain text.
builder.add_conditional_edges(
    "respond",
    lambda state: "tools" if state["messages"][-1].tool_calls else END,
    {"tools": "tools", END: END},
)
builder.add_edge("tools", "respond")

# In-process checkpoints: perfect for a classroom, gone on restart.
# Session 07 swaps in a durable checkpointer without touching the graph.
graph = builder.compile(checkpointer=InMemorySaver())
