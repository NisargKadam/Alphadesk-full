"""The LangGraph StateGraph behind AlphaDesk. Introduced in Session 02;
Session 06 turns the single agent into a team.

Same product, real architecture — typed state, checkpointed threads, one
honest human-in-the-loop interrupt, and (since Session 06) a supervisor
delegating to specialist sub-agents through *typed contracts*. (Old
tutorials build agents with `langgraph.prebuilt.create_react_agent`; that is
deprecated in 1.x in favor of `langchain.agents.create_agent`. We wire the
graph by hand because seeing the nodes and edges is the lesson.)
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

from .llm import SYSTEM_PROMPT  # the Session 01 prompt survives every rearchitecture
from .tools import (
    get_company_news,
    get_price_history,
    get_quote,
    rerank_filings,
    search_filings,
)


class DeskState(TypedDict):
    # `add_messages` is a reducer: what a node returns is *appended* to the
    # history instead of replacing it. Checkpointed per thread, this list is
    # exactly the memory Session 01 lacked.
    messages: Annotated[list, add_messages]
    route: str
    clarifying_question: Optional[str]
    # Plain fields (no reducer) are *replaced* on write — but they persist
    # across turns in the checkpoint, so the router clears them every turn.
    tasks: list
    results: list


# --- Session 06: the hand-off contracts -------------------------------------
# This is the whole lesson: a hand-off between agents is DATA WITH A SCHEMA,
# not a vibe in prose. Task goes out, AgentResult comes back — anything not
# in these models does not survive the trip, by design.


class Task(BaseModel):
    """One unit of delegated work."""

    agent: Literal["market", "filings", "news"]
    instruction: str = Field(description="A direct order naming the company/ticker.")


class Plan(BaseModel):
    """The supervisor's structured output: which specialists run, and why."""

    tasks: list[Task] = Field(
        description="The MINIMAL set of tasks. A plain price check is ONE market task."
    )


class AgentResult(BaseModel):
    """What returns across the hand-off. Evidence rides inside the contract —
    citations must survive two hops (analyst → report writer → user)."""

    agent: str
    summary: str
    evidence: list = Field(default_factory=list)


class RouteDecision(BaseModel):
    """Structured output for the router: the model must pick a label, not prose."""

    route: Literal["general", "desk", "unclear"] = Field(
        description="Where the latest user message should go."
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description="Only when route is 'unclear': the ONE question to ask the user.",
    )


llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"])

# Classification and planning want determinism, not creativity: the router
# and supervisor run at temperature 0 so the same question takes the same
# path in every class.
router_llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"], temperature=0)

ROUTER_PROMPT = (
    "You route requests arriving at an equity research desk. Classify the "
    "user's LATEST message given the conversation so far:\n"
    "- 'desk': anything needing live market data, news headlines, or the "
    "filings index — prices, reported figures, risk factors, briefings. The "
    "desk supervisor decides which specialists run; an unfamiliar company or "
    "product name is NOT a reason to ask the user.\n"
    "- 'general': definitions, concepts, anything answerable without "
    "documents or live data.\n"
    "- 'unclear': only when the message needs one clarifying answer that ONLY "
    "THE USER can give — an ambiguous back-reference like 'the other one'."
)

# The Session 02 router didn't get duplicated — it evolved: it still triages
# every message, and 'desk' traffic now flows to the supervisor below.
SUPERVISOR_PROMPT = (
    "You run an equity research desk with three specialists:\n"
    "- 'market': live quotes and price history (has market data tools)\n"
    "- 'news': recent headlines (has a news tool)\n"
    "- 'filings': the ingested annual-report index — the ONLY agent that can "
    "cite documents (risk factors, reported figures, policies)\n"
    "Emit the MINIMAL set of tasks for the user's request. A price check "
    "delegates to market alone; only a genuine multi-domain briefing fans "
    "out. AT MOST ONE task per specialist — fold related questions into a "
    "single instruction. Write each instruction as a direct order naming the "
    "company/ticker."
)


def announce(name: str) -> None:
    # Nodes announce themselves on LangGraph's custom stream channel; the
    # frontend renders the NodeTrail from these events. This is the visible
    # pipeline — the UI grows a layer as the architecture does.
    get_stream_writer()({"type": "node", "name": name})


def router(state: DeskState) -> dict:
    announce("router")
    decision = router_llm.with_structured_output(RouteDecision).invoke(
        [("system", ROUTER_PROMPT), *state["messages"]]
    )
    # tasks/results reset: last turn's plan must not leak into this turn.
    return {
        "route": decision.route,
        "clarifying_question": decision.clarifying_question,
        "tasks": [],
        "results": [],
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


def supervisor(state: DeskState) -> dict:
    announce("supervisor")
    plan = router_llm.with_structured_output(Plan).invoke(
        [("system", SUPERVISOR_PROMPT), *state["messages"]]
    )
    # Contract enforcement, not just prompt hope: one lane per specialist.
    # Two filings tasks would produce two evidence sets both numbered [1..n]
    # — colliding citations downstream. The schema allows it; we don't.
    deduped, seen = [], set()
    for task in plan.tasks:
        if task.agent not in seen:
            seen.add(task.agent)
            deduped.append(task)
    plan.tasks = deduped
    writer = get_stream_writer()
    for task in plan.tasks:
        # Each hand-off is announced on the wire — delegation is part of the
        # visible pipeline, not something students take on faith.
        writer({"type": "handoff", "from": "supervisor", "to": task.agent, "task": task.instruction})
    return {"tasks": [t.model_dump() for t in plan.tasks]}


# --- The specialists ---------------------------------------------------------

RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"

GROUNDED_RULES = (
    "Answer ONLY from the numbered sources below — not from memory. Every "
    "sentence that states a fact must end with the marker(s) of the sources "
    "backing it, like: The payout ratio was 44%. [1] If the sources do not "
    "cover the question, say so plainly instead of guessing."
)


def run_tool_agent(name: str, tools_: list, instruction: str) -> AgentResult:
    # A focused sub-agent = a narrow system prompt + a SUBSET of tools + the
    # same bounded tool loop as Session 04. Scoping the toolbox is the point:
    # the news agent cannot quote prices even if it wanted to.
    writer = get_stream_writer()
    registry = {t.name: t for t in tools_}
    model = llm.bind_tools(tools_)
    messages = [
        ("system", f"{SYSTEM_PROMPT}\nYou are the desk's {name} specialist. Use your tools; report facts, not filler."),
        ("user", instruction),
    ]
    reply = None
    for _ in range(4):  # bounded: a specialist that can't converge must not hang the desk
        reply = model.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            break
        for call in reply.tool_calls:
            # `agent` on tool events lets the UI nest each card under the
            # specialist that made the call.
            writer({"type": "tool_call", "name": call["name"], "args": call["args"], "agent": name})
            payload = registry[call["name"]].invoke(call["args"])
            writer({"type": "tool_result", "name": call["name"], "payload": payload, "agent": name})
            messages.append(ToolMessage(content=json.dumps(payload), tool_call_id=call["id"]))
    return AgentResult(agent=name, summary=reply.content or "(no findings)")


def run_filings_analyst(instruction: str) -> AgentResult:
    # The Session 03/05 retrieval pipeline, wearing an agent's badge. Its
    # citations ride back inside AgentResult.evidence — numbered here, and
    # the report writer is forbidden from renumbering them.
    hits = search_filings(instruction, k=10)
    sources = rerank_filings(instruction, hits) if RERANK_ENABLED and hits else hits[:4]
    if not sources:
        return AgentResult(agent="filings", summary="The filings index has nothing on this — run the ingest or add documents.")
    numbered = "\n\n".join(
        f"[{i + 1}] ({s['source']}, p.{s['page']}) {s['text']}" for i, s in enumerate(sources)
    )
    reply = llm.invoke(
        [("system", f"{SYSTEM_PROMPT}\n\n{GROUNDED_RULES}\n\nSOURCES:\n{numbered}"), ("user", instruction)]
    )
    evidence = [
        {"id": i + 1, "source": s["source"], "page": s["page"], "snippet": s["snippet"][:500],
         "score": s["score"], "stage": s["stage"]}
        for i, s in enumerate(sources)
    ]
    return AgentResult(agent="filings", summary=reply.content, evidence=evidence)


AGENT_FNS = {
    "market": lambda instr: run_tool_agent("market", [get_quote, get_price_history], instr),
    "news": lambda instr: run_tool_agent("news", [get_company_news], instr),
    "filings": run_filings_analyst,
}


def agents(state: DeskState) -> dict:
    announce("agents")
    writer = get_stream_writer()
    results = []
    # Sequential on purpose: one lane fills at a time and the class can watch
    # each hand-off complete. Production fans out with LangGraph's Send API —
    # same contracts, concurrent lanes.
    for task in state["tasks"]:
        writer({"type": "agent", "name": task["agent"], "status": "running"})
        result = AGENT_FNS[task["agent"]](task["instruction"])
        writer({"type": "agent", "name": task["agent"], "status": "done"})
        results.append(result.model_dump())
    return {"results": results}


REPORT_PROMPT = (
    "You are the desk's report writer. Compose ONE coherent briefing from the "
    "specialist results below. Square brackets are RESERVED for the numbered "
    "document citations: keep every [1]-style marker EXACTLY as written, and "
    "never write any other bracketed tag (no [news], no [market]) — attribute "
    "those findings in plain prose, e.g. 'Per the news desk, …'. Keep prices, "
    "changes, and figures exactly as reported. Add no facts of your own. If a "
    "specialist found nothing, say so in one line rather than papering over it."
)


def report(state: DeskState) -> dict:
    announce("report")
    # Evidence gathered by the filings analyst is re-emitted here with its
    # original numbering — this is the test that citations survive hand-offs.
    evidence = [e for r in state["results"] if r["agent"] == "filings" for e in r["evidence"]]
    if evidence:
        get_stream_writer()({"type": "citations", "items": evidence})
    reply = llm.invoke(
        [
            ("system", REPORT_PROMPT),
            *state["messages"],
            ("user", "Specialist results:\n" + json.dumps(state["results"], indent=1)),
        ]
    )
    return {"messages": [reply]}


def respond(state: DeskState) -> dict:
    announce("respond")
    # The general-conversation path: no tools, no retrieval — those belong to
    # the specialists now. Sessions 03–05 grew this node; Session 06 slimmed
    # it back down, which is its own lesson in where capabilities live.
    reply = llm.invoke([("system", SYSTEM_PROMPT), *state["messages"]])
    return {"messages": [reply]}


builder = StateGraph(DeskState)
builder.add_node("router", router)
builder.add_node("clarify", clarify)
builder.add_node("supervisor", supervisor)
builder.add_node("agents", agents)
builder.add_node("report", report)
builder.add_node("respond", respond)
builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "general": "respond",
        "desk": "supervisor",  # Session 06: specialists behind a planner
        "unclear": "clarify",
    },
)
# A clarified message goes back through triage — the user's answer may have
# turned "the other one?" into a market question.
builder.add_edge("clarify", "router")
builder.add_edge("supervisor", "agents")
builder.add_edge("agents", "report")
builder.add_edge("report", END)
builder.add_edge("respond", END)

# In-process checkpoints: perfect for a classroom, gone on restart.
# Session 07 swaps in a durable checkpointer without touching the graph.
graph = builder.compile(checkpointer=InMemorySaver())
