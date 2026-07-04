"""AlphaDesk API — one streaming chat endpoint, plus resume for HITL.

The SSE event vocabulary grows one session at a time (see README,
"Streaming protocol"). S01: token/done/error · S02: + node/interrupt ·
S03: + citations · S04: + tool_call/tool_result · S06: + agent/handoff.
"""

from dotenv import load_dotenv

load_dotenv()  # backend/.env — secrets stay out of code and out of git

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from langgraph.types import Command  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from .graph import graph  # noqa: E402  (imported after the env is loaded)
from .sse import event  # noqa: E402

app = FastAPI(title="AlphaDesk")

# The Vite dev server is a different origin, so the browser preflights our
# POST; without these headers the stream never starts.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    # Vite increments the port when 5173 is already occupied. Keep local
    # development working on that fallback port (and when opened via
    # 127.0.0.1) without allowing non-local origins.
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):517\d+$",
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str  # client-generated; names this conversation's checkpoint


class ResumeRequest(BaseModel):
    thread_id: str
    answer: str


async def stream_graph(graph_input, thread_id: str):
    # thread_id selects which checkpointed conversation this run extends —
    # state lives server-side in the checkpointer, not in the request.
    config = {"configurable": {"thread_id": thread_id}}
    try:
        # LangGraph's current streaming API: astream with multiple modes.
        #   "messages" → per-token chunks from LLM calls inside nodes
        #   "custom"   → whatever nodes write via get_stream_writer()
        #   "updates"  → node results; also where an interrupt surfaces
        async for mode, chunk in graph.astream(
            graph_input, config, stream_mode=["messages", "custom", "updates"]
        ):
            if mode == "custom":
                yield event(**chunk)
            elif mode == "messages":
                token, meta = chunk
                # Only answer-writing nodes stream to the user; the router,
                # supervisor, and sub-agents also produce tokens here and
                # must not leak. (First main.py change since Session 02 —
                # Session 06 added `report` as a second answering node.)
                if token.content and meta.get("langgraph_node") in ("respond", "report"):
                    yield event("token", text=token.content)
            elif mode == "updates" and "__interrupt__" in chunk:
                yield event("interrupt", question=chunk["__interrupt__"][0].value["question"])
        yield event("done")
    except Exception as exc:
        # A failure mid-stream must be an event, not a dead socket.
        yield event("error", message=str(exc))


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_graph({"messages": [("user", req.message)]}, req.thread_id),
        media_type="text/event-stream",
    )


@app.post("/api/chat/resume")
def resume(req: ResumeRequest) -> StreamingResponse:
    # Command(resume=...) hands the human's answer to the interrupt() call
    # that paused this thread; the graph picks up exactly where it stopped.
    return StreamingResponse(
        stream_graph(Command(resume=req.answer), req.thread_id),
        media_type="text/event-stream",
    )
