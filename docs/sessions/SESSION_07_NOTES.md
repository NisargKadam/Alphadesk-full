# Session 07 — Memory & State

## Learning objectives

- Draw the line between the desk's two memories: the **checkpointer** (where were we in THIS conversation? — keyed by `thread_id`) and the **long-term store** (what do we know about this USER? — keyed by `user_id`, crossing every thread).
- Make thread state durable: swap `InMemorySaver` for a SQLite checkpointer and retire Session 02's "gone on restart" caveat on camera.
- Learn where async resources are born: the async checkpointer binds to the running event loop, so the graph compiles in FastAPI's lifespan, not at import.
- Hand-roll a memory store — a table and two queries, not magic — and a structured-output extractor whose most important behavior is returning *nothing* most turns.
- Summarize, don't truncate: fold old turns into a briefing note with `RemoveMessage`, keeping the context window flat and the facts exact.

## Live-coding order

1. Terminal — `uv pip install langgraph-checkpoint-sqlite aiosqlite`; re-pin `requirements.txt`.
2. `backend/app/graph.py` — swap `InMemorySaver` for `SqliteSaver` at compile. Boot, curl once, and read the error out loud: *the sync saver refuses an async graph run.* First lesson free.
3. `backend/app/graph.py` — `AsyncSqliteSaver` at module level next: `RuntimeError: no running event loop`. Second lesson free: async resources are born where the loop lives.
4. `backend/app/graph.py` + `backend/app/main.py` — land it properly: `compile_graph()` + FastAPI `lifespan`. Second change to `main.py` since Session 02; say that out loud.
5. Terminal — curl a two-turn conversation on one `thread_id`, **kill uvicorn**, boot it again, continue the thread: it remembers. `desk_state.db` is the whole story; delete it to reset the desk. **~30 minutes in.**
6. `backend/app/memory.py` — `_conn()`, `remember()`, `recall()`: the long-term store is a table with an UPSERT. Note `check_same_thread=False` and why (uvicorn's threadpool).
7. `backend/app/memory.py` — `MemoryUpdate` + `extract_and_store()`: one cheap structured call per turn; the docstring rail ("a question is not a fact — usually EMPTY") is the whole difference between memory and creepiness.
8. `backend/app/graph.py` — the `memory` node: write path (`saved` event), read path (`recalled` event); wire `START → memory → router`. Memory runs before triage.
9. `backend/app/graph.py` — `_context_prefix()` into `router`, `respond`, and `report`: memory changes what the model *knows*, never what the user *said*. The router needs it too — "what's on my watchlist?" is only triageable if triage knows a watchlist exists.
10. `backend/app/graph.py` — summarization in the `memory` node: past 12 messages, fold all but the last 4 into `summary` and return `RemoveMessage`s. The checkpointer keeps history; the model carries a briefing note.
11. `backend/app/main.py` — `user_id: str = "default"` on `ChatRequest`, passed into graph input; the docstring's event vocabulary gains `memory`.
12. Terminal — the demo script. Two users, two processes, one desk that remembers.

## Demo script

1. **"Add Nvidia to my watchlist. I prefer low-risk positions."** (`user_id: "priya"`, thread `a1`) — a `memory saved` event with both facts, then `recalled` echoing the profile back.
2. **Kill the server. Restart it.** Then **"What's on my watchlist?"** with a *new* thread `a2`, same user — `memory recalled` fires and the answer names Nvidia. New thread, new process; the checkpointer starts blank, the store doesn't. That distinction is the session.
3. Same question, `user_id: "dev"` — nothing recalled, and the desk says so. Memory is per-person, not global.
4. **"What did I just ask you?"** on thread `a2` — still answered, from the *checkpointer*. Two memories, two mechanisms, both visibly at work.

## Where it breaks

- **`The SqliteSaver does not support async methods`** — the sync saver met `graph.astream`. Use `AsyncSqliteSaver` (steps 2–4 walk through it deliberately).
- **`RuntimeError: no running event loop` at boot.** `AsyncSqliteSaver` constructed at import time. It must be built inside the lifespan — that is what `compile_graph()` is for.
- **The desk "remembers" things nobody told it.** The extractor is hoovering questions into facts. The rail lives in `MemoryUpdate`'s field description and `EXTRACT_PROMPT` ("a question is not a fact", "no NEW fact → empty list") — if memory gets creepy, those lines got weakened.
- **"What's on my watchlist?" triggers a clarify interrupt.** The router can't see the profile — `_context_prefix(state)` must ride into the router's system prompt, not just `respond`'s. (We hit this live; triage without memory asks "which watchlist?")
- **Old threads break after changing `DeskState`.** Checkpoints store state by shape; a renamed field resumes as a missing key. Delete `desk_state.db` in class, or read state defensively (`state.get(...)`) the way the `memory` node does.

## Stretch

> **Budget note:** ~262 changed application lines against the ~250 target — on budget; nothing named for trimming.

- Swap the hand-rolled store for LangGraph's `SqliteStore` (`langgraph.store`) and compare: namespacing, TTLs, and semantic search over memories are what the framework buys you.
- Add a `forget` path: "remove Nvidia from my watchlist" should shrink the stored value — deletion is memory's hardest UX.
- Lower the summarization threshold to 4 messages and watch the `summary` fold mid-demo; print it and grade what survived.
- Store per-user disclosure: log every profile fact the desk *used* in an answer — the audit trail guardrails will want in Session 08.
