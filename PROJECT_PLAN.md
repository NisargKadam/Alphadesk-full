# AlphaDesk — Project Plan

AlphaDesk is an AI equity research copilot that shows its work: every answer renders
the pipeline that produced it — which route the agent took, which tools it called,
which filing passages it read — before the answer text itself. It is the running
capstone of a 10-session agentic-AI bootcamp for working software engineers who are
new to agents. The repo is built session by session; each session adds exactly one
architectural layer, ends in a working demo, and is captured as one git commit and
one git tag (`session-01` … `session-10`), so the classroom can jump to any session's
exact state. The sessions teach the backend only — Python, end to end; the desk's
UI ships once, as a single final commit after the curriculum, so nothing in class
time is spent on frontend code.

## Session map

1. **LLM & Agent Foundations** — streaming chat on a raw model call, deliberately
   stateless and tool-less. The two failures this exposes (no memory, unverifiable
   specifics) motivate everything that follows.
2. **LangChain & LangGraph** — the raw call becomes a LangGraph `StateGraph` with
   checkpointed thread state, a router node, and one honest human-in-the-loop
   interrupt.
3. **Retrieval & RAG** — filings ingest → chunking → Chroma → retrieval-augmented,
   cited answers. Session 01's hallucination demo gets fixed on camera.
4. **Tool Use, Function Calling & MCP** — live market data (quotes, price history,
   news) via function calling inside the graph, and the same tools exposed over MCP
   to external agents.
5. **Enterprise RAG at Scale** — the Session 03 pipeline, done properly: an eval
   harness (hand-rolled hit@k and MRR) so changes are judged by numbers, hybrid
   (BM25 + vector, RRF-fused) search, parent-child chunking, and cross-encoder
   re-ranking replacing the lightweight model-call re-rank.
6. **Multi-Agent Systems** — the single graph becomes a supervisor coordinating
   Market, Filings Analyst, and News specialists plus a Report Writer, with
   hand-offs as typed contracts (`Task` out, `AgentResult` back) so evidence
   survives the trip. Every delegation is announced on the wire: one `handoff`
   event per task, `agent` status events as each specialist runs.
7. **Memory & State** — the in-process checkpointer becomes a durable SQLite
   checkpointer (threads survive restarts; the graph compiles in the app's
   lifespan because async savers are born inside the loop), a hand-rolled
   long-term store carries the user's watchlist and preferences across
   threads, and long conversations get summarized instead of truncated.
   *← implemented through here*
8. **Guardrails & Safety** *(ROADMAP)* — input validation, policy rails that keep
   the desk on the right side of "not investment advice", structured-output checks,
   and a tour of realistic failure modes and how each layer catches them.
9. **Observability & LLMOps** *(ROADMAP)* — tracing across the graph, per-run
   token/cost accounting emitted on the stream (the pipeline grows a price tag),
   and evaluation runs wired into the workflow.
10. **Full-Stack Deployment** *(ROADMAP)* — Docker packaging, health checks, the
    architecture diagram, and shipping the desk.

After Session 10, one final commit (`ui-final`) delivers the complete UI bench:
every event the backend learned to emit — nodes, hand-offs, citations, memory,
guardrails, cost — gets its renderer, at once.
