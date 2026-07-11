# AlphaDesk

An AI equity research copilot that shows its work — every answer renders the
pipeline that produced it: route → tools → evidence → text.

Built session by session across a 10-session agentic-AI bootcamp. Each session is
one git commit and one tag; `git switch -d session-01` (etc.) jumps to any
session's exact state. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full map and
[docs/sessions/](docs/sessions/) for per-session notes.

> Educational project — not investment advice.

## Quickstart

One terminal (Python 3.11+). The UI ships as one final commit at the end of the
course — until then, `curl` is the client.

```bash
cd backend
uv venv .venv && uv pip install -r requirements.txt
# or: python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env       # then put your OpenAI API key in .env
.venv/bin/uvicorn app.main:app --reload --port 8000
```

```bash
curl -N localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"Brief me on Apple — price action, latest news, and risk factors.","thread_id":"t1"}'
```

## Streaming protocol

`POST /api/chat` with `{"message": "..."}` returns `text/event-stream`. Every
event is one JSON object on a `data:` line, with a `type` field the client
switches on:

```
data: {"type": "token", "text": "Apple"}\n\n
```

The vocabulary grows one session at a time — a client parses this envelope
once, and later sessions only add event types (the final-commit UI adds one
renderer per type):

| Event       | Since | Payload    | Meaning                                          |
| ----------- | ----- | ---------- | ------------------------------------------------ |
| `token`     | S01   | `text`     | One chunk of assistant text — append it.         |
| `done`      | S01   | —          | The stream is complete.                          |
| `error`     | S01   | `message`  | Something failed; render it, stop.               |
| `node`      | S02   | `name`     | A graph node started — the pipeline, visible.    |
| `interrupt` | S02   | `question` | The graph paused for a human answer (see below). |
| `citations` | S03   | `items`    | The numbered evidence the answer cites as `[n]`: `[{id, source, page, snippet}]`. Since S05 each item also carries `score` and `stage` (`"vector" \| "hybrid" \| "reranked"`). |
| `tool_call` | S04   | `name`, `args` | The model requested a tool call — work in flight. Since S06 carries `agent`, the specialist that called it. |
| `tool_result` | S04 | `name`, `payload` | The tool returned — live data (or its `error`). Since S06 carries `agent`. |
| `handoff`   | S06   | `from`, `to`, `task` | The supervisor delegated a task to a specialist. |
| `agent`     | S06   | `name`, `status` | A specialist started (`running`) or finished (`done`). |
| `memory`    | S07   | `action`, `items` | Long-term memory was `saved` (new facts) or `recalled` (the user's profile). |
| `guardrail` | S08   | `rail`, `action` | A rail fired: `input/blocked`, `advice/reframed`, or `schema/retried`. |
| `cost`      | S09   | `input_tokens`, `output_tokens`, `usd`, `by_model` | The run's price tag, just before `done`. `usd` is null if a model is missing from the price table. |

Since Session 02 every request also carries a client-generated `thread_id` —
the conversation state lives server-side in the graph checkpointer, keyed by
that id. When an `interrupt` event arrives, the graph is paused;
`POST /api/chat/resume` with `{"thread_id": "...", "answer": "..."}` resumes
it and returns a fresh SSE stream that continues the same reply.

Since Session 07 a request may also carry a `user_id` (default `"default"`) —
it names the *person*, not the conversation: long-term memory (watchlist,
preferences) is keyed by it and crosses every thread that user opens.

## Durable state (Session 07)

Threads and long-term memory both live in `backend/desk_state.db`
(gitignored): checkpoints from the SQLite checkpointer, the user profile
store beside them. Kill the server and resume any thread; delete the file to
reset the desk completely.

## Filings ingest (Session 03)

Drop annual reports or 10-K PDFs into `data/filings/` (a synthetic sample is
included — see [data/filings/README.md](data/filings/README.md)), then:

```bash
cd backend
.venv/bin/python -m app.ingest ../data/filings
```

First run downloads Chroma's local embedding model (~80 MB, no API key).
Re-runs are idempotent. The index lives in `backend/chroma_db/` (gitignored).
Since Session 05 the index is parent-child (small chunks matched, whole pages
read) — re-run the ingest once after pulling Session 05.

## Retrieval evaluation (Session 05)

```bash
cd backend
.venv/bin/python -m app.eval_retrieval
```

Scores vector-only vs. hybrid (BM25 + RRF) vs. hybrid + cross-encoder
re-rank on `data/eval/golden_questions.json` with hand-rolled hit@k and MRR.
Measure first, then improve — add golden pairs for your own filings.

## Observability (Session 09)

Cost needs no server at all: every `/api/chat` stream now ends with a `cost`
event — the run's token totals and a USD estimate — just before `done`.

Set `OPIK_TRACING=true` in `backend/.env` to also ship every run as a full
trace tree (router → supervisor → specialists → report, with prompts,
latencies, and token counts) to a locally hosted
[Opik](https://github.com/comet-ml/opik) instance — the SDK reads
`~/.opik.config` for the server URL. Tracing off or server down: nothing
changes; observability is strictly additive.

With Opik on, `python -m app.eval_retrieval` also logs each run as an
experiment over the `alphadesk-golden` dataset — retrieval changes get a
tracked history instead of terminal numbers that scroll away.

## Use AlphaDesk's tools from Claude (Session 04)

The same market tools the graph calls in-process are served over MCP by
`backend/app/mcp_server.py`. Register them with Claude Code (adjust the path):

```bash
claude mcp add alphadesk -- /path/to/alphadesk/backend/.venv/bin/python /path/to/alphadesk/backend/app/mcp_server.py
```

or drop the equivalent into `.mcp.json`:

```json
{
  "mcpServers": {
    "alphadesk": {
      "command": "/path/to/alphadesk/backend/.venv/bin/python",
      "args": ["/path/to/alphadesk/backend/app/mcp_server.py"]
    }
  }
}
```

Then ask Claude "use alphadesk to quote TSLA" — the exact functions your graph
calls, consumed by an external agent.

## Session map

1. **LLM & Agent Foundations** — streaming chat on a raw model call
2. **LangChain & LangGraph** — StateGraph, thread state, HITL interrupt
3. **Retrieval & RAG** — filings ingest, Chroma retrieval, cited answers
4. **Tool Use, Function Calling & MCP** — live market tools, inside and outside the graph
5. **Enterprise RAG at Scale** — hybrid retrieval, parent-child chunks, re-ranking, eval harness
6. **Multi-Agent Systems** — supervisor, typed hand-offs, specialist sub-agents
7. **Memory & State** — durable checkpoints, long-term store, summarization
8. **Guardrails & Safety** — input rails, policy rails, structured-output checks
9. **Observability & LLMOps** — Opik traces, per-run cost on the stream, eval experiments *(you are here)*
10. Roadmap: deployment — see [PROJECT_PLAN.md](PROJECT_PLAN.md).
