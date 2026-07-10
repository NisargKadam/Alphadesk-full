# Sessions 09 & 10 — Design

**Date:** 2026-07-11
**Scope:** Finish the AlphaDesk curriculum with its last two backend sessions, one
commit + one tag each (`session-09`, `session-10`), Python only. The post-curriculum
`ui-final` commit is explicitly **out of scope** for this work.

## Context

Sessions 01–08 are built and tagged. PROJECT_PLAN.md defines the remaining two:

- **Session 09 — Observability & LLMOps:** tracing across the graph, per-run
  token/cost accounting emitted on the stream, evaluation runs wired into the
  workflow.
- **Session 10 — Full-Stack Deployment:** Docker packaging, health checks, the
  architecture diagram, shipping the desk.

Standing constraints (from project memory): OpenAI stack only; verification runs on
port **8010** (8000 is occupied by another project); never commit
`docs/INSTRUCTOR_PLAYBOOK.md`; no pushes to remotes as part of this work.

**Decision (user):** observability and eval tracking use **Opik**, the open-source
LLM observability platform the user already runs locally — not LangSmith (managed,
costly). Session notes name LangSmith/Langfuse as the managed alternatives and say
why the local open-source option won here.

---

## Session 09 — Observability & LLMOps

### Tracing (Opik)

- `opik` joins `requirements.txt`. Its `OpikTracer` is a LangChain callback;
  `stream_graph()` in `main.py` attaches it to each run's `config["callbacks"]`, so
  every graph run lands in the local Opik instance as a full trace tree — router
  decision, supervisor plan, each specialist's tool loop, report writer — with
  per-call prompts, latencies, and token counts.
- Gated by `OPIK_TRACING` (default `false`) in `.env` / `.env.example`, alongside
  `OPIK_PROJECT_NAME=alphadesk`. The user's local Opik URL/workspace is confirmed at
  implementation time (the SDK reads `~/.opik.config` or `OPIK_URL_OVERRIDE`).
- **Failure isolation is a requirement:** tracer construction is wrapped; if Opik is
  disabled, unconfigured, or unreachable, the app behaves exactly as Session 08 left
  it. Observability must never take down the desk.

### Per-run cost on the stream

- New module `app/observability.py`:
  - A price table (USD per 1M input/output tokens) for the models this repo uses:
    `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o-mini`, `gpt-5.4-nano`. Unknown model →
    `usd: null` (tokens still reported).
  - `make_tracer()` → the env-gated, failure-wrapped OpikTracer (or `None`).
  - A helper that folds a `UsageMetadataCallbackHandler`'s per-model totals into the
    `cost` event payload.
- `stream_graph()` creates one `UsageMetadataCallbackHandler` per request, passes it
  in `config["callbacks"]`, and yields one new SSE event after the run, right before
  `done`:

  ```
  data: {"type":"cost","input_tokens":2841,"output_tokens":412,"usd":0.0018,"by_model":{...}}
  ```

- `graph.py`: both `ChatOpenAI` instances gain `stream_usage=True` so streamed calls
  report token usage (without it, streamed responses carry no usage metadata).
- README event table gains the row: `cost | S09 | input_tokens, output_tokens, usd,
  by_model | the run's price tag`.
- Error path unchanged: an exception mid-run still emits `error`; `cost` is only
  emitted on runs that complete (including runs that pause at an `interrupt` — the
  resume request accounts for its own spend).

### Evals wired into the workflow

- `eval_retrieval.py` keeps its CLI behavior byte-for-byte first (print the
  hit@1/hit@4/MRR table), then — only when Opik is configured — additionally:
  - creates/gets an Opik dataset from `data/eval/golden_questions.json`, and
  - logs the run as an Opik experiment: one experiment entry per retrieval config
    with its hit@1 / hit@4 / MRR scores.
- Result: retrieval changes get tracked history in the Opik dashboard instead of
  terminal numbers that scroll away. Failure-safe like the tracer: no Opik, no
  change from Session 05 behavior.

### Session 09 deliverables

- `app/observability.py` (new), `main.py` + `graph.py` (small diffs),
  `eval_retrieval.py` (additive Opik logging), `requirements.txt` (+`opik`),
  `.env.example` (+`OPIK_TRACING`, `OPIK_PROJECT_NAME`), README (event row +
  "Observability" section), `docs/sessions/SESSION_09_NOTES.md`.
- Verification: backend on 8010; curl a chat and see the `cost` event; with
  `OPIK_TRACING=true`, see the trace in local Opik; run `python -m
  app.eval_retrieval` and see the experiment logged.

---

## Session 10 — Full-Stack Deployment

### Packaging

- `backend/Dockerfile`: slim Python base matching the dev interpreter, requirements
  installed as a cached layer, app copied last, `EXPOSE 8000`, uvicorn entrypoint
  binding `0.0.0.0:8000`.
- `backend/.dockerignore`: `.venv`, `chroma_db`, `desk_state.db*`, `.env`,
  `__pycache__`.
- Root `docker-compose.yml`: one `backend` service; port mapping
  `${ALPHADESK_PORT:-8000}:8000` (classroom default 8000; local verification uses
  `ALPHADESK_PORT=8010` because 8000 is occupied); `env_file: backend/.env`; volumes
  persisting the three stateful paths (SQLite checkpoint/profile DB, `chroma_db/`,
  `data/filings/`) across container rebuilds.
- Honest caveat taught in the notes: `sentence-transformers` pulls torch, so the
  image is multi-GB; `RERANK_ENABLED=false` is the documented slim-runtime escape
  hatch (the dependency still installs; the model never loads).

### Health

- `GET /health` in `main.py`: `{status: "ok", model, graph_ready, filings_indexed}`
  — `graph_ready` from the lifespan-compiled graph, `filings_indexed` from the
  presence of the Chroma index.
- The compose file wires `/health` as the container `healthcheck`, so
  `docker compose ps` shows real readiness, not just "running".

### Architecture diagram + docs

- `docs/ARCHITECTURE.md`: Mermaid diagram of the finished system — client →
  FastAPI/SSE → graph (`memory → guard → router → {respond | clarify | supervisor →
  agents → report}`), specialists' tools, the stores (SQLite checkpoints + profile,
  Chroma index), externals (OpenAI, yfinance, local Opik), and the MCP server
  exposing the same tools outward.
- README: "Deploy (Session 10)" section (`docker compose up`), session map rows 9–10
  move from roadmap to done.
- PROJECT_PLAN.md: the *← implemented through here* marker moves to Session 10;
  roadmap annotations removed from 09/10.
- `docs/sessions/SESSION_10_NOTES.md`.

### Session 10 deliverables

- `backend/Dockerfile`, `backend/.dockerignore`, `docker-compose.yml`,
  `main.py` (+`/health`), `docs/ARCHITECTURE.md`, README, PROJECT_PLAN.md,
  `docs/sessions/SESSION_10_NOTES.md`.
- Verification: `docker build` succeeds; `ALPHADESK_PORT=8010 docker compose up`
  serves `/health` healthy and streams a chat; state survives a container restart.

---

## Error handling (both sessions)

- Observability (tracer, cost math, Opik eval logging) is strictly additive and
  failure-wrapped — a broken/absent Opik or an unpriced model never breaks a run.
- `/health` never throws: each probe is best-effort and reports `false` over 500s.

## Testing

Manual verification per session (this repo has no test suite; the course verifies by
demo), on port 8010, exactly as listed in each session's deliverables. Docker build
verified locally (Docker is installed and running on this machine).

## Out of scope

- The `ui-final` frontend commit.
- CI/GitHub Actions.
- Pushing to any remote (the user manages the classroom remote themselves).
