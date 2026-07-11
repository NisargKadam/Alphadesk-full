# Session 09 — Observability & LLMOps

## Learning objectives

- Observability as callbacks, not graph surgery: the token meter and the tracer both attach through LangChain's `callbacks` list — the graph itself never learns it's being watched.
- Cost accounting hand-rolled on purpose: `usage_metadata` × a price table is the whole lesson, and it only teaches something if you write the arithmetic yourself.
- Tracing delegated to Opik on purpose: trace *storage and UI* — the tree view, the run history, the dashboard — is bought, not built. LangSmith and Langfuse are the managed equivalents of this file; Opik is the one you can run for free, entirely on your own machine.
- Failure isolation as a hard requirement: observability may never take down the desk. A missing SDK, a broken import, or a dead Opik server all have to leave the stream byte-identical to tracing being off.

## Live-coding order

1. `backend/app/observability.py` — the module docstring first: two concerns, deliberately separate (COST is hand-rolled because the arithmetic is the lesson; TRACES are delegated to Opik because storage/UI is bought).
2. `backend/app/observability.py` — `PRICES_PER_MTOK`: USD per 1M tokens, keyed by the bare model alias. A model missing from the table still streams its token counts — the `usd` field goes null rather than guessing.
3. `backend/app/observability.py` — `cost_payload()`: fold a run's per-model usage into the `cost` event shape (`input_tokens`, `output_tokens`, `usd`, `by_model`).
4. `backend/app/observability.py` — `make_tracer()`: the env gate (`OPIK_TRACING`) and the `try/except` around the import. Point at *why* the import lives inside the function — a broken opik install must not break `main.py` at import time, and tracing off must not even pay opik's import cost.
5. `backend/app/graph.py` — `stream_usage=True` on both `ChatOpenAI` instances. Show the bug first: comment it out, run a question, read the `cost` event — `input_tokens`/`output_tokens` are both zero. Streamed calls only report usage when asked.
6. `backend/app/main.py` — a `UsageMetadataCallbackHandler` created per run, the tracer appended to the same `callbacks` list when it's not `None`, and the `cost` event emitted from `usage.usage_metadata` just before `done`.
7. `backend/app/eval_retrieval.py` — `log_to_opik()`: same golden questions, same three configs, now also written to Opik as one experiment per configuration over an `alphadesk-golden` dataset.

## Demo script

1. **General question** ("What is a P/E ratio?") — read the `cost` event on the wire, do the arithmetic on screen: `by_model` tokens × the price table equals the `usd` field.
2. **Desk briefing** (multi-agent fan-out) — cost is roughly 10x the general question. Router, supervisor, and every specialist each spend tokens; fan-out has a price, and now it's a number instead of a feeling.
3. Set `OPIK_TRACING=true`, re-run the same briefing — walk the trace tree in the local Opik UI: router → supervisor → specialists → report, with prompts, latencies, and token counts per node.
4. Stop the Opik server, run the briefing again — the stream is identical, no error surfaces to the client. Failure isolation, on camera.
5. Run `python -m app.eval_retrieval` with tracing on — three experiments land in the Opik dashboard, one per retrieval configuration.

## Where it breaks

- **Cost reads zero.** `stream_usage=True` is missing from the `ChatOpenAI` construction — streamed calls don't report `usage_metadata` unless asked.
- **The app won't boot after breaking the opik install.** The `import opik...` line escaped `make_tracer()` and now runs at module import time. It must live *inside* the function, gated behind `OPIK_TRACING`, so a bad install only breaks tracing, never the boot.
- **Every run priced null even though the model "is in the table."** This actually happened: OpenAI reports usage under the resolved snapshot name (`gpt-4.1-mini-2025-04-14`), but `PRICES_PER_MTOK` keys the bare alias (`gpt-4.1-mini`) — a plain `dict.get(model)` never matches, so `usd` is silently `None` on every single run, price table notwithstanding. Fixed with a `_price_for()` helper that checks the exact key first, then a `model.startswith(known + "-")` prefix match against each table entry. Verified live: `{"type":"cost","input_tokens":839,"output_tokens":101,"usd":0.000497,"by_model":{"gpt-4.1-mini-2025-04-14":{...}}}` — the `by_model` key is the giveaway that this is a snapshot-name problem, not a missing-model problem. The table also deliberately ships only the three models with verifiable list prices — a model absent from the table (e.g. `gpt-5.4-nano`) logs a warning and reports `usd: null`; add your model's row before class.
- **Eval logs "duplicate-looking" experiments on every re-run.** By design — experiments are per-run history, and the golden-question dataset (which Opik dedupes on insert) is what keeps re-runs from duplicating the *data*, not the *experiments*. A growing experiment list next to a stable dataset is the tool working correctly.
- **Traces missing with tracing on.** Check `~/.opik.config` for the right server URL, and that the server is actually up — another dev server may already be squatting on the Opik port (5173). Server down is the safe failure, and it was measured, not assumed: the eval CLI prints `(Opik logging skipped: …)` and exits 0 — its stdout with tracing off is MD5-identical before vs. after the Session 09 change (323 bytes either way) — and the `/api/chat` stream with tracing on and the server down shows the same event sequence as tracing off: same vocabulary, no `error` event, `cost` still present.
- **A blocked run still shows a small cost.** The memory extractor runs before the guard, so even a blocked message spends extraction tokens — the cost event reports it honestly; re-run the Session 08 blocked demo and read the price tag.

## Stretch

> **Budget note:** 163 insertions / 4 deletions across `backend/app` + `requirements.txt` + `.env.example` — comfortably under the ~250 target.

- Per-node cost breakdown: attribute tokens to the node that spent them (span-level accounting) instead of one total for the whole run.
- A `budget` guardrail that refuses runs past a per-thread USD cap — ties Session 08's rail pattern to Session 09's cost numbers.
- Alerting on p95 latency computed from Opik's trace data, instead of eyeballing the dashboard.
- RAGAS-style answer-quality metrics on the eval, alongside the existing rank-based hit@k and MRR.
