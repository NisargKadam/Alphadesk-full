# Session 05 — Enterprise RAG at Scale

## Learning objectives

- Build a retrieval eval harness and hand-roll hit@k and MRR — measure first, then improve; that ordering is the whole session.
- Add a second, lexical view of the corpus (BM25) and fuse it with vectors via Reciprocal Rank Fusion — keyword and semantic search fail differently.
- Split chunking's two jobs with parent-child: small children to *match*, whole pages to *read*.
- Replace the Session 03 LLM re-rank with a real cross-encoder and understand retrieval as a funnel: cheap search finds ten, the expensive model orders them.
- Watch every improvement land as a number in the same table before believing it.

## Live-coding order

1. `data/eval/golden_questions.json` — read three pairs aloud; note `expected_substring`: we score whether retrieval surfaced the *answering passage*, not just the right page (a 3-page sample saturates page-level metrics instantly).
2. `backend/app/eval_retrieval.py` — `rank_of_expected()` and `evaluate()`: hit@1, hit@4, MRR in ~20 hand-rolled lines. Name-check RAGAS as the production equivalent.
3. `backend/app/eval_retrieval.py` — the `CONFIGS` dict: a retrieval *policy* is just a function; the harness doesn't care what's inside.
4. Terminal — `python -m app.eval_retrieval` against the Session 03 index (vector-only will be the only working config at this point). **A baseline number exists, ~20 minutes in — everything after must beat it.**
5. Terminal — `uv pip install rank-bm25 sentence-transformers`; re-pin `requirements.txt`.
6. `backend/app/ingest.py` — parent-child: children (300/60) index precisely, parents (whole pages) read richly; per-document delete-then-upsert keeps re-ingest idempotent even across layout changes.
7. Terminal — re-run ingest; point at "3 pages → 22 child chunks".
8. `backend/app/tools.py` — `_keyword_index()`: BM25 in memory over every child; one comment on where this lives in production (the search engine).
9. `backend/app/tools.py` — `_rrf()`: ~10 lines; why rank fusion beats score fusion (incomparable scales).
10. `backend/app/tools.py` — `search_filings()` rebuilt: overfetch children → fuse → dedupe to parents → rows carry `score` and `stage`.
11. Terminal — eval again: hybrid ≥ vector. The number moved because the code did.
12. `backend/app/tools.py` — `rerank_filings()`: the cross-encoder funnel; lazy model load; config-level swap to a BGE reranker noted.
13. `backend/app/graph.py` — `retrieve` swaps the LLM re-rank for `rerank_filings` behind the same `RERANK_ENABLED` flag; citations gain `score`/`stage`. Delete the Session 03 `Ranking` model — code that leaves is as teachable as code that arrives.
14. Terminal — eval a third time: the full gradient (vector < hybrid < re-ranked) on screen. **The table is the demo.**

## Demo script

0. `python -m app.eval_retrieval` — the table shows the gradient live (our run: MRR 0.883 → 0.900 → 0.933, hit@1 0.80 → 0.80 → 0.90).
1. **"How many ProLine-9 controllers did SampleCo ship this year?"** — keyword-heavy: BM25 puts the exact-figure passage on top where pure vectors wobble. Read the `stage` labels in the `citations` payload.
2. **"How is SampleCo protecting itself from generic AI models?"** — a paraphrase sharing almost no words with the filing; watch the re-ranker reorder the candidates (flip `RERANK_ENABLED` and re-ask to see the difference).
3. Flip `RERANK_ENABLED=false` and re-ask question 2 — the evidence order in the `citations` event visibly degrades, and the eval table's "hybrid (RRF)" row *is* that configuration's number. Point at both: evaluation as a habit, not a ceremony.

## Where it breaks

- **Every config scores 1.00 and the table teaches nothing.** Metrics are matching at page level (or the corpus is one page). Match on `expected_substring` — and remember this moment when your production eval saturates: make the golden set harder.
- **First question after Session 05 hangs ~20s.** The cross-encoder loads lazily on first use (and downloads on first *ever* use — pre-download before class). Narrate it or warm it at boot.
- **`no such column: collections.topic` or similar Chroma errors after jumping sessions.** The index layout changed in Session 05; delete `backend/chroma_db/` and re-ingest for the session you're standing in.
- **Hybrid scores *below* vector-only.** Usually a tokenization mismatch — BM25 here is `lower().split()`; if you "improved" the query pipeline on one side only, fusion happily merges apples and oranges.

## Stretch

> **Budget note:** ~281 changed application lines against the ~250 target. The named trim: the cross-encoder re-ranker (`rerank_filings` + its wiring, ~35 lines) — hybrid + the eval harness carry the session's thesis; teach the re-ranker from here if time runs short.

- Add a `--per-question` flag to the eval that prints the rank each config gave every golden question — the fastest way to find *which* questions a change broke.
- Try children of 150 and 600 chars; re-run the eval; argue from the table, not taste.
- Swap the cross-encoder for `BAAI/bge-reranker-base` (one string) and measure whether the bigger model earns its latency on your corpus.
