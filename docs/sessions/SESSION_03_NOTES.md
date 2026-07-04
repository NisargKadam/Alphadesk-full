# Session 03 — Retrieval & RAG

## Learning objectives

- Build the offline half of RAG: page-aware extraction → chunking (and its trade-offs) → embeddings → a persistent vector store.
- Build the online half: retrieve → re-rank → inject numbered sources → answer *only* from them, with inline citations.
- Understand why grounding beats parametric memory for anything checkable — Session 01's hallucination demo gets fixed on camera.
- See idempotent ingestion (content-hash IDs) and a lightweight listwise re-rank as one cheap structured model call.
- Notice what didn't change: the SSE parser, the endpoints. Evidence is just one more event type.

## Live-coding order

1. Terminal — `uv pip install chromadb pypdf langchain-text-splitters`; re-pin `requirements.txt`.
2. `data/filings/` — open the synthetic SampleCo report; point at line 1 (it announces itself as synthetic) and the `\f` page breaks.
3. `backend/app/ingest.py` — `load_pages()`: PDFs page by page via pypdf, `.txt` passes through; citations will need page numbers.
4. `backend/app/ingest.py` — the splitter: 900 chars / 150 overlap; discuss the trade-off (too small → answers fall in the gaps, too big → chunks go off-topic). Strategies get *measured* in Session 05.
5. `backend/app/ingest.py` — Chroma `PersistentClient` + default embedding function (local, keyless); content-hash IDs make re-runs idempotent.
6. Terminal — `python -m app.ingest ../data/filings`, twice. Same count both times: that's the idempotency. **~25 minutes in.**
7. `backend/app/tools.py` — `search_filings(query, k)`: embed the query, return chunks + metadata; empty-index guard.
8. `backend/app/graph.py` — `sources` joins `DeskState`; the router clears it each turn (state persists across turns — what shouldn't linger must be cleared).
9. `backend/app/graph.py` — the `retrieve` node: top-10 search, then `rerank()` down to 4 — one cheap structured call (`Ranking`), flag-gated by `RERANK_ENABLED`; cross-encoders arrive in Session 05.
10. `backend/app/graph.py` — `retrieve` emits the `citations` event via the same custom stream channel as `node` events. `main.py` doesn't change at all — dwell on that.
11. `backend/app/graph.py` — `GROUNDED_RULES` in `respond`: answer ONLY from sources, cite as [n], admit gaps; wire `filings → retrieve → respond`.
12. Terminal — curl a filings question; watch `node retrieve`, then `citations`, then cited tokens.
13. `frontend/src/lib/stream.ts` — the `Citation` type and `citations` event. The parser still doesn't change.
14. `frontend/src/hooks/useChat.ts` — store `citations` on the streaming message.
15. `frontend/src/components/EvidencePanel.tsx` — gold chips, one per source; click expands the exact passage on a spring.
16. `frontend/src/components/MessageBubble.tsx` — `CitedText` tints inline `[n]` gold; mount the panel beneath the answer.
17. `frontend/src/components/ChatWindow.tsx` + `App.tsx` — Session 03 suggestions and label.
18. Browser — run the demo script; expand every citation.

## Demo script

0. `python -m app.ingest ../data/filings` — ingest the synthetic report before anything else.
1. **"What does the annual report say about AI risk?"** — cited answer; expand [1] and show the exact passage, page number and all. The trail now reads `router → retrieve → respond`.
2. **"What is SampleCo's dividend policy?"** — numbers with sources: 40–50% payout, $1.16 per share, all traceable to page 3.
3. **"What was Apple's exact revenue in fiscal 2025?"** — Session 01's hallucination prompt, re-asked. With only SampleCo ingested, the desk now says its sources don't cover Apple — grounded honesty. Drop a real Apple 10-K into `data/filings/`, re-ingest, re-ask: the arc closes with a cited figure.

## Where it breaks

- **No citation chips, answer sounds like Session 01.** The ingest CLI never ran (or ran against an empty folder), so `search_filings` returns nothing. Run step 0; the empty-index guard in `respond` even says so.
- **First filings question hangs for ~30s.** Chroma is downloading its local embedding model on first use. It's once per machine — do it before class, or narrate it.
- **The model cites confidently from memory, no [n] markers.** The grounded system prompt didn't make it into the `respond` call (usually: `route` never set to `filings`, so the plain prompt was used). Check the router decision and the `if state["route"] == "filings"` branch.

## Stretch

> **Budget note:** ~293 changed application lines against the ~250 target. If trimming live, the re-rank pass (`Ranking` + `rerank()`, ~30 lines) is self-contained — teach it here as the first stretch item instead.

- Add `where={"source": ...}` filtering to `search_filings` and a per-document dropdown in the UI.
- Log retrieval scores (`result["distances"]`) and show the worst-kept chunk — the first taste of retrieval evaluation before Session 05 does it properly.
- Swap `RERANK_ENABLED` off and compare answers side by side on the AI-risk question; discuss when the extra model call pays for itself.
