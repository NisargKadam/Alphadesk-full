"""Desk tools. Session 03: filings search. Session 04: live market tools.
Session 05: the search becomes hybrid (BM25 + vectors, RRF-fused), returns
parent pages for matched children, and re-ranks with a cross-encoder.

Everything the agent can *do* (beyond talking) lives in this file.
"""

import chromadb
import yfinance as yf
from chromadb.config import Settings
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi

_children = None
_parents = None
_bm25 = None
_child_index = None  # id → {text, source, page, parent_id}
_cross_encoder = None


def _collections():
    # Lazy singleton: Chroma only spins up (and loads its local embedding
    # model) the first time a filings question actually arrives.
    global _children, _parents
    if _children is None:
        client = chromadb.PersistentClient(
            path="./chroma_db", settings=Settings(anonymized_telemetry=False)
        )
        _children = client.get_or_create_collection("filings")
        _parents = client.get_or_create_collection("filings_parents")
    return _children, _parents


def _keyword_index():
    # An in-memory BM25 over every child chunk — a second, *lexical* view of
    # the same corpus, built lazily once per process. Production pushes this
    # into the search engine (OpenSearch, Vespa, pg trigram); the concept,
    # not the deployment, is what's taught here.
    global _bm25, _child_index
    if _bm25 is None:
        children, _ = _collections()
        rows = children.get(include=["documents", "metadatas"])
        _child_index = {
            cid: {"text": doc, **meta}
            for cid, doc, meta in zip(rows["ids"], rows["documents"], rows["metadatas"])
        }
        _bm25 = (BM25Okapi([doc.lower().split() for doc in rows["documents"]]), rows["ids"])
    return _bm25, _child_index


def _rrf(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    # Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank + 1) across lists.
    # Keyword and semantic retrieval fail differently — BM25 nails exact
    # figures and product names, embeddings nail paraphrase — and RRF merges
    # them on *rank alone*, sidestepping their incomparable score scales.
    # k=60 comes from the original RRF paper and is rarely worth tuning.
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def search_filings(query: str, k: int = 10, mode: str = "hybrid", group_parents: bool = True) -> list[dict]:
    """Retrieve filing passages: match on small child chunks, return their
    parent pages. mode='vector' skips BM25 (kept for the eval harness A/B).
    group_parents=False returns the raw child ranking — what the eval scores."""
    children, parents = _collections()
    if children.count() == 0:
        return []  # nothing ingested yet — the caller says so, honestly

    pool = max(k * 2, 20)  # overfetch: parent-dedup and fusion both shrink the list
    res = children.query(query_texts=[query], n_results=min(pool, children.count()))
    vector_ids = res["ids"][0]

    if mode == "hybrid":
        (bm25, all_ids), _ = _keyword_index()
        keyword_scores = bm25.get_scores(query.lower().split())
        ranked_kw = sorted(zip(all_ids, keyword_scores), key=lambda p: -p[1])[:pool]
        keyword_ids = [cid for cid, s in ranked_kw if s > 0]
        fused = _rrf([vector_ids, keyword_ids])
    else:
        fused = _rrf([vector_ids])

    _, child_index = _keyword_index()
    ranked = sorted(fused, key=lambda cid: -fused[cid])

    rows, seen_parents = [], set()
    for cid in ranked:
        child = child_index.get(cid)
        if child is None:
            continue
        if group_parents:
            if child["parent_id"] in seen_parents:
                continue  # the best child already represents this page
            seen_parents.add(child["parent_id"])
            text = parents.get(ids=[child["parent_id"]])["documents"][0]
        else:
            text = child["text"]
        rows.append(
            {
                "text": text,  # the model reads the whole page…
                "snippet": child["text"],  # …the citation shows the matched needle
                "source": child["source"],
                "page": child["page"],
                "score": round(fused[cid], 4),
                "stage": mode,
            }
        )
        if len(rows) == k:
            break
    return rows


def rerank_filings(query: str, hits: list[dict], top_n: int | None = 4) -> list[dict]:
    """Order hits with a cross-encoder — the proper version of Session 03's
    LLM re-rank, behind the same RERANK_ENABLED flag."""
    if not hits:
        return hits
    # A cross-encoder reads query and passage *together* in one transformer
    # pass — far more accurate than comparing two independently-computed
    # embeddings, and far too slow to run over a whole corpus. Retrieval is
    # therefore a funnel: cheap search finds ~10, this orders them properly.
    # Production swap is config-level: e.g. 'BAAI/bge-reranker-base'.
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder  # heavy import — lazy on purpose

        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = _cross_encoder.predict([(query, h["snippet"]) for h in hits])
    ranked = sorted(zip(hits, scores), key=lambda pair: -pair[1])
    return [
        {**h, "score": round(float(s), 2), "stage": "reranked"}
        for h, s in (ranked[:top_n] if top_n else ranked)
    ]


# --- Session 04: live market tools -----------------------------------------
# @tool turns a plain function into a schema the model can request calls
# against — the docstring and type hints ARE the interface the model reads.
# Every tool returns a typed error dict instead of raising: an uncaught
# exception would kill the whole graph run, while an {"error": ...} payload
# lets the model apologize, retry, or route around the failure.


@tool
def get_quote(ticker: str) -> dict:
    """Live quote for a stock ticker: current price, currency, and the day's
    absolute and percent change."""
    try:
        info = yf.Ticker(ticker).fast_info
        price, prev = info["lastPrice"], info["previousClose"]
        return {
            "ticker": ticker.upper(),
            "price": round(price, 2),
            "currency": info["currency"],
            "change": round(price - prev, 2),
            "change_pct": round((price - prev) / prev * 100, 2),
        }
    except Exception as exc:
        return {"error": f"Quote lookup failed for '{ticker}': {exc}"}


@tool
def get_price_history(ticker: str, period: str = "1mo") -> dict:
    """Daily closing prices for a ticker over a period ('5d', '1mo', '3mo',
    '6mo', '1y'). Use for trends and comparisons."""
    try:
        closes = yf.Ticker(ticker).history(period=period)["Close"]
        if closes.empty:
            return {"error": f"No price history for '{ticker}' over {period}."}
        step = max(1, len(closes) // 30)  # ≤30 points — plenty for a sparkline
        return {
            "ticker": ticker.upper(),
            "period": period,
            "closes": [round(c, 2) for c in closes.tolist()[::step]],
        }
    except Exception as exc:
        return {"error": f"History lookup failed for '{ticker}': {exc}"}


@tool
def get_company_news(ticker: str, limit: int = 5) -> dict:
    """Recent news headlines for a ticker."""
    try:
        headlines = []
        for item in yf.Ticker(ticker).news[:limit]:
            # Yahoo's news schema shifts under yfinance regularly — reach for
            # every field defensively rather than trusting today's shape.
            content = item.get("content", item) or {}
            provider = content.get("provider") or {}
            headlines.append(
                {
                    "title": content.get("title", "(untitled)"),
                    "publisher": provider.get("displayName", "unknown") if isinstance(provider, dict) else str(provider),
                    "date": (content.get("pubDate") or "")[:10],
                }
            )
        return {"ticker": ticker.upper(), "headlines": headlines}
    except Exception as exc:
        return {"error": f"News lookup failed for '{ticker}': {exc}"}


MARKET_TOOLS = [get_quote, get_price_history, get_company_news]
