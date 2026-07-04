"""Desk tools. Session 03: filings search. Session 04: live market tools.

Everything the agent can *do* (beyond talking) lives in this file.
"""

import chromadb
import yfinance as yf
from chromadb.config import Settings
from langchain_core.tools import tool

_filings = None


def _collection():
    # Lazy singleton: the Chroma client only spins up (and loads the local
    # embedding model) the first time a filings question actually arrives.
    global _filings
    if _filings is None:
        client = chromadb.PersistentClient(
            path="./chroma_db", settings=Settings(anonymized_telemetry=False)
        )
        _filings = client.get_or_create_collection("filings")
    return _filings


def search_filings(query: str, k: int = 10) -> list[dict]:
    """Dense retrieval over the ingested filings: embed the query, return the
    k nearest chunks with the metadata a citation needs."""
    if _collection().count() == 0:
        return []  # nothing ingested yet — the caller says so, honestly
    result = _collection().query(query_texts=[query], n_results=k)
    return [
        {"text": doc, "source": meta["source"], "page": meta["page"]}
        for doc, meta in zip(result["documents"][0], result["metadatas"][0])
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
