"""Session 09: observability — the pipeline grows a price tag and a tracer.

Two concerns, deliberately separate from the graph:

- COST: every LangChain model response carries `usage_metadata`; a
  UsageMetadataCallbackHandler (created per run in main.py) aggregates it
  per model, and this module prices the totals so every stream can end
  with one `cost` event. Hand-rolled on purpose — the arithmetic is the
  lesson.
- TRACES: OpikTracer ships the same run to a locally hosted Opik instance
  as a full trace tree. Env-gated and failure-wrapped: observability that
  can take down the desk is worse than no observability at all.
  (LangSmith and Langfuse are the managed equivalents of this file.)
"""

import os
import logging

LOGGER = logging.getLogger(__name__)

# USD per 1M tokens (input, output) — OpenAI list prices. A model missing
# from this table still streams its token counts; the usd field goes null
# rather than guessing.
# The estimate also ignores cached-input discounts (usage_metadata's
# input_token_details.cache_read) — close enough for a price tag, not a bill.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
}


def _price_for(model: str) -> tuple[float, float] | None:
    # OpenAI reports usage under the resolved snapshot name
    # ("gpt-4.1-mini-2025-04-14"); the table keys the bare alias.
    for known, prices in PRICES_PER_MTOK.items():
        if model == known or model.startswith(known + "-"):
            return prices
    return None


def make_tracer():
    """The env-gated, failure-wrapped Opik tracer — or None.

    The import lives inside the function: a missing or broken opik install
    must not break main.py at import time, and OPIK_TRACING=false must not
    pay opik's import cost at all.
    """
    if os.getenv("OPIK_TRACING", "false").lower() != "true":
        return None
    try:
        from opik.integrations.langchain import OpikTracer

        return OpikTracer(project_name=os.getenv("OPIK_PROJECT_NAME", "alphadesk"))
    except Exception:
        return None  # tracing is optional; the desk is not


def cost_payload(usage_by_model: dict) -> dict:
    """Fold a run's per-model usage into the `cost` event payload.

    Input shape (UsageMetadataCallbackHandler.usage_metadata):
        {"gpt-4.1-mini": {"input_tokens": 2841, "output_tokens": 412, ...}}
    """
    input_tokens = sum(u.get("input_tokens", 0) for u in usage_by_model.values())
    output_tokens = sum(u.get("output_tokens", 0) for u in usage_by_model.values())
    usd, priced = 0.0, True
    for model, usage in usage_by_model.items():
        prices = _price_for(model)
        if prices is None:
            priced = False
            LOGGER.warning(
                "No price for model %r — the cost event will carry usd: null", model
            )
            continue
        usd += usage.get("input_tokens", 0) / 1e6 * prices[0]
        usd += usage.get("output_tokens", 0) / 1e6 * prices[1]
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": round(usd, 6) if priced else None,
        "by_model": {
            model: {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }
            for model, usage in usage_by_model.items()
        },
    }
