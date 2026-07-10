"""Session 08: guardrails — the desk's policies as CODE, not vibes.

Three rails, three failure modes they catch:
- INPUT rail (screen_input): runs before any model call. Length caps and
  injection tripwires. Regexes are tripwires, not walls — say so in class;
  their job is to catch the casual case cheaply and *visibly*.
- POLICY rail (wants_personal_advice + POLICY_RULES + disclaimer_suffix):
  "not investment advice" stops being a hope in the system prompt and
  becomes structure. The reframe changes what the model is told to do; the
  disclaimer is APPENDED BY CODE — a prompt can be talked out of a rule,
  an append cannot.
- SCHEMA rail (try_structured): structured outputs are validated at the
  model boundary with one bounded retry, so a malformed plan degrades into
  a visible event instead of a stack trace mid-demo.

NeMo Guardrails and Llama Guard are the production versions of this file.
We hand-roll first because the lesson is WHERE the rails sit, not which
library provides them.
"""

import re
from typing import Literal

from pydantic import BaseModel

MAX_INPUT_CHARS = 4000  # a question fits; a pasted prompt-injection payload rarely does

# Tripwires for the casual case. A determined paraphrase walks past regex —
# that is the demo, and the argument for defense in depth (the system prompt
# and the reframe rail behind this one still hold).
INJECTION_PATTERNS = [
    r"ignore (all |your |any |previous |prior )*(instructions|rules|guidelines)",
    r"(reveal|print|show|repeat) (your |the )?(system|developer) (prompt|message)",
    r"\bjailbreak\b",
    r"you are now\b",
    r"pretend (you are|to be)\b",
]

# Personalized advice asks. These don't block — they reframe: the desk
# teaches the evaluation instead of recommending the action.
ADVICE_PATTERNS = [
    r"should i (buy|sell|short|hold)\b",
    r"is .{1,60} a good (buy|investment|stock to buy)",
    r"what (stock|shares?) should i",
    r"how should i invest",
    r"(would|do) you recommend (buying|selling|shorting)",
]

DISCLAIMER = "\n\n_Educational analysis, not investment advice._"


class Verdict(BaseModel):
    action: Literal["pass", "block"]
    rail: str = ""
    reason: str = ""


def screen_input(text: str) -> Verdict:
    """The input rail. Cheap, deterministic, and runs before a single token
    is spent — the order is the lesson."""
    if len(text) > MAX_INPUT_CHARS:
        return Verdict(action="block", rail="input",
                       reason=f"Message exceeds {MAX_INPUT_CHARS} characters.")
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return Verdict(action="block", rail="input",
                           reason="Message matches an instruction-override pattern.")
    return Verdict(action="pass")


def wants_personal_advice(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered) for p in ADVICE_PATTERNS)


POLICY_RULES = (
    "\nPOLICY: The user is asking for personalized investment advice. You must "
    "not recommend an action (buy/sell/hold/size). Reframe educationally: lay "
    "out the relevant factors, the data a professional would weigh, and how "
    "one would evaluate the decision — then stop short of making it for them."
)


def disclaimer_suffix(text: str) -> str:
    """Returns the disclaimer iff the text doesn't already carry it — the
    append is idempotent, so the rail can run on every desk answer."""
    return "" if "not investment advice" in text.lower() else DISCLAIMER


def try_structured(model, schema, messages, attempts: int = 2):
    """The schema rail: retry a structured call once before giving up.
    Returns (result, retries_used) so the caller can announce the save."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            result = model.with_structured_output(schema).invoke(messages)
            if result is not None:
                return result, attempt
        except Exception as exc:  # validation or transport — same rail
            last_exc = exc
    raise last_exc or ValueError(f"{schema.__name__}: no valid structured output")
