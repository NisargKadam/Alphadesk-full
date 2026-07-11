"""Session 07: long-term memory — durable facts that outlive any one thread.

Two kinds of remembering, two mechanisms, one lesson:
- The CHECKPOINTER (graph.py) answers "where were we in THIS conversation?"
  It stores whole graph states, keyed by thread_id.
- This STORE answers "what do we know about this USER?" — watchlist, risk
  preferences — keyed by user_id, shared across every thread they ever open.

Hand-rolled on sqlite3 so the mechanism stays visible: a memory store is a
table and two queries, not magic. LangGraph ships the productized version as
`langgraph.store` (SqliteStore/PostgresStore) once you want namespacing,
TTLs, and semantic search over memories.
"""

import os
import sqlite3

from pydantic import BaseModel, Field

# One durable file for the whole desk: the checkpointer's tables and ours
# live side by side in desk_state.db. Separate connections, shared fate —
# delete the file and the desk forgets everything, which is the reset story.
# Session 10: env-overridable so the container keeps state on a volume.
DB_PATH = os.getenv(
    "DESK_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "desk_state.db")
)


def _conn() -> sqlite3.Connection:
    # check_same_thread=False: uvicorn services requests from a threadpool,
    # and sqlite's default single-thread guard predates that world. Writes
    # are tiny and serialized by sqlite's own lock.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS memories (
            user_id    TEXT NOT NULL,
            key        TEXT NOT NULL,
            value      TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, key))"""
    )
    return conn


def remember(user_id: str, key: str, value: str) -> None:
    # UPSERT: a memory is the CURRENT value of a fact, not a log of its
    # history. "risk_preference" gets overwritten, not appended.
    with _conn() as conn:
        conn.execute(
            """INSERT INTO memories (user_id, key, value) VALUES (?, ?, ?)
               ON CONFLICT(user_id, key) DO UPDATE
               SET value = excluded.value, updated_at = CURRENT_TIMESTAMP""",
            (user_id, key, value),
        )


def recall(user_id: str) -> dict[str, str]:
    """Everything the desk knows about this user, as a flat profile dict."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT key, value FROM memories WHERE user_id = ?", (user_id,)
        ).fetchall()
    return dict(rows)


class MemoryFact(BaseModel):
    key: str = Field(description="snake_case slug, e.g. 'watchlist' or 'risk_preference'")
    value: str = Field(description="The fact, stated plainly.")


class MemoryUpdate(BaseModel):
    """Structured output for the extractor. The docstring below is the rail:
    questions are not facts, and most turns should store nothing."""

    facts: list[MemoryFact] = Field(
        default_factory=list,
        description=(
            "ONLY durable user facts explicitly stated in the message: "
            "watchlist changes, standing preferences, who the user is. "
            "A question is not a fact. Curiosity is not a preference. "
            "Usually this list is EMPTY."
        ),
    )


EXTRACT_PROMPT = (
    "You maintain an equity research desk's memory of its user. From the "
    "user's latest message, extract durable facts worth keeping across "
    "conversations. Merge additions into 'watchlist' as one comma-separated "
    "value when the current watchlist is provided. If the message states no "
    "NEW fact — it merely asks a question, or restates what the current "
    "profile already says — return an empty list."
)


def extract_and_store(llm, user_id: str, text: str) -> list[dict]:
    """One cheap structured call per turn; returns [] almost every time —
    that is the correct behavior, not a failure. (The alternative — stuffing
    every message into memory — is how assistants get creepy and wrong.)"""
    profile = recall(user_id)
    context = f"Current watchlist: {profile['watchlist']}\n" if "watchlist" in profile else ""
    update = llm.with_structured_output(MemoryUpdate).invoke(
        [("system", EXTRACT_PROMPT), ("user", f"{context}Message: {text}")]
    )
    for fact in update.facts:
        remember(user_id, fact.key, fact.value)
    return [f.model_dump() for f in update.facts]
