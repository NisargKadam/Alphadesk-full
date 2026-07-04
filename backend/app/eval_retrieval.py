"""Session 05: the retrieval eval harness. Run: python -m app.eval_retrieval

Why this file exists: retrieval tuning without measurement is folklore. This
harness turns "hybrid feels better" into a number, so every change to
chunking, fusion, or re-ranking is judged by the same ten questions.

The metrics are hand-rolled on purpose — they are ~20 lines, and demystifying
them is the lesson. RAGAS (and friends) are the production equivalents: same
ideas, plus answer-quality metrics that need an LLM judge.
"""

import json
import sys
from pathlib import Path

from .tools import rerank_filings, search_filings

K = 4  # the graph cites 4 sources, so hit@4 answers "would the answer have seen it?"


def rank_of_expected(results: list[dict], expected: dict) -> int | None:
    """1-based rank of the first result containing the answer, else None.

    Matching on the *exact answering passage* (a substring), not just the
    right page — on a 3-page sample document, page-level matching saturates
    to 1.0 for every config and teaches nothing. Passage-level is also the
    honest question: did retrieval surface the text the answer needs?
    """
    for i, r in enumerate(results):
        if (
            r["source"] == expected["expected_source"]
            and expected["expected_substring"].lower() in r["text"].lower()
        ):
            return i + 1
    return None


def evaluate(golden: list[dict], retrieve) -> tuple[float, float, float]:
    """hit@1, hit@K, MRR over the golden set for one retrieval configuration.

    - hit@k: was the right page anywhere in the top k? (recall-flavoured —
      the generator can only cite what retrieval surfaces)
    - MRR (mean reciprocal rank): 1/rank of the first correct result,
      averaged. Rewards putting the right page *first*, which is what a
      4-source context window actually needs.
    """
    hit1 = hitk = rr = 0.0
    for item in golden:
        rank = rank_of_expected(retrieve(item["question"]), item)
        hit1 += rank == 1
        hitk += bool(rank and rank <= K)
        rr += 1 / rank if rank else 0
    n = len(golden)
    return hit1 / n, hitk / n, rr / n


# Each configuration is one retrieval *policy*, scored on the raw child
# ranking (group_parents=False): with a 3-page sample document, page-level
# results saturate instantly — child-level ranking is where configs differ.
CONFIGS = {
    "vector only": lambda q: search_filings(q, k=10, mode="vector", group_parents=False),
    "hybrid (RRF)": lambda q: search_filings(q, k=10, mode="hybrid", group_parents=False),
    "hybrid + re-rank": lambda q: rerank_filings(
        q, search_filings(q, k=10, mode="hybrid", group_parents=False), top_n=None
    ),
}


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/eval/golden_questions.json")
    golden = json.loads(path.read_text())
    print(f"{len(golden)} golden questions · k={K}\n")
    print(f"{'configuration':<20} {'hit@1':>6} {'hit@4':>6} {'MRR':>6}")
    print("-" * 42)
    for name, retrieve in CONFIGS.items():
        h1, hk, mrr = evaluate(golden, retrieve)
        print(f"{name:<20} {h1:>6.2f} {hk:>6.2f} {mrr:>6.3f}")
    print("\nRe-run with RERANK_ENABLED=false in .env to feel the difference in the app itself.")


if __name__ == "__main__":
    main()
