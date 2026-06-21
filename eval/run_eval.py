"""Week 3 — Retrieval evaluation: recall@k, vector-only vs reranked.

Loads eval/questions.jsonl, runs each question through retrieve() both ways,
and reports recall@k. A question is "recalled" if any of its expected
source:page citations appears in the top-k results.

Run from the project root so the `app` package is importable:
    cd E:\\rag-chatbot
    python eval/run_eval.py
    python eval/run_eval.py --k 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the backend package importable when running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.retrieve import retrieve  # noqa: E402

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.jsonl"


def load_questions(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def recall_hit(chunks, expected: set[str]) -> bool:
    """True if any retrieved chunk's citation is in the expected set."""
    return any(c.cite() in expected for c in chunks)


def evaluate(questions: list[dict], k: int, rerank: bool) -> tuple[int, list[bool]]:
    hits = []
    for q in questions:
        expected = set(q["expected"])
        # Pull top_n candidates as configured; keep top_k for scoring.
        chunks = retrieve(q["question"], top_k=k, rerank=rerank)
        hits.append(recall_hit(chunks, expected))
    return sum(hits), hits


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate retrieval recall@k.")
    p.add_argument("--k", type=int, default=5, help="top-k cutoff for recall (default: 5)")
    args = p.parse_args()
    k = args.k

    questions = load_questions(QUESTIONS_PATH)
    n = len(questions)
    print(f"Loaded {n} eval questions. Computing recall@{k}...\n")

    vec_total, vec_hits = evaluate(questions, k, rerank=False)
    rer_total, rer_hits = evaluate(questions, k, rerank=True)

    # Per-question breakdown
    print(f"{'#':<3}{'vec':<5}{'rerank':<8}question")
    print("-" * 70)
    for i, (q, vh, rh) in enumerate(zip(questions, vec_hits, rer_hits), start=1):
        vmark = "OK " if vh else " X "
        rmark = "OK " if rh else " X "
        print(f"{i:<3}{vmark:<5}{rmark:<8}{q['question'][:50]}")

    print("\n" + "=" * 70)
    print(f"Recall@{k}  vector-only : {vec_total}/{n}  =  {vec_total / n:.1%}")
    print(f"Recall@{k}  reranked    : {rer_total}/{n}  =  {rer_total / n:.1%}")
    delta = (rer_total - vec_total) / n
    sign = "+" if delta >= 0 else ""
    print(f"Delta from reranking    : {sign}{delta:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
