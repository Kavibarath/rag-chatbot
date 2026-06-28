"""Week 4 — Grounded chat over the IT lecture notes.

Pipeline: question -> retrieve top-K chunks -> format prompt -> LLM -> answer.

The answer is constrained by SYSTEM_PROMPT to:
  - cite [source:page] for every claim
  - say "I don't have that in the provided notes." if not supported

Usage:
    python -m app.chat "what is a system call"
    python -m app.chat "compare AMP and SMP" --k 5
    python -m app.chat "what is a system call" --no-rerank   # vector-only retrieval
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Load .env BEFORE importing the SDK so GROQ_API_KEY is in os.environ
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_PATH)

from groq import Groq

from app.config import (
    LLM_MODEL,
    MAX_OUTPUT_TOKENS,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    RERANK_TOP_K,
)
from app.retrieve import retrieve, Chunk


# --- API key setup --------------------------------------------------------

def _get_client() -> Groq:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key or key == "your-key-here":
        print(
            "ERROR: GROQ_API_KEY is not set.\n"
            "Check that backend/.env exists and contains your real key:\n"
            "    GROQ_API_KEY=gsk_...\n"
            "Get one (free) at https://console.groq.com/keys",
            file=sys.stderr,
        )
        sys.exit(1)
    return Groq(api_key=key)


# --- Prompt assembly ------------------------------------------------------

def format_context(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as a numbered block the model can cite from."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] {c.cite()}\n{c.text}")
    return "\n\n".join(parts)


# --- The actual chat call -------------------------------------------------

def answer(question: str, top_k: int = RERANK_TOP_K, rerank: bool = True) -> dict:
    """Run the full RAG pipeline. Returns {'answer', 'sources'}."""
    chunks = retrieve(question, top_k=top_k, rerank=rerank)
    if not chunks:
        return {
            "answer": "I don't have that in the provided notes.",
            "sources": [],
        }

    context = format_context(chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,  # low temp = less invention, more faithful citations
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = (response.choices[0].message.content or "").strip()

    return {
        "answer": text,
        "sources": [{"cite": c.cite(), "score": round(c.score, 3)} for c in chunks],
    }


# --- CLI ------------------------------------------------------------------

def _print_result(question: str, result: dict) -> None:
    print(f"\nQuestion: {question}")
    print("=" * 60)
    print(result["answer"])
    print("\n" + "-" * 60)
    print("Retrieved chunks (top-k):")
    for s in result["sources"]:
        print(f"  - {s['cite']}  (score={s['score']})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grounded chat over your lecture notes.")
    p.add_argument("question")
    p.add_argument("--k", type=int, default=RERANK_TOP_K, help="chunks to feed the LLM")
    p.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder rerank stage")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = answer(args.question, top_k=args.k, rerank=not args.no_rerank)
    _print_result(args.question, result)
