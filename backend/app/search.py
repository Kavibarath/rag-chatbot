"""Week 2 — Sanity-check semantic search.

Tiny CLI: given a query, embed it with the same model and print the top-k
most similar chunks from Chroma. No reranker, no LLM — just raw retrieval
to verify the vector store works end-to-end.

Usage:
    python -m app.search "what is a system call"
    python -m app.search "context switching" --k 3
"""
from __future__ import annotations

import argparse
import os
import sys

# Silence Chroma's telemetry noise before importing chromadb
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_DIR, EMBED_MODEL, COLLECTION_NAME


def search(query: str, k: int) -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"ERROR: collection '{COLLECTION_NAME}' not found. "
              f"Run `python -m app.embed` first.\n{e}", file=sys.stderr)
        sys.exit(1)

    model = SentenceTransformer(EMBED_MODEL)
    query_vec = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_vec,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    print(f"\nQuery: {query!r}")
    print(f"Top {len(docs)} results:\n" + "=" * 60)
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        # cosine distance -> similarity (1 - d)
        sim = 1 - dist
        print(f"\n[{i}] {meta['source']}  page {meta['page']}  (sim={sim:.3f})")
        print("-" * 60)
        snippet = doc[:400].replace("\n", " ")
        print(snippet + ("..." if len(doc) > 400 else ""))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semantic search smoke test.")
    p.add_argument("query", help="Search query")
    p.add_argument("--k", type=int, default=5, help="Number of results (default: 5)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    search(args.query, args.k)
