"""Week 3 — Two-stage retrieval: Chroma recall + cross-encoder rerank.

Stage 1 pulls RETRIEVE_TOP_N candidates from Chroma (fast bi-encoder).
Stage 2 rescores those with a cross-encoder and keeps RERANK_TOP_K.

The retrieve() function is the reusable entrypoint — Week 4 (chat.py) and
Week 5 (FastAPI) will both call it. Models are loaded once and cached so
repeated calls in a long-running process stay fast.

Usage:
    python -m app.retrieve "what is a system call"
    python -m app.retrieve "real time operating system" --no-rerank   # stage 1 only
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

# Silence Chroma telemetry noise before importing chromadb
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import (
    CHROMA_DIR,
    EMBED_MODEL,
    RERANK_MODEL,
    COLLECTION_NAME,
    RETRIEVE_TOP_N,
    RERANK_TOP_K,
)


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    score: float  # cosine sim (stage 1) or rerank score (stage 2)

    def cite(self) -> str:
        return f"{self.source}:{self.page}"


# --- lazy singletons so models load once per process ----------------------

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception as e:
            print(
                f"ERROR: collection '{COLLECTION_NAME}' not found. "
                f"Run `python -m app.embed` first.\n{e}",
                file=sys.stderr,
            )
            sys.exit(1)
    return _collection


# --- retrieval ------------------------------------------------------------

def _stage1(query: str, top_n: int) -> list[Chunk]:
    """Vector search in Chroma — returns top_n candidates."""
    collection = _get_collection()
    qvec = _get_embedder().encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).tolist()
    res = collection.query(
        query_embeddings=qvec,
        n_results=top_n,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for cid, doc, meta, dist in zip(
        res["ids"][0],
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        chunks.append(
            Chunk(
                id=cid,
                text=doc,
                source=meta["source"],
                page=meta["page"],
                score=1 - dist,  # cosine distance -> similarity
            )
        )
    return chunks


def _stage2(query: str, candidates: list[Chunk], top_k: int) -> list[Chunk]:
    """Rerank candidates with the cross-encoder, keep top_k."""
    if not candidates:
        return []
    pairs = [(query, c.text) for c in candidates]
    scores = _get_reranker().predict(pairs)
    for c, s in zip(candidates, scores):
        c.score = float(s)
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


def retrieve(
    query: str,
    top_n: int = RETRIEVE_TOP_N,
    top_k: int = RERANK_TOP_K,
    rerank: bool = True,
) -> list[Chunk]:
    """Main entrypoint. Returns the final ranked chunks for a query."""
    candidates = _stage1(query, top_n)
    if not rerank:
        return candidates[:top_k]
    return _stage2(query, candidates, top_k)


# --- CLI ------------------------------------------------------------------

def _print_results(query: str, chunks: list[Chunk], reranked: bool) -> None:
    label = "RERANKED" if reranked else "VECTOR-ONLY"
    print(f"\nQuery: {query!r}  [{label}]")
    print("=" * 60)
    for i, c in enumerate(chunks, start=1):
        print(f"\n[{i}] {c.cite()}  (score={c.score:.3f})")
        print("-" * 60)
        snippet = c.text[:400].replace("\n", " ")
        print(snippet + ("..." if len(c.text) > 400 else ""))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage retrieval (vector + rerank).")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=RERANK_TOP_K, help="final results to keep")
    p.add_argument("--n", type=int, default=RETRIEVE_TOP_N, help="candidates before rerank")
    p.add_argument("--no-rerank", action="store_true", help="skip stage 2 (vector only)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    do_rerank = not args.no_rerank
    results = retrieve(args.query, top_n=args.n, top_k=args.k, rerank=do_rerank)
    _print_results(args.query, results, reranked=do_rerank)
