"""Week 2 — Embed chunks and upsert them into ChromaDB.

Reads data/chunks/chunks.jsonl, encodes every chunk with all-MiniLM-L6-v2,
and stores them in a persistent Chroma collection at data/chroma/.

Usage:
    python -m app.embed                  # default paths, resets the collection
    python -m app.embed --no-reset       # add to existing collection (won't dedupe)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import (
    CHUNKS_PATH,
    CHROMA_DIR,
    EMBED_MODEL,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
)


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: chunks file not found: {path}", file=sys.stderr)
        print("Run `python -m app.ingest` first.", file=sys.stderr)
        sys.exit(1)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ! skipping malformed line {line_no}: {e}", file=sys.stderr)
    return rows


def get_collection(client: chromadb.PersistentClient, reset: bool):
    """Return the mph_notes collection, optionally wiping it first."""
    existing = [c.name for c in client.list_collections()]
    if reset and COLLECTION_NAME in existing:
        print(f"Resetting existing collection '{COLLECTION_NAME}'")
        client.delete_collection(COLLECTION_NAME)
    # cosine similarity matches what most sentence-transformers models are trained for
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run(chunks_path: Path, chroma_dir: Path, reset: bool) -> None:
    chunks = load_chunks(chunks_path)
    if not chunks:
        print("No chunks to embed. Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    print(f"Loading model: {EMBED_MODEL}")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL)
    print(f"  model ready in {time.time() - t0:.1f}s")

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = get_collection(client, reset=reset)

    total = len(chunks)
    done = 0
    t_embed = 0.0
    for batch in batched(chunks, EMBED_BATCH_SIZE):
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [
            {"source": c["source"], "page": c["page"], "char_count": c["char_count"]}
            for c in batch
        ]

        t1 = time.time()
        embeddings = model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine + L2-normalized = dot product
        )
        t_embed += time.time() - t1

        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )

        done += len(batch)
        print(f"  {done}/{total} embedded")

    final_count = collection.count()
    print(
        f"\nDone. Collection '{COLLECTION_NAME}' has {final_count} vectors. "
        f"Total embed time: {t_embed:.1f}s. Persisted to {chroma_dir}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed chunks into ChromaDB.")
    p.add_argument("chunks_path", nargs="?", default=str(CHUNKS_PATH))
    p.add_argument("chroma_dir", nargs="?", default=str(CHROMA_DIR))
    p.add_argument(
        "--no-reset",
        action="store_true",
        help="Don't drop the existing collection (default: drop + recreate)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.chunks_path), Path(args.chroma_dir), reset=not args.no_reset)
