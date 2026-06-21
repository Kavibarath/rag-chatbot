"""Central config for the RAG chatbot.

All paths are resolved relative to the project root (E:\\rag-chatbot\\),
so scripts work whether you run them from backend/ or the project root.
"""
from pathlib import Path

# Project root = two levels up from this file (backend/app/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
CHROMA_DIR = DATA_DIR / "chroma"

# Week 1 output
CHUNKS_PATH = CHUNKS_DIR / "chunks.jsonl"

# Chunking parameters (characters, not tokens)
# 500 chars ~ 125 tokens for English — good for MiniLM (max 256 tokens).
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Drop pages with less than this many chars after cleaning (likely blank/title pages)
MIN_PAGE_CHARS = 50

# If a line appears on more than this fraction of pages, treat it as header/footer noise
HEADER_FOOTER_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Week 2 — Embeddings & Vector Store
# ---------------------------------------------------------------------------

# Sentence-transformers model. 384 dims, CPU-fast, free, max 256 input tokens.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chroma collection name
COLLECTION_NAME = "mph_notes"

# Number of chunks encoded per forward pass — bigger = faster, more RAM.
# 64 is safe on a typical laptop. Bump to 128/256 if you have headroom.
EMBED_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Week 3 — Reranking
# ---------------------------------------------------------------------------

# Cross-encoder reranker. Scores (query, chunk) pairs jointly — slow but precise.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Stage 1: how many candidates to pull from Chroma before reranking.
RETRIEVE_TOP_N = 20

# Stage 2: how many to keep after reranking — this is what feeds the LLM.
RERANK_TOP_K = 5

