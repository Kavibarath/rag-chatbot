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
