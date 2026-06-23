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

# ---------------------------------------------------------------------------
# Week 4 — LLM (Gemini)
# ---------------------------------------------------------------------------

# Google Gemini model. 2.0 Flash: fast, free tier, 1M context window.
# Swap to "gemini-2.5-pro" later if you want higher quality (free tier slower).
GEMINI_MODEL = "gemini-2.0-flash"

# Cap the answer length so we don't spend tokens on rambling responses.
MAX_OUTPUT_TOKENS = 1024

# The system prompt — the anti-hallucination guardrail lives here.
SYSTEM_PROMPT = """You are a study assistant for the user's university IT coursework
(operating systems, system administration, and related topics).
Answer the user's question using ONLY the provided context from their lecture notes.

Rules:
1. If the answer isn't fully supported by the context, say exactly:
   "I don't have that in the provided notes."
   Then optionally suggest what the user might search for instead.
2. Cite every factual claim with [source:page] using the exact source filename
   and page number from the context. Multiple citations are fine: [a.pdf:3][b.pdf:7].
3. Be concise. Answer in 2-5 sentences unless the question explicitly needs more.
4. Do not invent facts, page numbers, or sources. Do not use outside knowledge.
"""

# The user-turn template — context + question, formatted for the model.
USER_PROMPT_TEMPLATE = """Context (top retrieved chunks from the notes):
{context}

Question: {question}

Answer (with [source:page] citations):"""


