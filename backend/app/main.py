"""Week 5 — FastAPI server.

Wraps the RAG pipeline as a streaming HTTP API.

Endpoints:
    GET  /health  -> {"status": "ok", "vectors": <count>}
    POST /chat    -> Server-Sent Events stream of {sources, tokens..., done}

Run:
    cd backend
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Load .env before anything imports the Groq SDK
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_PATH)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.chat import stream_answer
from app.config import COLLECTION_NAME, CHROMA_DIR, RERANK_TOP_K
from app.retrieve import _get_collection, _get_embedder, _get_reranker


# ---------------------------------------------------------------------------
# Startup: pre-load models so the first /chat request is fast
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming up models...")
    _get_embedder()
    _get_reranker()
    _get_collection()
    print(f"Ready. Collection '{COLLECTION_NAME}' has {_get_collection().count()} vectors.")
    yield


app = FastAPI(
    title="RAG Chatbot API",
    description="Grounded Q&A over university IT lecture notes.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server (5173) and Vercel deploys (Week 6)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # Add your deployed frontend origin in Week 6, e.g. "https://your-app.vercel.app"
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=RERANK_TOP_K, ge=1, le=20)
    rerank: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "vectors": _get_collection().count(),
        "collection": COLLECTION_NAME,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """Stream the grounded answer as Server-Sent Events."""

    def event_stream():
        for event in stream_answer(req.question, top_k=req.top_k, rerank=req.rerank):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx, Render)
        },
    )
