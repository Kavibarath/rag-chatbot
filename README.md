# RAG Chatbot — University IT Lecture Assistant

A production-grade Retrieval-Augmented Generation chatbot that answers questions **grounded exclusively in your own course materials**. Upload lecture PDFs and PowerPoint slides; ask questions in natural language; get precise, cited answers — with an explicit refusal when the notes don't cover it.

Built end-to-end from ingestion to a streaming chat UI, with a measurable retrieval eval and a clean two-stage retrieval pipeline (semantic search + cross-encoder reranking).

---

## Demo

> **Live demo:** _coming soon (Week 6 deployment)_
>
> **Screenshots:** _add screenshots to `docs/` and link here_

---

## Why this exists

Generic LLM chatbots hallucinate — they'll confidently give you an answer even when they have no idea. That's fine for casual questions, useless for studying. This chatbot fixes both problems for course material:

1. **Grounded** — every answer is drawn from your own lecture notes, retrieved before the model generates. No outside knowledge.
2. **Honest** — when the notes don't cover a question, the model says so explicitly (`"I don't have that in the provided notes."`) instead of bluffing.
3. **Cited** — every answer shows exactly which slides/pages it was built from, so you can go verify.

---

## Features

- **Multi-format ingestion** — PDFs, DOCX, and PowerPoint (`.pptx`) including speaker notes
- **Header/footer detection** — automatically strips repeating slide headers and page counters
- **Semantic search** — 384-dim `all-MiniLM-L6-v2` embeddings stored in ChromaDB
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` reranks the top-20 candidates down to top-5 for LLM context
- **Grounded generation** — Llama 3.3 70B (via Groq) constrained by a strict system prompt with anti-hallucination guardrails
- **Explicit refusal** — canonical `"I don't have that in the provided notes."` when retrieval fails
- **Streaming responses** — Server-Sent Events push tokens to the frontend as they generate
- **Modern chat UI** — React + Vite, dark theme, conversation history (localStorage), source citation chips
- **Retrieval eval** — reproducible `recall@k` measurement comparing vector-only vs reranked retrieval

---

## Architecture

```
┌───────────────────┐        ┌──────────────────────────────────────────┐
│  React (Vite)     │        │  FastAPI + Uvicorn                       │
│  Chat UI          │  HTTP  │                                          │
│  ─────────────    │───────▶│  POST /chat  (Server-Sent Events)        │
│  Message stream   │  SSE   │  GET  /health                            │
│  Sidebar history  │        │                                          │
│  localStorage     │        │  Loads at startup:                       │
└───────────────────┘        │   • MiniLM embedder                      │
                             │   • Cross-encoder reranker               │
                             │   • ChromaDB persistent client           │
                             │   • Groq API client                      │
                             └──────────┬───────────────────────────────┘
                                        │
                    ┌───────────────────┼──────────────────┐
                    ▼                   ▼                  ▼
        ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐
        │  ChromaDB        │  │  Cross-encoder   │  │  Groq API      │
        │  (vector store)  │  │  (rerank top-20  │  │  (Llama 3.3    │
        │  876 vectors     │  │   → top-5)       │  │   70B chat)    │
        └──────────────────┘  └──────────────────┘  └────────────────┘
```

### The RAG pipeline in one paragraph

Documents are chunked into ~500-char overlapping windows, embedded with MiniLM, and stored in ChromaDB. When you ask a question: (1) the query is embedded with the same model, (2) ChromaDB returns the 20 nearest chunks by cosine similarity, (3) a cross-encoder re-scores each `(query, chunk)` pair jointly and keeps the top 5, (4) those 5 chunks are formatted into a strict prompt that instructs Llama to answer *only* from that context (or refuse), and (5) the answer streams back to the browser token-by-token via SSE, followed by citation chips showing the source pages.

---

## Tech Stack

| Layer            | Technology                                       |
| ---------------- | ------------------------------------------------ |
| Language         | Python 3.11, JavaScript (React 19)               |
| Document parsing | `pypdf`, `python-docx`, `python-pptx`            |
| Chunking         | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) |
| Embeddings       | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim, CPU) |
| Reranker         | `sentence-transformers` — `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector DB        | ChromaDB (local persistent, HNSW, cosine)        |
| LLM              | Groq API — Llama 3.3 70B Versatile               |
| Backend          | FastAPI + Uvicorn (async, SSE streaming)         |
| Frontend         | React 19 + Vite (dark theme, localStorage history) |
| Eval             | Custom recall@k script with per-question audit   |
| VCS              | Git + GitHub                                     |

---

## Retrieval Eval

Measured on a **13-question eval set** with known-answer pages, over a **876-chunk corpus** covering operating systems and web-technology lectures.

| Metric                   | Vector-only | + Reranker | Δ         |
| ------------------------ | ----------- | ---------- | --------- |
| **Recall@1**             | 92.3%       | **100.0%** | **+7.7pp** |
| **Recall@5**             | 100.0%      | 100.0%     | 0.0pp     |

The reranker's real win shows at k=1 (the model gets exactly one shot at the right chunk) — it fixes ordering errors where the bi-encoder retrieved the right page but ranked it 2nd or 3rd. Recall@5 was already at ceiling on this corpus size; expect the reranker's edge to widen as the corpus grows.

Run the eval yourself:

```bash
python eval/run_eval.py --k 1
```

---

## Project structure

```
rag-chatbot/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py       # paths, chunk sizes, model IDs, prompt template
│   │   ├── ingest.py       # PDF/DOCX/PPTX → cleaned JSONL chunks
│   │   ├── embed.py        # JSONL → ChromaDB vectors
│   │   ├── retrieve.py     # two-stage retrieval (vector + rerank)
│   │   ├── search.py       # standalone semantic-search CLI
│   │   ├── chat.py         # RAG orchestration + Groq client (streaming + non-streaming)
│   │   └── main.py         # FastAPI server (POST /chat SSE, GET /health)
│   ├── requirements.txt
│   ├── .env.example        # commit this template; real .env is gitignored
│   └── .env                # your real GROQ_API_KEY (gitignored)
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # chat UI, sidebar, SSE stream parser
│   │   ├── App.css         # dark theme, layout, animations
│   │   ├── index.css       # global reset + fonts
│   │   └── main.jsx        # React entrypoint
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── data/                   # ALL contents gitignored — reproducible from source docs
│   ├── raw/                # your PDFs + PPTX files
│   ├── chunks/             # chunks.jsonl (ingest output)
│   └── chroma/             # ChromaDB persistent files
├── eval/
│   ├── questions.jsonl     # ~15 questions with expected source:page
│   └── run_eval.py         # recall@k with vs without reranking
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python 3.11** (not 3.12+ — some ML packages don't yet ship wheels for newer versions)
- **Node.js 20+** (for the React frontend)
- A free **Groq API key** — https://console.groq.com/keys (no credit card required)

### 1. Clone and set up the backend

```bash
git clone https://github.com/Kavibarath/rag-chatbot.git
cd rag-chatbot

# Create Python 3.11 virtualenv
py -3.11 -m venv venv           # Windows
# python3.11 -m venv venv       # macOS/Linux
.\venv\Scripts\Activate.ps1     # Windows
# source venv/bin/activate      # macOS/Linux

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 2. Configure your API key

```bash
cp backend/.env.example backend/.env
# then open backend/.env and paste your key:
# GROQ_API_KEY=gsk_...
```

### 3. Add your documents

Drop your lecture PDFs and PowerPoint files into `data/raw/`. Subfolders are fine — the ingestion script walks recursively.

### 4. Ingest and embed

```bash
cd backend
python -m app.ingest        # PDF/PPTX → chunks.jsonl
python -m app.embed         # chunks.jsonl → ChromaDB vectors
```

You'll see progress per file. When done, your `data/chroma/` folder holds the persistent vector store.

### 5. Set up the frontend

```bash
cd ../frontend
npm install
```

### 6. Run the app (two terminals)

**Terminal 1 — backend:**
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Wait for `Ready. Collection 'mph_notes' has N vectors.`

**Terminal 2 — frontend:**
```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** and start asking questions.

---

## Usage

### CLI mode

You can also chat without the web UI:

```bash
cd backend
python -m app.chat "what is a system call"
```

### Standalone semantic search

To inspect what retrieval alone returns (no LLM):

```bash
python -m app.search "context switching" --k 5
```

### Compare reranked vs vector-only retrieval

```bash
python -m app.retrieve "real time operating system"                # reranked
python -m app.retrieve "real time operating system" --no-rerank    # vector-only
```

---

## How it works — a deeper look

### Ingestion (`ingest.py`)

- Walks `data/raw/` recursively for `.pdf`, `.docx`, `.pptx`
- Extracts per-page (PDF) or per-slide (PPTX) text; PPTX extraction includes **speaker notes**
- Detects lines that repeat across >50% of pages (headers, footers, page counters like `12/26`) and strips them via a normalized-line dedup pass
- Chunks with `RecursiveCharacterTextSplitter(500, overlap=50)` using natural boundaries (`\n\n`, `\n`, `. `, ` `)
- Emits JSONL with `{id, source, page, text, char_count}` per chunk

### Embedding (`embed.py`)

- Loads the JSONL and encodes every chunk with `all-MiniLM-L6-v2` (`normalize_embeddings=True` for cosine=dot)
- Upserts into a Chroma collection with `hnsw:space=cosine`
- Stores `source`, `page`, `char_count` as metadata for later citation

### Retrieval (`retrieve.py`)

Two-stage:

1. **Vector recall** — encode the query, ask Chroma for top-20 nearest chunks
2. **Cross-encoder rerank** — score each `(query, chunk_text)` pair jointly with the cross-encoder, keep top-5

Cross-encoders are much slower than bi-encoders but much more accurate at pairwise relevance — the two-stage pattern gets the best of both.

### Generation (`chat.py`)

- Formats retrieved chunks as a numbered context block with `[source:page]` tags
- Sends to Groq's chat completions API with a system prompt that:
  - Forbids external knowledge
  - Requires the exact refusal phrase for unsupported questions
  - Forbids inline citations (sources are rendered separately in the UI)
- Streams token deltas over Server-Sent Events

### Frontend (`App.jsx`)

- Parses SSE stream chunk-by-chunk (`data: {...}\n\n`), extracts three event types: `sources`, `token`, `done`
- Renders tokens as they arrive with a blinking cursor
- Displays retrieved sources as citation chips below each answer
- Persists conversations to `localStorage` with auto-generated titles from the first question
- Regex-strips any leftover inline citations for defense-in-depth

---

## Design decisions

**Why not just use LangChain end-to-end?**
Only `langchain-text-splitters` is installed. The full LangChain framework adds heavy dependencies and abstractions that obscure what's actually happening in a RAG pipeline. This project builds the RAG loop from primitives specifically so each component is inspectable.

**Why Groq + Llama and not OpenAI/Claude?**
Free tier without a credit card, extremely low latency (~500 tok/s), and Llama 3.3 70B is strong enough for grounded Q&A. The `chat.py` module is generic enough to swap for any OpenAI-compatible endpoint with ~10 lines changed.

**Why Server-Sent Events instead of WebSockets?**
SSE is one-way (server → client), which is exactly what streaming a chat completion needs. It's simpler than WebSockets, works over plain HTTP, and browsers support it natively. No extra library needed.

**Why localStorage for chat history and not a database?**
Zero backend state means zero database to provision, secure, back up, or migrate. Perfect for a free-tier deployment. Each user's history stays private to their own browser. Trade-off: history doesn't sync across devices — an acceptable limitation for a portfolio demo.

---

## Roadmap

- [ ] Docker + `docker-compose.yml` for one-command deployment
- [ ] Deploy backend to Render (free tier)
- [ ] Deploy frontend to Vercel (free tier)
- [ ] Add live demo link + screenshots to this README
- [ ] Screenshot / demo GIF
- [ ] Optional: switch to Ollama for a fully-local LLM path
- [ ] Optional: multi-user auth + server-side chat history sync

---

## Author

**Kavibarath (Sitharthan Kavibarath)** — [GitHub](https://github.com/Kavibarath)

Feedback and issues welcome.
