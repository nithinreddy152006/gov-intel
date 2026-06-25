# 🏛️ Gov-Intel — Regulatory Intelligence Engine

Dual-mode RAG + Semantic Similarity pipeline for the Ministry of Education (MoE), India.
Reasons over fragmented government regulations, gazettes, and visual scheme posters.

---

## Architecture

```
Streamlit UI (port 8501)
    │
    ▼  REST
FastAPI Backend (port 8000)
    ├── /api/search   → Mode A: Hybrid search + BGE-Reranker (no LLM)
    ├── /api/chat     → Mode B: Retrieve → Rerank → Gemini synthesis
    ├── /api/compare  → Mode C: Draft policy conflict detection
    ├── /api/ingest   → PDF upload & ingestion pipeline
    └── /api/scrape   → MoE/UGC/AICTE auto-scraper
    │
    ├── Qdrant (port 6333) — Hybrid Dense+Sparse vectors
    ├── BAAI/bge-m3        — Dense embeddings (local, 1024-dim)
    ├── SPLADE             — Sparse embeddings (keyword matching)
    ├── BGE-Reranker-v2-m3 — Cross-encoder reranking
    └── Gemini 2.5 Flash   — LLM synthesis + VLM image description
```

---

## Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Qdrant)
- Google API Key (for Gemini)

### 1. Set up environment

```bash
cp .env.example .env
# Edit .env and set your GOOGLE_API_KEY
```

### 2. Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.9.2
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ingest your PDFs

```bash
# Place PDFs in data/policies/
python scripts/ingest_local.py
```

### 5. Start the backend

```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Start the frontend

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501

---

## Full Docker Deployment

```bash
cp .env.example .env  # fill in your keys
docker-compose up --build
```

Services:
| Service  | URL                        |
|----------|----------------------------|
| Frontend | http://localhost:8501       |
| Backend  | http://localhost:8000       |
| Qdrant   | http://localhost:6333       |
| API Docs | http://localhost:8000/docs  |

---

## Operational Modes

| Mode | Endpoint | Description |
|------|----------|-------------|
| **A — Similarity Search** | `POST /api/search` | Hybrid retrieval + reranking. No LLM. Deterministic. |
| **B — RAG Chatbot** | `POST /api/chat` | Full RAG pipeline → Gemini synthesis with page citations |
| **C — Policy Comparison** | `POST /api/compare` | Upload draft PDF → conflict/overlap/duplicate detection |

---

## Enabling Local LLM (Ollama)

1. Install [Ollama](https://ollama.ai/)
2. Pull the model: `ollama pull llama3:8b-instruct-q4_K_M`
3. Set in `.env`:
   ```
   USE_LOCAL_LLM=true
   OLLAMA_BASE_URL=http://localhost:11434
   ```
4. The system will use Llama-3 for synthesis and fall back to Gemini if Ollama is unreachable.

---

## Project Structure

```
project/
├── backend/
│   ├── api/           # FastAPI routes
│   ├── core/          # Config, logging
│   ├── etl/           # Parser, chunker, ingestor, scraper, scheduler
│   ├── models/        # Pydantic schemas
│   └── rag/           # Embedder, reranker, Qdrant store, retriever, LLM
├── frontend/
│   └── app.py         # Streamlit UI (3 modes + Admin)
├── scripts/
│   ├── ingest_local.py
│   └── run_scraper.py
├── data/
│   ├── policies/      # Place PDFs here
│   └── scraped/       # Auto-downloaded PDFs
├── docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
├── docker-compose.yml
└── .env               # Your secrets (never commit)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Dense Embeddings | `BAAI/bge-m3` (1024-dim, local) |
| Sparse Embeddings | `SPLADE` via fastembed (keyword matching) |
| Vector DB | `Qdrant` (hybrid RRF fusion search) |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder |
| LLM (cloud) | `Gemini 2.5 Flash/Pro` |
| LLM (local) | `Llama-3-8B-Instruct` via Ollama |
| VLM | `Gemini Vision` (image→text for scheme posters) |
| PDF Parser | `LlamaParse` (cloud) / `PyMuPDF` (local fallback) |
| Scraper | `BeautifulSoup4` + `requests` + `APScheduler` |
| Backend | `FastAPI` + `uvicorn` |
| Frontend | `Streamlit` |
