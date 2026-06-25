"""
backend/api/main.py
FastAPI application entry-point for Gov-Intel backend.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import get_settings
from backend.core.logging_config import setup_logging, get_logger
from backend.api.routes import search, chat, ingest, compare, scrape

setup_logging()
log = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    log.info("=== Gov-Intel Backend Starting ===")

    # Ensure Qdrant collection exists
    try:
        from backend.rag.qdrant_store import ensure_collection
        ensure_collection()
        log.info("Qdrant collection ready.")
    except Exception as e:
        log.warning("Qdrant not reachable at startup: %s", e)

    # Start background scraper scheduler
    try:
        from backend.etl.scheduler import start_scheduler
        start_scheduler()
    except Exception as e:
        log.warning("Scheduler startup failed: %s", e)

    log.info("=== Gov-Intel Backend Ready ===")
    yield

    # Shutdown
    try:
        from backend.etl.scheduler import stop_scheduler
        stop_scheduler()
    except Exception:
        pass
    log.info("=== Gov-Intel Backend Stopped ===")


app = FastAPI(
    title="Gov-Intel Regulatory Intelligence Engine",
    description=(
        "Dual-mode RAG pipeline for MoE policy documents. "
        "Mode A: Semantic Similarity Search. Mode B: RAG Chatbot. "
        "Mode C: Draft Policy Comparison."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(search.router,  prefix="/api", tags=["Search"])
app.include_router(chat.router,    prefix="/api", tags=["Chat"])
app.include_router(ingest.router,  prefix="/api", tags=["Ingest"])
app.include_router(compare.router, prefix="/api", tags=["Compare"])
app.include_router(scrape.router,  prefix="/api", tags=["Scraper"])


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check():
    from backend.rag.qdrant_store import collection_info
    info = collection_info()
    return {
        "status": "ok",
        "version": "1.0.0",
        "qdrant_connected": info.get("exists", False),
        "collection_exists": info.get("exists", False),
        "total_chunks": info.get("total_chunks", 0),
        "embedding_mode": "cloud-gemini" if settings.use_cloud_embeddings else "local-bge-m3",
        "llm_mode": f"local-ollama ({settings.local_llm_model})" if settings.use_local_llm else f"cloud-gemini ({settings.gemini_llm_model})",
    }
