"""
backend/rag/embedder.py
Dual-mode embedding: local BAAI/bge-m3 (dense) + SPLADE (sparse)
or Gemini Embedding API for cloud mode.

Dense vectors  → semantic meaning
Sparse vectors → exact keyword match (legal terms, section numbers)
Both are required for Qdrant hybrid search.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Dict, List, Tuple

from backend.core.config import get_settings
from backend.core.logging_config import get_logger

log = get_logger(__name__)
settings = get_settings()

# ─── Lazy singletons ─────────────────────────────────────────────────────────
_dense_model = None
_sparse_model = None
_genai_client = None


def _load_dense_model():
    global _dense_model
    if _dense_model is None:
        log.info("Loading dense embedding model: %s", settings.embedding_model)
        t0 = time.time()
        from sentence_transformers import SentenceTransformer
        _dense_model = SentenceTransformer(settings.embedding_model)
        log.info("Dense model loaded in %.1fs", time.time() - t0)
    return _dense_model


def _load_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        log.info("Loading sparse (SPLADE) model: %s", settings.sparse_model)
        t0 = time.time()
        try:
            from fastembed import SparseTextEmbedding
            _sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)
        except Exception as e:
            log.warning("SPLADE model load failed (%s) — sparse search disabled", e)
            _sparse_model = None
        if _sparse_model:
            log.info("Sparse model loaded in %.1fs", time.time() - t0)
    return _sparse_model


def _load_genai():
    global _genai_client
    if _genai_client is None:
        import google.generativeai as genai
        genai.configure(api_key=settings.google_api_key)
        _genai_client = genai
    return _genai_client


# ─── Public API ──────────────────────────────────────────────────────────────

def embed_dense(texts: List[str]) -> List[List[float]]:
    """Return dense float vectors for a list of texts."""
    if settings.use_cloud_embeddings:
        genai = _load_genai()
        vectors = []
        for text in texts:
            result = genai.embed_content(
                model=settings.embedding_model
                if "embedding" in settings.embedding_model
                else "models/embedding-001",
                content=text,
                task_type="RETRIEVAL_DOCUMENT",
            )
            vectors.append(result["embedding"])
        return vectors
    else:
        model = _load_dense_model()
        return model.encode(texts, batch_size=4, normalize_embeddings=True).tolist()


def embed_dense_query(query: str) -> List[float]:
    """Return dense vector for a single query string."""
    if settings.use_cloud_embeddings:
        genai = _load_genai()
        result = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="RETRIEVAL_QUERY",
        )
        return result["embedding"]
    else:
        model = _load_dense_model()
        return model.encode([query], normalize_embeddings=True)[0].tolist()


def embed_sparse(texts: List[str]) -> List[Dict]:
    """
    Return sparse vector dicts [{indices: [...], values: [...]}, ...].
    Returns list of None if sparse model is unavailable.
    """
    model = _load_sparse_model()
    if model is None:
        return [None] * len(texts)
    results = []
    for embedding in model.embed(texts, batch_size=4):
        results.append({
            "indices": embedding.indices.tolist(),
            "values": embedding.values.tolist(),
        })
    return results


def embed_sparse_query(query: str) -> Dict | None:
    """Return sparse vector dict for a single query or None."""
    sparse_list = embed_sparse([query])
    return sparse_list[0] if sparse_list else None


def get_vector_size() -> int:
    """Return the dense vector dimension for this model."""
    if settings.use_cloud_embeddings:
        return 768   # Gemini embedding-001
    # bge-m3 produces 1024-dim
    return 1024
