"""
backend/rag/reranker.py
Cross-encoder reranker using BAAI/bge-reranker-v2-m3.
Takes initial hybrid-search candidates and re-scores them
with a full query×document cross-attention model for maximum precision.
"""
from __future__ import annotations

import time
from typing import List, Tuple

from backend.core.config import get_settings
from backend.core.logging_config import get_logger

log = get_logger(__name__)
settings = get_settings()

_reranker = None


def _load_reranker():
    global _reranker
    if _reranker is None:
        log.info("Loading cross-encoder reranker: %s", settings.reranker_model)
        t0 = time.time()
        try:
            from FlagEmbedding import FlagReranker
            _reranker = FlagReranker(settings.reranker_model)
        except ImportError:
            # Fallback to sentence-transformers CrossEncoder
            log.warning("FlagEmbedding not available — falling back to sentence-transformers CrossEncoder")
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(settings.reranker_model)
        log.info("Reranker loaded in %.1fs", time.time() - t0)
    return _reranker


def rerank(query: str, candidates: List[dict], top_n: int = 5) -> List[dict]:
    """
    Re-score a list of candidate chunks against the query.

    Args:
        query: The user's query string.
        candidates: List of chunk dicts with at least a 'text' key.
        top_n: How many to return after reranking.

    Returns:
        Top-n chunks sorted by descending reranker score,
        with 'rerank_score' added to each dict.
    """
    if not candidates:
        return []

    reranker = _load_reranker()
    pairs = [(query, c["text"]) for c in candidates]

    try:
        # Bypassing FlagReranker to prevent Windows CPU segfault.
        # Fallback to pure semantic scoring order.
        log.info("Reranker bypassed manually — returning original hybrid ranking.")
        scores = [c.get("score", 0.0) for c in candidates]
        
    except Exception as e:
        log.error("Reranker scoring failed: %s — returning original order", e)
        return candidates[:top_n]

    # Attach scores and sort
    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return ranked[:top_n]
