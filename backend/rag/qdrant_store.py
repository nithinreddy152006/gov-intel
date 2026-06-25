"""
backend/rag/qdrant_store.py
Singleton Qdrant client + collection management for Gov-Intel.

Hybrid search is implemented as:
  1. Dense search (cosine, BAAI/bge-m3)
  2. Sparse search (SPLADE) — skipped gracefully if unavailable
  3. Reciprocal Rank Fusion (RRF) to merge both result lists
     → identical result quality to server-side fusion, fully client-side.

This approach works with all qdrant-client versions >= 1.5.x.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
    NamedVector,
    NamedSparseVector,
    Filter,
    FieldCondition,
    MatchValue,
)

from backend.core.config import get_settings
from backend.core.logging_config import get_logger
from backend.rag.embedder import get_vector_size

log = get_logger(__name__)
settings = get_settings()

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

_client: Optional[QdrantClient] = None


# ─── Client singleton ─────────────────────────────────────────────────────────

def get_client() -> QdrantClient:
    """Return a singleton QdrantClient."""
    global _client
    if _client is None:
        log.info("Connecting to Qdrant at %s", settings.qdrant_url)
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=60,
        )
        log.info("Qdrant client connected.")
    return _client


# ─── Collection management ────────────────────────────────────────────────────

def ensure_collection() -> None:
    """
    Create the gov_policies collection if it doesn't already exist.
    Uses named vectors: 'dense' (cosine) + 'sparse' (SPLADE).
    """
    client = get_client()
    col = settings.qdrant_collection_name

    existing = [c.name for c in client.get_collections().collections]
    if col in existing:
        log.info("Collection '%s' already exists — skipping creation.", col)
        return

    dim = get_vector_size()
    log.info("Creating collection '%s' (dense_dim=%d) ...", col, dim)

    client.create_collection(
        collection_name=col,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=dim,
                distance=Distance.COSINE,
                on_disk=False,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    log.info("Collection '%s' created successfully.", col)


# ─── Upsert ───────────────────────────────────────────────────────────────────

def upsert_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Embed and upsert a list of chunk dicts into Qdrant.

    Each chunk must have:
      - text (str)
      - metadata (dict): file_name, page_number, chunk_index, doc_id, url
    """
    if not chunks:
        return 0

    from backend.rag.embedder import embed_dense, embed_sparse

    texts = [c["text"] for c in chunks]
    dense_vecs = embed_dense(texts)
    sparse_vecs = embed_sparse(texts)

    points = []
    for chunk, d_vec, s_vec in zip(chunks, dense_vecs, sparse_vecs):
        point_id = chunk.get("chunk_id") or str(uuid.uuid4())

        named_vectors: Dict[str, Any] = {DENSE_VECTOR_NAME: d_vec}

        # Attach sparse vector only when available
        if s_vec is not None:
            named_vectors[SPARSE_VECTOR_NAME] = SparseVector(
                indices=s_vec["indices"],
                values=s_vec["values"],
            )

        payload = {
            "text": chunk["text"],
            "file_name": chunk["metadata"].get("file_name", ""),
            "page_number": chunk["metadata"].get("page_number", ""),
            "chunk_index": chunk["metadata"].get("chunk_index", 0),
            "doc_id": chunk["metadata"].get("doc_id", ""),
            "url": chunk["metadata"].get("url", ""),
        }

        points.append(
            PointStruct(id=point_id, vector=named_vectors, payload=payload)
        )

    client = get_client()
    client.upsert(
        collection_name=settings.qdrant_collection_name,
        points=points,
        wait=True,
    )
    log.info("Upserted %d points into '%s'.", len(points), settings.qdrant_collection_name)
    return len(points)


# ─── RRF Fusion (client-side) ─────────────────────────────────────────────────

def _rrf_merge(
    dense_hits: List[Any],
    sparse_hits: List[Any],
    k: int = 60,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion of two ranked lists.

    RRF score = 1/(k + rank_dense) + 1/(k + rank_sparse)
    where ranks are 1-based. Missing results get no contribution.

    k=60 is the standard default (Robertson & Zaragoza 2009).
    """
    scores: Dict[str, float] = {}
    payloads: Dict[str, Dict[str, Any]] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        pid = str(hit.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        payload = dict(hit.payload)
        payload["chunk_id"] = pid
        payloads[pid] = payload

    for rank, hit in enumerate(sparse_hits, start=1):
        pid = str(hit.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        if pid not in payloads:
            payload = dict(hit.payload)
            payload["chunk_id"] = pid
            payloads[pid] = payload

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for pid in sorted_ids[:top_n]:
        item = dict(payloads[pid])
        item["score"] = round(scores[pid], 6)
        results.append(item)

    return results


def _dense_only_results(hits: List[Any]) -> List[Dict[str, Any]]:
    """Convert dense-only hits to payload dicts."""
    results = []
    for hit in hits:
        payload = dict(hit.payload)
        payload["chunk_id"] = str(hit.id)
        payload["score"] = round(hit.score, 6)
        results.append(payload)
    return results


# ─── Public search API ────────────────────────────────────────────────────────

def hybrid_search(
    dense_vec: List[float],
    sparse_vec: Optional[Dict],
    top_k: int = 20,
    filter_doc_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid (dense + sparse) search using client-side RRF fusion.
    Falls back to pure dense if sparse vector/results are unavailable.

    Args:
        dense_vec:     Dense query vector.
        sparse_vec:    Sparse query dict {indices: [...], values: [...]} or None.
        top_k:         Number of results to return.
        filter_doc_id: Optional doc_id to restrict results to.

    Returns:
        List of payload dicts with 'score' and 'chunk_id' attached.
    """
    client = get_client()
    col = settings.qdrant_collection_name

    query_filter = None
    if filter_doc_id:
        query_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=filter_doc_id))]
        )

    fetch_limit = top_k * 2  # over-fetch for better RRF quality

    # ── Dense search ──────────────────────────────────────────────────────────
    dense_hits = client.search(
        collection_name=col,
        query_vector=NamedVector(name=DENSE_VECTOR_NAME, vector=dense_vec),
        limit=fetch_limit,
        query_filter=query_filter,
        with_payload=True,
    )

    # ── Sparse search (optional) ──────────────────────────────────────────────
    sparse_hits = []
    if sparse_vec is not None:
        try:
            sparse_hits = client.search(
                collection_name=col,
                query_vector=NamedSparseVector(
                    name=SPARSE_VECTOR_NAME,
                    vector=SparseVector(
                        indices=sparse_vec["indices"],
                        values=sparse_vec["values"],
                    ),
                ),
                limit=fetch_limit,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as e:
            log.warning("Sparse search failed (%s) — using dense only.", e)
            sparse_hits = []

    # ── Merge ──────────────────────────────────────────────────────────────────
    if sparse_hits:
        log.debug(
            "Hybrid RRF: dense=%d results, sparse=%d results",
            len(dense_hits), len(sparse_hits),
        )
        return _rrf_merge(dense_hits, sparse_hits, top_n=top_k)
    else:
        log.debug("Dense-only search: %d results", len(dense_hits))
        return _dense_only_results(dense_hits[:top_k])


# ─── Utility helpers ──────────────────────────────────────────────────────────

def doc_exists(doc_id: str) -> bool:
    """Check if any chunk with the given doc_id exists in the collection."""
    client = get_client()
    results, _ = client.scroll(
        collection_name=settings.qdrant_collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(results) > 0


def collection_info() -> Dict[str, Any]:
    """Return basic stats about the collection."""
    client = get_client()
    try:
        info = client.get_collection(settings.qdrant_collection_name)
        return {
            "exists": True,
            "total_chunks": info.points_count,
            "status": str(info.status),
        }
    except Exception:
        return {"exists": False, "total_chunks": 0, "status": "not_found"}
