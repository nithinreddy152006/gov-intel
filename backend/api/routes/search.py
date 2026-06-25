"""
backend/api/routes/search.py
Mode A — Semantic Similarity Search endpoint.
No LLM involved. Returns reranked document chunks with citations.
"""
from fastapi import APIRouter, HTTPException
from backend.models.schemas import SearchRequest, SearchResponse
from backend.rag.retriever import similarity_search
from backend.core.logging_config import get_logger

log = get_logger(__name__)
router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def api_search(request: SearchRequest):
    """
    Mode A: Semantic Similarity Search.
    Embeds the query, performs hybrid dense+sparse search,
    reranks with BGE cross-encoder, returns top-k chunk citations.
    No LLM synthesis — deterministic retrieval only.
    """
    log.info("Search request: '%s...' (top_k=%d)", request.query[:60], request.top_k)
    try:
        results = similarity_search(request.query, top_k=request.top_k)
        return SearchResponse(
            query=request.query,
            results=results,
            total_found=len(results),
        )
    except Exception as e:
        log.error("Search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
