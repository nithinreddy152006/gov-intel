"""
backend/rag/retriever.py
High-level retrieval functions used by the API routes.

Both modes use the same pipeline:
  Embed query (dense + sparse) → Qdrant hybrid search → BGE-Reranker → return results
"""
from __future__ import annotations

from typing import List, Optional

from backend.core.config import get_settings
from backend.core.logging_config import get_logger
from backend.models.schemas import SourceCitation, ChatResponse
from backend.rag.embedder import embed_dense_query, embed_sparse_query
from backend.rag.qdrant_store import hybrid_search
from backend.rag.reranker import rerank
from backend.etl.graph import get_kg
import json
import re

log = get_logger(__name__)
settings = get_settings()


def _retrieve_chunks(query: str, top_k: int) -> List[dict]:
    """
    Multi-Layer Retrieval Pipeline:
      Layer 1: Hybrid Vector Search (Dense + Sparse)
      Layer 2: Knowledge Graph Traversal (Entities/Relations)
      Layer 3: Cross-Encoder Reranking
    """
    fetch_k = top_k * settings.retrieval_multiplier

    # Layer 1: Hybrid Search
    dense_vec = embed_dense_query(query)
    sparse_vec = embed_sparse_query(query)
    candidates = hybrid_search(dense_vec, sparse_vec, top_k=fetch_k)
    log.info("Hybrid search returned %d candidates.", len(candidates))

    # Layer 2: Knowledge Graph Lookup
    kg = get_kg()
    # Simple entity extraction: look for Capitalized words or acronyms
    entities = list(set(re.findall(r'\b[A-Z][a-zA-Z]{2,}\b|\b[A-Z]{2,}\b', query)))
    if entities:
        graph_relations = kg.query_graph(entities, max_hops=1)
        if graph_relations:
            log.info("Graph-Search found %d semantic relations.", len(graph_relations))
            graph_chunk = {
                "chunk_id": "graph_knowledge",
                "text": "The following document relationships were discovered in the Knowledge Graph:\n" + "\n".join(graph_relations[:10]),
                "file_name": "Knowledge Graph",
                "page_number": "N/A",
                "score": 0.9,
                "url": None
            }
            candidates.insert(0, graph_chunk)

    # Layer 3: Reranking
    reranked = rerank(query, candidates, top_n=top_k)
    log.info("Reranked to %d results.", len(reranked))

    return reranked


def similarity_search(query: str, top_k: int = 5) -> List[SourceCitation]:
    """
    Mode A — Semantic Similarity Search (no LLM).
    Returns ranked document chunks with page-level citations.
    """
    chunks = _retrieve_chunks(query, top_k)
    results = []
    for chunk in chunks:
        results.append(
            SourceCitation(
                chunk_id=chunk.get("chunk_id", ""),
                text=chunk.get("text", ""),
                file_name=chunk.get("file_name", "Unknown"),
                page_number=str(chunk.get("page_number", "?")),
                score=round(chunk.get("rerank_score", chunk.get("score", 0.0)), 4),
                url=chunk.get("url") or None,
            )
        )
    return results


def rag_chat(
    query: str,
    conversation_history: Optional[List[dict]] = None,
    temporary_context: Optional[str] = None,
    top_k: int = 15,
) -> ChatResponse:
    """
    Mode B — RAG Chatbot.
    Retrieves relevant chunks, optionally prepends temporary context, passes them to the LLM, returns synthesis + citations.
    """
    from backend.rag.llm import generate_rag_answer

    chunks = _retrieve_chunks(query, top_k)
    
    if temporary_context:
        # Inject the uploaded document as the highest priority chunk
        synthetic_chunk = {
            "chunk_id": "temp_upload",
            "text": temporary_context,
            "file_name": "Uploaded Document",
            "page_number": "1-End",
            "score": 1.0,
            "rerank_score": 1.0,
            "url": None
        }
        chunks.insert(0, synthetic_chunk)

    if not chunks:
        return ChatResponse(
            answer=(
                "No relevant documents were found in the knowledge base for this query. "
                "Please ensure policies have been ingested first."
            ),
            sources=[],
            model_used="none",
        )

    answer, model_used = generate_rag_answer(
        query=query,
        context_chunks=chunks,
        conversation_history=conversation_history or [],
    )

    sources = []
    for chunk in chunks:
        sources.append(
            SourceCitation(
                chunk_id=chunk.get("chunk_id", ""),
                text=chunk.get("text", "")[:350] + "…",
                file_name=chunk.get("file_name", "Unknown"),
                page_number=str(chunk.get("page_number", "?")),
                score=round(chunk.get("rerank_score", chunk.get("score", 0.0)), 4),
                url=chunk.get("url") or None,
            )
        )

    return ChatResponse(answer=answer, sources=sources, model_used=model_used)

def rag_chat_stream(
    query: str,
    conversation_history: Optional[List[dict]] = None,
    temporary_context: Optional[str] = None,
    top_k: int = 15,
    use_local_llm: Optional[bool] = None,
):
    import json
    from backend.rag.llm import generate_rag_answer_stream

    yield json.dumps({"type": "status", "message": "Performing Multi-Layer Retrieval (Vectors + Graph)..."}) + "\n"
    chunks = _retrieve_chunks(query, top_k)
    
    if temporary_context:
        synthetic_chunk = {
            "chunk_id": "temp_upload",
            "text": temporary_context,
            "file_name": "Uploaded Document",
            "page_number": "1-End",
            "score": 1.0,
            "rerank_score": 1.0,
            "url": None
        }
        chunks.insert(0, synthetic_chunk)
        
    yield json.dumps({"type": "sources", "data": chunks}) + "\n"

    if not chunks:
        yield json.dumps({"type": "status", "message": "No context found. Answering from general knowledge..."}) + "\n"
    else:
        yield json.dumps({"type": "status", "message": "Synthesising with Gemini..."}) + "\n"

    model_used = "none"
    for token, model in generate_rag_answer_stream(query, chunks, conversation_history):
        model_used = model
        yield json.dumps({"type": "chunk", "data": token}) + "\n"
        
    yield json.dumps({"type": "done", "model": model_used}) + "\n"


def compare_document(draft_chunks: List[dict], top_k: int = 5) -> List[dict]:
    """
    Mode C — Document Comparison.
    For each draft chunk, find the most similar existing chunks.
    Returns a flat list of matches with similarity scores.
    """
    all_matches = []
    seen_ids = set()

    for draft_chunk in draft_chunks[:10]:  # Limit to first 10 draft chunks
        dense_vec = embed_dense_query(draft_chunk["text"])
        sparse_vec = embed_sparse_query(draft_chunk["text"])
        candidates = hybrid_search(dense_vec, sparse_vec, top_k=top_k * 2)
        reranked = rerank(draft_chunk["text"], candidates, top_n=top_k)

        for match in reranked:
            cid = match.get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_matches.append(match)

    # Sort all matches by score descending
    all_matches.sort(
        key=lambda x: x.get("rerank_score", x.get("score", 0.0)), reverse=True
    )
    return all_matches[:20]
