"""
backend/api/routes/chat.py
Mode B — RAG Chatbot endpoint.
Retrieve → Rerank → LLM Synthesis → Cited Response.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.models.schemas import ChatRequest, ChatResponse
from backend.rag.retriever import rag_chat, rag_chat_stream
from backend.core.logging_config import get_logger

log = get_logger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def api_chat(request: ChatRequest):
    """
    Mode B: RAG Chatbot.
    Retrieves relevant policy chunks, reranks them, then passes to
    Gemini (or Ollama/Llama-3 if enabled) for synthesis.
    Returns a cited answer with source page references.
    """
    log.info("Chat request: '%s...'", request.query[:60])
    try:
        history = [m.model_dump() for m in request.conversation_history]
        response = rag_chat(
            query=request.query,
            conversation_history=history,
            temporary_context=request.temporary_context,
        )
        return response
    except Exception as e:
        log.error("Chat failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@router.post("/chat_stream")
async def api_chat_stream(request: ChatRequest):
    """
    Mode B: RAG Chatbot (Streaming).
    Yields NDJSON events for retrieval, reranking, and LLM text generation.
    """
    log.info("Chat stream request: '%s...'", request.query[:60])
    try:
        history = [m.model_dump() for m in request.conversation_history]
        gen = rag_chat_stream(
            query=request.query,
            conversation_history=history,
            temporary_context=request.temporary_context,
            use_local_llm=request.use_local_llm
        )
        return StreamingResponse(gen, media_type="application/x-ndjson")
    except Exception as e:
        log.error("Chat stream failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat stream error: {str(e)}")
