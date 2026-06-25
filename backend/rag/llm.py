"""
backend/rag/llm.py
LLM synthesis layer for Gov-Intel.

Priority:
  1. Ollama / Llama-3-8B (local, if USE_LOCAL_LLM=true and Ollama reachable)
  2. Gemini 2.5 Flash / Pro (cloud, always available if GOOGLE_API_KEY is set)

The local LLM path is fully coded and activates automatically when
USE_LOCAL_LLM=true in .env. Add Ollama later and flip the flag.
"""
from __future__ import annotations

import json
from typing import List, Optional

from backend.core.config import get_settings
from backend.core.logging_config import get_logger

log = get_logger(__name__)
settings = get_settings()

# ─── System prompt ────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are Gov-Intel, an expert regulatory intelligence assistant for the \
Ministry of Education (MoE), India.

You answer questions about government policies, schemes, UGC regulations, AICTE circulars, \
and higher-education guidelines with absolute precision.

Rules:
1. Base your answer ONLY on the provided context chunks. Summarize and synthesize the information. DO NOT just copy the context as your answer.
2. If the context does not contain sufficient information, clearly state: \
   "The retrieved documents do not contain enough information to answer this question."
3. Always cite the source document and page number for every claim using the format \
   [Source: <file_name>, Page <page_number>].
4. Never hallucinate laws, section numbers, or policy names.
5. Be concise, professional, and authoritative.
"""

COMPARE_SYSTEM_PROMPT = """You are a regulatory conflict-detection engine.
Given a draft policy excerpt and a list of similar existing regulatory chunks,
identify: (a) exact duplications, (b) direct contradictions, (c) related/overlapping content.
Return a brief plain-language summary of your findings. Keep it under 200 words."""


# ─── Ollama (local) ───────────────────────────────────────────────────────────

def _ollama_generate(prompt: str, system: str = "") -> str:
    """Call Ollama's /api/generate endpoint synchronously."""
    import requests as req
    payload = {
        "model": settings.local_llm_model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    try:
        resp = req.post(
            f"{settings.ollama_base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        log.warning("Ollama call failed: %s — falling back to Gemini", e)
        return None

def _ollama_generate_stream(prompt: str, system: str = ""):
    """Stream Ollama local LLM generation."""
    import requests as req
    import json
    payload = {
        "model": settings.local_llm_model,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    try:
        resp = req.post(
            f"{settings.ollama_base_url}/api/generate",
            json=payload,
            timeout=120,
            stream=True
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
    except Exception as e:
        log.warning("Ollama stream failed: %s", e)
        # Yielding None acts as a signal to fallback
        yield None


# ─── Gemini (cloud) ───────────────────────────────────────────────────────────

def _gemini_generate(prompt: str, system: str = "") -> str:
    if not prompt:
        return ""

    import google.generativeai as genai
    genai.configure(api_key=settings.google_api_key)
    
    model_kwargs = {
        "model_name": settings.gemini_llm_model,
        "generation_config": {"temperature": 0.1, "max_output_tokens": 2048},
    }
    if system:
        model_kwargs["system_instruction"] = system
        
    model = genai.GenerativeModel(**model_kwargs)
    
    try:
        response = model.generate_content(prompt)
        # Check if response was blocked
        if not response or not response.candidates:
            log.warning("Gemini returned no candidates (blocked?).")
            return ""
        return response.text.strip()
    except Exception as e:
        log.error("Gemini generation error: %s", e)
        return ""

def _gemini_generate_stream(prompt: str, system: str = ""):
    """Call Gemini via google-generativeai SDK and stream."""
    import google.generativeai as genai
    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(
        model_name=settings.gemini_llm_model,
        system_instruction=system,
        generation_config={"temperature": 0.1, "max_output_tokens": 2048},
    )
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_rag_answer(
    query: str,
    context_chunks: List[dict],
    conversation_history: Optional[List[dict]] = None,
) -> tuple[str, str]:
    """
    Synthesise an answer from retrieved context chunks.

    Returns:
        (answer_text, model_name_used)
    """
    # Build context block
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Chunk {i}] Source: {chunk.get('file_name', 'Unknown')}, "
            f"Page {chunk.get('page_number', '?')}\n{chunk.get('text', '')}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    # Build history block
    history_block = ""
    if conversation_history:
        turns = []
        for msg in conversation_history[-6:]:  # last 3 exchanges
            role = "User" if msg.get("role") == "user" else "Assistant"
            turns.append(f"{role}: {msg.get('content', '')}")
        history_block = "Previous conversation:\n" + "\n".join(turns) + "\n\n"

    full_prompt = (
        f"{history_block}"
        f"Context Documents:\n{context_block}\n\n"
        f"Instructions:\n"
        f"1. If the user's question is a general conversational greeting (e.g., 'hi', 'hello', 'good morning'), please respond warmly and offer to help them query the regulatory knowledge base. Ignore context documents for this.\n"
        f"2. For all other queries, based ONLY on the context documents provided above, please answer the question. "
        f"Synthesize the information into a comprehensive answer (with citations) rather than just returning the raw context.\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    # Try local LLM first
    if settings.use_local_llm:
        answer = _ollama_generate(full_prompt, system=RAG_SYSTEM_PROMPT)
        if answer:
            return answer, settings.local_llm_model

    # Gemini fallback / primary
    answer = _gemini_generate(full_prompt, system=RAG_SYSTEM_PROMPT)
    return answer, settings.gemini_llm_model

def generate_rag_answer_stream(
    query: str,
    context_chunks: List[dict],
    conversation_history: Optional[List[dict]] = None,
    use_local_llm: Optional[bool] = None,
):
    """
    Synthesise an answer from retrieved context chunks, streaming tokens.
    Yields (token, model_name_used).
    """
    # Override via parameter if explicitly requested
    is_local = settings.use_local_llm if use_local_llm is None else use_local_llm

    # Build context block
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Chunk {i}] Source: {chunk.get('file_name', 'Unknown')}, "
            f"Page {chunk.get('page_number', '?')}\n{chunk.get('text', '')}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    # Build history block
    history_block = ""
    if conversation_history:
        turns = []
        for msg in conversation_history[-6:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            turns.append(f"{role}: {msg.get('content', '')}")
        history_block = "Previous conversation:\n" + "\n".join(turns) + "\n\n"

    full_prompt = (
        f"{history_block}"
        f"Context Documents:\n{context_block}\n\n"
        f"Based ONLY on the context documents above, please answer the following question. "
        f"Synthesize the information into a comprehensive answer (with citations) rather than just returning the raw context.\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    if is_local:
        gen = _ollama_generate_stream(full_prompt, system=RAG_SYSTEM_PROMPT)
        try:
            first_token = next(gen)
            if first_token is not None:
                yield first_token, settings.local_llm_model
                for token in gen:
                    if token is not None:
                        yield token, settings.local_llm_model
                return
        except StopIteration:
            pass

    # Fallback to Gemini stream
    for token in _gemini_generate_stream(full_prompt, system=RAG_SYSTEM_PROMPT):
        yield token, settings.gemini_llm_model


def generate_compare_summary(
    draft_excerpt: str,
    matches: List[dict],
) -> str:
    """
    Summarise conflicts/overlaps between a draft and existing chunks.
    """
    matches_block = "\n\n---\n\n".join(
        f"[Existing] {m.get('file_name', '')} p.{m.get('page_number', '')} "
        f"(score={m.get('rerank_score', m.get('score', 0)):.3f}):\n{m.get('text', '')}"
        for m in matches[:5]
    )
    prompt = (
        f"Draft policy excerpt:\n{draft_excerpt}\n\n"
        f"Most similar existing regulations:\n{matches_block}\n\n"
        f"Analysis:"
    )

    if settings.use_local_llm:
        result = _ollama_generate(prompt, system=COMPARE_SYSTEM_PROMPT)
        if result:
            return result

    return _gemini_generate(prompt, system=COMPARE_SYSTEM_PROMPT)


def describe_image_with_vlm(image_bytes: bytes, context_hint: str = "") -> str:
    """
    Pass an image through Gemini Vision and return a descriptive text.
    Used during ingestion for scheme posters, flowcharts, org charts, etc.
    Local VLM (LLaVA) hook: swap this function body if Ollama Vision is available.
    """
    import google.generativeai as genai
    from PIL import Image
    import io

    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(settings.gemini_vision_model)

    image = Image.open(io.BytesIO(image_bytes))
    prompt = (
        "You are analysing a page image from an Indian government policy document. "
        "Describe all text, tables, flowcharts, scheme details, eligibility criteria, "
        "budget figures, and hierarchy information visible in this image. "
        "Be detailed and comprehensive so the description is searchable. "
        f"Context hint: {context_hint}"
    )
    try:
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        log.warning("VLM image description failed: %s", e)
        return f"[Image — description unavailable: {e}]"
