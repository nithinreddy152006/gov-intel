"""
backend/models/schemas.py
All Pydantic request/response models for the Gov-Intel API.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Shared sub-models ────────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    """A single retrieved source chunk with metadata."""
    chunk_id: str
    text: str
    file_name: str
    page_number: str
    score: float = 0.0
    url: Optional[str] = None


class MessageRole(str):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


# ─── Search (Mode A) ──────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    results: List[SourceCitation]
    total_found: int


# ─── Chat (Mode B) ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    conversation_history: List[ChatMessage] = Field(default_factory=list)
    temporary_context: Optional[str] = None
    use_local_llm: Optional[bool] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    model_used: str


# ─── Ingest ───────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    job_id: str
    status: str           # "accepted" | "processing" | "done" | "failed"
    file_name: str
    message: str = ""


class IngestStatusResponse(BaseModel):
    job_id: str
    status: str
    chunks_indexed: int = 0
    error: Optional[str] = None


# ─── Compare (Mode C) ─────────────────────────────────────────────────────────

class SimilarityMatch(BaseModel):
    file_name: str
    page_number: str
    similarity_score: float       # 0.0 – 1.0
    overlap_type: str             # "duplicate" | "conflict" | "related"
    excerpt: str


class CompareResponse(BaseModel):
    draft_name: str
    max_similarity: float
    matches: List[SimilarityMatch]
    summary: str


# ─── Scraper ──────────────────────────────────────────────────────────────────

class ScrapeStatusResponse(BaseModel):
    last_run: Optional[str] = None
    files_discovered: int = 0
    files_indexed: int = 0
    next_scheduled_run: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    qdrant_connected: bool
    collection_exists: bool
    total_chunks: int
    embedding_mode: str
    llm_mode: str
