"""
backend/etl/chunker.py
Sentence-aware text chunker.
Produces chunk dicts ready for embedding and Qdrant upsert.

Each chunk:
  {
      "chunk_id": str (deterministic hash),
      "text":     str,
      "metadata": {
          "file_name":    str,
          "page_number":  str,
          "chunk_index":  int,
          "doc_id":       str,
          "url":          str,
      }
  }
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List

from backend.core.config import get_settings
from backend.core.logging_config import get_logger

log = get_logger(__name__)
settings = get_settings()


def _split_sentences(text: str) -> List[str]:
    """Split on sentence boundaries while preserving structure."""
    # Split on period/exclamation/question followed by whitespace + capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\d\(])', text)
    return [s.strip() for s in sentences if s.strip()]


def _make_chunk_id(doc_id: str, page: str, index: int) -> str:
    """Deterministic UUID-like chunk ID from doc+page+index."""
    raw = f"{doc_id}::page{page}::chunk{index}"
    return hashlib.md5(raw.encode()).hexdigest()


def chunk_pages(
    pages: List[Dict[str, Any]],
    doc_id: str,
    file_name: str,
    url: str = "",
) -> List[Dict[str, Any]]:
    """
    Convert a list of PageResult dicts into overlapping text chunks.

    Uses a simple sliding-window approach on sentences to respect
    chunk_size (in chars) and chunk_overlap.
    """
    chunk_size = settings.chunk_size * 4   # chars ≈ tokens * 4
    chunk_overlap_chars = settings.chunk_overlap * 4

    chunks = []
    global_chunk_index = 0

    for page in pages:
        page_num = page["page_number"]
        text = page.get("text", "").strip()

        if not text or len(text) < 20:
            continue

        sentences = _split_sentences(text)
        if not sentences:
            continue

        buffer = ""
        for sentence in sentences:
            if len(buffer) + len(sentence) + 1 <= chunk_size:
                buffer = (buffer + " " + sentence).strip()
            else:
                if buffer:
                    chunks.append({
                        "chunk_id": _make_chunk_id(doc_id, page_num, global_chunk_index),
                        "text": buffer,
                        "metadata": {
                            "file_name": file_name,
                            "page_number": page_num,
                            "chunk_index": global_chunk_index,
                            "doc_id": doc_id,
                            "url": url,
                        }
                    })
                    global_chunk_index += 1
                    # Overlap: keep last overlap_chars of the previous buffer
                    overlap_text = buffer[-chunk_overlap_chars:] if chunk_overlap_chars > 0 else ""
                    buffer = (overlap_text + " " + sentence).strip()
                else:
                    buffer = sentence

        # Flush remaining buffer
        if buffer and len(buffer) > 20:
            chunks.append({
                "chunk_id": _make_chunk_id(doc_id, page_num, global_chunk_index),
                "text": buffer,
                "metadata": {
                    "file_name": file_name,
                    "page_number": page_num,
                    "chunk_index": global_chunk_index,
                    "doc_id": doc_id,
                    "url": url,
                }
            })
            global_chunk_index += 1

    log.info("Chunked %d pages → %d chunks for doc '%s'", len(pages), len(chunks), file_name)
    return chunks


def chunk_text(
    text: str,
    doc_id: str,
    file_name: str,
    page_number: str = "1",
    url: str = "",
) -> List[Dict[str, Any]]:
    """Convenience: chunk a raw text string directly (e.g. for draft comparison)."""
    fake_page = [{"page_number": page_number, "text": text, "images": []}]
    return chunk_pages(fake_page, doc_id, file_name, url)
