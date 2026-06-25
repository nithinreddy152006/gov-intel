"""
backend/etl/ingestor.py
Orchestrates the full ingestion pipeline for a single PDF:
  1. Parse with LlamaParse / PyMuPDF
  2. For each page image → Gemini Vision → text description
  3. Chunk text (including VLM descriptions)
  4. Embed (dense + sparse)
  5. Upsert to Qdrant
  6. Idempotency: skip if doc_id already exists

Used by:
  - backend/api/routes/ingest.py  (file upload)
  - backend/etl/scraper.py        (automated scraping)
  - scripts/ingest_local.py       (CLI)
"""
from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from backend.core.config import get_settings
from backend.core.logging_config import get_logger
from backend.etl.parser import parse_pdf
from backend.etl.chunker import chunk_pages
from backend.rag.qdrant_store import upsert_chunks, doc_exists, ensure_collection
from backend.etl.graph import get_kg

log = get_logger(__name__)
settings = get_settings()

# ─── In-memory job tracking ───────────────────────────────────────────────────
# For production, replace with Redis or a DB-backed job store.
_jobs: Dict[str, Dict[str, Any]] = {}


def _make_doc_id(file_name: str) -> str:
    """Stable doc ID derived from the file name."""
    return hashlib.md5(file_name.encode()).hexdigest()


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    return _jobs.get(job_id)


def _update_job(job_id: str, **kwargs):
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)


# ─── VLM image enrichment ─────────────────────────────────────────────────────

def _enrich_pages_with_vlm(pages: List[Dict[str, Any]], context_hint: str) -> List[Dict[str, Any]]:
    """
    For every page that has embedded images, run Gemini Vision and
    append the description to the page text.
    """
    if not settings.has_google_key:
        log.warning("No Google API key — skipping VLM image enrichment.")
        return pages

    from backend.rag.llm import describe_image_with_vlm

    for page in pages:
        images = page.get("images", [])
        if not images:
            continue

        image_descriptions = []
        for img_bytes in images:
            if len(img_bytes) < 500:   # Skip tiny/noise images
                continue
            desc = describe_image_with_vlm(img_bytes, context_hint=context_hint)
            if desc:
                image_descriptions.append(desc)

        if image_descriptions:
            description_block = "\n\n[IMAGE CONTENT]\n" + "\n\n".join(image_descriptions)
            page["text"] = page.get("text", "") + description_block
            log.debug("Page %s: appended %d image description(s).", page["page_number"], len(image_descriptions))

    return pages


# ─── Core ingest function ─────────────────────────────────────────────────────

def ingest_pdf(
    pdf_path: Path,
    url: str = "",
    job_id: Optional[str] = None,
    force_reindex: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Full ingestion pipeline for a single PDF.

    Args:
        pdf_path:          Path to the PDF file.
        url:               Source URL (for scraped documents).
        job_id:            Job ID for status tracking (auto-generated if None).
        force_reindex:     If True, re-ingest even if doc already exists.
        progress_callback: Optional callback(message) for progress updates.

    Returns:
        {"job_id": str, "status": str, "chunks_indexed": int, "error": str|None}
    """
    pdf_path = Path(pdf_path)
    job_id = job_id or str(uuid.uuid4())
    file_name = pdf_path.name
    doc_id = _make_doc_id(file_name)

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "file_name": file_name,
        "chunks_indexed": 0,
        "error": None,
    }

    def _progress(msg: str):
        log.info("[Job %s] %s", job_id, msg)
        if progress_callback:
            progress_callback(msg)

    try:
        # Ensure collection exists
        ensure_collection()

        # Idempotency check
        if not force_reindex and doc_exists(doc_id):
            _progress(f"Document '{file_name}' already indexed — skipping.")
            _update_job(job_id, status="done", chunks_indexed=0)
            return _jobs[job_id]

        # 1. Parse
        _progress(f"Parsing '{file_name}'...")
        t0 = time.time()
        pages = parse_pdf(pdf_path)
        _progress(f"Parsed {len(pages)} pages in {time.time()-t0:.1f}s")

        # 2. VLM image enrichment
        _progress("Running VLM on embedded images...")
        pages = _enrich_pages_with_vlm(pages, context_hint=file_name)

        # 3. Chunk
        _progress("Chunking text...")
        chunks = chunk_pages(pages, doc_id=doc_id, file_name=file_name, url=url)
        _progress(f"Created {len(chunks)} chunks")

        if not chunks:
            raise ValueError("No indexable content found in the PDF.")

        # 4. Embed + Upsert (in batches of 64)
        batch_size = 64
        total_upserted = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            upserted = upsert_chunks(batch)
            total_upserted += upserted
            _progress(f"Upserted batch {i//batch_size + 1} ({total_upserted}/{len(chunks)} chunks)")
            _update_job(job_id, chunks_indexed=total_upserted)

        # 5. Graph Extraction (NEW: Multi-Layer GraphRAG)
        _progress("Building Knowledge Graph (Extracting entities/relations)...")
        kg = get_kg()
        total_triplets = 0
        for i, chunk in enumerate(chunks):
            if i % 2 == 0:
                msg = f"Graph extraction progress: {i}/{len(chunks)} chunks"
                _progress(msg)
                _update_job(job_id, status=msg)
            
            # Extract triplets using LLM
            metadata = {
                "file_name": file_name,
                "page_number": chunk.get("metadata", {}).get("page_number", ""),
                "doc_id": doc_id
            }
            count = kg.extract_triplets_from_text(chunk.get("text", ""), metadata)
            total_triplets += count
        
        kg.save()
        _progress(f"Knowledge Graph updated: added {total_triplets} new relationships.")

        _update_job(job_id, status="done", chunks_indexed=total_upserted)
        _progress(f"✓ Ingestion complete — {total_upserted} chunks indexed.")
        return _jobs[job_id]

    except Exception as e:
        log.error("[Job %s] Ingestion failed: %s", job_id, e, exc_info=True)
        _update_job(job_id, status="failed", error=str(e))
        return _jobs[job_id]


def ingest_directory(directory: Path, force_reindex: bool = False) -> List[Dict[str, Any]]:
    """Ingest all PDFs in a directory. Returns list of job results."""
    directory = Path(directory)
    pdf_files = list(directory.glob("*.pdf"))
    log.info("Found %d PDF(s) in %s", len(pdf_files), directory)

    results = []
    for pdf_path in pdf_files:
        result = ingest_pdf(pdf_path, force_reindex=force_reindex)
        results.append(result)

    return results
