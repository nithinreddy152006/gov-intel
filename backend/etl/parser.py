"""
backend/etl/parser.py
Document parser with two tiers:
  1. LlamaParse (cloud) — if LLAMA_PARSE_API_KEY is set, uses advanced layout parsing
  2. PyMuPDF (local)   — high-quality fallback, extracts text + images per page

Returns a list of PageResult dicts:
  {
      "page_number": str,
      "text": str,
      "images": [bytes, ...],   # raw PNG/JPEG bytes of extracted images
  }
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Dict, List

from backend.core.config import get_settings
from backend.core.logging_config import get_logger

log = get_logger(__name__)
settings = get_settings()


# ─── PyMuPDF parser ──────────────────────────────────────────────────────────

def _parse_with_pymupdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract text and images from every page using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        # Extract embedded images
        image_bytes_list = []
        for img_index, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes_list.append(base_image["image"])
            except Exception as e:
                log.debug("Could not extract image %d on page %d: %s", img_index, page_num + 1, e)

        pages.append({
            "page_number": str(page_num + 1),
            "text": text,
            "images": image_bytes_list,
        })

    doc.close()
    log.info("PyMuPDF parsed %d pages from %s", len(pages), pdf_path.name)
    return pages


# ─── LlamaParse parser ────────────────────────────────────────────────────────

def _parse_with_llamaparse(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Parse with LlamaParse. Falls back to PyMuPDF on any error.
    LlamaParse returns markdown-formatted text with better table/layout handling.
    """
    try:
        from llama_parse import LlamaParse

        parser = LlamaParse(
            api_key=settings.llama_parse_api_key,
            result_type="markdown",
            verbose=False,
            language="en",
        )
        documents = parser.load_data(str(pdf_path))

        pages = []
        for i, doc in enumerate(documents):
            pages.append({
                "page_number": str(i + 1),
                "text": doc.text,
                "images": [],  # LlamaParse doesn't expose raw images
            })

        log.info("LlamaParse parsed %d pages from %s", len(pages), pdf_path.name)
        return pages

    except Exception as e:
        log.warning("LlamaParse failed (%s) — falling back to PyMuPDF", e)
        return _parse_with_pymupdf(pdf_path)


# ─── Public entry point ───────────────────────────────────────────────────────

def parse_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a PDF and return a list of PageResult dicts.
    Automatically selects LlamaParse or PyMuPDF based on config.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    log.info("Parsing %s (size=%.1f KB) ...", pdf_path.name, pdf_path.stat().st_size / 1024)

    if settings.has_llama_parse_key:
        return _parse_with_llamaparse(pdf_path)
    else:
        return _parse_with_pymupdf(pdf_path)
