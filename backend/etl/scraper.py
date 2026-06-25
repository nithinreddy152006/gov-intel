"""
backend/etl/scraper.py
Automated web scrapers for MoE, UGC, and AICTE portals.

Each scraper:
  1. Fetches the portal's notifications/circulars listing page
  2. Identifies new PDF links not yet in the known-files registry
  3. Downloads them to data/scraped/
  4. Triggers ingestor

Design priorities:
  - Polite crawling (delay between requests)
  - Idempotent (tracks downloaded URLs in a local JSON registry)
  - Robust to page structure changes (logs warnings, doesn't crash)
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from backend.core.config import get_settings
from backend.core.logging_config import get_logger

log = get_logger(__name__)
settings = get_settings()

REGISTRY_FILE = Path("data/scraped/.registry.json")
SCRAPE_DELAY_SECONDS = 2   # polite delay between requests
REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

# ─── Portal definitions ───────────────────────────────────────────────────────

PORTALS = [
    {
        "name": "MoE_Schemes",
        "list_url": "https://www.education.gov.in/schemes-guidelines",
        "base_url": "https://www.education.gov.in",
        "pdf_selector": "a[href$='.pdf'], a[href$='.PDF']",
    },
    {
        "name": "UGC_Regulations",
        "list_url": "https://www.ugc.gov.in/regulations",
        "base_url": "https://www.ugc.gov.in",
        "pdf_selector": "a[href$='.pdf'], a[href$='.PDF']",
    },
    {
        "name": "MoE_OpenData",
        "list_url": "https://www.data.gov.in/catalogs/?ministry=Ministry%20of%20Education",
        "base_url": "https://www.data.gov.in",
        "pdf_selector": "a[href$='.pdf'], a[href$='.PDF']",
    },
    {
        "name": "NEP_2020",
        "list_url": "https://www.education.gov.in/nep/about-nep",
        "base_url": "https://www.education.gov.in",
        "pdf_selector": "a[href$='.pdf'], a[href$='.PDF']",
    },
]


# ─── Registry ─────────────────────────────────────────────────────────────────

def _load_registry() -> Dict[str, str]:
    """Load the URL→filename registry from disk."""
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_registry(registry: Dict[str, str]):
    REGISTRY_FILE.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ─── Scraper logic ────────────────────────────────────────────────────────────

def _fetch_pdf_links(portal: dict) -> List[str]:
    """Fetch listing page and extract PDF URLs."""
    try:
        resp = requests.get(
            portal["list_url"], headers=HEADERS, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for tag in soup.select(portal["pdf_selector"]):
            href = tag.get("href", "")
            if not href:
                continue
            full_url = urljoin(portal["base_url"], href)
            links.append(full_url)
        log.info("[%s] Found %d PDF links", portal["name"], len(links))
        return links
    except Exception as e:
        log.warning("[%s] Failed to fetch listing page: %s", portal["name"], e)
        return []


def _safe_filename(url: str, portal_name: str) -> str:
    """Derive a safe filename from a URL."""
    path = urlparse(url).path
    name = Path(path).name or "document.pdf"
    # Prefix with portal name
    if not name.startswith(portal_name):
        name = f"{portal_name}_{name}"
    return name


def _download_pdf(url: str, dest_dir: Path, file_name: str) -> Optional[Path]:
    """Download a PDF and return the saved path, or None on failure."""
    dest_path = dest_dir / file_name
    if dest_path.exists():
        return dest_path

    try:
        time.sleep(SCRAPE_DELAY_SECONDS)
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            log.debug("Skipping non-PDF URL: %s", url)
            return None

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        log.info("Downloaded: %s (%.1f KB)", file_name, dest_path.stat().st_size / 1024)
        return dest_path
    except Exception as e:
        log.warning("Failed to download %s: %s", url, e)
        return None


# ─── Scrape state tracking ────────────────────────────────────────────────────

_scrape_state: Dict = {
    "last_run": None,
    "files_discovered": 0,
    "files_indexed": 0,
    "errors": [],
}


def get_scrape_state() -> Dict:
    return dict(_scrape_state)


# ─── Main scrape function ─────────────────────────────────────────────────────

def run_scraper() -> Dict:
    """
    Run all portal scrapers, download new PDFs, and ingest them.
    Returns a summary dict.
    """
    from backend.etl.ingestor import ingest_pdf

    log.info("=== Starting Gov-Intel Scraper ===")
    registry = _load_registry()
    dest_dir = settings.scraped_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    files_discovered = 0
    files_indexed = 0
    errors = []

    for portal in PORTALS:
        log.info("Scraping portal: %s", portal["name"])
        pdf_links = _fetch_pdf_links(portal)
        files_discovered += len(pdf_links)

        for url in pdf_links:
            if url in registry:
                log.debug("Already registered: %s", url)
                continue

            file_name = _safe_filename(url, portal["name"])
            pdf_path = _download_pdf(url, dest_dir, file_name)

            if pdf_path is None:
                continue

            # Ingest
            result = ingest_pdf(pdf_path, url=url)
            if result["status"] == "done":
                files_indexed += 1
                registry[url] = file_name
            else:
                errors.append(f"{portal['name']}: {result.get('error', 'unknown error')}")

    _save_registry(registry)

    _scrape_state.update({
        "last_run": datetime.now().isoformat(),
        "files_discovered": files_discovered,
        "files_indexed": files_indexed,
        "errors": errors,
    })

    log.info(
        "=== Scraper done — discovered=%d indexed=%d errors=%d ===",
        files_discovered, files_indexed, len(errors),
    )
    return _scrape_state
