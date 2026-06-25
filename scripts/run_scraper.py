"""
scripts/run_scraper.py
CLI script to manually trigger the MoE/UGC/AICTE web scrapers.

Usage:
    python scripts/run_scraper.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.logging_config import setup_logging
setup_logging()

from backend.rag.qdrant_store import ensure_collection
from backend.etl.scraper import run_scraper


def main():
    print("\n" + "="*60)
    print("  Gov-Intel Web Scraper")
    print("  Portals: MoE · UGC · AICTE")
    print("="*60 + "\n")

    print("Ensuring Qdrant collection exists...")
    ensure_collection()

    print("Starting scraper...\n")
    result = run_scraper()

    print("\n" + "="*60)
    print(f"  Discovered:  {result.get('files_discovered', 0)} PDF links")
    print(f"  Indexed:     {result.get('files_indexed', 0)} new documents")
    print(f"  Errors:      {len(result.get('errors', []))}")
    if result.get("errors"):
        for err in result["errors"]:
            print(f"    ✗ {err}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
