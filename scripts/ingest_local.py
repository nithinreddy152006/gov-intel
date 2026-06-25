"""
scripts/ingest_local.py
CLI script to run the full ingestion pipeline locally (outside Docker).

Usage:
    # Ingest all PDFs in data/policies/
    python scripts/ingest_local.py

    # Ingest a specific PDF
    python scripts/ingest_local.py --file data/policies/ugc_rules.pdf

    # Force re-index even if already indexed
    python scripts/ingest_local.py --force
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.logging_config import setup_logging
setup_logging()

from backend.core.config import get_settings
from backend.rag.qdrant_store import ensure_collection
from backend.etl.ingestor import ingest_pdf, ingest_directory


def main():
    parser = argparse.ArgumentParser(description="Gov-Intel Local Ingestion CLI")
    parser.add_argument("--file", type=str, help="Path to a specific PDF file to ingest")
    parser.add_argument("--dir", type=str, help="Path to directory of PDFs (default: data/policies/)")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if already indexed")
    args = parser.parse_args()

    settings = get_settings()
    print("\n" + "="*60)
    print("  Gov-Intel Ingestion Pipeline")
    print("="*60)
    print(f"  Qdrant URL:     {settings.qdrant_url}")
    print(f"  Collection:     {settings.qdrant_collection_name}")
    print(f"  Embeddings:     {'Cloud (Gemini)' if settings.use_cloud_embeddings else 'Local (BAAI/bge-m3)'}")
    print(f"  Parser:         {'LlamaParse' if settings.has_llama_parse_key else 'PyMuPDF'}")
    print("="*60 + "\n")

    # Ensure collection
    print("Ensuring Qdrant collection exists...")
    ensure_collection()

    if args.file:
        pdf_path = Path(args.file)
        print(f"Ingesting single file: {pdf_path}")
        result = ingest_pdf(pdf_path, force_reindex=args.force)
        _print_result(result)
    else:
        target_dir = Path(args.dir) if args.dir else settings.policies_dir
        print(f"Ingesting directory: {target_dir}")
        results = ingest_directory(target_dir, force_reindex=args.force)
        for r in results:
            _print_result(r)

    print("\n✓ All done.\n")


def _print_result(result: dict):
    status = result.get("status", "?")
    file_name = result.get("file_name", "?")
    chunks = result.get("chunks_indexed", 0)
    error = result.get("error")

    if status == "done":
        print(f"  ✓ {file_name} → {chunks} chunks indexed")
    elif error:
        print(f"  ✗ {file_name} → FAILED: {error}")
    else:
        print(f"  ○ {file_name} → {status} (already indexed / skipped)")


if __name__ == "__main__":
    main()
