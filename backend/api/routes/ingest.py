"""
backend/api/routes/ingest.py
Ingestion endpoints — upload PDF → background ingestion → status polling.
"""
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from backend.core.config import get_settings
from backend.core.logging_config import get_logger
from backend.etl.ingestor import ingest_pdf, get_job_status
from backend.models.schemas import IngestResponse, IngestStatusResponse

log = get_logger(__name__)
router = APIRouter()
settings = get_settings()

UPLOAD_TMP_DIR = Path("data/uploads")


@router.post("/ingest", response_model=IngestResponse)
async def api_ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF and trigger background ingestion.
    Returns a job_id you can poll at GET /api/ingest/status/{job_id}.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    save_path = UPLOAD_TMP_DIR / f"{job_id}_{file.filename}"

    # Stream file to disk
    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        log.info("Saved upload: %s (%.1f KB)", save_path.name, save_path.stat().st_size / 1024)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save error: {e}")

    # Kick off background ingestion
    background_tasks.add_task(_run_ingest, save_path, job_id)

    return IngestResponse(
        job_id=job_id,
        status="accepted",
        file_name=file.filename,
        message="Ingestion started. Poll /api/ingest/status/{job_id} for progress.",
    )


@router.get("/ingest/status/{job_id}", response_model=IngestStatusResponse)
def api_ingest_status(job_id: str):
    """Poll the status of an ingestion job."""
    job = get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return IngestStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        chunks_indexed=job.get("chunks_indexed", 0),
        error=job.get("error"),
    )


@router.post("/ingest/directory", response_model=list)
def api_ingest_directory(background_tasks: BackgroundTasks):
    """
    Trigger ingestion of all PDFs in the configured policies directory.
    Returns immediately with a list of job IDs.
    """
    policies_dir = settings.policies_dir
    if not policies_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Policies directory not found: {policies_dir}"
        )

    pdf_files = list(policies_dir.glob("*.pdf"))
    if not pdf_files:
        raise HTTPException(status_code=404, detail="No PDF files found in policies directory.")

    jobs = []
    for pdf_path in pdf_files:
        job_id = str(uuid.uuid4())
        background_tasks.add_task(_run_ingest, pdf_path, job_id)
        jobs.append({"job_id": job_id, "file_name": pdf_path.name, "status": "accepted"})

    return jobs


def _run_ingest(pdf_path: Path, job_id: str):
    """Background task helper."""
    try:
        ingest_pdf(pdf_path, job_id=job_id)
    except Exception as e:
        log.error("Background ingest failed for %s: %s", pdf_path, e, exc_info=True)


@router.post("/extract_text")
async def api_extract_text(file: UploadFile = File(...)):
    """
    Extract raw text from an uploaded PDF for temporary context usage (e.g. Chat).
    Returns immediately with the parsed text.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_id = str(uuid.uuid4())
    save_path = UPLOAD_TMP_DIR / f"temp_{temp_id}_{file.filename}"

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        from backend.etl.parser import parse_pdf
        pages = parse_pdf(save_path)
        full_text = "\n\n".join([p["text"] for p in pages])
        
        # Clean up temp file
        save_path.unlink()
        
        return {"file_name": file.filename, "text": full_text}
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {e}")
