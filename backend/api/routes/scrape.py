"""
backend/api/routes/scrape.py
Scraper management endpoints — trigger manually and check status.
"""
from fastapi import APIRouter, BackgroundTasks
from backend.core.logging_config import get_logger
from backend.models.schemas import ScrapeStatusResponse

log = get_logger(__name__)
router = APIRouter()


@router.post("/scrape/trigger", response_model=dict)
def api_scrape_trigger(background_tasks: BackgroundTasks):
    """
    Manually trigger the MoE/UGC/AICTE scraper in the background.
    The scraper will download new PDFs and ingest them automatically.
    """
    log.info("Manual scrape trigger received.")
    background_tasks.add_task(_run_scrape)
    return {
        "status": "accepted",
        "message": "Scraper started in background. Check /api/scrape/status for progress.",
    }


@router.get("/scrape/status", response_model=ScrapeStatusResponse)
def api_scrape_status():
    """Return the status of the last completed scrape run."""
    from backend.etl.scraper import get_scrape_state
    state = get_scrape_state()
    return ScrapeStatusResponse(
        last_run=state.get("last_run"),
        files_discovered=state.get("files_discovered", 0),
        files_indexed=state.get("files_indexed", 0),
        next_scheduled_run=_get_next_run(),
        errors=state.get("errors", []),
    )


def _run_scrape():
    try:
        from backend.etl.scraper import run_scraper
        run_scraper()
    except Exception as e:
        log.error("Background scrape failed: %s", e, exc_info=True)


def _get_next_run() -> str | None:
    """Return next scheduled run time from APScheduler."""
    try:
        from backend.etl.scheduler import _scheduler
        if _scheduler and _scheduler.running:
            job = _scheduler.get_job("gov_intel_scraper")
            if job and job.next_run_time:
                return job.next_run_time.isoformat()
    except Exception:
        pass
    return None
