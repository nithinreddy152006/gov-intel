"""
backend/etl/scheduler.py
APScheduler-based cron job for automated policy scraping.
Runs on server startup (via FastAPI lifespan) in background.
"""
from __future__ import annotations

from backend.core.config import get_settings
from backend.core.logging_config import get_logger

log = get_logger(__name__)
settings = get_settings()

_scheduler = None


def start_scheduler():
    """Start the background APScheduler with the configured cron trigger."""
    global _scheduler

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        log.warning("APScheduler not installed — automated scraping disabled.")
        return

    if _scheduler is not None and _scheduler.running:
        log.info("Scheduler already running.")
        return

    _scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

    _scheduler.add_job(
        func=_run_scrape_job,
        trigger=CronTrigger(
            hour=settings.scraper_cron_hour,
            minute=settings.scraper_cron_minute,
            timezone="Asia/Kolkata",
        ),
        id="gov_intel_scraper",
        name="Gov-Intel Policy Scraper",
        replace_existing=True,
        misfire_grace_time=3600,   # run even if missed by up to 1 hour
    )

    _scheduler.start()
    log.info(
        "Scheduler started — scraper cron: %02d:%02d IST daily",
        settings.scraper_cron_hour,
        settings.scraper_cron_minute,
    )


def stop_scheduler():
    """Gracefully stop the background scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("Scheduler stopped.")


def _run_scrape_job():
    """Wrapper executed by APScheduler."""
    log.info("Scheduled scrape job triggered.")
    try:
        from backend.etl.scraper import run_scraper
        run_scraper()
    except Exception as e:
        log.error("Scheduled scrape job failed: %s", e, exc_info=True)
