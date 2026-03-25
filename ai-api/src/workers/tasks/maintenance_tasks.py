from __future__ import annotations

from src.utils.logging import get_logger
from src.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(name="maintenance.reap_stale_jobs")
def reap_stale_jobs():
    """Mark stale pending/processing jobs as failed.

    Runs periodically via Celery beat to prevent zombie jobs from
    permanently consuming backpressure slots (RS-3).
    """
    from src.db.repositories.sync_job_repo import SyncJobRepository
    from src.db.sync_session import get_sync_session

    with get_sync_session() as session:
        repo = SyncJobRepository(session)
        count = repo.reap_stale_jobs(max_age_seconds=3900)
        session.commit()

    if count > 0:
        logger.info("Reaped stale jobs", count=count)
    return count


@celery_app.task(name="maintenance.cleanup_old_jobs")
def cleanup_old_jobs():
    """Delete completed/failed jobs older than the retention period (RS-4).

    Runs periodically via Celery beat to prevent unbounded table growth.
    """
    from src.config import get_settings
    from src.db.repositories.sync_job_repo import SyncJobRepository
    from src.db.sync_session import get_sync_session

    settings = get_settings()

    with get_sync_session() as session:
        repo = SyncJobRepository(session)
        count = repo.cleanup_old_jobs(retention_days=settings.JOB_RETENTION_DAYS)
        session.commit()

    if count > 0:
        logger.info("Cleaned up old jobs", count=count, retention_days=settings.JOB_RETENTION_DAYS)
    return count
