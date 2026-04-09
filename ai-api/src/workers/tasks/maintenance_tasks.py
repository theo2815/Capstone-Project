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


@celery_app.task(name="maintenance.cleanup_stale_blobs")
def cleanup_stale_blobs():
    """Remove blob staging directories older than 2 hours.

    Safety net for jobs that crashed before cleanup ran in complete_job/fail_job.
    """
    from src.utils.blob_store import cleanup_stale_blobs as _cleanup

    count = _cleanup(max_age_seconds=7200)
    return count


@celery_app.task(name="maintenance.finalize_mega_batch")
def finalize_mega_batch(chunk_results: list, parent_job_id: str):
    """Merge results from chunked mega-batch sub-tasks into the parent job.

    Called as the Celery chord callback after all chunk tasks complete.
    Each element of *chunk_results* is a list[dict] from one sub-task.
    """
    from itertools import chain

    from src.workers.helpers import complete_job

    # Flatten sub-task results. Celery chord delivers results in task
    # submission order, so the flat list is already in global image order.
    flat = list(chain.from_iterable(
        chunk if isinstance(chunk, list) else [chunk]
        for chunk in chunk_results
    ))
    # Re-assign globally consistent indices. Sub-tasks produce local
    # indices (0..chunk_size-1) because they don't know their global
    # offset. Chord order guarantees the flat list matches upload order.
    for global_idx, result in enumerate(flat):
        result["index"] = global_idx
    merged = flat

    complete_job(parent_job_id, merged)
    logger.info(
        "Mega-batch finalized",
        parent_job_id=parent_job_id,
        total_results=len(merged),
    )
