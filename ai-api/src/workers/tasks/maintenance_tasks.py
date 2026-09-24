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
    from src.config import get_settings
    from src.db.repositories.sync_job_repo import SyncJobRepository
    from src.db.sync_session import get_sync_session

    with get_sync_session() as session:
        repo = SyncJobRepository(session)
        count = repo.reap_stale_jobs(
            max_age_seconds=get_settings().CELERY_VISIBILITY_TIMEOUT + 300)
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
    """Retain active inputs; remove terminal or orphan batches after the retry window."""
    import uuid
    from datetime import UTC, datetime

    from src.config import get_settings
    from src.db.repositories.sync_job_repo import SyncJobRepository
    from src.db.sync_session import get_sync_session
    from src.utils.blob_store import cleanup_batch, list_batches

    cutoff = datetime.now(UTC).timestamp() - get_settings().BLOB_RETENTION_SECONDS
    removed = 0
    with get_sync_session() as session:
        repo = SyncJobRepository(session)
        for job_id, modified in list_batches():
            try:
                job = repo.get(uuid.UUID(job_id))
            except ValueError:
                logger.warning("Invalid staged job ID", job_id=job_id)
                continue
            if job is None:
                expired = modified <= cutoff
            else:
                expired = (job.status in ("completed", "failed")
                           and job.completed_at is not None
                           and job.completed_at.timestamp() <= cutoff)
            if expired:
                try:
                    cleanup_batch(job_id)
                    removed += 1
                except Exception as exc:
                    logger.error("Blob cleanup failed", job_id=job_id,
                                 error_type=type(exc).__name__)
    return removed


@celery_app.task(name="maintenance.finalize_mega_batch")
def finalize_mega_batch(chunk_results: list, parent_job_id: str):
    """Merge results from chunked mega-batch sub-tasks into the parent job.

    Called as the Celery chord callback after all chunk tasks complete.
    Each element of *chunk_results* is a list[dict] from one sub-task.
    """
    import uuid
    from itertools import chain

    from src.db.repositories.sync_job_repo import SyncJobRepository
    from src.db.sync_session import get_sync_session
    from src.workers.helpers import complete_job

    # Flatten sub-task results. Celery chord delivers results in task
    # submission order, so the flat list is already in global image order.
    flat = list(chain.from_iterable(chunk_results))
    with get_sync_session() as session:
        parent = SyncJobRepository(session).get(uuid.UUID(parent_job_id))
        expected = parent.total_items if parent else None
    if expected is None or len(flat) != expected or any(not isinstance(r, dict) for r in flat):
        raise ValueError("Mega-batch results are incomplete")
    # Re-assign globally consistent indices. Sub-tasks produce local
    # indices (0..chunk_size-1) because they don't know their global
    # offset. Chord order guarantees the flat list matches upload order.
    for global_idx, result in enumerate(flat):
        result["index"] = global_idx
    merged = flat

    if complete_job(parent_job_id, merged):
        logger.info("Mega-batch finalized", parent_job_id=parent_job_id,
                    total_results=len(merged))


@celery_app.task(bind=True, name="maintenance.fail_mega_batch")
def fail_mega_batch(self, failed_task_id: str, parent_job_id: str):
    """A failed child or callback leaves one terminal parent failure."""
    from src.workers.helpers import fail_job

    fail_job(parent_job_id, "Mega-batch processing failed")
