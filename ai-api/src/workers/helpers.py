from __future__ import annotations

import base64
import uuid

import cv2
import numpy as np

from src.db.repositories.sync_job_repo import SyncJobRepository
from src.db.repositories.sync_webhook_repo import SyncWebhookRepository
from src.db.sync_session import get_sync_session
from src.utils.logging import get_logger

logger = get_logger(__name__)


def decode_base64_image(b64_data: str) -> np.ndarray | None:
    """Decode a base64 string to a BGR numpy array."""
    try:
        image_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


def update_job_progress(job_id: str, processed: int, total: int) -> None:
    """Update job progress in the database."""
    progress = processed / total if total > 0 else 0.0
    with get_sync_session() as session:
        repo = SyncJobRepository(session)
        repo.update_progress(uuid.UUID(job_id), processed, progress)


def complete_job(job_id: str, results: list[dict]) -> None:
    """Mark a job as completed and dispatch webhooks."""
    with get_sync_session() as session:
        repo = SyncJobRepository(session)
        repo.complete(uuid.UUID(job_id), results)

    dispatch_webhook_sync(
        event="job.completed",
        payload={"job_id": job_id, "result_count": len(results)},
    )


def fail_job(job_id: str, error: str) -> None:
    """Mark a job as failed and dispatch webhooks."""
    with get_sync_session() as session:
        repo = SyncJobRepository(session)
        repo.fail(uuid.UUID(job_id), error)

    dispatch_webhook_sync(
        event="job.failed",
        payload={"job_id": job_id, "error": error},
    )


def dispatch_webhook_sync(event: str, payload: dict) -> int:
    """Query matching webhook subscriptions and queue delivery tasks."""
    count = 0
    with get_sync_session() as session:
        repo = SyncWebhookRepository(session)
        webhooks = repo.list_by_event(event)

    for wh in webhooks:
        from src.workers.tasks.webhook_tasks import deliver_webhook

        deliver_webhook.delay(
            url=wh.url,
            event=event,
            payload=payload,
            secret=wh.secret,
        )
        count += 1

    if count > 0:
        logger.info("Webhooks dispatched from worker", event=event, count=count)
    return count
