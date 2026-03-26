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


def decode_base64_image(
    b64_data: str, *, raw_bytes: bytes | None = None
) -> np.ndarray | None:
    """Decode a base64 string to a BGR numpy array with EXIF rotation.

    Uses PIL for decoding to match the single-image pipeline
    (``validate_and_decode``), ensuring EXIF orientation is applied
    consistently for phone photos. Applies ``downscale_for_inference()``
    to cap image dimensions, preventing OOM on high-resolution batch images.

    Args:
        b64_data: Base64-encoded image data.
        raw_bytes: Pre-decoded bytes to avoid redundant base64 decode.
    """
    try:
        import io

        from PIL import Image, ImageOps

        from src.utils.image_utils import downscale_for_inference

        if raw_bytes is None:
            raw_bytes = base64.b64decode(b64_data)
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        rgb_array = np.array(pil_img.convert("RGB"))
        image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        image = downscale_for_inference(image)
        return image
    except Exception:
        return None


def update_job_progress(
    job_id: str, processed: int, total: int, every_n: int = 10
) -> None:
    """Update job progress in the database, throttled to every N items.

    Always writes on the final item to ensure 100% progress is recorded.
    """
    if processed < total and processed % every_n != 0:
        return
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

    from src.utils.crypto import decrypt_secret

    for wh in webhooks:
        from src.workers.tasks.webhook_tasks import deliver_webhook

        deliver_webhook.delay(
            url=wh.url,
            event=event,
            payload=payload,
            secret=decrypt_secret(wh.secret) if wh.secret else None,
        )
        count += 1

    if count > 0:
        logger.info("Webhooks dispatched from worker", event=event, count=count)
    return count
