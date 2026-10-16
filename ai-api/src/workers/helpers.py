from __future__ import annotations

import base64
import os
import uuid

import cv2
import numpy as np

from src.db.repositories.sync_job_repo import SyncJobRepository
from src.db.repositories.sync_webhook_repo import SyncWebhookRepository
from src.db.sync_session import get_sync_session
from src.utils.logging import get_logger

logger = get_logger(__name__)


def claim_job(job_id: str) -> tuple[str, str, list | dict | None]:
    """Acquire this delivery's DB lease before accessing any staged input."""
    token = uuid.uuid4().hex
    from src.config import get_settings

    with get_sync_session() as session:
        repo = SyncJobRepository(session)
        state = repo.claim(uuid.UUID(job_id), token, get_settings().JOB_LEASE_SECONDS)
        result = repo.get(uuid.UUID(job_id)).result if state == "terminal" else None
    return state, token, result


def touch_mega_job(job_id: str) -> bool:
    """Refresh the parent lease without changing aggregate result progress."""
    from src.config import get_settings

    with get_sync_session() as session:
        return SyncJobRepository(session).touch_mega(uuid.UUID(job_id),
                                                     get_settings().JOB_LEASE_SECONDS)


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
        if raw_bytes is None:
            raw_bytes = base64.b64decode(b64_data)
        return _decode_raw_bytes(raw_bytes)
    except Exception:
        return None


def decode_image_from_path(path: str, max_dim: int = 0) -> np.ndarray | None:
    """Load an image from the blob store and decode to BGR numpy array.

    Args:
        path: Blob store file path.
        max_dim: Max dimension override (0 = use MAX_INFERENCE_DIMENSION).
            The blur classify worker passes BLUR_CLASSIFY_DECODE_DIM, the
            calibrated value shared with its endpoint.
    """
    try:
        from src.utils.blob_store import load_blob

        raw_bytes = load_blob(path)
        return _decode_raw_bytes(raw_bytes, max_dim=max_dim)
    except Exception:
        return None


def _decode_raw_bytes(raw_bytes: bytes, max_dim: int = 0) -> np.ndarray | None:
    """Shared decode logic: raw bytes -> EXIF-corrected, downscaled BGR array.

    Args:
        raw_bytes: Raw image file bytes.
        max_dim: Pipeline-specific max dimension override. 0 uses the global
            MAX_INFERENCE_DIMENSION setting.
    """
    import io

    from PIL import Image, ImageOps

    from src.utils.image_utils import downscale_for_inference

    pil_img = Image.open(io.BytesIO(raw_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)
    # Skip .convert("RGB") when already RGB (always true for JPEG after EXIF
    # transpose). np.asarray gives a zero-copy view when possible.
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    rgb_array = np.asarray(pil_img)
    image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    image = downscale_for_inference(image, max_dim=max_dim) if max_dim else downscale_for_inference(image)
    return image


def decode_gray_from_path(path: str, max_dim: int = 0) -> np.ndarray | None:
    """Load an image from the blob store and decode directly to grayscale.

    Uses the same OpenCV grayscale decode and resize as both blur endpoints.

    Args:
        path: Blob store file path.
        max_dim: Max dimension to downscale to (0 uses BLUR_DETECT_DECODE_DIM).
    """
    try:
        from src.utils.blob_store import load_blob

        raw_bytes = load_blob(path)
        from src.config import get_settings
        from src.utils.image_utils import decode_gray_for_blur

        return decode_gray_for_blur(
            raw_bytes, max_dim or get_settings().BLUR_DETECT_DECODE_DIM
        )
    except Exception:
        return None


def decode_grays_from_paths(
    paths: list[str], max_dim: int = 0, max_workers: int = 0,
) -> list[np.ndarray | None]:
    """Decode multiple images to grayscale in parallel using a thread pool.

    Same pattern as decode_images_from_paths but uses the fast grayscale path.
    When *max_workers* is 0 (default), scales to ``min(cpu_count, 8)``.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _decode_one(path: str) -> np.ndarray | None:
        return decode_gray_from_path(path, max_dim=max_dim)

    if max_workers <= 0:
        max_workers = min(os.cpu_count() or 4, 8)
    workers = min(max_workers, len(paths)) or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_decode_one, paths))


def decode_images_from_paths(
    paths: list[str], max_dim: int = 0, max_workers: int = 0,
) -> list[np.ndarray | None]:
    """Decode multiple images in parallel using a thread pool.

    PIL/cv2 decode releases the GIL during C-level work, so threads provide
    real parallelism for overlapping disk I/O with CPU decode.

    Args:
        paths: Blob store file paths.
        max_dim: Max dimension override (0 = use MAX_INFERENCE_DIMENSION).
        max_workers: Thread pool size. 0 (default) scales to ``min(cpu_count, 8)``.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _decode_one(path: str) -> np.ndarray | None:
        return decode_image_from_path(path, max_dim=max_dim)

    if max_workers <= 0:
        max_workers = min(os.cpu_count() or 4, 8)
    workers = min(max_workers, len(paths)) or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_decode_one, paths))


def update_job_progress(
    job_id: str, processed: int, total: int, every_n: int | None = None,
    attempt_token: str | None = None,
) -> None:
    """Update job progress in the database, throttled to every N items.

    Always writes on the final item to ensure 100% progress is recorded.
    When *every_n* is None, scales adaptively to produce ~20 updates
    regardless of batch size.
    """
    if not job_id:
        return  # mega-batch sub-tasks have no individual job to update
    if every_n is None:
        every_n = max(10, total // 5)  # ~5 updates per task (was 20)
    if processed < total and processed % every_n != 0:
        return
    progress = processed / total if total > 0 else 0.0
    with get_sync_session() as session:
        repo = SyncJobRepository(session)
        from src.config import get_settings
        repo.update_progress(uuid.UUID(job_id), processed, progress, attempt_token,
                             get_settings().JOB_LEASE_SECONDS)


def complete_job(job_id: str, results: list[dict], attempt_token: str | None = None) -> bool:
    """Commit one result; duplicates never replace it or send another webhook."""
    with get_sync_session() as session:
        repo = SyncJobRepository(session)
        changed = repo.complete(uuid.UUID(job_id), results, attempt_token)
        if not changed:
            return False
        job = repo.get(uuid.UUID(job_id))
        api_key_id = job.api_key_id

    dispatch_webhook_sync(
        event="job.completed",
        payload={"job_id": job_id, "result_count": len(results)},
        api_key_id=api_key_id,
    )
    return True


def fail_job(job_id: str, error: str, attempt_token: str | None = None) -> bool:
    """Commit one failure; retained inputs are removed by maintenance later."""
    with get_sync_session() as session:
        repo = SyncJobRepository(session)
        changed = repo.fail(uuid.UUID(job_id), error, attempt_token)
        if not changed:
            return False
        job = repo.get(uuid.UUID(job_id))
        api_key_id = job.api_key_id

    dispatch_webhook_sync(
        event="job.failed",
        payload={"job_id": job_id, "error": error},
        api_key_id=api_key_id,
    )
    return True


def dispatch_webhook_sync(event: str, payload: dict, api_key_id: str | None = None) -> int:
    """Query matching webhook subscriptions and queue delivery tasks.

    Scoped by *api_key_id* (the owner of the job that fired the event) so a
    tenant's job completion only fans out to that tenant's subscriptions.
    """
    count = 0
    with get_sync_session() as session:
        repo = SyncWebhookRepository(session)
        webhooks = repo.list_by_event(event, api_key_id)

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
