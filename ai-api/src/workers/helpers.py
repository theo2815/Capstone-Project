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
        max_dim: Pipeline-specific max dimension (0 = use global setting).
            Blur=640, face=768, bib=1024.
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


def decode_gray_from_path(path: str, max_dim: int = 640) -> np.ndarray | None:
    """Load an image from the blob store and decode directly to grayscale.

    Optimised for blur detection: uses cv2.imdecode(IMREAD_GRAYSCALE) which is
    ~30% faster than colour decode, skips EXIF rotation (irrelevant for blur
    metrics), and avoids PIL entirely. Mirrors the fast path in the streaming
    blur endpoint.

    Args:
        path: Blob store file path.
        max_dim: Max dimension to downscale to (default 640 for blur pipeline).
    """
    try:
        from src.utils.blob_store import load_blob

        raw_bytes = load_blob(path)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        h, w = gray.shape
        if max_dim > 0 and max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gray
    except Exception:
        return None


def decode_grays_from_paths(
    paths: list[str], max_dim: int = 640, max_workers: int = 0,
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
        max_dim: Pipeline-specific max dimension (0 = global setting).
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
    job_id: str, processed: int, total: int, every_n: int | None = None
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
        repo.update_progress(uuid.UUID(job_id), processed, progress)


def complete_job(job_id: str, results: list[dict]) -> None:
    """Mark a job as completed, clean up blobs, and dispatch webhooks."""
    with get_sync_session() as session:
        repo = SyncJobRepository(session)
        repo.complete(uuid.UUID(job_id), results)

    # Clean up staged image blobs (no-op if blobs don't exist)
    from src.utils.blob_store import cleanup_batch

    cleanup_batch(job_id)

    dispatch_webhook_sync(
        event="job.completed",
        payload={"job_id": job_id, "result_count": len(results)},
    )


def fail_job(job_id: str, error: str) -> None:
    """Mark a job as failed, clean up blobs, and dispatch webhooks."""
    with get_sync_session() as session:
        repo = SyncJobRepository(session)
        repo.fail(uuid.UUID(job_id), error)

    # Clean up staged image blobs (no-op if blobs don't exist)
    from src.utils.blob_store import cleanup_batch

    cleanup_batch(job_id)

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
