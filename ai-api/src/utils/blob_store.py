"""Filesystem blob store for batch image staging.

Replaces base64-encoded image data in Celery task messages with file-path
references.  Image bytes are written to a shared Docker volume and cleaned
up after the job completes (or by the periodic stale-blob reaper).

Typical flow:
    API endpoint → store_batch(job_id, raw_bytes_list) → list[str]  (paths)
    Celery task  → load_blob(path)                     → bytes
    complete_job → cleanup_batch(job_id)                → None
"""
from __future__ import annotations

import os
import shutil
import time

from src.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _blob_root() -> str:
    return get_settings().BLOB_STORE_PATH


def store_batch(job_id: str, raw_bytes_list: list[bytes]) -> list[str]:
    """Write image bytes to ``{BLOB_ROOT}/{job_id}/{index:05d}.bin``.

    Uses atomic write (tmp -> rename) so workers never read a partial file.
    Writes are parallelised with a thread pool — filesystem I/O releases
    the GIL, so threads provide real concurrency here.
    Returns a list of absolute paths in the same order as *raw_bytes_list*.
    """
    from concurrent.futures import ThreadPoolExecutor

    job_dir = os.path.join(_blob_root(), job_id)
    os.makedirs(job_dir, exist_ok=True)

    def _write_one(idx_raw: tuple[int, bytes]) -> str:
        idx, raw = idx_raw
        final_path = os.path.join(job_dir, f"{idx:05d}.bin")
        tmp_path = final_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(raw)
        os.replace(tmp_path, final_path)
        return final_path

    max_workers = min(8, len(raw_bytes_list)) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        paths = list(pool.map(_write_one, enumerate(raw_bytes_list)))

    logger.debug(
        "Blob batch stored",
        job_id=job_id,
        count=len(paths),
        total_bytes=sum(len(b) for b in raw_bytes_list),
    )
    return paths


def load_blob(path: str) -> bytes:
    """Read and return the raw image bytes at *path*.

    Raises ``FileNotFoundError`` with a clear message if the blob was
    already cleaned up (e.g. job completed before the worker fetched it).
    """
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Image blob not found: {path}. "
            "The job may have already completed or been cleaned up."
        )


def cleanup_batch(job_id: str) -> None:
    """Remove the staging directory for *job_id*.

    Called by ``complete_job`` and ``fail_job``.  Safe to call if the
    directory does not exist (e.g. job used the streaming path).
    """
    job_dir = os.path.join(_blob_root(), job_id)
    shutil.rmtree(job_dir, ignore_errors=True)


def cleanup_stale_blobs(max_age_seconds: int = 7200) -> int:
    """Remove blob directories older than *max_age_seconds*.

    Catches jobs that crashed before cleanup.  Returns the number of
    directories removed.
    """
    root = _blob_root()
    if not os.path.isdir(root):
        return 0

    cutoff = time.time() - max_age_seconds
    removed = 0
    for entry in os.scandir(root):
        if entry.is_dir() and entry.stat().st_mtime < cutoff:
            shutil.rmtree(entry.path, ignore_errors=True)
            removed += 1

    if removed:
        logger.info("Cleaned up stale blob directories", removed=removed)
    return removed
