"""Mega-batch utilities: accept up to 500 images, auto-chunk into sub-tasks.

Dispatches sub-tasks as a Celery chord that feeds into
``finalize_mega_batch`` to merge results into the parent job.
"""
from __future__ import annotations

from celery import chord, group

from src.api.v1.batch_utils import store_blobs_and_get_paths
from src.config import get_settings


def dispatch_mega_batch(
    job_id: str,
    raw_bytes_list: list[bytes],
    task_func,
    extra_kwargs: dict | None = None,
    refs: list[str] | None = None,
) -> None:
    """Store blobs, chunk paths, and dispatch as a Celery chord.

    Args:
        job_id: Parent job ID.
        raw_bytes_list: Validated raw image bytes.
        task_func: The Celery task to call for each chunk (e.g. blur_detect_batch).
        extra_kwargs: Additional keyword arguments passed to each sub-task
            (e.g. blur_type, operation, person_name).
        refs: Optional per-image caller references (e.g. the source filename),
            aligned with *raw_bytes_list*. When provided, each sub-task is given
            its chunk slice as a ``refs`` kwarg so per-image results can echo a
            stable caller ref instead of relying solely on positional index.
            Only pass for tasks whose signature accepts ``refs``.
    """
    from src.workers.tasks.maintenance_tasks import finalize_mega_batch

    settings = get_settings()
    chunk_size = settings.MAX_BATCH_SIZE

    image_paths = store_blobs_and_get_paths(job_id, raw_bytes_list)

    # Build sub-task signatures
    extra = extra_kwargs or {}
    tasks = []
    for chunk_start in range(0, len(image_paths), chunk_size):
        chunk_paths = image_paths[chunk_start:chunk_start + chunk_size]

        # Re-index: each sub-task works with a subset but indices must be
        # globally consistent.  We pass the chunk paths and the global offset.
        # The task receives standard (job_id, image_paths, ...) — the
        # finalize callback uses the original index stored in each result dict.
        #
        # Sub-tasks use a throwaway job_id (empty string) since the parent
        # job is the real tracking record.  Progress updates are skipped for
        # sub-tasks and the parent is updated once by finalize_mega_batch.
        chunk_kwargs = dict(extra)
        if refs is not None:
            chunk_kwargs["refs"] = refs[chunk_start:chunk_start + chunk_size]
        sig = task_func.s("", chunk_paths, **chunk_kwargs)
        tasks.append(sig)

    # Chord: run all sub-tasks in parallel → finalize merges results
    callback = finalize_mega_batch.s(job_id)
    chord(group(tasks))(callback)
