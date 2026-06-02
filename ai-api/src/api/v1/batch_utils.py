"""Shared utilities for batch endpoints.

Extracts the duplicated file validation, blob storage, job creation,
and response formatting that all batch endpoints share.

Image bytes are written to a shared filesystem volume (blob store) and
only file-path references are passed in Celery task messages.  This
replaces the old base64-in-Redis approach which consumed ~333 MB per
50-image task and caused silent eviction at scale.
"""
from __future__ import annotations

import asyncio

from fastapi import Request, UploadFile
from fastapi.responses import JSONResponse

from src.schemas.common import APIResponse
from src.schemas.jobs import JobCreateResponse
from src.utils.image_utils import validate_batch_file


async def validate_batch_files(
    request: Request,
    files: list[UploadFile],
    max_batch_size: int,
    max_file_size: int,
) -> list[bytes] | JSONResponse:
    """Validate batch files and return raw bytes (no base64 encoding).

    Returns a list of raw byte strings on success, or a JSONResponse error.
    """
    request_id = getattr(request.state, "request_id", "")

    if len(files) == 0:
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                success=False,
                request_id=request_id,
                error={"code": "EMPTY_BATCH", "message": "No files provided"},
            ).model_dump(mode="json"),
        )

    if len(files) > max_batch_size:
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                success=False,
                request_id=request_id,
                error={
                    "code": "BATCH_TOO_LARGE",
                    "message": f"Maximum {max_batch_size} files per batch",
                },
            ).model_dump(mode="json"),
        )

    async def _read_and_validate(f: UploadFile) -> bytes:
        raw = await f.read()
        validate_batch_file(raw, f.filename or "unknown", max_file_size, f.content_type)
        return raw

    raw_bytes_list = list(await asyncio.gather(
        *[_read_and_validate(f) for f in files]
    ))
    return raw_bytes_list


# Keep the old name as an alias so any lingering imports don't break
# at import time.  Callers should migrate to validate_batch_files.
validate_and_encode_batch = validate_batch_files


def store_blobs_and_get_paths(job_id: str, raw_bytes_list: list[bytes]) -> list[str]:
    """Write validated image bytes to the blob store and return file paths.

    Must be called AFTER ``create_batch_job`` so that the *job_id* directory
    can be cleaned up reliably on job completion or failure.
    """
    from src.utils.blob_store import store_batch

    return store_batch(job_id, raw_bytes_list)


async def create_batch_job(
    request: Request,
    job_type: str,
    total_items: int,
    api_key_id: str | None,
    rate_tier: str | None = None,
) -> str | JSONResponse:
    """Create a job record and return the job ID string.

    Returns a 429 JSONResponse if the caller already has too many active jobs
    (SCALE-2 backpressure). Internal-tier callers get a higher active-job
    ceiling (bulk indexing submits several jobs per drain).
    """
    from src.db.repositories.job_repo import JobRepository
    from src.db.session import get_session_ctx

    settings = request.app.state.settings
    max_active = (
        settings.MAX_ACTIVE_JOBS_PER_KEY_INTERNAL
        if rate_tier == "internal"
        else settings.MAX_ACTIVE_JOBS_PER_KEY
    )

    async with get_session_ctx() as session:
        repo = JobRepository(session)

        if api_key_id and max_active > 0:
            active_count = await repo.count_active_by_key(api_key_id)
            if active_count >= max_active:
                return JSONResponse(
                    status_code=429,
                    content=APIResponse(
                        success=False,
                        request_id=getattr(request.state, "request_id", ""),
                        error={
                            "code": "TOO_MANY_JOBS",
                            "message": (
                                f"You already have {active_count} active jobs. "
                                f"Maximum is {max_active}. Wait for existing jobs to complete."
                            ),
                        },
                    ).model_dump(mode="json"),
                )

        job = await repo.create(
            job_type=job_type,
            total_items=total_items,
            api_key_id=api_key_id,
        )
        return str(job.id)


def batch_accepted_response(
    request: Request, job_id: str, total_items: int
) -> JSONResponse:
    """Return a standard 202 Accepted response for batch jobs."""
    data = JobCreateResponse(
        job_id=job_id,
        status="pending",
        total_items=total_items,
        poll_url=f"/api/v1/jobs/{job_id}",
    )
    return JSONResponse(
        status_code=202,
        content=APIResponse(
            success=True,
            request_id=getattr(request.state, "request_id", ""),
            data=data.model_dump(mode="json"),
        ).model_dump(mode="json"),
    )
