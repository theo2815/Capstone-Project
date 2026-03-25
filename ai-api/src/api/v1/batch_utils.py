"""Shared utilities for batch endpoints.

Extracts the duplicated file validation, base64 encoding, job creation,
and response formatting that all 4 batch endpoints share.
"""
from __future__ import annotations

import base64

from fastapi import Request, UploadFile
from fastapi.responses import JSONResponse

from src.schemas.common import APIResponse
from src.schemas.jobs import JobCreateResponse
from src.utils.image_utils import validate_batch_file


async def validate_and_encode_batch(
    request: Request,
    files: list[UploadFile],
    max_batch_size: int,
    max_file_size: int,
) -> list[str] | JSONResponse:
    """Validate batch files and return base64-encoded data.

    Returns a list of base64 strings on success, or a JSONResponse error.
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

    image_data_list = []
    for f in files:
        raw = await f.read()
        validate_batch_file(raw, f.filename or "unknown", max_file_size)
        image_data_list.append(base64.b64encode(raw).decode("ascii"))

    return image_data_list


async def create_batch_job(
    request: Request,
    job_type: str,
    total_items: int,
    api_key_id: str | None,
) -> str | JSONResponse:
    """Create a job record and return the job ID string.

    Returns a 429 JSONResponse if the caller already has too many active jobs
    (SCALE-2 backpressure).
    """
    from src.db.repositories.job_repo import JobRepository
    from src.db.session import get_session_ctx

    settings = request.app.state.settings
    max_active = settings.MAX_ACTIVE_JOBS_PER_KEY

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
