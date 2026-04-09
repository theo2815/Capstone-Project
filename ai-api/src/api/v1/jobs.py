from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, Request

from src.middleware.auth import check_scope, verify_api_key
from src.schemas.common import APIResponse
from src.schemas.jobs import JobStatusResponse

router = APIRouter(prefix="/jobs", tags=["Async Jobs"])


@router.get("/{job_id}", response_model=APIResponse)
async def get_job_status(
    request: Request,
    job_id: uuid.UUID,
    offset: int = Query(default=0, ge=0, description="Result pagination offset"),
    limit: int = Query(default=100, ge=1, le=500, description="Result page size"),
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Get the status and results of an async job (tenant-isolated).

    When the job is completed, results are paginated via *offset* and *limit*
    to avoid sending multi-MB JSON blobs for large batches.
    """
    check_scope("jobs:read", key_meta)
    from src.db.repositories.job_repo import JobRepository
    from src.db.session import get_session_ctx

    async with get_session_ctx() as session:
        repo = JobRepository(session)
        key_id = key_meta.get("key_id", "")
        job = await repo.get_by_owner(job_id, key_id)
        if job is None:
            return APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={"code": "NOT_FOUND", "message": f"Job {job_id} not found"},
            )

        result_page = None
        result_total = None
        result_offset = None
        result_limit = None

        if job.status == "completed" and job.result:
            full_result = job.result
            result_total = len(full_result)
            result_page = full_result[offset:offset + limit]
            result_offset = offset
            result_limit = limit

        data = JobStatusResponse(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            total_items=job.total_items,
            processed_items=job.processed_items,
            created_at=job.created_at,
            completed_at=job.completed_at,
            result=result_page,
            result_total=result_total,
            result_offset=result_offset,
            result_limit=result_limit,
            error=job.error if job.status == "failed" else None,
        )
        return APIResponse(
            success=True,
            request_id=getattr(request.state, "request_id", ""),
            data=data.model_dump(),
        )
