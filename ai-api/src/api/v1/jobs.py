from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request

from src.middleware.auth import verify_api_key
from src.schemas.common import APIResponse
from src.schemas.jobs import JobStatusResponse

router = APIRouter(prefix="/jobs", tags=["Async Jobs"])


@router.get("/{job_id}", response_model=APIResponse)
async def get_job_status(
    request: Request,
    job_id: uuid.UUID,
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Get the status and results of an async job."""
    from src.db.repositories.job_repo import JobRepository
    from src.db.session import get_session

    async for session in get_session():
        repo = JobRepository(session)
        job = await repo.get(job_id)
        if job is None:
            return APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={"code": "NOT_FOUND", "message": f"Job {job_id} not found"},
            )

        data = JobStatusResponse(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            total_items=job.total_items,
            processed_items=job.processed_items,
            created_at=job.created_at,
            completed_at=job.completed_at,
            result=job.result if job.status == "completed" else None,
            error=job.error if job.status == "failed" else None,
        )
        return APIResponse(
            success=True,
            request_id=getattr(request.state, "request_id", ""),
            data=data.model_dump(),
        )
