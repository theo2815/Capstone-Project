from __future__ import annotations

from src.utils.logging import get_logger

logger = get_logger(__name__)


class JobService:
    """Business logic for async job lifecycle management."""

    def __init__(self, job_repo) -> None:
        self.job_repo = job_repo

    async def create_job(self, job_type: str, total_items: int) -> dict:
        job = await self.job_repo.create(job_type=job_type, total_items=total_items)
        return {
            "job_id": str(job.id),
            "status": job.status,
            "total_items": job.total_items,
        }

    async def get_status(self, job_id):
        return await self.job_repo.get(job_id)
