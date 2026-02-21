from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Job


class JobRepository:
    """Repository for async job tracking."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self, job_type: str, total_items: int = 0
    ) -> Job:
        job = Job(job_type=job_type, total_items=total_items, status="pending")
        self.session.add(job)
        await self.session.flush()
        return job

    async def get(self, job_id: uuid.UUID) -> Job | None:
        result = await self.session.execute(
            select(Job).where(Job.id == job_id)
        )
        return result.scalar_one_or_none()

    async def update_progress(
        self, job_id: uuid.UUID, processed_items: int, progress: float
    ) -> None:
        job = await self.get(job_id)
        if job:
            job.processed_items = processed_items
            job.progress = progress
            job.status = "processing"
            await self.session.flush()

    async def complete(
        self, job_id: uuid.UUID, result: dict | list
    ) -> None:
        job = await self.get(job_id)
        if job:
            job.status = "completed"
            job.progress = 1.0
            job.processed_items = job.total_items
            job.result = result
            job.completed_at = datetime.utcnow()
            await self.session.flush()

    async def fail(self, job_id: uuid.UUID, error: str) -> None:
        job = await self.get(job_id)
        if job:
            job.status = "failed"
            job.error = error
            job.completed_at = datetime.utcnow()
            await self.session.flush()
