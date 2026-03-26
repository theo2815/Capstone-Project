from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Job


class JobRepository:
    """Repository for async job tracking."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self, job_type: str, total_items: int = 0, api_key_id: str | None = None
    ) -> Job:
        job = Job(
            job_type=job_type,
            total_items=total_items,
            status="pending",
            api_key_id=api_key_id,
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def get(self, job_id: uuid.UUID) -> Job | None:
        result = await self.session.execute(
            select(Job).where(Job.id == job_id)
        )
        return result.scalar_one_or_none()

    async def get_by_owner(
        self, job_id: uuid.UUID, api_key_id: str
    ) -> Job | None:
        result = await self.session.execute(
            select(Job).where(Job.id == job_id, Job.api_key_id == api_key_id)
        )
        return result.scalar_one_or_none()

    async def update_progress(
        self, job_id: uuid.UUID, processed_items: int, progress: float
    ) -> None:
        result = await self.session.execute(
            select(Job).where(Job.id == job_id).with_for_update()
        )
        job = result.scalar_one_or_none()
        if job:
            job.processed_items = processed_items
            job.progress = progress
            job.status = "processing"
            await self.session.flush()

    async def complete(
        self, job_id: uuid.UUID, result: dict | list
    ) -> None:
        row = await self.session.execute(
            select(Job).where(Job.id == job_id).with_for_update()
        )
        job = row.scalar_one_or_none()
        if job:
            job.status = "completed"
            job.progress = 1.0
            job.processed_items = job.total_items
            job.result = result
            job.completed_at = datetime.now(UTC)
            await self.session.flush()

    async def count_active_by_key(self, api_key_id: str) -> int:
        """Count pending + processing jobs for a given API key.

        Excludes stale jobs older than task_time_limit + buffer (3600s + 300s)
        to prevent zombie jobs from permanently consuming backpressure slots.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=3600 + 300)
        result = await self.session.execute(
            select(func.count())
            .select_from(Job)
            .where(
                Job.api_key_id == api_key_id,
                Job.status.in_(["pending", "processing"]),
                Job.created_at > cutoff,
            )
        )
        return result.scalar_one()

    async def fail(self, job_id: uuid.UUID, error: str) -> None:
        row = await self.session.execute(
            select(Job).where(Job.id == job_id).with_for_update()
        )
        job = row.scalar_one_or_none()
        if job:
            job.status = "failed"
            job.error = error
            job.completed_at = datetime.now(UTC)
            await self.session.flush()

    async def reap_stale_jobs(self, max_age_seconds: int = 3900) -> int:
        """Mark stale pending/processing jobs as failed.

        Returns the number of jobs reaped.
        """
        from sqlalchemy import update

        cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
        result = await self.session.execute(
            update(Job)
            .where(
                Job.status.in_(["pending", "processing"]),
                Job.created_at <= cutoff,
            )
            .values(
                status="failed",
                error="Worker timeout — job expired",
                completed_at=datetime.now(UTC),
            )
        )
        await self.session.flush()
        return result.rowcount

    async def cleanup_old_jobs(self, retention_days: int = 7) -> int:
        """Delete completed/failed jobs older than the retention period.

        Returns the number of jobs deleted.
        """
        from sqlalchemy import delete

        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        result = await self.session.execute(
            delete(Job).where(
                Job.status.in_(["completed", "failed"]),
                Job.created_at <= cutoff,
            )
        )
        await self.session.flush()
        return result.rowcount
