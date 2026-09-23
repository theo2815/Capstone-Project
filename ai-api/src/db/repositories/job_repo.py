from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, func, or_, select, text, update
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
        await self.session.execute(
            update(Job).where(Job.id == job_id, Job.status.in_(["pending", "processing"]),
                              Job.processed_items <= processed_items)
            .values(processed_items=processed_items, progress=progress, status="processing")
        )

    async def complete(
        self, job_id: uuid.UUID, result: dict | list
    ) -> None:
        await self.session.execute(
            update(Job).where(Job.id == job_id, Job.status.in_(["pending", "processing"]))
            .values(status="completed", progress=1.0, processed_items=Job.total_items,
                    result=result, completed_at=datetime.now(UTC), lease_expires_at=None)
        )

    async def lock_admission(self, api_key_id: str) -> None:
        """Serialize submissions for one key until the surrounding transaction commits."""
        await self.session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
            {"key": api_key_id},
        )

    async def count_active_by_key(self, api_key_id: str) -> int:
        """Count active jobs; the reaper, not admission, releases stale slots."""
        result = await self.session.execute(
            select(func.count())
            .select_from(Job)
            .where(
                Job.api_key_id == api_key_id,
                Job.status.in_(["pending", "processing"]),
            )
        )
        return result.scalar_one()

    async def fail(self, job_id: uuid.UUID, error: str) -> bool:
        changed = await self.session.execute(
            update(Job).where(Job.id == job_id, Job.status.in_(["pending", "processing"]))
            .values(status="failed", error=error, completed_at=datetime.now(UTC),
                    lease_expires_at=None)
        )
        return bool(changed.rowcount)

    async def reap_stale_jobs(self, max_age_seconds: int = 3900) -> int:
        """Mark stale pending/processing jobs as failed.

        Returns the number of jobs reaped.
        """
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=max_age_seconds)
        result = await self.session.execute(
            update(Job)
            .where(
                or_(
                    and_(Job.status == "pending", Job.created_at <= cutoff),
                    and_(Job.status == "processing", Job.lease_expires_at.is_not(None),
                         Job.lease_expires_at <= cutoff),
                    and_(Job.status == "processing", Job.lease_expires_at.is_(None),
                         Job.created_at <= cutoff),
                ),
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
                Job.completed_at <= cutoff,
            )
        )
        await self.session.flush()
        return result.rowcount
