from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from src.db.models import Job


class SyncJobRepository:
    """Synchronous repository for job tracking, used by Celery tasks."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, job_id: uuid.UUID) -> Job | None:
        result = self.session.execute(select(Job).where(Job.id == job_id))
        return result.scalar_one_or_none()

    def update_progress(
        self, job_id: uuid.UUID, processed_items: int, progress: float
    ) -> None:
        result = self.session.execute(
            select(Job).where(Job.id == job_id).with_for_update()
        )
        job = result.scalar_one_or_none()
        if job:
            job.processed_items = processed_items
            job.progress = progress
            job.status = "processing"
            self.session.flush()

    def complete(self, job_id: uuid.UUID, result: dict | list) -> None:
        job = self.get(job_id)
        if job:
            job.status = "completed"
            job.progress = 1.0
            job.processed_items = job.total_items
            job.result = result
            job.completed_at = datetime.now(UTC)
            self.session.flush()

    def fail(self, job_id: uuid.UUID, error: str) -> None:
        job = self.get(job_id)
        if job:
            job.status = "failed"
            job.error = error
            job.completed_at = datetime.now(UTC)
            self.session.flush()

    def reap_stale_jobs(self, max_age_seconds: int = 3900) -> int:
        """Mark stale pending/processing jobs as failed."""
        cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
        result = self.session.execute(
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
        self.session.flush()
        return result.rowcount

    def cleanup_old_jobs(self, retention_days: int = 7) -> int:
        """Delete completed/failed jobs older than the retention period."""
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        result = self.session.execute(
            delete(Job).where(
                Job.status.in_(["completed", "failed"]),
                Job.created_at <= cutoff,
            )
        )
        self.session.flush()
        return result.rowcount
