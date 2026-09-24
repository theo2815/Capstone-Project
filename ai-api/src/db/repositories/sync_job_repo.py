from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, delete, or_, select, update
from sqlalchemy.orm import Session

from src.db.models import Job


class SyncJobRepository:
    """Synchronous repository for job tracking, used by Celery tasks."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, job_id: uuid.UUID) -> Job | None:
        result = self.session.execute(select(Job).where(Job.id == job_id))
        return result.scalar_one_or_none()

    def claim(self, job_id: uuid.UUID, attempt_token: str, lease_seconds: int = 3900) -> str:
        """Claim a pending or expired attempt; terminal jobs are immutable."""
        now = datetime.now(UTC)
        changed = self.session.execute(
            update(Job).where(Job.id == job_id, or_(
                Job.status == "pending",
                and_(Job.status == "processing", Job.lease_expires_at <= now),
            )).values(status="processing", attempt_token=attempt_token,
                      lease_expires_at=now + timedelta(seconds=lease_seconds))
        )
        if changed.rowcount:
            return "claimed"
        job = self.get(job_id)
        if job is None:
            return "missing"
        return "busy" if job.status == "processing" else "terminal"

    def update_progress(
        self, job_id: uuid.UUID, processed_items: int, progress: float,
        attempt_token: str | None = None, lease_seconds: int = 3900,
    ) -> None:
        condition = [Job.id == job_id, Job.status.in_(["pending", "processing"]),
                     Job.processed_items <= processed_items]
        if attempt_token is not None:
            condition.extend([Job.status == "processing", Job.attempt_token == attempt_token])
        values = dict(processed_items=processed_items, progress=progress, status="processing")
        if attempt_token is not None:
            values["lease_expires_at"] = datetime.now(UTC) + timedelta(seconds=lease_seconds)
        self.session.execute(
            update(Job)
            .where(*condition)
            .values(**values)
        )
        self.session.flush()

    def touch_mega(self, job_id: uuid.UUID, lease_seconds: int) -> bool:
        """Keep a parent alive while one of its chord children is running."""
        changed = self.session.execute(
            update(Job).where(Job.id == job_id, Job.status.in_(["pending", "processing"]))
            .values(status="processing",
                    lease_expires_at=datetime.now(UTC) + timedelta(seconds=lease_seconds))
        )
        return bool(changed.rowcount)

    def complete(self, job_id: uuid.UUID, result: dict | list,
                 attempt_token: str | None = None) -> bool:
        condition = [Job.id == job_id, Job.status.in_(["pending", "processing"])]
        if attempt_token is not None:
            condition.extend([Job.status == "processing", Job.attempt_token == attempt_token])
        changed = self.session.execute(
            update(Job).where(*condition)
            .values(status="completed", progress=1.0, processed_items=Job.total_items,
                    result=result, completed_at=datetime.now(UTC), lease_expires_at=None)
        )
        return bool(changed.rowcount)

    def fail(self, job_id: uuid.UUID, error: str,
             attempt_token: str | None = None) -> bool:
        condition = [Job.id == job_id, Job.status.in_(["pending", "processing"])]
        if attempt_token is not None:
            condition.extend([Job.status == "processing", Job.attempt_token == attempt_token])
        changed = self.session.execute(
            update(Job).where(*condition)
            .values(status="failed", error=error, completed_at=datetime.now(UTC),
                    lease_expires_at=None)
        )
        return bool(changed.rowcount)

    def reap_stale_jobs(self, max_age_seconds: int = 3900) -> int:
        """Mark stale pending/processing jobs as failed."""
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=max_age_seconds)
        result = self.session.execute(
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
        self.session.flush()
        return result.rowcount

    def cleanup_old_jobs(self, retention_days: int = 7) -> int:
        """Delete completed/failed jobs older than the retention period."""
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        result = self.session.execute(
            delete(Job).where(
                Job.status.in_(["completed", "failed"]),
                Job.completed_at <= cutoff,
            )
        )
        self.session.flush()
        return result.rowcount
