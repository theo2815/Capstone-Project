from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
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
        job = self.get(job_id)
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
