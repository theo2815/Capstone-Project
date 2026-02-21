from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobCreateResponse(BaseModel):
    job_id: UUID
    status: JobStatus = JobStatus.PENDING
    total_items: int
    poll_url: str


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    progress: float = Field(ge=0, le=1.0)
    total_items: int
    processed_items: int
    created_at: datetime
    completed_at: datetime | None = None
    result: list[dict] | None = None
    error: str | None = None
