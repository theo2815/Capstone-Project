from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None = None


class APIResponse(BaseModel):
    """Standard response wrapper for all endpoints."""

    success: bool
    request_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict | list | None = None
    error: ErrorDetail | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    models_loaded: bool
    database: bool
    redis: bool


class JobResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    progress: float = Field(ge=0, le=1.0)
    created_at: datetime
    completed_at: datetime | None = None
    result: dict | list | None = None
