from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


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


class ReadinessResponse(BaseModel):
    models_loaded: bool
    database: bool
    redis: bool
