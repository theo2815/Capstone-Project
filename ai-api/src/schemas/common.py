from __future__ import annotations

from datetime import UTC, datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None = None


class APIResponse(BaseModel, Generic[T]):
    """Standard response wrapper for all endpoints.

    Generic over T for typed ``data`` payloads. Backwards-compatible:
    ``APIResponse(data=...)`` still accepts dict/list/None.
    """

    success: bool
    request_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: T | dict | list | None = None
    error: ErrorDetail | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    models_loaded: bool
    database: bool
    redis: bool
