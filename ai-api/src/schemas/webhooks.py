from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, field_validator

ALLOWED_EVENTS = {"job.completed", "job.failed"}


class WebhookCreateRequest(BaseModel):
    url: HttpUrl
    events: list[str] = Field(..., min_length=1, max_length=10)
    secret: str | None = Field(default=None, max_length=256)

    @field_validator("events")
    @classmethod
    def validate_events(cls, v: list[str]) -> list[str]:
        invalid = set(v) - ALLOWED_EVENTS
        if invalid:
            raise ValueError(f"Invalid events: {invalid}. Allowed: {ALLOWED_EVENTS}")
        return v


class WebhookResponse(BaseModel):
    id: UUID
    url: str
    events: list[str]
    active: bool
    created_at: datetime


class WebhookListResponse(BaseModel):
    webhooks: list[WebhookResponse]
    total: int
