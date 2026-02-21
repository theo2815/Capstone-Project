from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, HttpUrl


class WebhookCreateRequest(BaseModel):
    url: HttpUrl
    events: list[str]
    secret: str | None = None


class WebhookResponse(BaseModel):
    id: UUID
    url: str
    events: list[str]
    active: bool
    created_at: datetime


class WebhookListResponse(BaseModel):
    webhooks: list[WebhookResponse]
    total: int
