from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import WebhookSubscription


class SyncWebhookRepository:
    """Synchronous webhook repo for Celery tasks to look up subscriptions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def list_by_event(self, event: str) -> list[WebhookSubscription]:
        from sqlalchemy import cast
        from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB

        result = self.session.execute(
            select(WebhookSubscription).where(
                WebhookSubscription.active.is_(True),
                WebhookSubscription.events.op("@>")(cast([event], PG_JSONB)),
            )
        )
        return list(result.scalars().all())
