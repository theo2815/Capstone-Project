from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import WebhookSubscription


class SyncWebhookRepository:
    """Synchronous webhook repo for Celery tasks to look up subscriptions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def list_by_event(
        self, event: str, api_key_id: str | None = None
    ) -> list[WebhookSubscription]:
        from sqlalchemy import cast
        from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB

        conditions = [
            WebhookSubscription.active.is_(True),
            WebhookSubscription.events.op("@>")(cast([event], PG_JSONB)),
        ]
        # Tenant scoping: only fan out to the subscriptions owned by the API key
        # whose job fired the event. Without this, a job.completed for tenant A
        # would dispatch to EVERY tenant's webhook subscribed to that event,
        # leaking job IDs across tenants.
        if api_key_id is not None:
            conditions.append(WebhookSubscription.api_key_id == api_key_id)

        result = self.session.execute(
            select(WebhookSubscription).where(*conditions)
        )
        return list(result.scalars().all())
