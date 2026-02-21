from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import WebhookSubscription


class WebhookRepository:
    """Repository for webhook subscription management."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        url: str,
        events: list[str],
        secret: str | None = None,
        api_key_id: str | None = None,
    ) -> WebhookSubscription:
        webhook = WebhookSubscription(
            url=url, events=events, secret=secret, api_key_id=api_key_id
        )
        self.session.add(webhook)
        await self.session.flush()
        return webhook

    async def get(self, webhook_id: uuid.UUID) -> WebhookSubscription | None:
        result = await self.session.execute(
            select(WebhookSubscription).where(WebhookSubscription.id == webhook_id)
        )
        return result.scalar_one_or_none()

    async def list_all(self, api_key_id: str | None = None) -> list[WebhookSubscription]:
        query = select(WebhookSubscription).where(WebhookSubscription.active.is_(True))
        if api_key_id:
            query = query.where(WebhookSubscription.api_key_id == api_key_id)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def list_by_event(self, event: str) -> list[WebhookSubscription]:
        result = await self.session.execute(
            select(WebhookSubscription).where(
                WebhookSubscription.active.is_(True),
            )
        )
        return [
            wh for wh in result.scalars().all() if event in wh.events
        ]

    async def delete(self, webhook_id: uuid.UUID) -> bool:
        webhook = await self.get(webhook_id)
        if webhook is None:
            return False
        await self.session.delete(webhook)
        await self.session.flush()
        return True
