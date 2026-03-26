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
        from src.utils.crypto import encrypt_secret

        encrypted_secret = encrypt_secret(secret) if secret else None
        webhook = WebhookSubscription(
            url=url, events=events, secret=encrypted_secret, api_key_id=api_key_id
        )
        self.session.add(webhook)
        await self.session.flush()
        return webhook

    async def get(self, webhook_id: uuid.UUID) -> WebhookSubscription | None:
        result = await self.session.execute(
            select(WebhookSubscription).where(WebhookSubscription.id == webhook_id)
        )
        return result.scalar_one_or_none()

    async def list_all(
        self,
        api_key_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[WebhookSubscription]:
        query = (
            select(WebhookSubscription)
            .where(WebhookSubscription.active.is_(True))
            .order_by(WebhookSubscription.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if api_key_id:
            query = query.where(WebhookSubscription.api_key_id == api_key_id)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def count_all(self, api_key_id: str | None = None) -> int:
        from sqlalchemy import func

        query = (
            select(func.count())
            .select_from(WebhookSubscription)
            .where(WebhookSubscription.active.is_(True))
        )
        if api_key_id:
            query = query.where(WebhookSubscription.api_key_id == api_key_id)
        result = await self.session.execute(query)
        return result.scalar_one()

    async def list_with_count(
        self,
        api_key_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[WebhookSubscription], int]:
        """List webhooks and count in one call, ensuring consistent filters."""
        from sqlalchemy import func

        base_filter = [WebhookSubscription.active.is_(True)]
        if api_key_id:
            base_filter.append(WebhookSubscription.api_key_id == api_key_id)

        count_q = select(func.count()).select_from(WebhookSubscription).where(*base_filter)
        count_result = await self.session.execute(count_q)
        total = count_result.scalar_one()

        list_q = (
            select(WebhookSubscription)
            .where(*base_filter)
            .order_by(WebhookSubscription.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        list_result = await self.session.execute(list_q)
        webhooks = list(list_result.scalars().all())

        return webhooks, total

    async def list_by_event(self, event: str) -> list[WebhookSubscription]:
        from sqlalchemy import cast
        from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB

        result = await self.session.execute(
            select(WebhookSubscription).where(
                WebhookSubscription.active.is_(True),
                WebhookSubscription.events.op("@>")(cast([event], PG_JSONB)),
            )
        )
        return list(result.scalars().all())

    async def delete(self, webhook_id: uuid.UUID) -> bool:
        webhook = await self.get(webhook_id)
        if webhook is None:
            return False
        await self.session.delete(webhook)
        await self.session.flush()
        return True
