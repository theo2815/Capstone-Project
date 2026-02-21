from __future__ import annotations

from src.utils.logging import get_logger

logger = get_logger(__name__)


class WebhookService:
    """Business logic for webhook management and dispatch."""

    def __init__(self, webhook_repo) -> None:
        self.webhook_repo = webhook_repo

    async def dispatch(self, event: str, payload: dict) -> int:
        """Dispatch a webhook event to all matching subscribers.

        Returns the number of webhooks queued for delivery.
        """
        webhooks = await self.webhook_repo.list_by_event(event)
        count = 0
        for wh in webhooks:
            from src.workers.tasks.webhook_tasks import deliver_webhook

            deliver_webhook.delay(
                url=wh.url,
                event=event,
                payload=payload,
                secret=wh.secret,
            )
            count += 1

        logger.info("Webhooks dispatched", event=event, count=count)
        return count
