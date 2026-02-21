from __future__ import annotations

import hmac
import hashlib
import json

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.workers.celery_app import celery_app
from src.utils.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(bind=True, name="webhooks.deliver")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
def deliver_webhook(
    self,
    url: str,
    event: str,
    payload: dict,
    secret: str | None = None,
    timeout: int = 10,
):
    """Deliver a webhook callback to a registered URL.

    Args:
        url: The webhook URL to POST to.
        event: The event type (e.g., 'job.completed').
        payload: The JSON payload to send.
        secret: Optional HMAC secret for signature verification.
        timeout: Request timeout in seconds.
    """
    body = json.dumps({"event": event, **payload})
    headers = {"Content-Type": "application/json"}

    if secret:
        signature = hmac.new(
            secret.encode(), body.encode(), hashlib.sha256
        ).hexdigest()
        headers["X-EventAI-Signature"] = f"sha256={signature}"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, content=body, headers=headers)
            response.raise_for_status()
            logger.info("Webhook delivered", url=url, event=event, status=response.status_code)
    except httpx.HTTPError as e:
        logger.error("Webhook delivery failed", url=url, event=event, error=str(e))
        raise
