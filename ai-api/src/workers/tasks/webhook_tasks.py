from __future__ import annotations

import asyncio
import hmac
import hashlib
import ipaddress
import json
import socket
import uuid
from urllib.parse import urlparse, urlunparse

import httpx
from sqlalchemy.exc import SQLAlchemyError

from src.config import get_settings
from src.db.repositories.sync_webhook_repo import SyncWebhookRepository
from src.db.sync_session import get_sync_session
from src.utils.crypto import decrypt_secret
from src.workers.celery_app import celery_app
from src.utils.logging import get_logger

logger = get_logger(__name__)

# NAT64 addresses count as global, but they wrap an IPv4 address in their last 32 bits.
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def _is_public(answer: str) -> bool:
    ip = ipaddress.ip_address(answer)
    if ip in _NAT64:
        ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip.is_global


def _validate_webhook_url(url: str) -> tuple[str, str]:
    """Resolve once; return (ip, hostname) only if EVERY answer is a public address.

    deliver_webhook then connects to that checked ip, so no second lookup can pick
    an internal address.
    """
    parsed = urlparse(url)
    if "@" in parsed.netloc:
        raise ValueError("Webhook URL must not contain user info")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid webhook URL: no hostname")
    try:
        answers = [info[4][0] for info in socket.getaddrinfo(hostname, None)]
    except socket.gaierror as e:
        raise ValueError(f"Cannot resolve webhook hostname '{hostname}': {e}")
    if not answers:
        raise ValueError(f"Cannot resolve webhook hostname '{hostname}'")
    for answer in answers:
        if not _is_public(answer):
            raise ValueError(f"Webhook URL resolves to a non-public address ({answer})")
    return answers[0], hostname


@celery_app.task(
    bind=True,
    name="webhooks.deliver",
    autoretry_for=(httpx.HTTPError, SQLAlchemyError),
    retry_backoff=True,
    retry_backoff_max=30,
    max_retries=3,
)
def deliver_webhook(
    self,
    url: str,
    event: str,
    payload: dict,
    secret: str | None = None,
    timeout: int = 10,
    webhook_id: str | None = None,
):
    """Deliver a webhook callback to a registered URL.

    Args:
        url: The webhook URL to POST to.
        event: The event type (e.g., 'job.completed').
        payload: The JSON payload to send.
        secret: Plain HMAC secret; only messages queued before webhook_id existed carry it.
        timeout: Request timeout in seconds.
        webhook_id: Subscription whose stored secret signs the request.
    """
    # SSRF protection: resolve DNS once and validate before making request.
    # The HTTP request uses the pre-resolved IP to prevent DNS rebinding attacks.
    try:
        resolved_ip, hostname = _validate_webhook_url(url)
    except ValueError as e:
        logger.error("Webhook blocked by SSRF protection", url=url, error=str(e))
        return

    if webhook_id and not secret:
        with get_sync_session() as session:
            webhook = SyncWebhookRepository(session).get(uuid.UUID(webhook_id))
        if webhook is None:
            logger.warning("Webhook deleted before delivery", webhook_id=webhook_id)
            return
        secret = decrypt_secret(webhook.secret) if webhook.secret else None

    body = json.dumps({"event": event, **payload})
    parsed = urlparse(url)
    host = f"[{resolved_ip}]" if ":" in resolved_ip else resolved_ip
    netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
    target_url = urlunparse(
        (parsed.scheme, netloc, parsed.path or "/", parsed.params, parsed.query, "")
    )
    headers = {"Content-Type": "application/json", "Host": parsed.netloc}

    if secret:
        signature = hmac.new(
            secret.encode(), body.encode(), hashlib.sha256
        ).hexdigest()
        headers["X-QuickPitik-Signature"] = f"sha256={signature}"

    async def post() -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout, verify=True) as client:
            return await client.post(target_url, content=body, headers=headers,
                                     extensions={"sni_hostname": hostname})

    # timeout limits each read; the deadline limits the whole delivery, so a receiver
    # answering one byte at a time cannot hold the worker.
    deadline = get_settings().WEBHOOK_DELIVERY_DEADLINE_SECONDS
    try:
        response = asyncio.run(asyncio.wait_for(post(), deadline))
        response.raise_for_status()
        # structlog's first parameter is named "event": log the webhook's under another key.
        logger.info(
            "Webhook delivered", url=url, webhook_event=event, status=response.status_code
        )
    except TimeoutError as e:
        logger.error("Webhook delivery deadline exceeded", url=url, webhook_event=event,
                     deadline=deadline)
        raise httpx.TimeoutException("Webhook delivery deadline exceeded") from e
    except httpx.HTTPError as e:
        logger.error("Webhook delivery failed", url=url, webhook_event=event, error=str(e))
        raise
