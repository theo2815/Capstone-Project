from __future__ import annotations

import hmac
import hashlib
import ipaddress
import json
import socket
from urllib.parse import urlparse

import httpx

from src.workers.celery_app import celery_app
from src.utils.logging import get_logger

logger = get_logger(__name__)

# SSRF protection: block private/internal IP ranges
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _validate_webhook_url(url: str) -> tuple[str, str]:
    """Resolve DNS once and validate that the webhook URL doesn't target internal networks.

    Returns (resolved_ip, hostname) to prevent DNS rebinding TOCTOU attacks.

    Raises:
        ValueError: If URL resolves to a blocked network.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid webhook URL: no hostname")

    try:
        addr_info = socket.getaddrinfo(hostname, None)
        resolved_ip = addr_info[0][4][0]
        ip = ipaddress.ip_address(resolved_ip)
    except (socket.gaierror, ValueError, IndexError) as e:
        raise ValueError(f"Cannot resolve webhook hostname '{hostname}': {e}")

    for network in _BLOCKED_NETWORKS:
        if ip in network:
            raise ValueError(
                f"Webhook URL targets blocked internal network ({network})"
            )

    return resolved_ip, hostname


@celery_app.task(
    bind=True,
    name="webhooks.deliver",
    autoretry_for=(httpx.HTTPError,),
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
):
    """Deliver a webhook callback to a registered URL.

    Args:
        url: The webhook URL to POST to.
        event: The event type (e.g., 'job.completed').
        payload: The JSON payload to send.
        secret: Optional HMAC secret for signature verification.
        timeout: Request timeout in seconds.
    """
    # SSRF protection: resolve DNS once and validate before making request.
    # The HTTP request uses the pre-resolved IP to prevent DNS rebinding attacks.
    try:
        resolved_ip, hostname = _validate_webhook_url(url)
    except ValueError as e:
        logger.error("Webhook blocked by SSRF protection", url=url, error=str(e))
        return

    body = json.dumps({"event": event, **payload})
    headers = {"Content-Type": "application/json", "Host": hostname}

    if secret:
        signature = hmac.new(
            secret.encode(), body.encode(), hashlib.sha256
        ).hexdigest()
        headers["X-EventAI-Signature"] = f"sha256={signature}"

    # Replace hostname with resolved IP to prevent DNS rebinding TOCTOU
    target_url = url.replace(hostname, resolved_ip, 1)

    try:
        with httpx.Client(timeout=timeout, verify=True) as client:
            response = client.post(target_url, content=body, headers=headers)
            response.raise_for_status()
            logger.info("Webhook delivered", url=url, event=event, status=response.status_code)
    except httpx.HTTPError as e:
        logger.error("Webhook delivery failed", url=url, event=event, error=str(e))
        raise
