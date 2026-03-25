from __future__ import annotations

import ipaddress
import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Query, Request

from src.middleware.auth import check_scope, verify_api_key
from src.schemas.common import APIResponse
from src.schemas.webhooks import WebhookCreateRequest, WebhookListResponse, WebhookResponse

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


def _validate_webhook_url_basic(url: str) -> str | None:
    """Reject webhook URLs that target obviously private IPs at registration time.

    Returns an error message if blocked, None if OK.
    Hostname-based URLs are allowed here (full DNS resolution check at delivery time).
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return "Invalid webhook URL: no hostname"
    if not parsed.scheme or parsed.scheme not in ("http", "https"):
        return "Invalid webhook URL: scheme must be http or https"
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return "Webhook URL must not target private/internal networks"
    except ValueError:
        pass  # Hostname (not IP literal) — allow, validated at delivery
    return None


@router.post("", response_model=APIResponse)
async def register_webhook(
    request: Request,
    body: WebhookCreateRequest,
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Register a webhook URL for event callbacks."""
    check_scope("webhooks:write", key_meta)

    # SEC-10: Validate URL at registration time (IP-literal check)
    url_error = _validate_webhook_url_basic(str(body.url))
    if url_error:
        return APIResponse(
            success=False,
            request_id=getattr(request.state, "request_id", ""),
            error={"code": "INVALID_WEBHOOK_URL", "message": url_error},
        )

    from src.db.repositories.webhook_repo import WebhookRepository
    from src.db.session import get_session_ctx

    async with get_session_ctx() as session:
        repo = WebhookRepository(session)
        webhook = await repo.create(
            url=str(body.url),
            events=body.events,
            secret=body.secret,
            api_key_id=key_meta.get("key_id"),
        )

        data = WebhookResponse(
            id=webhook.id,
            url=webhook.url,
            events=webhook.events,
            active=webhook.active,
            created_at=webhook.created_at,
        )
        return APIResponse(
            success=True,
            request_id=getattr(request.state, "request_id", ""),
            data=data.model_dump(),
        )


@router.get("", response_model=APIResponse)
async def list_webhooks(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """List registered webhooks for the current API key (paginated)."""
    check_scope("webhooks:read", key_meta)
    from src.db.repositories.webhook_repo import WebhookRepository
    from src.db.session import get_session_ctx

    api_key_id = key_meta.get("key_id")

    async with get_session_ctx() as session:
        repo = WebhookRepository(session)
        webhooks = await repo.list_all(
            api_key_id=api_key_id, limit=limit, offset=offset
        )
        total = await repo.count_all(api_key_id=api_key_id)

        data = WebhookListResponse(
            webhooks=[
                WebhookResponse(
                    id=wh.id,
                    url=wh.url,
                    events=wh.events,
                    active=wh.active,
                    created_at=wh.created_at,
                )
                for wh in webhooks
            ],
            total=total,
        )
        return APIResponse(
            success=True,
            request_id=getattr(request.state, "request_id", ""),
            data=data.model_dump(),
        )


@router.delete("/{webhook_id}", response_model=APIResponse)
async def delete_webhook(
    request: Request,
    webhook_id: uuid.UUID,
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Remove a registered webhook (tenant-isolated)."""
    check_scope("webhooks:write", key_meta)
    from src.db.repositories.webhook_repo import WebhookRepository
    from src.db.session import get_session_ctx

    async with get_session_ctx() as session:
        repo = WebhookRepository(session)
        webhook = await repo.get(webhook_id)
        if webhook is None or webhook.api_key_id != key_meta.get("key_id"):
            return APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={"code": "NOT_FOUND", "message": "Webhook not found"},
            )

        deleted = await repo.delete(webhook_id)

        return APIResponse(
            success=True,
            request_id=getattr(request.state, "request_id", ""),
            data={"deleted": True, "webhook_id": str(webhook_id)},
        )
