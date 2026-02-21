from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request

from src.middleware.auth import verify_api_key
from src.schemas.common import APIResponse
from src.schemas.webhooks import WebhookCreateRequest, WebhookListResponse, WebhookResponse

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


@router.post("", response_model=APIResponse)
async def register_webhook(
    request: Request,
    body: WebhookCreateRequest,
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Register a webhook URL for event callbacks."""
    from src.db.repositories.webhook_repo import WebhookRepository
    from src.db.session import get_session

    async for session in get_session():
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
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """List all registered webhooks for the current API key."""
    from src.db.repositories.webhook_repo import WebhookRepository
    from src.db.session import get_session

    async for session in get_session():
        repo = WebhookRepository(session)
        webhooks = await repo.list_all(api_key_id=key_meta.get("key_id"))

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
            total=len(webhooks),
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
    """Remove a registered webhook."""
    from src.db.repositories.webhook_repo import WebhookRepository
    from src.db.session import get_session

    async for session in get_session():
        repo = WebhookRepository(session)
        deleted = await repo.delete(webhook_id)
        if not deleted:
            return APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={"code": "NOT_FOUND", "message": "Webhook not found"},
            )

        return APIResponse(
            success=True,
            request_id=getattr(request.state, "request_id", ""),
            data={"deleted": True, "webhook_id": str(webhook_id)},
        )
