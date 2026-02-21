from __future__ import annotations

import hashlib

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from src.utils.logging import get_logger

logger = get_logger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> dict:
    """Validate API key and return key metadata (scopes, rate tier).

    In development mode (DEBUG=true), a missing key is allowed.
    """
    settings = request.app.state.settings

    if settings.DEBUG and not api_key:
        return {"scopes": ["*"], "rate_tier": "internal", "key_id": "debug"}

    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # Check Redis cache first
    redis = getattr(request.app.state, "redis", None)
    if redis:
        cached = await redis.get(f"apikey:{key_hash}")
        if cached:
            import json

            return json.loads(cached)

    # Fallback: check database
    from sqlalchemy import select

    from src.db.models import APIKey
    from src.db.session import get_session

    async for session in get_session():
        result = await session.execute(
            select(APIKey).where(APIKey.key_hash == key_hash, APIKey.active.is_(True))
        )
        db_key = result.scalar_one_or_none()

        if db_key is None:
            raise HTTPException(status_code=401, detail="Invalid API key")

        key_meta = {
            "key_id": str(db_key.id),
            "scopes": db_key.scopes,
            "rate_tier": db_key.rate_tier,
        }

        # Cache in Redis for 5 minutes
        if redis:
            import json

            await redis.set(f"apikey:{key_hash}", json.dumps(key_meta), ex=300)

        return key_meta

    raise HTTPException(status_code=401, detail="Invalid API key")


def check_scope(required_scope: str, key_meta: dict) -> None:
    """Check if the API key has the required scope."""
    scopes = key_meta.get("scopes", [])
    if "*" in scopes:
        return
    if required_scope not in scopes:
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient permissions. Required scope: {required_scope}",
        )
