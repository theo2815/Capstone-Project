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
    """Validate API key, enforce rate limits, and return key metadata.

    In development mode (DEBUG=true), a missing key is allowed.
    """
    settings = request.app.state.settings

    if settings.DEBUG and not api_key:
        logger.warning("Auth bypassed in DEBUG mode — do not use in production")
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

            key_meta = json.loads(cached)
            await _enforce_rate_limit(request, key_meta)
            return key_meta

    # Fallback: check database
    from sqlalchemy import select

    from src.db.models import APIKey
    from src.db.session import get_session_ctx

    async with get_session_ctx() as session:
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

        await _enforce_rate_limit(request, key_meta)
        return key_meta


async def invalidate_api_key_cache(redis, api_key: str) -> bool:
    """Invalidate the Redis cache entry for a revoked API key.

    Call this when deactivating an API key to close the 5-minute cache window.
    Returns True if a cached entry was deleted, False otherwise.
    """
    if redis is None:
        return False
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    deleted = await redis.delete(f"apikey:{key_hash}")
    if deleted:
        logger.info("API key cache invalidated", key_hash_prefix=key_hash[:8])
    return bool(deleted)


async def _enforce_rate_limit(request: Request, key_meta: dict) -> None:
    """Apply rate limiting after successful authentication."""
    from src.middleware.rate_limit import check_rate_limit

    await check_rate_limit(request, key_meta)


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
