from __future__ import annotations

import hashlib

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from redis.exceptions import RedisError

from src.middleware.rate_limit import redis_unavailable
from src.utils.logging import get_logger

logger = get_logger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> dict:
    """Validate API key, enforce rate limits, and return key metadata.

    In development mode (DEBUG=true), a missing key is allowed.
    Cached key metadata expires after 300 seconds; revocation should also
    invalidate the cache entry via ``invalidate_api_key_cache``.
    A key the database does not know is remembered as refused for 60 seconds,
    so a key created within 60 s of being tried while invalid is refused for up to 60 s.
    """
    settings = request.app.state.settings
    cached = getattr(request.state, "blur_key_meta", None)
    if isinstance(cached, dict):
        return cached

    if settings.DEBUG and not api_key:
        logger.warning("Auth bypassed in DEBUG mode — do not use in production")
        return {"scopes": ["*"], "rate_tier": "internal", "key_id": "debug"}

    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # Check Redis cache first
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise redis_unavailable()
    try:
        cached = await redis.get(f"apikey:{key_hash}")
        refused = not cached and await redis.get(f"apikey:miss:{key_hash}")
    except RedisError as exc:
        logger.warning("API key cache unavailable", error_type=type(exc).__name__)
        raise redis_unavailable() from exc
    if cached:
        import json

        key_meta = json.loads(cached)
        await _enforce_rate_limit(request, key_meta)
        return key_meta
    if refused:
        raise HTTPException(status_code=401, detail="Invalid API key")

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
            try:
                await redis.set(f"apikey:miss:{key_hash}", "1", ex=60)
            except RedisError as exc:
                logger.warning("API key cache unavailable", error_type=type(exc).__name__)
                raise redis_unavailable() from exc
            raise HTTPException(status_code=401, detail="Invalid API key")

        key_meta = {
            "key_id": str(db_key.id),
            "scopes": db_key.scopes,
            "rate_tier": db_key.rate_tier,
        }

        # Cache in Redis for 5 minutes
        import json

        try:
            await redis.set(f"apikey:{key_hash}", json.dumps(key_meta), ex=300)
        except RedisError as exc:
            logger.warning("API key cache unavailable", error_type=type(exc).__name__)
            raise redis_unavailable() from exc

        await _enforce_rate_limit(request, key_meta)
        return key_meta


async def invalidate_api_key_hash(redis, key_hash: str) -> bool:
    """Drop the cached entry for a key hash (closes the 5-minute window after revocation)."""
    if redis is None:
        return False
    deleted = await redis.delete(f"apikey:{key_hash}")
    if deleted:
        logger.info("API key cache invalidated", key_hash_prefix=key_hash[:8])
    return bool(deleted)


async def invalidate_api_key_cache(redis, api_key: str) -> bool:
    """Invalidate the Redis cache entry for a revoked raw API key."""
    return await invalidate_api_key_hash(redis, hashlib.sha256(api_key.encode()).hexdigest())


async def _enforce_rate_limit(request: Request, key_meta: dict) -> None:
    """Apply rate limiting after successful authentication.

    Stores rate_info on request.state so the RateLimitHeadersMiddleware
    can set X-RateLimit-* headers on the response (RS-5).
    """
    from src.middleware.rate_limit import check_rate_limit

    rate_info = await check_rate_limit(request, key_meta)
    request.state.rate_info = rate_info


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
