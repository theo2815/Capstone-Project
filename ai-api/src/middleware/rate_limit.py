from __future__ import annotations

import time

from fastapi import HTTPException, Request

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Lua script for atomic token bucket rate limiting in Redis
TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local max_tokens = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1])
local last_refill = tonumber(bucket[2])

if tokens == nil then
    tokens = max_tokens
    last_refill = now
end

local elapsed = now - last_refill
local new_tokens = math.min(max_tokens, tokens + elapsed * refill_rate)

if new_tokens >= 1 then
    new_tokens = new_tokens - 1
    redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 120)
    return {1, math.floor(new_tokens), math.ceil((1 - new_tokens) / refill_rate)}
else
    local retry_after = math.ceil((1 - new_tokens) / refill_rate)
    return {0, 0, retry_after}
end
"""

RATE_TIERS = {
    "free": {"max_tokens": 60, "refill_rate": 1.0},       # 60/min
    "pro": {"max_tokens": 300, "refill_rate": 5.0},       # 300/min
    "internal": {"max_tokens": 1000, "refill_rate": 16.7}, # 1000/min
}


async def check_rate_limit(request: Request, key_meta: dict) -> dict:
    """Check rate limit for the given API key.

    Returns rate limit info dict with remaining, limit, and reset.
    Raises HTTPException 429 if limit exceeded.
    """
    redis = getattr(request.app.state, "redis", None)
    if not redis:
        # No Redis = no rate limiting (development fallback)
        return {"remaining": -1, "limit": -1, "reset": 0}

    rate_tier = key_meta.get("rate_tier", "free")
    tier_config = RATE_TIERS.get(rate_tier, RATE_TIERS["free"])
    key_id = key_meta.get("key_id", "unknown")

    bucket_key = f"ratelimit:{key_id}"
    now = time.time()

    result = await redis.eval(
        TOKEN_BUCKET_SCRIPT,
        1,
        bucket_key,
        tier_config["max_tokens"],
        tier_config["refill_rate"],
        now,
    )

    allowed, remaining, retry_after = result

    rate_info = {
        "remaining": int(remaining),
        "limit": tier_config["max_tokens"],
        "reset": int(now) + int(retry_after),
    }

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(tier_config["max_tokens"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(rate_info["reset"]),
            },
        )

    return rate_info
