from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
import redis.asyncio as aioredis
from fastapi import HTTPException

from src.middleware.auth import check_scope, invalidate_api_key_cache, verify_api_key
from src.middleware.rate_limit import check_rate_limit


@pytest.mark.asyncio
async def test_real_redis_pause_fails_closed_then_recovers():
    url = os.getenv("TEST_REDIS_URL")
    if not url:
        pytest.skip("Set TEST_REDIS_URL to a disposable local Redis instance")
    if urlparse(url).hostname not in {"localhost", "127.0.0.1"}:
        pytest.skip("Redis outage test only runs against a disposable local instance")

    redis = aioredis.from_url(url, decode_responses=True, socket_timeout=0.1)
    admin = aioredis.from_url(url, decode_responses=True)
    api_key = secrets.token_hex(16)
    cache_key = f"apikey:{hashlib.sha256(api_key.encode()).hexdigest()}"
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(
            settings=SimpleNamespace(DEBUG=False), redis=redis
        )),
        state=SimpleNamespace(),
    )
    key_meta = {"key_id": api_key, "scopes": ["blur:read"], "rate_tier": "free"}
    try:
        await redis.set(cache_key, json.dumps(key_meta), ex=30)
        assert (await verify_api_key(request, api_key))["key_id"] == api_key

        await admin.execute_command("CLIENT", "PAUSE", 1000, "ALL")
        with pytest.raises(HTTPException) as auth_error:
            await verify_api_key(request, api_key)
        assert auth_error.value.status_code == 503
        with pytest.raises(HTTPException) as limit_error:
            await check_rate_limit(request, key_meta)
        assert limit_error.value.status_code == 503

        await asyncio.sleep(1.1)
        recovered = await verify_api_key(request, api_key)
        assert recovered["key_id"] == api_key
        with pytest.raises(HTTPException) as scope_error:
            check_scope("faces:read", recovered)
        assert scope_error.value.status_code == 403
        assert await invalidate_api_key_cache(redis, api_key)
        assert await redis.get(cache_key) is None
    finally:
        await admin.execute_command("CLIENT", "UNPAUSE")
        await redis.delete(cache_key)
        await redis.aclose()
        await admin.aclose()
