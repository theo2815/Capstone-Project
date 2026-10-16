from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest


def _session_returning(key_hash):
    session = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = key_hash
    session.execute = AsyncMock(return_value=result)

    @asynccontextmanager
    async def ctx():
        yield session

    return session, ctx


async def test_revoke_deactivates_then_clears_the_cache():
    from scripts.revoke_api_key import revoke

    session, ctx = _session_returning("abc123")
    redis = MagicMock(delete=AsyncMock(return_value=1))
    assert await revoke(str(uuid.uuid4()), session_ctx=ctx, redis=redis) is True
    statement = str(session.execute.await_args.args[0])
    assert statement.startswith("UPDATE api_keys") and "active" in statement
    redis.delete.assert_awaited_once_with("apikey:abc123")


async def test_revoke_unknown_key_changes_nothing():
    from scripts.revoke_api_key import revoke

    _session, ctx = _session_returning(None)
    redis = MagicMock(delete=AsyncMock())
    assert await revoke(str(uuid.uuid4()), session_ctx=ctx, redis=redis) is False
    redis.delete.assert_not_awaited()


async def test_main_refuses_unless_both_target_urls_are_in_the_environment(monkeypatch):
    # A missing REDIS_URL would come from .env or the localhost default, so the cache
    # clear would miss the target while the script still printed "Key revoked".
    from scripts.revoke_api_key import _main

    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@target:5432/db")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr("src.db.session.init_db", AsyncMock(side_effect=AssertionError))
    with pytest.raises(SystemExit, match="REDIS_URL"):
        await _main(str(uuid.uuid4()))


async def test_cache_is_cleared_by_hash():
    from src.middleware.auth import invalidate_api_key_hash

    redis = MagicMock(delete=AsyncMock(return_value=1))
    assert await invalidate_api_key_hash(redis, "abc123") is True
    redis.delete.assert_awaited_once_with("apikey:abc123")
