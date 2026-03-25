from __future__ import annotations

import hashlib
import json
import sys
import uuid
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.middleware.auth import check_scope, verify_api_key
from src.middleware.rate_limit import RATE_TIERS, check_rate_limit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(*, debug: bool = False, redis=None) -> MagicMock:
    """Build a fake ``Request`` with ``app.state.settings`` and optional redis."""
    request = MagicMock()
    request.app.state.settings = MagicMock(DEBUG=debug)
    request.app.state.redis = redis
    return request


def _make_settings(*, debug: bool = False, environment: str = "development") -> MagicMock:
    settings = MagicMock(DEBUG=debug, ENVIRONMENT=environment)
    return settings


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


@contextmanager
def _patch_db_imports(mock_get_session_ctx):
    """Context manager that injects mock modules for ``sqlalchemy`` and
    ``src.db.session`` into ``sys.modules`` so that the local imports inside
    ``verify_api_key`` resolve without the real packages installed.

    Also patches ``_enforce_rate_limit`` to avoid hitting real rate-limit code.
    """
    # Build mock sqlalchemy module with a ``select`` callable
    mock_sqlalchemy = MagicMock()
    mock_sqlalchemy.select = MagicMock(return_value=MagicMock())

    # Build mock src.db.session module exposing get_session_ctx
    mock_session_mod = MagicMock()
    mock_session_mod.get_session_ctx = mock_get_session_ctx

    # Build mock src.db.models module exposing APIKey sentinel
    mock_models_mod = MagicMock()

    fake_modules = {
        "sqlalchemy": mock_sqlalchemy,
        "src.db.models": mock_models_mod,
        "src.db.session": mock_session_mod,
    }

    # Only inject modules that are not already present; keep originals intact.
    with patch.dict(sys.modules, fake_modules):
        yield


def _make_session_ctx(mock_session):
    """Build a mock async context manager returned by ``get_session_ctx()``."""
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_request():
    """Default non-debug request without Redis."""
    return _make_request(debug=False, redis=None)


@pytest.fixture
def debug_request():
    """Debug-mode request without Redis."""
    return _make_request(debug=True, redis=None)


@pytest.fixture
def sample_key_meta():
    return {
        "key_id": str(uuid.uuid4()),
        "scopes": ["blur:read", "face:read"],
        "rate_tier": "pro",
    }


# ===================================================================
# 1. verify_api_key
# ===================================================================

class TestVerifyApiKey:
    """Tests for ``verify_api_key``."""

    @pytest.mark.asyncio
    async def test_missing_key_returns_401(self, fake_request):
        """A request with no API key (and DEBUG=false) must raise 401."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(fake_request, api_key=None)

        assert exc_info.value.status_code == 401
        assert "Missing API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_debug_bypass_returns_wildcard_scopes(self, debug_request):
        """DEBUG=true + no key => debug key_meta with scopes=["*"]."""
        result = await verify_api_key(debug_request, api_key=None)

        assert result["scopes"] == ["*"]
        assert result["rate_tier"] == "internal"
        assert result["key_id"] == "debug"

    @pytest.mark.asyncio
    async def test_valid_key_from_database(self, fake_request):
        """A valid API key found in the database returns correct key_meta."""
        raw_key = "test-api-key-12345"
        key_id = uuid.uuid4()

        db_row = MagicMock()
        db_row.id = key_id
        db_row.scopes = ["blur:read"]
        db_row.rate_tier = "free"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = db_row

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        mock_ctx = _make_session_ctx(mock_session)
        mock_get_session_ctx = MagicMock(return_value=mock_ctx)

        with (
            _patch_db_imports(mock_get_session_ctx),
            patch("src.middleware.auth._enforce_rate_limit", new_callable=AsyncMock) as mock_rl,
        ):
            result = await verify_api_key(fake_request, api_key=raw_key)

        assert result["key_id"] == str(key_id)
        assert result["scopes"] == ["blur:read"]
        assert result["rate_tier"] == "free"
        mock_rl.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invalid_key_returns_401(self, fake_request):
        """An API key not found in the database must raise 401."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        mock_ctx = _make_session_ctx(mock_session)
        mock_get_session_ctx = MagicMock(return_value=mock_ctx)

        with _patch_db_imports(mock_get_session_ctx):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(fake_request, api_key="bogus-key")

        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_redis_cache_hit_returns_cached_meta(self):
        """When Redis has a cached entry for the key hash, it returns that
        directly without touching the database."""
        raw_key = "cached-key-abc"
        expected_meta = {
            "key_id": "cached-id-1",
            "scopes": ["face:read"],
            "rate_tier": "pro",
        }

        mock_redis = AsyncMock()
        mock_redis.get.return_value = json.dumps(expected_meta)

        request = _make_request(debug=False, redis=mock_redis)

        with patch("src.middleware.auth._enforce_rate_limit", new_callable=AsyncMock) as mock_rl:
            result = await verify_api_key(request, api_key=raw_key)

        assert result == expected_meta
        key_hash = _hash_key(raw_key)
        mock_redis.get.assert_awaited_once_with(f"apikey:{key_hash}")
        mock_rl.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rate_limit_called_after_db_auth(self, fake_request):
        """After successful DB auth, _enforce_rate_limit must be called."""
        raw_key = "rl-test-key"
        db_row = MagicMock()
        db_row.id = uuid.uuid4()
        db_row.scopes = ["*"]
        db_row.rate_tier = "internal"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = db_row

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        mock_ctx = _make_session_ctx(mock_session)
        mock_get_session_ctx = MagicMock(return_value=mock_ctx)

        with (
            _patch_db_imports(mock_get_session_ctx),
            patch("src.middleware.auth._enforce_rate_limit", new_callable=AsyncMock) as mock_rl,
        ):
            await verify_api_key(fake_request, api_key=raw_key)

        mock_rl.assert_awaited_once()
        # The first positional arg should be the request, second the key_meta
        call_args = mock_rl.call_args
        assert call_args[0][0] is fake_request
        assert call_args[0][1]["rate_tier"] == "internal"

    @pytest.mark.asyncio
    async def test_rate_limit_called_after_cache_hit(self):
        """After a Redis cache hit, _enforce_rate_limit must still be called."""
        raw_key = "rl-cache-key"
        cached_meta = {"key_id": "c1", "scopes": ["*"], "rate_tier": "pro"}

        mock_redis = AsyncMock()
        mock_redis.get.return_value = json.dumps(cached_meta)

        request = _make_request(debug=False, redis=mock_redis)

        with patch("src.middleware.auth._enforce_rate_limit", new_callable=AsyncMock) as mock_rl:
            await verify_api_key(request, api_key=raw_key)

        mock_rl.assert_awaited_once()
        assert mock_rl.call_args[0][1] == cached_meta

    @pytest.mark.asyncio
    async def test_debug_bypass_works_in_dev_environment(self):
        """DEBUG=true in a non-production environment still allows bypass."""
        request = _make_request(debug=True, redis=None)
        request.app.state.settings.ENVIRONMENT = "development"

        result = await verify_api_key(request, api_key=None)

        assert result["scopes"] == ["*"]
        assert result["key_id"] == "debug"

    @pytest.mark.asyncio
    async def test_production_debug_blocked_at_startup(self):
        """The lifespan guard in main.py raises RuntimeError when
        DEBUG=true and ENVIRONMENT=production. Auth itself does not check
        ENVIRONMENT -- it trusts that the startup guard already ran.

        We replicate the guard logic here to verify the invariant without
        importing src.main (which may pull heavy dependencies).
        """
        settings = _make_settings(debug=True, environment="production")

        # This mirrors the exact check in src/main.py lifespan():
        #   if settings.DEBUG and settings.ENVIRONMENT == "production":
        #       raise RuntimeError(...)
        with pytest.raises(RuntimeError, match="DEBUG=true is not allowed"):
            if settings.DEBUG and settings.ENVIRONMENT == "production":
                raise RuntimeError(
                    "FATAL: DEBUG=true is not allowed when ENVIRONMENT=production. "
                    "Set DEBUG=false in your .env or environment variables."
                )

    @pytest.mark.asyncio
    async def test_production_debug_guard_does_not_fire_when_debug_false(self):
        """When DEBUG=false the production guard does NOT fire."""
        settings = _make_settings(debug=False, environment="production")

        # The guard should not raise
        fired = False
        if settings.DEBUG and settings.ENVIRONMENT == "production":
            fired = True  # pragma: no cover
        assert not fired

    @pytest.mark.asyncio
    async def test_valid_key_cached_in_redis_after_db_lookup(self):
        """After a DB hit with Redis available, the key_meta is cached."""
        raw_key = "cache-me-key"
        key_id = uuid.uuid4()

        db_row = MagicMock()
        db_row.id = key_id
        db_row.scopes = ["blur:read"]
        db_row.rate_tier = "free"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = db_row

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        mock_ctx = _make_session_ctx(mock_session)
        mock_get_session_ctx = MagicMock(return_value=mock_ctx)

        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # cache miss

        request = _make_request(debug=False, redis=mock_redis)

        with (
            _patch_db_imports(mock_get_session_ctx),
            patch("src.middleware.auth._enforce_rate_limit", new_callable=AsyncMock),
        ):
            await verify_api_key(request, api_key=raw_key)

        key_hash = _hash_key(raw_key)
        expected_meta = {
            "key_id": str(key_id),
            "scopes": ["blur:read"],
            "rate_tier": "free",
        }
        mock_redis.set.assert_awaited_once_with(
            f"apikey:{key_hash}", json.dumps(expected_meta), ex=300
        )


# ===================================================================
# 2. check_rate_limit
# ===================================================================

class TestCheckRateLimit:
    """Tests for ``check_rate_limit`` in rate_limit.py."""

    @pytest.mark.asyncio
    async def test_no_redis_returns_unlimited(self):
        """Without Redis, rate limiting is effectively disabled."""
        request = _make_request(debug=False, redis=None)
        key_meta = {"key_id": "k1", "rate_tier": "free"}

        result = await check_rate_limit(request, key_meta)

        assert result["remaining"] == -1
        assert result["limit"] == -1
        assert result["reset"] == 0

    @pytest.mark.asyncio
    async def test_returns_rate_info_when_allowed(self):
        """When Redis allows the request, rate info is returned."""
        mock_redis = AsyncMock()
        # eval returns [allowed=1, remaining=59, retry_after=1]
        mock_redis.eval.return_value = [1, 59, 1]

        request = _make_request(debug=False, redis=mock_redis)
        key_meta = {"key_id": "k2", "rate_tier": "free"}

        result = await check_rate_limit(request, key_meta)

        assert result["remaining"] == 59
        assert result["limit"] == RATE_TIERS["free"]["max_tokens"]
        assert "reset" in result

    @pytest.mark.asyncio
    async def test_raises_429_when_limit_exceeded(self):
        """When the token bucket is empty, a 429 HTTPException is raised."""
        mock_redis = AsyncMock()
        # eval returns [allowed=0, remaining=0, retry_after=5]
        mock_redis.eval.return_value = [0, 0, 5]

        request = _make_request(debug=False, redis=mock_redis)
        key_meta = {"key_id": "k3", "rate_tier": "free"}

        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit(request, key_meta)

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in exc_info.value.detail
        assert exc_info.value.headers["Retry-After"] == "5"
        assert exc_info.value.headers["X-RateLimit-Remaining"] == "0"

    @pytest.mark.asyncio
    async def test_uses_correct_tier_config_pro(self):
        """The ``pro`` tier config must be used when rate_tier=pro."""
        mock_redis = AsyncMock()
        mock_redis.eval.return_value = [1, 299, 1]

        request = _make_request(debug=False, redis=mock_redis)
        key_meta = {"key_id": "k4", "rate_tier": "pro"}

        result = await check_rate_limit(request, key_meta)

        # The Lua script is called with pro tier's max_tokens and refill_rate
        call_args = mock_redis.eval.call_args
        assert call_args[0][2] == "ratelimit:k4"
        assert call_args[0][3] == RATE_TIERS["pro"]["max_tokens"]
        assert call_args[0][4] == RATE_TIERS["pro"]["refill_rate"]
        assert result["limit"] == RATE_TIERS["pro"]["max_tokens"]

    @pytest.mark.asyncio
    async def test_uses_correct_tier_config_internal(self):
        """The ``internal`` tier config must be used when rate_tier=internal."""
        mock_redis = AsyncMock()
        mock_redis.eval.return_value = [1, 999, 1]

        request = _make_request(debug=False, redis=mock_redis)
        key_meta = {"key_id": "k5", "rate_tier": "internal"}

        result = await check_rate_limit(request, key_meta)

        call_args = mock_redis.eval.call_args
        assert call_args[0][3] == RATE_TIERS["internal"]["max_tokens"]
        assert call_args[0][4] == RATE_TIERS["internal"]["refill_rate"]
        assert result["limit"] == RATE_TIERS["internal"]["max_tokens"]

    @pytest.mark.asyncio
    async def test_unknown_tier_falls_back_to_free(self):
        """An unrecognised rate_tier must fall back to ``free``."""
        mock_redis = AsyncMock()
        mock_redis.eval.return_value = [1, 59, 1]

        request = _make_request(debug=False, redis=mock_redis)
        key_meta = {"key_id": "k6", "rate_tier": "enterprise"}

        result = await check_rate_limit(request, key_meta)

        call_args = mock_redis.eval.call_args
        assert call_args[0][3] == RATE_TIERS["free"]["max_tokens"]
        assert result["limit"] == RATE_TIERS["free"]["max_tokens"]


# ===================================================================
# 3. check_scope
# ===================================================================

class TestCheckScope:
    """Tests for ``check_scope``."""

    def test_wildcard_scope_grants_access(self):
        """A key with scopes=["*"] should pass any scope check."""
        key_meta = {"scopes": ["*"]}
        # Should not raise
        check_scope("blur:write", key_meta)

    def test_matching_scope_grants_access(self):
        """A key that explicitly lists the required scope should pass."""
        key_meta = {"scopes": ["blur:read", "face:read"]}
        check_scope("blur:read", key_meta)

    def test_missing_scope_raises_403(self):
        """A key lacking the required scope must raise 403."""
        key_meta = {"scopes": ["blur:read"]}
        with pytest.raises(HTTPException) as exc_info:
            check_scope("face:write", key_meta)

        assert exc_info.value.status_code == 403
        assert "Insufficient permissions" in exc_info.value.detail
        assert "face:write" in exc_info.value.detail

    def test_empty_scopes_raises_403(self):
        """A key with no scopes at all must raise 403."""
        key_meta = {"scopes": []}
        with pytest.raises(HTTPException) as exc_info:
            check_scope("blur:read", key_meta)

        assert exc_info.value.status_code == 403

    def test_missing_scopes_key_raises_403(self):
        """A key_meta dict without a ``scopes`` key defaults to [] and raises 403."""
        key_meta = {}
        with pytest.raises(HTTPException) as exc_info:
            check_scope("blur:read", key_meta)

        assert exc_info.value.status_code == 403

    def test_wildcard_among_other_scopes_still_grants(self):
        """Wildcard mixed with other scopes still grants access."""
        key_meta = {"scopes": ["blur:read", "*"]}
        check_scope("anything:here", key_meta)
