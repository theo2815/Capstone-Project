"""Switch one API key off and drop its cached entry, so it stops working at once.

Usage (from ai-api/, with DATABASE_URL and REDIS_URL of the target in the environment):
    python -m scripts.revoke_api_key <key id (UUID)>
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid


async def revoke(key_id: str, session_ctx, redis) -> bool:
    """Deactivate first, then clear the cache, so the next lookup reads the inactive row."""
    from sqlalchemy import update

    from src.db.models import APIKey
    from src.middleware.auth import invalidate_api_key_hash

    async with session_ctx() as session:
        result = await session.execute(
            update(APIKey).where(APIKey.id == uuid.UUID(key_id)).values(active=False)
            .returning(APIKey.key_hash)
        )
        key_hash = result.scalar_one_or_none()
    if key_hash is None:
        return False
    # ponytail: a request that read the old row just before the update can re-cache it;
    # run it twice, or wait 5 minutes, if a key must be gone with certainty.
    await invalidate_api_key_hash(redis, key_hash)
    return True


async def _main(key_id: str) -> int:
    # Settings would fill a missing URL from .env or localhost, so the update or the
    # cache clear could hit a different system than the operator meant.
    missing = [name for name in ("DATABASE_URL", "REDIS_URL") if not os.environ.get(name)]
    if missing:
        sys.exit(f"Set {' and '.join(missing)} of the target in the environment.")

    import redis.asyncio as aioredis

    from src.config import get_settings
    from src.db.session import close_db, get_session_ctx, init_db

    await init_db()
    redis = aioredis.from_url(get_settings().REDIS_URL, decode_responses=True)
    try:
        found = await revoke(key_id, get_session_ctx, redis)
    finally:
        await redis.aclose()
        await close_db()
    print("Key revoked" if found else "No key with that id")
    return 0 if found else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(asyncio.run(_main(sys.argv[1])))
