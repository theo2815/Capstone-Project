from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_sync_engine = None
_sync_session_factory = None


def _get_sync_url() -> str:
    """Derive a synchronous database URL from the async one."""
    settings = get_settings()
    return settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")


def init_sync_db() -> None:
    """Initialize the synchronous database engine. Called once at worker startup."""
    global _sync_engine, _sync_session_factory
    if _sync_engine is not None:
        return

    _sync_engine = create_engine(
        _get_sync_url(),
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
    )
    _sync_session_factory = sessionmaker(bind=_sync_engine, expire_on_commit=False)
    logger.info("Sync database engine initialized")


def close_sync_db() -> None:
    """Dispose of the synchronous database engine."""
    global _sync_engine
    if _sync_engine:
        _sync_engine.dispose()
        logger.info("Sync database engine disposed")


@contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    """Yield a synchronous database session with auto-commit/rollback."""
    if _sync_session_factory is None:
        init_sync_db()
    session = _sync_session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
