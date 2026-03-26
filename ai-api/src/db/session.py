from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_engine = None
_session_factory = None
_engine_pid: int | None = None


async def init_db() -> None:
    """Initialize the async database engine and session factory."""
    global _engine, _session_factory, _engine_pid
    settings = get_settings()

    _engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.SQL_ECHO,
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True,
        pool_timeout=30,
        pool_recycle=3600,
        connect_args={"timeout": 10, "command_timeout": 30},
    )
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    _engine_pid = os.getpid()
    logger.info("Database engine initialized")


async def close_db() -> None:
    """Dispose of the database engine."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("Database engine disposed")


def _check_fork_safety() -> None:
    """Guard against using a pre-fork connection pool in a child process."""
    if _engine_pid is not None and os.getpid() != _engine_pid:
        raise RuntimeError(
            f"Database engine was created in PID {_engine_pid} but is being used "
            f"in PID {os.getpid()}. asyncpg connections are not fork-safe. "
            "Ensure init_db() is called after fork, not before (disable --preload)."
        )


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session (for FastAPI Depends only)."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    _check_fork_safety()
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_session_ctx() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions.

    Use this in endpoint code (``async with get_session_ctx() as session``).
    The session is committed on normal exit and rolled back on exception.
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    _check_fork_safety()
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_readonly_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for read-only database sessions.

    Rolls back instead of committing — avoids the unnecessary COMMIT
    round-trip for GET/read-only endpoints.
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    _check_fork_safety()
    async with _session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()


async def check_db_health() -> bool:
    """Check if the database is reachable."""
    if _engine is None:
        return False
    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return False
