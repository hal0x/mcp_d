"""Database utilities for Policy MCP."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import Settings, get_settings


class Base(DeclarativeBase):
    """Base class for all ORM models."""


_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine(settings: Optional[Settings] = None) -> AsyncEngine:
    """Create (or return cached) async engine."""
    global _engine, _session_factory

    if _engine is None:
        settings = settings or get_settings()
        engine_kwargs: dict[str, object] = {"echo": settings.DB_ECHO}

        if settings.DB_POOL_SIZE is not None:
            engine_kwargs["pool_size"] = settings.DB_POOL_SIZE
        if settings.DB_MAX_OVERFLOW is not None:
            engine_kwargs["max_overflow"] = settings.DB_MAX_OVERFLOW

        _engine = create_async_engine(settings.DB_URL, **engine_kwargs)
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)

    return _engine


def get_session_factory(settings: Optional[Settings] = None) -> async_sessionmaker[AsyncSession]:
    """Return async session factory, initializing engine if needed."""
    global _session_factory

    if _session_factory is None:
        get_engine(settings=settings)

    if _session_factory is None:  # pragma: no cover - defensive
        raise RuntimeError("Session factory is not initialized")

    return _session_factory


@asynccontextmanager
async def get_session(settings: Optional[Settings] = None) -> AsyncIterator[AsyncSession]:
    """Provide an async session context manager."""
    session_factory = get_session_factory(settings=settings)
    async with session_factory() as session:
        yield session


async def init_db(settings: Optional[Settings] = None) -> None:
    """Initialize database schema (idempotent)."""
    engine = get_engine(settings=settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """Dispose engine (used in tests/shutdown)."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
