"""Database utilities for Supervisor MCP."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import Settings, get_settings


class Base(DeclarativeBase):
    """Base class for ORM models."""


_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine(settings: Optional[Settings] = None) -> AsyncEngine:
    """Return global async engine instance (initialize on first call)."""
    global _engine, _session_factory
    if _engine is None:
        settings = settings or get_settings()

        kwargs: dict[str, object] = {"echo": settings.db_echo}
        if settings.db_pool_size is not None:
            kwargs["pool_size"] = settings.db_pool_size
        if settings.db_max_overflow is not None:
            kwargs["max_overflow"] = settings.db_max_overflow

        _engine = create_async_engine(settings.db_url, **kwargs)
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)

    return _engine


def get_session_factory(settings: Optional[Settings] = None) -> async_sessionmaker[AsyncSession]:
    """Return async session factory."""
    global _session_factory
    if _session_factory is None:
        get_engine(settings=settings)
    if _session_factory is None:  # pragma: no cover
        raise RuntimeError("Session factory initialization failed")
    return _session_factory


@asynccontextmanager
async def get_session(settings: Optional[Settings] = None) -> AsyncIterator[AsyncSession]:
    """Async context manager yielding DB session."""
    session_factory = get_session_factory(settings=settings)
    async with session_factory() as session:
        yield session


async def init_db(settings: Optional[Settings] = None) -> None:
    """Ensure database schema is created."""
    engine = get_engine(settings=settings)
    async with engine.begin() as conn:
        from .pydantic_models import orm  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """Dispose engine (useful in tests)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
