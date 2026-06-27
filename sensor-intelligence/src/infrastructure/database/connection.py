"""Database connection management.

Provides async SQLAlchemy engine and session factory.
Handles SQLite-specific configuration (e.g. enabling WAL mode, foreign keys).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.infrastructure.config.settings import get_settings

settings = get_settings()

# ── Engine Configuration ──
# SQLite requires special handling: no connection pool, enable foreign keys.
# PostgreSQL uses connection pooling with sensible defaults.
if settings.is_sqlite:
    engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        connect_args={"check_same_thread": False},
    )
else:
    engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )

# ── Session Factory ──
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that provides a scoped async session.

    The session is automatically committed on success and rolled back
    on exception. It is always closed when the request completes.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
