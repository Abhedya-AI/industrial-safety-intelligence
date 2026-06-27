"""FastAPI dependency injection container (Composition Root).

Provides application settings, database sessions, and event publishers.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import Settings, get_settings
from app.shared.database.connection import get_async_session

# We will define abstract interfaces in a shared location or modularly.
# Currently event publisher is generic, so it can be shared.
# Let's import it from a shared place or use direct classes.
from app.sensor_intelligence.repositories.noop_publisher import NoOpPublisher


def get_app_settings() -> Settings:
    """Provide application settings."""
    return get_settings()


async def get_db_session(
    session: AsyncSession = Depends(get_async_session),
) -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session for the request lifecycle."""
    yield session


# Event Publisher Singleton
_event_publisher = None


def _get_publisher(settings: Settings):
    global _event_publisher
    if _event_publisher is None:
        # Defaults to NoOpPublisher for local/hackathon dev
        _event_publisher = NoOpPublisher()
    return _event_publisher


def get_event_publisher(
    settings: Settings = Depends(get_app_settings),
):
    """Provide the configured event publisher."""
    return _get_publisher(settings)
