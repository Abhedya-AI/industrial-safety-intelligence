"""FastAPI dependency injection container (Composition Root).

Provides application settings, database sessions, event publishers,
and service-layer dependencies.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import Settings, get_settings
from app.shared.database.connection import get_async_session

# Event publisher (generic, shared)
from app.sensor_intelligence.repositories.noop_publisher import NoOpPublisher


def get_app_settings() -> Settings:
    """Provide application settings."""
    return get_settings()


async def get_db_session(
    session: AsyncSession = Depends(get_async_session),
) -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session for the request lifecycle."""
    yield session


# ── Event Publisher ──

_event_publisher = None


def _get_publisher(settings: Settings):
    global _event_publisher
    if _event_publisher is None:
        _event_publisher = NoOpPublisher()
    return _event_publisher


def get_event_publisher(
    settings: Settings = Depends(get_app_settings),
):
    """Provide the configured event publisher."""
    return _get_publisher(settings)


# ── Sensor Intelligence Dependencies ──


def get_sensor_service(
    session: AsyncSession = Depends(get_async_session),
):
    """Provide a SensorService wired to a SQLAlchemy repository."""
    from app.sensor_intelligence.repositories.sqlalchemy_sensor_repo import (
        SQLAlchemySensorRepository,
    )
    from app.sensor_intelligence.services.sensor_service import SensorService

    repo = SQLAlchemySensorRepository(session)
    return SensorService(repo, session)


def get_reading_service(
    session: AsyncSession = Depends(get_async_session),
):
    """Provide a ReadingService wired to SQLAlchemy repositories."""
    from app.sensor_intelligence.repositories.sqlalchemy_reading_repo import (
        SQLAlchemyReadingRepository,
    )
    from app.sensor_intelligence.repositories.sqlalchemy_sensor_repo import (
        SQLAlchemySensorRepository,
    )
    from app.sensor_intelligence.services.reading_service import ReadingService

    reading_repo = SQLAlchemyReadingRepository(session)
    sensor_repo = SQLAlchemySensorRepository(session)
    return ReadingService(reading_repo, sensor_repo)


# ── Risk Prediction Dependencies ──


def get_risk_prediction_service(
    session: AsyncSession = Depends(get_async_session),
):
    """Provide a RiskPredictionService wired to a SQLAlchemy repository."""
    from app.risk_prediction.repositories.sqlalchemy_risk_prediction_repo import (
        SQLAlchemyRiskPredictionRepository,
    )
    from app.risk_prediction.services.risk_prediction_service import (
        RiskPredictionService,
    )

    repo = SQLAlchemyRiskPredictionRepository(session)
    return RiskPredictionService(repo)

