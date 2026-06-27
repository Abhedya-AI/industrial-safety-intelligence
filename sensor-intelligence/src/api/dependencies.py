"""FastAPI dependency injection wiring.

This module is the composition root — it resolves all dependencies
and provides them to FastAPI route handlers via Depends().

The key principle: all wiring happens here, not in the routes.
Routes receive fully constructed use cases / repositories.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.application.interfaces.event_publisher import EventPublisher
from src.infrastructure.config.settings import Settings, get_settings
from src.infrastructure.database.connection import get_async_session
from src.infrastructure.messaging.noop_publisher import NoOpPublisher


# ── Settings ──


def get_app_settings() -> Settings:
    """Provide application settings."""
    return get_settings()


# ── Database Session ──


async def get_db_session(
    session: AsyncSession = Depends(get_async_session),
) -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session for the request lifecycle."""
    yield session


# ── Event Publisher ──

# Singleton publisher instance (created once, reused across requests)
_event_publisher: EventPublisher | None = None


def _get_publisher(settings: Settings) -> EventPublisher:
    """Create the event publisher based on configuration.

    Currently only NoOp is implemented. Future:
    - "kafka" → KafkaPublisher(settings.kafka_bootstrap_servers)
    - "mqtt"  → MQTTPublisher(settings.mqtt_broker_url)
    """
    global _event_publisher
    if _event_publisher is None:
        match settings.event_broker:
            case "noop":
                _event_publisher = NoOpPublisher()
            # case "kafka":
            #     from src.infrastructure.messaging.kafka_publisher import KafkaPublisher
            #     _event_publisher = KafkaPublisher(settings.kafka_bootstrap_servers)
            # case "mqtt":
            #     from src.infrastructure.messaging.mqtt_publisher import MQTTPublisher
            #     _event_publisher = MQTTPublisher(settings.mqtt_broker_url)
            case _:
                _event_publisher = NoOpPublisher()
    return _event_publisher


def get_event_publisher(
    settings: Settings = Depends(get_app_settings),
) -> EventPublisher:
    """Provide the event publisher for the current configuration."""
    return _get_publisher(settings)


# ── Repository Providers ──
# These will be fleshed out when repository implementations are built.
# For now, they serve as the DI wiring points.

# async def get_sensor_repository(
#     session: AsyncSession = Depends(get_db_session),
# ) -> SensorRepository:
#     return SQLAlchemySensorRepository(session)

# async def get_reading_repository(
#     session: AsyncSession = Depends(get_db_session),
# ) -> ReadingRepository:
#     return SQLAlchemyReadingRepository(session)

# ... (repeat for anomaly, alert, threshold repos)


# ── Use Case Providers ──
# These will be added when use cases are implemented.
# Each use case receives its required repositories via DI.

# async def get_ingest_reading_use_case(
#     sensor_repo: SensorRepository = Depends(get_sensor_repository),
#     reading_repo: ReadingRepository = Depends(get_reading_repository),
#     anomaly_repo: AnomalyRepository = Depends(get_anomaly_repository),
#     alert_repo: AlertRepository = Depends(get_alert_repository),
#     threshold_repo: ThresholdRepository = Depends(get_threshold_repository),
#     publisher: EventPublisher = Depends(get_event_publisher),
# ) -> IngestReadingUseCase:
#     return IngestReadingUseCase(
#         sensor_repo, reading_repo, anomaly_repo,
#         alert_repo, threshold_repo, publisher,
#     )
