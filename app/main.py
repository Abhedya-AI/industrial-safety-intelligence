"""FastAPI application factory for Industrial Safety Intelligence Monolith."""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from app.core.logging import setup_logging
from app.core.middleware import setup_middleware
from app.core.settings import get_settings
from app.core.dependencies import (
    start_consumers,
    stop_consumers,
    recover_twin_from_snapshot,
)
from app.sensor_intelligence.api.router import sensor_intelligence_router
from app.risk_prediction.api.router import risk_prediction_router
from app.compound_risk.api.router import compound_risk_router
from app.hazard_propagation.api.router import hazard_propagation_router
from app.digital_twin.api.router import digital_twin_router

# Initialize unified logging
setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    logger.info(
        "Starting %s v%s [%s]",
        settings.app_name,
        settings.app_version,
        settings.app_env,
    )
    logger.info(
        "Database: %s",
        settings.database_url.split("@")[-1] if "@" in settings.database_url else settings.database_url
    )

    # In development, auto-create tables
    if not settings.is_production:
        from app.shared.database.base import Base
        from app.shared.database.connection import engine
        # Import all ORM models to register them on Base.metadata
        from app.sensor_intelligence.models import (  # noqa: F401
            alert_model,
            anomaly_model,
            reading_model,
            sensor_model,
            threshold_model,
        )
        from app.risk_prediction.models import risk_prediction_model  # noqa: F401
        from app.compound_risk.models import compound_risk_model  # noqa: F401
        from app.digital_twin.models import (  # noqa: F401
            facility_snapshot_model,
            zone_state_model,
        )

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created/verified")

    # Recover Digital Twin from latest snapshot (before consumers start)
    try:
        await recover_twin_from_snapshot(settings)
    except Exception:
        logger.exception(
            "Snapshot recovery failed — Digital Twin starts fresh"
        )

    # Start Kafka consumers (noop if EVENT_BROKER != kafka)
    start_consumers(settings)
    logger.info("Event broker: %s", settings.event_broker)

    yield

    logger.info("Shutting down %s", settings.app_name)

    # Stop Kafka consumers
    stop_consumers()

    from app.shared.database.connection import engine
    await engine.dispose()
    logger.info("Database connection pool closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title=settings.app_name,
        description=(
            "Unified backend for the Industrial Safety Intelligence Platform "
            "comprising Sensor Intelligence, Risk Prediction, Compound Risk, "
            "Hazard Propagation, and Digital Twin modules."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware & Error Handlers
    setup_middleware(application)

    # Mount v1 Routers
    application.include_router(sensor_intelligence_router, prefix=settings.api_prefix)
    application.include_router(risk_prediction_router, prefix=settings.api_prefix)
    application.include_router(compound_risk_router, prefix=settings.api_prefix)
    application.include_router(hazard_propagation_router, prefix=settings.api_prefix)
    application.include_router(digital_twin_router, prefix=settings.api_prefix)

    return application


app = create_app()
