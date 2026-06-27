"""FastAPI application factory.

Creates and configures the FastAPI app with all middleware,
routers, and lifecycle hooks.
"""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from src.api.middleware.cors import setup_cors
from src.api.middleware.error_handler import register_error_handlers
from src.api.middleware.request_logging import RequestLoggingMiddleware
from src.api.v1.router import v1_router
from src.infrastructure.config.settings import get_settings

# ── Logging Setup ──
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Application Lifecycle ──


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown events.

    Startup:
        - Log configuration
        - Create database tables (dev only — production uses Alembic)
        - Initialize connections

    Shutdown:
        - Close database connections
        - Disconnect event publisher
    """
    # ── Startup ──
    logger.info(
        "Starting %s v%s [%s]",
        settings.app_name,
        settings.app_version,
        settings.app_env,
    )
    logger.info("Database: %s", settings.database_url.split("@")[-1] if "@" in settings.database_url else settings.database_url)
    logger.info("Event broker: %s", settings.event_broker)

    # In development, auto-create tables if they don't exist.
    # In production, Alembic migrations handle schema management.
    if not settings.is_production:
        from src.infrastructure.database.base import Base
        from src.infrastructure.database.connection import engine
        # Import all models so they register with Base.metadata
        from src.infrastructure.database.models import (  # noqa: F401
            alert_model,
            anomaly_model,
            reading_model,
            sensor_model,
            threshold_model,
        )

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created/verified")

    yield

    # ── Shutdown ──
    logger.info("Shutting down %s", settings.app_name)
    from src.infrastructure.database.connection import engine

    await engine.dispose()
    logger.info("Database connections closed")


# ── App Factory ──


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title=settings.app_name,
        description=(
            "Sensor Intelligence Service for the SentinelAI "
            "Industrial Safety Intelligence Platform. "
            "Ingests IoT sensor data, detects anomalies, "
            "scores sensor health, and generates real-time alerts."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware (order matters — last added = first executed)
    application.add_middleware(RequestLoggingMiddleware)
    setup_cors(application)

    # Error handlers
    register_error_handlers(application)

    # Routers
    application.include_router(v1_router, prefix=settings.api_prefix)

    return application


# ── App Instance ──
app = create_app()
