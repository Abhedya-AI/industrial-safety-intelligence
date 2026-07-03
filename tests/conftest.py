"""Shared test fixtures for all test types."""

import asyncio
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.shared.database.base import Base
from app.shared.database.connection import get_async_session

# Import all models so they register with Base.metadata
from app.sensor_intelligence.models.sensor_model import SensorModel  # noqa: F401
from app.sensor_intelligence.models.reading_model import ReadingModel  # noqa: F401
from app.sensor_intelligence.models.anomaly_model import AnomalyModel  # noqa: F401
from app.sensor_intelligence.models.alert_model import AlertModel  # noqa: F401
from app.sensor_intelligence.models.threshold_model import ThresholdModel  # noqa: F401
from app.sensor_intelligence.models.sensor_health_model import SensorHealthModel  # noqa: F401
from app.sensor_intelligence.models.sensor_baseline_model import SensorBaselineModel  # noqa: F401
from app.risk_prediction.models.risk_prediction_model import RiskPredictionModel  # noqa: F401


# ── In-Memory Test Database ──

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)


# Enable FK enforcement for SQLite (required for FK constraint tests)
from sqlalchemy import event  # noqa: E402


@event.listens_for(test_engine.sync_engine, "connect")
def _enable_sqlite_fks(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

TestSessionFactory = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a clean database session for each test."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionFactory() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Provide an async HTTP client for e2e tests."""
    from app.main import app

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_async_session] = override_get_session

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()
