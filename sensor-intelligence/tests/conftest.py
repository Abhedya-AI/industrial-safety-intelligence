"""Shared test fixtures for all test types.

Provides:
    - In-memory SQLite async engine/session for integration tests
    - FastAPI TestClient for e2e tests
    - Pre-created table schema
"""

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

from src.infrastructure.database.base import Base
from src.infrastructure.database.connection import get_async_session

# Import all models so they register with Base.metadata
from src.infrastructure.database.models.sensor_model import SensorModel  # noqa: F401
from src.infrastructure.database.models.reading_model import ReadingModel  # noqa: F401
from src.infrastructure.database.models.anomaly_model import AnomalyModel  # noqa: F401
from src.infrastructure.database.models.alert_model import AlertModel  # noqa: F401
from src.infrastructure.database.models.threshold_model import ThresholdModel  # noqa: F401


# ── In-Memory Test Database ──

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

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
    """Provide a clean database session for each test.

    Creates all tables before the test and drops them after,
    ensuring complete test isolation.
    """
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionFactory() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Provide an async HTTP client for e2e tests.

    Overrides the database session dependency to use the test database.
    """
    from src.main import app

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_async_session] = override_get_session

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()
