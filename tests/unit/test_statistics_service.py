"""Unit tests for the StatisticsService (integration with reading repository).

Covers:
  - Empty datasets (no readings)
  - Single reading
  - Multiple readings with full analysis
  - Named time ranges
  - Trend computation
  - Descriptive-only computation
  - Nonexistent sensor → 404
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.sensor_intelligence.models.reading_model import ReadingModel
from app.sensor_intelligence.models.sensor_model import SensorModel
from app.sensor_intelligence.repositories.sqlalchemy_reading_repo import (
    SQLAlchemyReadingRepository,
)
from app.sensor_intelligence.repositories.sqlalchemy_sensor_repo import (
    SQLAlchemySensorRepository,
)
from app.sensor_intelligence.services.statistics_service import StatisticsService
from app.shared.exceptions.domain_exceptions import ResourceNotFoundError


# ── Helpers ──


def _uid() -> str:
    return str(uuid.uuid4())


def _make_sensor(**overrides) -> SensorModel:
    defaults = {
        "sensor_id": f"S-{uuid.uuid4().hex[:6]}",
        "sensor_name": "Test Sensor",
        "sensor_type": "GAS",
        "status": "NORMAL",
        "unit": "ppm",
    }
    defaults.update(overrides)
    return SensorModel(**defaults)


def _make_reading(sensor_pk: str, **overrides) -> ReadingModel:
    defaults = {
        "id": _uid(),
        "sensor_id": sensor_pk,
        "value": 42.0,
        "timestamp": datetime.now(timezone.utc),
        "confidence": 95.0,
    }
    defaults.update(overrides)
    return ReadingModel(**defaults)


# ── Fixtures ──


@pytest_asyncio.fixture
async def sensor_repo(db_session: AsyncSession) -> SQLAlchemySensorRepository:
    return SQLAlchemySensorRepository(db_session)


@pytest_asyncio.fixture
async def reading_repo(db_session: AsyncSession) -> SQLAlchemyReadingRepository:
    return SQLAlchemyReadingRepository(db_session)


@pytest_asyncio.fixture
async def service(
    reading_repo: SQLAlchemyReadingRepository,
    sensor_repo: SQLAlchemySensorRepository,
) -> StatisticsService:
    return StatisticsService(reading_repo, sensor_repo)


@pytest_asyncio.fixture
async def sensor(sensor_repo: SQLAlchemySensorRepository) -> SensorModel:
    return await sensor_repo.create_sensor(_make_sensor(sensor_id="S001"))


async def _seed_readings(
    reading_repo: SQLAlchemyReadingRepository,
    sensor_pk: str,
    values: list[float],
    base: datetime | None = None,
) -> list[ReadingModel]:
    """Insert readings with increasing timestamps."""
    if base is None:
        base = datetime.now(timezone.utc) - timedelta(hours=10)
    readings = []
    for i, v in enumerate(values):
        r = await reading_repo.create_reading(
            _make_reading(
                sensor_pk,
                value=v,
                timestamp=base + timedelta(hours=i),
            )
        )
        readings.append(r)
    return readings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# compute_statistics — empty
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_statistics_no_readings(
    service: StatisticsService, sensor: SensorModel
):
    result = await service.compute_statistics("S001")
    assert result.reading_count == 0
    assert result.descriptive is None
    assert result.trend is None
    assert result.rate_of_change == []
    assert result.rolling_avg is None
    assert result.rolling_std is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# compute_statistics — single reading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_statistics_single_reading(
    service: StatisticsService,
    sensor: SensorModel,
    reading_repo: SQLAlchemyReadingRepository,
):
    await _seed_readings(reading_repo, sensor.id, [42.0])

    result = await service.compute_statistics("S001")
    assert result.reading_count == 1
    assert result.descriptive is not None
    assert result.descriptive.mean == 42.0
    assert result.descriptive.std_dev == 0.0
    assert result.trend is None  # Need >= 2 for trend
    assert result.rate_of_change == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# compute_statistics — multiple readings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_statistics_multiple_readings(
    service: StatisticsService,
    sensor: SensorModel,
    reading_repo: SQLAlchemyReadingRepository,
):
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    await _seed_readings(reading_repo, sensor.id, values)

    result = await service.compute_statistics("S001")
    assert result.reading_count == 5
    assert result.descriptive is not None
    assert result.descriptive.mean == 30.0
    assert result.descriptive.median == 30.0
    assert result.descriptive.minimum == 10.0
    assert result.descriptive.maximum == 50.0
    assert result.descriptive.count == 5

    # Trend should be increasing
    assert result.trend is not None
    assert result.trend.direction == "increasing"
    assert result.trend.slope > 0

    # Rate of change
    assert len(result.rate_of_change) == 4
    assert all(r == 10.0 for r in result.rate_of_change)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# compute_statistics — constant values
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_statistics_constant_values(
    service: StatisticsService,
    sensor: SensorModel,
    reading_repo: SQLAlchemyReadingRepository,
):
    await _seed_readings(reading_repo, sensor.id, [7.0] * 5)

    result = await service.compute_statistics("S001")
    assert result.descriptive is not None
    assert result.descriptive.mean == 7.0
    assert result.descriptive.variance == 0.0
    assert result.descriptive.std_dev == 0.0
    assert result.trend is not None
    assert result.trend.direction == "stable"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# compute_statistics — decreasing values
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_statistics_decreasing_values(
    service: StatisticsService,
    sensor: SensorModel,
    reading_repo: SQLAlchemyReadingRepository,
):
    await _seed_readings(reading_repo, sensor.id, [50.0, 40.0, 30.0, 20.0, 10.0])

    result = await service.compute_statistics("S001")
    assert result.trend is not None
    assert result.trend.direction == "decreasing"
    assert result.trend.slope < 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# compute_statistics — rolling window
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_statistics_with_rolling_window(
    service: StatisticsService,
    sensor: SensorModel,
    reading_repo: SQLAlchemyReadingRepository,
):
    base = datetime.now(timezone.utc) - timedelta(hours=24)
    values = list(range(1, 21))  # 1..20
    await _seed_readings(reading_repo, sensor.id, [float(v) for v in values], base=base)

    result = await service.compute_statistics(
        "S001",
        start=base - timedelta(minutes=1),
        end=base + timedelta(hours=20),
        rolling_window=5,
    )
    assert result.reading_count == 20
    assert result.rolling_avg is not None
    assert result.rolling_avg.window_size == 5
    assert len(result.rolling_avg.values) == 16  # 20 - 5 + 1
    assert result.rolling_std is not None
    assert len(result.rolling_std.values) == 16


async def test_statistics_rolling_window_too_large(
    service: StatisticsService,
    sensor: SensorModel,
    reading_repo: SQLAlchemyReadingRepository,
):
    await _seed_readings(reading_repo, sensor.id, [10.0, 20.0, 30.0])

    result = await service.compute_statistics("S001", rolling_window=10)
    assert result.rolling_avg is None
    assert result.rolling_std is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# compute_statistics — named time ranges
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_statistics_time_range_filter(
    service: StatisticsService,
    sensor: SensorModel,
    reading_repo: SQLAlchemyReadingRepository,
):
    """Readings outside the time range should be excluded."""
    base = datetime.now(timezone.utc) - timedelta(hours=48)
    # Insert at -48h, -47h, ..., -44h (5 readings — all > 24h ago)
    await _seed_readings(reading_repo, sensor.id, [10.0] * 5, base=base)

    result = await service.compute_statistics("S001", time_range="24h")
    assert result.reading_count == 0  # All readings are older than 24h


async def test_statistics_explicit_start_end(
    service: StatisticsService,
    sensor: SensorModel,
    reading_repo: SQLAlchemyReadingRepository,
):
    base = datetime.now(timezone.utc) - timedelta(hours=10)
    await _seed_readings(reading_repo, sensor.id, [10.0, 20.0, 30.0, 40.0, 50.0], base=base)

    start = base + timedelta(hours=1)
    end = base + timedelta(hours=3)
    result = await service.compute_statistics("S001", start=start, end=end)
    assert result.reading_count == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# compute_descriptive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_compute_descriptive_success(
    service: StatisticsService,
    sensor: SensorModel,
    reading_repo: SQLAlchemyReadingRepository,
):
    base = datetime.now(timezone.utc) - timedelta(hours=5)
    await _seed_readings(reading_repo, sensor.id, [10.0, 20.0, 30.0], base=base)

    start = base - timedelta(minutes=1)
    end = base + timedelta(hours=5)
    result = await service.compute_descriptive("S001", start, end)
    assert result is not None
    assert result.mean == 20.0
    assert result.count == 3


async def test_compute_descriptive_empty(
    service: StatisticsService,
    sensor: SensorModel,
):
    start = datetime.now(timezone.utc) - timedelta(hours=1)
    end = datetime.now(timezone.utc)
    result = await service.compute_descriptive("S001", start, end)
    assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# compute_trend
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_compute_trend_increasing(
    service: StatisticsService,
    sensor: SensorModel,
    reading_repo: SQLAlchemyReadingRepository,
):
    base = datetime.now(timezone.utc) - timedelta(hours=5)
    await _seed_readings(reading_repo, sensor.id, [10.0, 20.0, 30.0], base=base)

    start = base - timedelta(minutes=1)
    end = base + timedelta(hours=5)
    result = await service.compute_trend("S001", start, end)
    assert result is not None
    assert result.direction == "increasing"


async def test_compute_trend_insufficient_data(
    service: StatisticsService,
    sensor: SensorModel,
    reading_repo: SQLAlchemyReadingRepository,
):
    base = datetime.now(timezone.utc) - timedelta(hours=5)
    await _seed_readings(reading_repo, sensor.id, [42.0], base=base)

    start = base - timedelta(minutes=1)
    end = base + timedelta(hours=5)
    result = await service.compute_trend("S001", start, end)
    assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Error: nonexistent sensor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_statistics_nonexistent_sensor_raises_404(
    service: StatisticsService,
):
    with pytest.raises(ResourceNotFoundError):
        await service.compute_statistics("NONEXISTENT")


async def test_compute_descriptive_nonexistent_sensor(
    service: StatisticsService,
):
    with pytest.raises(ResourceNotFoundError):
        start = datetime.now(timezone.utc) - timedelta(hours=1)
        end = datetime.now(timezone.utc)
        await service.compute_descriptive("NONEXISTENT", start, end)


async def test_compute_trend_nonexistent_sensor(
    service: StatisticsService,
):
    with pytest.raises(ResourceNotFoundError):
        start = datetime.now(timezone.utc) - timedelta(hours=1)
        end = datetime.now(timezone.utc)
        await service.compute_trend("NONEXISTENT", start, end)
