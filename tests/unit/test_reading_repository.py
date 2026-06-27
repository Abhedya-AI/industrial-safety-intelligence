"""Comprehensive unit tests for the SensorReading repository layer.

Covers:
  - Reading creation (single and batch)
  - Reading retrieval by ID
  - Latest reading lookup
  - Sensor history (time-range queries)
  - list_readings with pagination and filtering
  - reading_exists existence check
  - count_for_sensor
  - get_stats aggregation
  - Empty result scenarios
  - Database error handling (duplicate PK, FK violation)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.sensor_intelligence.models.reading_model import ReadingModel
from app.sensor_intelligence.models.sensor_model import SensorModel
from app.sensor_intelligence.repositories.sqlalchemy_reading_repo import (
    SQLAlchemyReadingRepository,
)
from app.sensor_intelligence.repositories.sqlalchemy_sensor_repo import (
    SQLAlchemySensorRepository,
)


# ── Helpers ──


def _uid() -> str:
    return str(uuid.uuid4())


def _make_sensor(**overrides) -> SensorModel:
    defaults = {
        "sensor_id": f"S-{uuid.uuid4().hex[:6]}",
        "sensor_name": "Test Sensor",
        "sensor_type": "GAS",
        "status": "NORMAL",
        "location_zone": "ZONE_A",
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
async def repo(db_session: AsyncSession) -> SQLAlchemyReadingRepository:
    return SQLAlchemyReadingRepository(db_session)


@pytest_asyncio.fixture
async def sensor(sensor_repo: SQLAlchemySensorRepository) -> SensorModel:
    """Pre-register a sensor for FK references."""
    return await sensor_repo.create_sensor(
        _make_sensor(sensor_id="S001")
    )


@pytest_asyncio.fixture
async def sensor_b(sensor_repo: SQLAlchemySensorRepository) -> SensorModel:
    """Second sensor for cross-sensor tests."""
    return await sensor_repo.create_sensor(
        _make_sensor(sensor_id="S002", location_zone="ZONE_B")
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# create_reading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_create_reading_persists_and_returns_model(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    reading = _make_reading(sensor.id, value=100.0, confidence=99.0)
    result = await repo.create_reading(reading)

    assert result.id == reading.id
    assert result.sensor_id == sensor.id
    assert result.value == 100.0
    assert result.confidence == 99.0
    assert result.received_at is not None


async def test_create_reading_auto_sets_received_at(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    reading = _make_reading(sensor.id)
    result = await repo.create_reading(reading)
    assert result.received_at is not None


async def test_create_reading_duplicate_id_raises_integrity_error(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    reading_id = _uid()
    await repo.create_reading(_make_reading(sensor.id, id=reading_id))
    with pytest.raises(IntegrityError):
        await repo.create_reading(_make_reading(sensor.id, id=reading_id))


async def test_create_reading_invalid_sensor_fk_raises_integrity_error(
    repo: SQLAlchemyReadingRepository,
):
    with pytest.raises(IntegrityError):
        await repo.create_reading(_make_reading("nonexistent-sensor-pk"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# create_readings_batch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_create_readings_batch_persists_all(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    readings = [
        _make_reading(sensor.id, value=float(i * 10))
        for i in range(5)
    ]
    results = await repo.create_readings_batch(readings)

    assert len(results) == 5
    assert {r.value for r in results} == {0.0, 10.0, 20.0, 30.0, 40.0}


async def test_create_readings_batch_empty_list(
    repo: SQLAlchemyReadingRepository,
):
    results = await repo.create_readings_batch([])
    assert results == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# get_reading_by_id
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_get_reading_by_id_returns_reading(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    reading = await repo.create_reading(_make_reading(sensor.id, value=77.7))
    fetched = await repo.get_reading_by_id(reading.id)

    assert fetched is not None
    assert fetched.id == reading.id
    assert fetched.value == 77.7


async def test_get_reading_by_id_returns_none_for_unknown(
    repo: SQLAlchemyReadingRepository,
):
    result = await repo.get_reading_by_id(_uid())
    assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# get_latest_reading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_get_latest_reading_returns_most_recent(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    t1 = datetime.now(timezone.utc) - timedelta(hours=3)
    t2 = datetime.now(timezone.utc) - timedelta(hours=2)
    t3 = datetime.now(timezone.utc) - timedelta(hours=1)

    await repo.create_reading(_make_reading(sensor.id, value=10.0, timestamp=t1))
    await repo.create_reading(_make_reading(sensor.id, value=20.0, timestamp=t2))
    await repo.create_reading(_make_reading(sensor.id, value=30.0, timestamp=t3))

    latest = await repo.get_latest_reading(sensor.id)
    assert latest is not None
    assert latest.value == 30.0


async def test_get_latest_reading_returns_none_when_no_readings(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    result = await repo.get_latest_reading(sensor.id)
    assert result is None


async def test_get_latest_reading_scoped_to_sensor(
    repo: SQLAlchemyReadingRepository,
    sensor: SensorModel,
    sensor_b: SensorModel,
):
    """Latest reading for sensor A does not return sensor B readings."""
    t1 = datetime.now(timezone.utc) - timedelta(hours=2)
    t2 = datetime.now(timezone.utc) - timedelta(hours=1)

    await repo.create_reading(
        _make_reading(sensor.id, value=10.0, timestamp=t1)
    )
    await repo.create_reading(
        _make_reading(sensor_b.id, value=99.0, timestamp=t2)
    )

    latest_a = await repo.get_latest_reading(sensor.id)
    assert latest_a is not None
    assert latest_a.value == 10.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# get_sensor_history
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_get_sensor_history_returns_range(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    base = datetime.now(timezone.utc) - timedelta(hours=10)
    for i in range(5):
        await repo.create_reading(
            _make_reading(
                sensor.id,
                value=float(i * 10),
                timestamp=base + timedelta(hours=i),
            )
        )

    # Query middle 3 readings (hours 1, 2, 3)
    start_time = base + timedelta(hours=1)
    end_time = base + timedelta(hours=3)
    results = await repo.get_sensor_history(sensor.id, start_time, end_time)

    assert len(results) == 3
    # Should be ordered ascending
    assert results[0].value == 10.0
    assert results[-1].value == 30.0


async def test_get_sensor_history_empty_range(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    start = datetime.now(timezone.utc) - timedelta(hours=2)
    end = datetime.now(timezone.utc) - timedelta(hours=1)
    results = await repo.get_sensor_history(sensor.id, start, end)
    assert results == []


async def test_get_sensor_history_respects_limit(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    base = datetime.now(timezone.utc) - timedelta(hours=10)
    for i in range(10):
        await repo.create_reading(
            _make_reading(
                sensor.id,
                value=float(i),
                timestamp=base + timedelta(hours=i),
            )
        )

    start = base
    end = base + timedelta(hours=20)
    results = await repo.get_sensor_history(sensor.id, start, end, limit=3)
    assert len(results) == 3


async def test_get_sensor_history_scoped_to_sensor(
    repo: SQLAlchemyReadingRepository,
    sensor: SensorModel,
    sensor_b: SensorModel,
):
    base = datetime.now(timezone.utc) - timedelta(hours=5)
    await repo.create_reading(
        _make_reading(sensor.id, value=10.0, timestamp=base)
    )
    await repo.create_reading(
        _make_reading(sensor_b.id, value=99.0, timestamp=base)
    )

    start = base - timedelta(hours=1)
    end = base + timedelta(hours=1)
    results = await repo.get_sensor_history(sensor.id, start, end)
    assert len(results) == 1
    assert results[0].value == 10.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# list_readings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_list_readings_returns_all_newest_first(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    base = datetime.now(timezone.utc) - timedelta(hours=5)
    for i in range(3):
        await repo.create_reading(
            _make_reading(
                sensor.id,
                value=float(i * 10),
                timestamp=base + timedelta(hours=i),
            )
        )

    results = await repo.list_readings()
    assert len(results) == 3
    # Newest first
    assert results[0].value == 20.0
    assert results[-1].value == 0.0


async def test_list_readings_filter_by_sensor(
    repo: SQLAlchemyReadingRepository,
    sensor: SensorModel,
    sensor_b: SensorModel,
):
    await repo.create_reading(_make_reading(sensor.id, value=10.0))
    await repo.create_reading(_make_reading(sensor_b.id, value=20.0))

    results = await repo.list_readings(sensor_pk=sensor.id)
    assert len(results) == 1
    assert results[0].value == 10.0


async def test_list_readings_pagination(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    base = datetime.now(timezone.utc) - timedelta(hours=10)
    for i in range(5):
        await repo.create_reading(
            _make_reading(
                sensor.id,
                value=float(i),
                timestamp=base + timedelta(hours=i),
            )
        )

    page1 = await repo.list_readings(offset=0, limit=2)
    page2 = await repo.list_readings(offset=2, limit=2)
    page3 = await repo.list_readings(offset=4, limit=2)

    assert len(page1) == 2
    assert len(page2) == 2
    assert len(page3) == 1


async def test_list_readings_empty(repo: SQLAlchemyReadingRepository):
    results = await repo.list_readings()
    assert results == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# reading_exists
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_reading_exists_returns_true(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    reading = await repo.create_reading(_make_reading(sensor.id))
    assert await repo.reading_exists(reading.id) is True


async def test_reading_exists_returns_false(
    repo: SQLAlchemyReadingRepository,
):
    assert await repo.reading_exists(_uid()) is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# count_for_sensor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_count_for_sensor_returns_correct_count(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    for _ in range(4):
        await repo.create_reading(_make_reading(sensor.id))
    assert await repo.count_for_sensor(sensor.id) == 4


async def test_count_for_sensor_zero_when_empty(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    assert await repo.count_for_sensor(sensor.id) == 0


async def test_count_for_sensor_scoped(
    repo: SQLAlchemyReadingRepository,
    sensor: SensorModel,
    sensor_b: SensorModel,
):
    await repo.create_reading(_make_reading(sensor.id))
    await repo.create_reading(_make_reading(sensor.id))
    await repo.create_reading(_make_reading(sensor_b.id))

    assert await repo.count_for_sensor(sensor.id) == 2
    assert await repo.count_for_sensor(sensor_b.id) == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# get_stats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_get_stats_computes_aggregates(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    base = datetime.now(timezone.utc) - timedelta(hours=10)
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    for i, v in enumerate(values):
        await repo.create_reading(
            _make_reading(
                sensor.id,
                value=v,
                timestamp=base + timedelta(hours=i),
            )
        )

    start = base - timedelta(minutes=1)
    end = base + timedelta(hours=10)
    stats = await repo.get_stats(sensor.id, start, end)

    assert stats is not None
    assert stats.count == 5
    assert stats.min_value == 10.0
    assert stats.max_value == 50.0
    assert stats.mean == 30.0  # (10+20+30+40+50)/5
    assert stats.std_dev > 0


async def test_get_stats_returns_none_when_empty(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    start = datetime.now(timezone.utc) - timedelta(hours=2)
    end = datetime.now(timezone.utc)
    stats = await repo.get_stats(sensor.id, start, end)
    assert stats is None


async def test_get_stats_single_reading(
    repo: SQLAlchemyReadingRepository, sensor: SensorModel
):
    ts = datetime.now(timezone.utc) - timedelta(hours=1)
    await repo.create_reading(_make_reading(sensor.id, value=42.0, timestamp=ts))

    start = ts - timedelta(minutes=1)
    end = ts + timedelta(minutes=1)
    stats = await repo.get_stats(sensor.id, start, end)

    assert stats is not None
    assert stats.count == 1
    assert stats.mean == 42.0
    assert stats.std_dev == 0.0
    assert stats.min_value == 42.0
    assert stats.max_value == 42.0
