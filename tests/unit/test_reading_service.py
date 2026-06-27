"""Comprehensive unit tests for the SensorReading service layer.

Covers all 8 business rules:
  1. Sensor must exist
  2. Reject readings for OFFLINE sensors
  3. Value must be finite (no NaN / Inf)
  4. Timestamp must not be in the future
  5. Value must be within sensor min/max range
  6. Confidence must be 0-100
  7. Duplicate reading prevention (same sensor + timestamp)
  8. Batch all-or-nothing validation

Plus:
  - Successful ingestion (single and batch)
  - Query methods (latest, range, stats)
  - Edge cases
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.sensor_intelligence.models.sensor_model import SensorModel
from app.sensor_intelligence.repositories.sqlalchemy_reading_repo import (
    SQLAlchemyReadingRepository,
)
from app.sensor_intelligence.repositories.sqlalchemy_sensor_repo import (
    SQLAlchemySensorRepository,
)
from app.sensor_intelligence.schemas.reading_schemas import ReadingCreateRequest
from app.sensor_intelligence.services.reading_service import ReadingService
from app.shared.exceptions.domain_exceptions import (
    BusinessRuleViolationError,
    InvalidReadingError,
    ResourceNotFoundError,
    ValidationError,
)


# ── Helpers ──


def _make_sensor(**overrides) -> SensorModel:
    defaults = {
        "sensor_id": f"S-{uuid.uuid4().hex[:6]}",
        "sensor_name": "Test Sensor",
        "sensor_type": "GAS",
        "status": "NORMAL",
        "location_zone": "ZONE_A",
        "unit": "ppm",
        "min_value": 0.0,
        "max_value": 1000.0,
    }
    defaults.update(overrides)
    return SensorModel(**defaults)


def _reading_request(sensor_id: str = "S001", **overrides) -> ReadingCreateRequest:
    defaults = {
        "sensor_id": sensor_id,
        "value": 42.0,
        "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        "confidence": 95.0,
    }
    defaults.update(overrides)
    return ReadingCreateRequest(**defaults)


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
) -> ReadingService:
    return ReadingService(reading_repo, sensor_repo)


@pytest_asyncio.fixture
async def registered_sensor(
    sensor_repo: SQLAlchemySensorRepository,
) -> SensorModel:
    """Pre-register an ACTIVE sensor for reading tests."""
    return await sensor_repo.create_sensor(
        _make_sensor(sensor_id="S001", min_value=0.0, max_value=1000.0)
    )


@pytest_asyncio.fixture
async def offline_sensor(
    sensor_repo: SQLAlchemySensorRepository,
) -> SensorModel:
    """Pre-register an OFFLINE sensor."""
    return await sensor_repo.create_sensor(
        _make_sensor(sensor_id="S-OFFLINE", status="OFFLINE")
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 1: Sensor must exist
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_ingest_nonexistent_sensor_raises_error(service: ReadingService):
    with pytest.raises(ResourceNotFoundError) as exc_info:
        await service.ingest_reading(_reading_request(sensor_id="NONEXISTENT"))
    assert "Sensor" in exc_info.value.message


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 2: Reject OFFLINE sensors
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_ingest_offline_sensor_raises_error(
    service: ReadingService, offline_sensor: SensorModel
):
    with pytest.raises(BusinessRuleViolationError) as exc_info:
        await service.ingest_reading(_reading_request(sensor_id="S-OFFLINE"))
    assert "OFFLINE" in exc_info.value.message


async def test_ingest_normal_sensor_succeeds(
    service: ReadingService, registered_sensor: SensorModel
):
    """NORMAL status should be accepted."""
    reading = await service.ingest_reading(_reading_request())
    assert reading.id is not None


async def test_ingest_warning_sensor_succeeds(
    service: ReadingService, sensor_repo: SQLAlchemySensorRepository
):
    """WARNING status should still be accepted."""
    sensor = await sensor_repo.create_sensor(
        _make_sensor(sensor_id="S-WARN", status="WARNING")
    )
    reading = await service.ingest_reading(_reading_request(sensor_id="S-WARN"))
    assert reading.id is not None


async def test_ingest_critical_sensor_succeeds(
    service: ReadingService, sensor_repo: SQLAlchemySensorRepository
):
    """CRITICAL status should still be accepted (only OFFLINE blocks)."""
    await sensor_repo.create_sensor(
        _make_sensor(sensor_id="S-CRIT", status="CRITICAL")
    )
    reading = await service.ingest_reading(_reading_request(sensor_id="S-CRIT"))
    assert reading.id is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 3: Value must be finite
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_ingest_nan_value_raises_error(
    service: ReadingService, registered_sensor: SensorModel
):
    with pytest.raises(InvalidReadingError) as exc_info:
        await service.ingest_reading(
            _reading_request(value=float("nan"))
        )
    assert "finite" in exc_info.value.message


async def test_ingest_inf_value_raises_error(
    service: ReadingService, registered_sensor: SensorModel
):
    with pytest.raises(InvalidReadingError):
        await service.ingest_reading(
            _reading_request(value=float("inf"))
        )


async def test_ingest_negative_inf_value_raises_error(
    service: ReadingService, registered_sensor: SensorModel
):
    with pytest.raises(InvalidReadingError):
        await service.ingest_reading(
            _reading_request(value=float("-inf"))
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 4: Timestamp must not be in the future
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_ingest_future_timestamp_raises_error(
    service: ReadingService, registered_sensor: SensorModel
):
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    with pytest.raises(InvalidReadingError) as exc_info:
        await service.ingest_reading(
            _reading_request(timestamp=future.isoformat())
        )
    assert "future" in exc_info.value.message


async def test_ingest_slightly_future_within_tolerance(
    service: ReadingService, registered_sensor: SensorModel
):
    """Timestamps within 5 minutes of now should be accepted."""
    almost_now = datetime.now(timezone.utc) + timedelta(minutes=3)
    reading = await service.ingest_reading(
        _reading_request(timestamp=almost_now.isoformat())
    )
    assert reading.id is not None


async def test_ingest_past_timestamp_succeeds(
    service: ReadingService, registered_sensor: SensorModel
):
    past = datetime.now(timezone.utc) - timedelta(hours=6)
    reading = await service.ingest_reading(
        _reading_request(timestamp=past.isoformat())
    )
    assert reading.id is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 5: Value within sensor min/max range
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_ingest_value_below_min_raises_error(
    service: ReadingService, registered_sensor: SensorModel
):
    with pytest.raises(InvalidReadingError) as exc_info:
        await service.ingest_reading(_reading_request(value=-1.0))
    assert "below" in exc_info.value.message


async def test_ingest_value_above_max_raises_error(
    service: ReadingService, registered_sensor: SensorModel
):
    with pytest.raises(InvalidReadingError) as exc_info:
        await service.ingest_reading(_reading_request(value=1001.0))
    assert "exceeds" in exc_info.value.message


async def test_ingest_value_at_min_boundary_succeeds(
    service: ReadingService, registered_sensor: SensorModel
):
    reading = await service.ingest_reading(_reading_request(value=0.0))
    assert reading.value == 0.0


async def test_ingest_value_at_max_boundary_succeeds(
    service: ReadingService, registered_sensor: SensorModel
):
    reading = await service.ingest_reading(_reading_request(value=1000.0))
    assert reading.value == 1000.0


async def test_ingest_no_min_max_configured_accepts_any_value(
    service: ReadingService, sensor_repo: SQLAlchemySensorRepository
):
    """Sensor with no min/max should accept any finite value."""
    await sensor_repo.create_sensor(
        _make_sensor(
            sensor_id="S-NORANGE",
            min_value=None,
            max_value=None,
        )
    )
    reading = await service.ingest_reading(
        _reading_request(sensor_id="S-NORANGE", value=999999.0)
    )
    assert reading.value == 999999.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 6: Confidence 0-100
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_ingest_negative_confidence_rejected_by_schema():
    """Pydantic schema rejects confidence < 0."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        _reading_request(confidence=-1.0)


async def test_ingest_over_100_confidence_rejected_by_schema():
    with pytest.raises(Exception):
        _reading_request(confidence=101.0)


async def test_ingest_confidence_at_boundaries_succeeds(
    service: ReadingService, registered_sensor: SensorModel
):
    r0 = await service.ingest_reading(
        _reading_request(
            value=10.0, confidence=0.0,
            timestamp=(datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
        )
    )
    r100 = await service.ingest_reading(
        _reading_request(
            value=20.0, confidence=100.0,
            timestamp=(datetime.now(timezone.utc) - timedelta(minutes=11)).isoformat(),
        )
    )
    assert r0.confidence == 0.0
    assert r100.confidence == 100.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 7: Duplicate reading prevention
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_duplicate_reading_same_sensor_and_timestamp_raises_error(
    service: ReadingService, registered_sensor: SensorModel
):
    ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    await service.ingest_reading(_reading_request(value=10.0, timestamp=ts))
    with pytest.raises(BusinessRuleViolationError) as exc_info:
        await service.ingest_reading(_reading_request(value=20.0, timestamp=ts))
    assert "Duplicate" in exc_info.value.message


async def test_same_timestamp_different_sensors_succeeds(
    service: ReadingService,
    registered_sensor: SensorModel,
    sensor_repo: SQLAlchemySensorRepository,
):
    await sensor_repo.create_sensor(_make_sensor(sensor_id="S002"))
    ts = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    r1 = await service.ingest_reading(
        _reading_request(sensor_id="S001", value=10.0, timestamp=ts)
    )
    r2 = await service.ingest_reading(
        _reading_request(sensor_id="S002", value=20.0, timestamp=ts)
    )
    assert r1.id != r2.id


async def test_same_sensor_different_timestamps_succeeds(
    service: ReadingService, registered_sensor: SensorModel
):
    t1 = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
    t2 = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    r1 = await service.ingest_reading(_reading_request(value=10.0, timestamp=t1))
    r2 = await service.ingest_reading(_reading_request(value=20.0, timestamp=t2))
    assert r1.id != r2.id


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Happy path: single ingestion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_ingest_single_reading_success(
    service: ReadingService, registered_sensor: SensorModel
):
    reading = await service.ingest_reading(_reading_request())

    assert reading.id is not None
    assert reading.value == 42.0
    assert reading.sensor_id == registered_sensor.id  # FK to UUID PK
    assert reading.confidence == 95.0
    assert reading.received_at is not None


async def test_ingest_persists_to_database(
    service: ReadingService,
    registered_sensor: SensorModel,
    reading_repo: SQLAlchemyReadingRepository,
):
    reading = await service.ingest_reading(_reading_request())
    fetched = await reading_repo.get_reading_by_id(reading.id)
    assert fetched is not None
    assert fetched.value == 42.0


async def test_ingest_stores_metadata(
    service: ReadingService,
    registered_sensor: SensorModel,
    reading_repo: SQLAlchemyReadingRepository,
):
    reading = await service.ingest_reading(
        _reading_request(metadata={"equipment_id": "EQ-001"})
    )
    fetched = await reading_repo.get_reading_by_id(reading.id)
    assert fetched is not None
    assert "EQ-001" in fetched.raw_metadata


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 8: Batch ingestion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_ingest_batch_success(
    service: ReadingService, registered_sensor: SensorModel
):
    base = datetime.now(timezone.utc) - timedelta(hours=10)
    requests = [
        _reading_request(
            value=float(i * 10),
            timestamp=(base + timedelta(hours=i)).isoformat(),
        )
        for i in range(3)
    ]
    readings = await service.ingest_batch(requests)
    assert len(readings) == 3
    assert {r.value for r in readings} == {0.0, 10.0, 20.0}


async def test_ingest_batch_rejects_all_if_one_invalid_sensor(
    service: ReadingService, registered_sensor: SensorModel
):
    """Entire batch fails if one reading references a nonexistent sensor."""
    requests = [
        _reading_request(value=10.0),
        _reading_request(sensor_id="NONEXISTENT", value=20.0),
    ]
    with pytest.raises(ResourceNotFoundError):
        await service.ingest_batch(requests)


async def test_ingest_batch_rejects_all_if_one_out_of_range(
    service: ReadingService, registered_sensor: SensorModel
):
    base = datetime.now(timezone.utc) - timedelta(hours=10)
    requests = [
        _reading_request(
            value=10.0,
            timestamp=(base + timedelta(hours=1)).isoformat(),
        ),
        _reading_request(
            value=99999.0,  # above max 1000
            timestamp=(base + timedelta(hours=2)).isoformat(),
        ),
    ]
    with pytest.raises(InvalidReadingError):
        await service.ingest_batch(requests)


async def test_ingest_batch_empty_raises_error(service: ReadingService):
    with pytest.raises(ValidationError):
        await service.ingest_batch([])


async def test_ingest_batch_with_offline_sensor_rejects_all(
    service: ReadingService,
    registered_sensor: SensorModel,
    offline_sensor: SensorModel,
):
    requests = [
        _reading_request(sensor_id="S001", value=10.0),
        _reading_request(sensor_id="S-OFFLINE", value=20.0),
    ]
    with pytest.raises(BusinessRuleViolationError):
        await service.ingest_batch(requests)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Query: get_latest_reading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_get_latest_reading(
    service: ReadingService, registered_sensor: SensorModel
):
    t1 = datetime.now(timezone.utc) - timedelta(hours=4)
    t2 = datetime.now(timezone.utc) - timedelta(hours=3)

    await service.ingest_reading(
        _reading_request(value=10.0, timestamp=t1.isoformat())
    )
    await service.ingest_reading(
        _reading_request(value=99.0, timestamp=t2.isoformat())
    )

    latest = await service.get_latest_reading("S001")
    assert latest is not None
    assert latest.value == 99.0


async def test_get_latest_reading_none_when_no_readings(
    service: ReadingService, registered_sensor: SensorModel
):
    result = await service.get_latest_reading("S001")
    assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Query: get_readings_range
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_get_readings_range(
    service: ReadingService, registered_sensor: SensorModel
):
    base = datetime.now(timezone.utc) - timedelta(hours=10)
    for i in range(5):
        await service.ingest_reading(
            _reading_request(
                value=float(i * 10),
                timestamp=(base + timedelta(hours=i)).isoformat(),
            )
        )

    # Query middle 3 hours
    from_dt = base + timedelta(hours=1)
    to_dt = base + timedelta(hours=3)
    readings = await service.get_readings_range("S001", from_dt, to_dt)

    assert len(readings) == 3
    assert readings[0].value == 10.0  # oldest first (asc order)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Query: get_reading_stats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_get_reading_stats(
    service: ReadingService, registered_sensor: SensorModel
):
    base = datetime.now(timezone.utc) - timedelta(hours=10)
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    for i, v in enumerate(values):
        await service.ingest_reading(
            _reading_request(
                value=v,
                timestamp=(base + timedelta(hours=i)).isoformat(),
            )
        )

    from_dt = base - timedelta(minutes=1)
    to_dt = base + timedelta(hours=10)
    stats = await service.get_reading_stats("S001", from_dt, to_dt)

    assert stats is not None
    assert stats.count == 5
    assert stats.min_value == 10.0
    assert stats.max_value == 50.0
    assert stats.mean == 30.0


async def test_get_reading_stats_empty_returns_none(
    service: ReadingService, registered_sensor: SensorModel
):
    from_dt = datetime.now(timezone.utc) - timedelta(hours=1)
    to_dt = datetime.now(timezone.utc)
    stats = await service.get_reading_stats("S001", from_dt, to_dt)
    assert stats is None
