"""Comprehensive unit tests for SQLAlchemySensorRepository.

Tests every method in the repository interface against a real in-memory
SQLite database to validate SQL correctness and edge cases.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.sensor_intelligence.models.sensor_model import SensorModel
from app.sensor_intelligence.repositories.sqlalchemy_sensor_repo import (
    SQLAlchemySensorRepository,
)


# ── Helpers ──


def _make_sensor(**overrides) -> SensorModel:
    """Build a SensorModel with sensible defaults, overridable per test."""
    defaults = {
        "sensor_id": f"S-{uuid.uuid4().hex[:6]}",
        "sensor_name": "Test Sensor",
        "sensor_type": "GAS",
        "status": "NORMAL",
        "location_zone": "ZONE_A",
        "equipment_id": "EQ001",
        "manufacturer": "Dräger",
        "model": "POLYTRON 8700",
        "unit": "ppm",
        "min_value": 0.0,
        "max_value": 10000.0,
        "accuracy_rating": 0.99,
    }
    defaults.update(overrides)
    return SensorModel(**defaults)


@pytest_asyncio.fixture
async def repo(db_session: AsyncSession) -> SQLAlchemySensorRepository:
    """Provide a repository wired to the test session."""
    return SQLAlchemySensorRepository(db_session)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# create_sensor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_create_sensor_persists_and_returns_model(
    repo: SQLAlchemySensorRepository,
):
    sensor = _make_sensor(sensor_id="S001")
    result = await repo.create_sensor(sensor)

    assert result.sensor_id == "S001"
    assert result.id is not None  # UUID auto-generated
    assert result.created_at is not None
    assert result.status == "NORMAL"


async def test_create_sensor_auto_generates_uuid(
    repo: SQLAlchemySensorRepository,
):
    sensor = _make_sensor(sensor_id="S002")
    result = await repo.create_sensor(sensor)
    # Should be a valid UUID string
    uuid.UUID(result.id)  # Raises ValueError if invalid


async def test_create_sensor_duplicate_raises_integrity_error(
    repo: SQLAlchemySensorRepository,
):
    from sqlalchemy.exc import IntegrityError

    await repo.create_sensor(_make_sensor(sensor_id="S001"))
    with pytest.raises(IntegrityError):
        await repo.create_sensor(_make_sensor(sensor_id="S001"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# get_sensor_by_id
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_get_sensor_by_id_returns_sensor(
    repo: SQLAlchemySensorRepository,
):
    created = await repo.create_sensor(_make_sensor(sensor_id="S010"))
    fetched = await repo.get_sensor_by_id(uuid.UUID(created.id))

    assert fetched is not None
    assert fetched.sensor_id == "S010"


async def test_get_sensor_by_id_returns_none_for_unknown(
    repo: SQLAlchemySensorRepository,
):
    result = await repo.get_sensor_by_id(uuid.uuid4())
    assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# get_sensor_by_code
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_get_sensor_by_code_returns_sensor(
    repo: SQLAlchemySensorRepository,
):
    await repo.create_sensor(_make_sensor(sensor_id="S020"))
    fetched = await repo.get_sensor_by_code("S020")

    assert fetched is not None
    assert fetched.sensor_id == "S020"
    assert fetched.sensor_type == "GAS"


async def test_get_sensor_by_code_returns_none_for_unknown(
    repo: SQLAlchemySensorRepository,
):
    result = await repo.get_sensor_by_code("NONEXISTENT")
    assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# list_sensors
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_list_sensors_returns_all(repo: SQLAlchemySensorRepository):
    await repo.create_sensor(_make_sensor(sensor_id="L001"))
    await repo.create_sensor(_make_sensor(sensor_id="L002"))
    await repo.create_sensor(_make_sensor(sensor_id="L003"))

    result = await repo.list_sensors()
    assert len(result) == 3


async def test_list_sensors_filter_by_type(repo: SQLAlchemySensorRepository):
    await repo.create_sensor(_make_sensor(sensor_id="F001", sensor_type="GAS"))
    await repo.create_sensor(
        _make_sensor(sensor_id="F002", sensor_type="TEMPERATURE", unit="°C")
    )

    gas = await repo.list_sensors(sensor_type="GAS")
    assert len(gas) == 1
    assert gas[0].sensor_id == "F001"

    temp = await repo.list_sensors(sensor_type="TEMPERATURE")
    assert len(temp) == 1
    assert temp[0].sensor_id == "F002"


async def test_list_sensors_filter_by_status(repo: SQLAlchemySensorRepository):
    await repo.create_sensor(_make_sensor(sensor_id="FS01", status="NORMAL"))
    await repo.create_sensor(_make_sensor(sensor_id="FS02", status="WARNING"))

    result = await repo.list_sensors(status="WARNING")
    assert len(result) == 1
    assert result[0].sensor_id == "FS02"


async def test_list_sensors_filter_by_zone(repo: SQLAlchemySensorRepository):
    await repo.create_sensor(
        _make_sensor(sensor_id="FZ01", location_zone="ZONE_A")
    )
    await repo.create_sensor(
        _make_sensor(sensor_id="FZ02", location_zone="ZONE_B")
    )

    result = await repo.list_sensors(location_zone="ZONE_B")
    assert len(result) == 1
    assert result[0].sensor_id == "FZ02"


async def test_list_sensors_combined_filters(repo: SQLAlchemySensorRepository):
    await repo.create_sensor(
        _make_sensor(
            sensor_id="CF01",
            sensor_type="GAS",
            status="CRITICAL",
            location_zone="ZONE_A",
        )
    )
    await repo.create_sensor(
        _make_sensor(
            sensor_id="CF02",
            sensor_type="GAS",
            status="NORMAL",
            location_zone="ZONE_A",
        )
    )

    result = await repo.list_sensors(
        sensor_type="GAS", status="CRITICAL", location_zone="ZONE_A"
    )
    assert len(result) == 1
    assert result[0].sensor_id == "CF01"


async def test_list_sensors_pagination(repo: SQLAlchemySensorRepository):
    for i in range(5):
        await repo.create_sensor(_make_sensor(sensor_id=f"P{i:03d}"))

    page1 = await repo.list_sensors(offset=0, limit=2)
    page2 = await repo.list_sensors(offset=2, limit=2)
    page3 = await repo.list_sensors(offset=4, limit=2)

    assert len(page1) == 2
    assert len(page2) == 2
    assert len(page3) == 1

    # No overlap between pages
    ids = [s.sensor_id for s in page1 + page2 + page3]
    assert len(ids) == len(set(ids))


async def test_list_sensors_ordered_by_sensor_id(repo: SQLAlchemySensorRepository):
    await repo.create_sensor(_make_sensor(sensor_id="ZZZ"))
    await repo.create_sensor(_make_sensor(sensor_id="AAA"))
    await repo.create_sensor(_make_sensor(sensor_id="MMM"))

    result = await repo.list_sensors()
    ids = [s.sensor_id for s in result]
    assert ids == sorted(ids)


async def test_list_sensors_empty_returns_empty_list(
    repo: SQLAlchemySensorRepository,
):
    result = await repo.list_sensors()
    assert result == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# count
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_count_all(repo: SQLAlchemySensorRepository):
    await repo.create_sensor(_make_sensor(sensor_id="C001"))
    await repo.create_sensor(_make_sensor(sensor_id="C002"))

    assert await repo.count() == 2


async def test_count_with_filters(repo: SQLAlchemySensorRepository):
    await repo.create_sensor(_make_sensor(sensor_id="CF01", sensor_type="GAS"))
    await repo.create_sensor(
        _make_sensor(sensor_id="CF02", sensor_type="TEMPERATURE", unit="°C")
    )

    assert await repo.count(sensor_type="GAS") == 1
    assert await repo.count(sensor_type="TEMPERATURE") == 1
    assert await repo.count(sensor_type="PRESSURE") == 0


async def test_count_empty_table(repo: SQLAlchemySensorRepository):
    assert await repo.count() == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# count_by_status
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_count_by_status(repo: SQLAlchemySensorRepository):
    await repo.create_sensor(_make_sensor(sensor_id="CS01", status="NORMAL"))
    await repo.create_sensor(_make_sensor(sensor_id="CS02", status="NORMAL"))
    await repo.create_sensor(_make_sensor(sensor_id="CS03", status="WARNING"))
    await repo.create_sensor(_make_sensor(sensor_id="CS04", status="CRITICAL"))

    counts = await repo.count_by_status()
    assert counts["NORMAL"] == 2
    assert counts["WARNING"] == 1
    assert counts["CRITICAL"] == 1
    assert counts.get("OFFLINE", 0) == 0


async def test_count_by_status_empty_table(repo: SQLAlchemySensorRepository):
    counts = await repo.count_by_status()
    assert counts == {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# sensor_exists
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_sensor_exists_returns_true(repo: SQLAlchemySensorRepository):
    await repo.create_sensor(_make_sensor(sensor_id="EX01"))
    assert await repo.sensor_exists("EX01") is True


async def test_sensor_exists_returns_false(repo: SQLAlchemySensorRepository):
    assert await repo.sensor_exists("NONEXISTENT") is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# update_sensor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_update_sensor_persists_changes(repo: SQLAlchemySensorRepository):
    sensor = await repo.create_sensor(_make_sensor(sensor_id="U001"))
    assert sensor.status == "NORMAL"

    sensor.status = "WARNING"
    sensor.sensor_name = "Updated Name"
    updated = await repo.update_sensor(sensor)

    assert updated.status == "WARNING"
    assert updated.sensor_name == "Updated Name"

    # Verify via fresh fetch
    fetched = await repo.get_sensor_by_code("U001")
    assert fetched is not None
    assert fetched.status == "WARNING"
    assert fetched.sensor_name == "Updated Name"


async def test_update_sensor_changes_location_zone(
    repo: SQLAlchemySensorRepository,
):
    sensor = await repo.create_sensor(
        _make_sensor(sensor_id="U002", location_zone="ZONE_A")
    )
    sensor.location_zone = "ZONE_B"
    updated = await repo.update_sensor(sensor)
    assert updated.location_zone == "ZONE_B"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# delete_sensor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_delete_sensor_removes_from_db(repo: SQLAlchemySensorRepository):
    sensor = await repo.create_sensor(_make_sensor(sensor_id="D001"))
    await repo.delete_sensor(sensor)

    assert await repo.get_sensor_by_code("D001") is None
    assert await repo.sensor_exists("D001") is False


async def test_delete_sensor_does_not_affect_others(
    repo: SQLAlchemySensorRepository,
):
    s1 = await repo.create_sensor(_make_sensor(sensor_id="D010"))
    await repo.create_sensor(_make_sensor(sensor_id="D011"))

    await repo.delete_sensor(s1)

    assert await repo.count() == 1
    assert await repo.sensor_exists("D011") is True
