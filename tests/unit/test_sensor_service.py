"""Unit tests for SensorService business rules.

Tests use the real SQLAlchemy repository against an in-memory SQLite DB
(integration-style) to verify end-to-end business logic without mocking
the repository, which gives higher confidence.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.sensor_intelligence.models.reading_model import ReadingModel
from app.sensor_intelligence.models.sensor_model import SensorModel
from app.sensor_intelligence.repositories.sqlalchemy_sensor_repo import (
    SQLAlchemySensorRepository,
)
from app.sensor_intelligence.schemas.sensor_schemas import (
    SensorCreateRequest,
    SensorUpdateRequest,
)
from app.sensor_intelligence.services.sensor_service import SensorService
from app.shared.exceptions.domain_exceptions import (
    BusinessRuleViolationError,
    DuplicateResourceError,
    ResourceNotFoundError,
    ValidationError,
)


# ── Fixtures ──


@pytest_asyncio.fixture
async def service(db_session: AsyncSession) -> SensorService:
    """Provide a SensorService wired to a real repo + session."""
    repo = SQLAlchemySensorRepository(db_session)
    return SensorService(repo, db_session)


def _create_request(**overrides) -> SensorCreateRequest:
    """Build a valid SensorCreateRequest with sensible defaults."""
    defaults = {
        "sensor_id": f"S-{uuid.uuid4().hex[:6]}",
        "sensor_name": "Test Sensor",
        "sensor_type": "GAS",
        "location_zone": "ZONE_A",
        "equipment_id": "EQ001",
        "manufacturer": "Dräger",
        "model": "POLYTRON 8700",
        "unit": "ppm",
        "min_value": 0.0,
        "max_value": 10000.0,
        "accuracy_rating": 0.99,
        "installation_date": "2024-01-15",
        "last_calibration": "2025-06-01",
        "next_calibration_due": "2025-09-01",
    }
    defaults.update(overrides)
    return SensorCreateRequest(**defaults)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 1: sensor_id must be unique
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_create_sensor_success(service: SensorService):
    req = _create_request(sensor_id="S001")
    sensor = await service.create_sensor(req)

    assert sensor.sensor_id == "S001"
    assert sensor.status == "NORMAL"
    assert sensor.id is not None


async def test_create_sensor_duplicate_raises_error(service: SensorService):
    await service.create_sensor(_create_request(sensor_id="S001"))

    with pytest.raises(DuplicateResourceError) as exc_info:
        await service.create_sensor(_create_request(sensor_id="S001"))

    assert "S001" in str(exc_info.value.message)
    assert exc_info.value.error_code == "DUPLICATE_RESOURCE"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 2: calibration_date >= installation_date
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_create_sensor_calibration_before_install_raises_error(
    service: SensorService,
):
    with pytest.raises(ValidationError) as exc_info:
        await service.create_sensor(
            _create_request(
                sensor_id="CAL01",
                installation_date="2025-06-01",
                last_calibration="2024-01-01",  # before installation
            )
        )

    assert "calibration_date" in exc_info.value.message
    assert "installation_date" in exc_info.value.message


async def test_create_sensor_calibration_same_as_install_succeeds(
    service: SensorService,
):
    sensor = await service.create_sensor(
        _create_request(
            sensor_id="CAL02",
            installation_date="2025-01-15",
            last_calibration="2025-01-15",  # same day — valid
        )
    )
    assert sensor.sensor_id == "CAL02"


async def test_create_sensor_no_calibration_date_succeeds(service: SensorService):
    """When calibration_date is None, no validation is needed."""
    sensor = await service.create_sensor(
        _create_request(
            sensor_id="CAL03",
            installation_date="2025-01-15",
            last_calibration=None,
        )
    )
    assert sensor.sensor_id == "CAL03"


async def test_create_sensor_no_installation_date_succeeds(service: SensorService):
    """When installation_date is None, no validation is needed."""
    sensor = await service.create_sensor(
        _create_request(
            sensor_id="CAL04",
            installation_date=None,
            last_calibration="2025-06-01",
        )
    )
    assert sensor.sensor_id == "CAL04"


async def test_update_sensor_calibration_before_install_raises_error(
    service: SensorService,
):
    """Validation applies on update too (merged with existing values)."""
    await service.create_sensor(
        _create_request(
            sensor_id="UCAL01",
            installation_date="2025-06-01",
            last_calibration="2025-06-15",
        )
    )

    with pytest.raises(ValidationError) as exc_info:
        await service.update_sensor(
            "UCAL01",
            SensorUpdateRequest(last_calibration=date(2024, 1, 1)),
        )

    assert "calibration_date" in exc_info.value.message


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 3: min_value < max_value
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_create_sensor_min_gte_max_raises_error(service: SensorService):
    with pytest.raises(ValidationError) as exc_info:
        await service.create_sensor(
            _create_request(
                sensor_id="MM01",
                min_value=100.0,
                max_value=50.0,
            )
        )

    assert "min_value" in exc_info.value.message
    assert "max_value" in exc_info.value.message


async def test_create_sensor_min_equals_max_raises_error(service: SensorService):
    with pytest.raises(ValidationError):
        await service.create_sensor(
            _create_request(
                sensor_id="MM02",
                min_value=100.0,
                max_value=100.0,
            )
        )


async def test_create_sensor_only_min_provided_succeeds(service: SensorService):
    """When only one of min/max is provided, no cross-validation needed."""
    sensor = await service.create_sensor(
        _create_request(sensor_id="MM03", min_value=0.0, max_value=None)
    )
    assert sensor.sensor_id == "MM03"


async def test_create_sensor_neither_min_nor_max_succeeds(service: SensorService):
    sensor = await service.create_sensor(
        _create_request(sensor_id="MM04", min_value=None, max_value=None)
    )
    assert sensor.sensor_id == "MM04"


async def test_update_sensor_min_gte_max_raises_error(service: SensorService):
    """Validation uses merged values: existing max + new min."""
    await service.create_sensor(
        _create_request(sensor_id="UMM01", min_value=0.0, max_value=100.0)
    )

    with pytest.raises(ValidationError):
        await service.update_sensor(
            "UMM01",
            SensorUpdateRequest(min_value=200.0),  # 200 >= existing max 100
        )


async def test_update_sensor_valid_min_max_change_succeeds(service: SensorService):
    await service.create_sensor(
        _create_request(sensor_id="UMM02", min_value=0.0, max_value=100.0)
    )

    updated = await service.update_sensor(
        "UMM02",
        SensorUpdateRequest(min_value=10.0, max_value=200.0),
    )
    assert updated.min_value == 10.0
    assert updated.max_value == 200.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 4: status defaults to NORMAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_create_sensor_defaults_to_normal_status(service: SensorService):
    sensor = await service.create_sensor(_create_request(sensor_id="ST01"))
    assert sensor.status == "NORMAL"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 5: Cannot delete sensors with associated readings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_delete_sensor_without_readings_succeeds(service: SensorService):
    await service.create_sensor(_create_request(sensor_id="DEL01"))
    await service.delete_sensor("DEL01")

    with pytest.raises(ResourceNotFoundError):
        await service.get_sensor("DEL01")


async def test_delete_sensor_with_readings_raises_error(
    service: SensorService, db_session: AsyncSession
):
    """Insert a reading manually, then verify deletion is blocked."""
    sensor = await service.create_sensor(_create_request(sensor_id="DEL02"))

    # Manually insert a reading associated to this sensor
    from datetime import datetime

    reading = ReadingModel(
        id=str(uuid.uuid4()),
        sensor_id=sensor.id,  # FK to sensors.id (UUID PK)
        value=42.0,
        timestamp=datetime.utcnow(),
        confidence=0.95,
    )
    db_session.add(reading)
    await db_session.flush()

    with pytest.raises(BusinessRuleViolationError) as exc_info:
        await service.delete_sensor("DEL02")

    assert "associated readings" in exc_info.value.message
    assert exc_info.value.error_code == "INVALID_REQUEST"


async def test_delete_sensor_not_found_raises_error(service: SensorService):
    with pytest.raises(ResourceNotFoundError):
        await service.delete_sensor("NONEXISTENT")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# get_sensor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_get_sensor_success(service: SensorService):
    await service.create_sensor(_create_request(sensor_id="GET01"))
    sensor = await service.get_sensor("GET01")
    assert sensor.sensor_id == "GET01"


async def test_get_sensor_not_found(service: SensorService):
    with pytest.raises(ResourceNotFoundError):
        await service.get_sensor("NONEXISTENT")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# update_sensor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_update_sensor_partial_update(service: SensorService):
    await service.create_sensor(_create_request(sensor_id="UPD01"))

    updated = await service.update_sensor(
        "UPD01",
        SensorUpdateRequest(sensor_name="New Name", status="WARNING"),
    )
    assert updated.sensor_name == "New Name"
    assert updated.status == "WARNING"
    # Untouched fields should remain
    assert updated.manufacturer == "Dräger"


async def test_update_sensor_not_found(service: SensorService):
    with pytest.raises(ResourceNotFoundError):
        await service.update_sensor(
            "NONEXISTENT",
            SensorUpdateRequest(sensor_name="Name"),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# list_sensors
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_list_sensors_returns_items_and_count(service: SensorService):
    await service.create_sensor(_create_request(sensor_id="LS01"))
    await service.create_sensor(_create_request(sensor_id="LS02"))

    items, total = await service.list_sensors()
    assert total == 2
    assert len(items) == 2
