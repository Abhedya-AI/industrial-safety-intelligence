"""Comprehensive unit tests for the Alert Management System.

Covers:
  1.  SQLAlchemyAlertRepository — CRUD, dedup, resolve, summary
  2.  AlertService — threshold evaluation, anomaly scoring, multi-violation,
      duplicate prevention, auto-resolve, event publishing
  3.  AlertThresholdConfig — configurable thresholds
  4.  Edge cases — no violations, boundary values, unknown sensor types
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.sensor_intelligence.domain.entities.alert import Alert
from app.sensor_intelligence.domain.value_objects.alert_level import AlertLevel
from app.sensor_intelligence.models.sensor_model import SensorModel
from app.sensor_intelligence.repositories.alert_repository import AlertSummary
from app.sensor_intelligence.repositories.noop_publisher import NoOpPublisher
from app.sensor_intelligence.repositories.sqlalchemy_alert_repo import (
    SQLAlchemyAlertRepository,
)
from app.sensor_intelligence.repositories.sqlalchemy_sensor_repo import (
    SQLAlchemySensorRepository,
)
from app.sensor_intelligence.services.alert_service import (
    AlertService,
    AlertThresholdConfig,
    ReadingContext,
    SensorThreshold,
)


# ── Helpers ──


def _make_sensor(**overrides) -> SensorModel:
    defaults = {
        "sensor_id": f"S-{uuid.uuid4().hex[:6]}",
        "sensor_name": "Test Sensor",
        "sensor_type": "GAS",
        "status": "NORMAL",
        "location_zone": "ZONE_A",
        "equipment_id": "EQ-001",
        "unit": "ppm",
        "min_value": 0.0,
        "max_value": 10000.0,
    }
    defaults.update(overrides)
    return SensorModel(**defaults)


def _make_context(
    sensor_pk: str,
    sensor_code: str = "S001",
    sensor_type: str = "GAS",
    value: float = 10.0,
    anomaly_score: float = 0.0,
    anomaly_status: str = "NORMAL",
    **kwargs,
) -> ReadingContext:
    return ReadingContext(
        sensor_id=uuid.UUID(sensor_pk),
        sensor_code=sensor_code,
        sensor_type=sensor_type,
        value=value,
        anomaly_score=anomaly_score,
        anomaly_status=anomaly_status,
        equipment_id=kwargs.get("equipment_id", "EQ-001"),
        zone_id=kwargs.get("zone_id", "ZONE_A"),
    )


class RecordingPublisher(NoOpPublisher):
    """Captures published events for test assertions."""

    def __init__(self):
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        self.events.append((topic, event))


# ── Fixtures ──


@pytest_asyncio.fixture
async def sensor_repo(db_session: AsyncSession) -> SQLAlchemySensorRepository:
    return SQLAlchemySensorRepository(db_session)


@pytest_asyncio.fixture
async def alert_repo(db_session: AsyncSession) -> SQLAlchemyAlertRepository:
    return SQLAlchemyAlertRepository(db_session)


@pytest_asyncio.fixture
async def publisher() -> RecordingPublisher:
    return RecordingPublisher()


@pytest_asyncio.fixture
async def config() -> AlertThresholdConfig:
    return AlertThresholdConfig()


@pytest_asyncio.fixture
async def service(
    alert_repo: SQLAlchemyAlertRepository,
    publisher: RecordingPublisher,
    config: AlertThresholdConfig,
) -> AlertService:
    return AlertService(alert_repo, publisher, config)


@pytest_asyncio.fixture
async def registered_sensor(
    sensor_repo: SQLAlchemySensorRepository,
) -> SensorModel:
    return await sensor_repo.create_sensor(
        _make_sensor(sensor_id="S001", sensor_type="GAS")
    )


@pytest_asyncio.fixture
async def temp_sensor(
    sensor_repo: SQLAlchemySensorRepository,
) -> SensorModel:
    return await sensor_repo.create_sensor(
        _make_sensor(sensor_id="S-TEMP", sensor_type="TEMPERATURE", unit="°C")
    )


@pytest_asyncio.fixture
async def pressure_sensor(
    sensor_repo: SQLAlchemySensorRepository,
) -> SensorModel:
    return await sensor_repo.create_sensor(
        _make_sensor(sensor_id="S-PRESS", sensor_type="PRESSURE", unit="bar")
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AlertRepository CRUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_save_and_retrieve_alert(
    alert_repo: SQLAlchemyAlertRepository, registered_sensor: SensorModel
):
    alert = Alert(
        sensor_id=uuid.UUID(registered_sensor.id),
        level=AlertLevel.WARNING,
        title="TEST_ALERT",
        message="Test message",
    )
    saved = await alert_repo.save(alert)
    fetched = await alert_repo.get_by_id(saved.id)
    assert fetched is not None
    assert fetched.title == "TEST_ALERT"
    assert fetched.level == AlertLevel.WARNING


async def test_list_all_filters_by_level(
    alert_repo: SQLAlchemyAlertRepository, registered_sensor: SensorModel
):
    sid = uuid.UUID(registered_sensor.id)
    await alert_repo.save(Alert(sensor_id=sid, level=AlertLevel.WARNING, title="W1", message=""))
    await alert_repo.save(Alert(sensor_id=sid, level=AlertLevel.CRITICAL, title="C1", message=""))
    await alert_repo.save(Alert(sensor_id=sid, level=AlertLevel.WARNING, title="W2", message=""))

    warnings = await alert_repo.list_all(level=AlertLevel.WARNING)
    assert len(warnings) == 2
    criticals = await alert_repo.list_all(level=AlertLevel.CRITICAL)
    assert len(criticals) == 1


async def test_acknowledge_alert(
    alert_repo: SQLAlchemyAlertRepository, registered_sensor: SensorModel
):
    sid = uuid.UUID(registered_sensor.id)
    alert = await alert_repo.save(
        Alert(sensor_id=sid, level=AlertLevel.WARNING, title="ACK_TEST", message="")
    )
    assert not alert.is_acknowledged

    now = datetime.now(timezone.utc)
    acked = await alert_repo.acknowledge(alert.id, "operator1", now)
    assert acked.is_acknowledged is True
    assert acked.acknowledged_by == "operator1"


async def test_get_unacknowledged(
    alert_repo: SQLAlchemyAlertRepository, registered_sensor: SensorModel
):
    sid = uuid.UUID(registered_sensor.id)
    await alert_repo.save(Alert(sensor_id=sid, level=AlertLevel.WARNING, title="A1", message=""))
    a2 = await alert_repo.save(Alert(sensor_id=sid, level=AlertLevel.CRITICAL, title="A2", message=""))
    now = datetime.now(timezone.utc)
    await alert_repo.acknowledge(a2.id, "op", now)

    unacked = await alert_repo.get_unacknowledged()
    assert len(unacked) == 1
    assert unacked[0].title == "A1"


async def test_get_summary(
    alert_repo: SQLAlchemyAlertRepository, registered_sensor: SensorModel
):
    sid = uuid.UUID(registered_sensor.id)
    await alert_repo.save(Alert(sensor_id=sid, level=AlertLevel.WARNING, title="W", message=""))
    await alert_repo.save(Alert(sensor_id=sid, level=AlertLevel.CRITICAL, title="C", message=""))
    await alert_repo.save(Alert(sensor_id=sid, level=AlertLevel.EMERGENCY, title="E", message=""))

    summary = await alert_repo.get_summary()
    assert summary.warning == 1
    assert summary.critical == 1
    assert summary.emergency == 1
    assert summary.total == 3
    assert summary.unacknowledged == 3


async def test_get_active_alert_for_sensor_dedup(
    alert_repo: SQLAlchemyAlertRepository, registered_sensor: SensorModel
):
    sid = uuid.UUID(registered_sensor.id)
    await alert_repo.save(
        Alert(sensor_id=sid, level=AlertLevel.WARNING, title="HIGH_GAS_CONCENTRATION", message="")
    )

    existing = await alert_repo.get_active_alert_for_sensor(sid, "HIGH_GAS_CONCENTRATION")
    assert existing is not None

    no_match = await alert_repo.get_active_alert_for_sensor(sid, "HIGH_TEMPERATURE")
    assert no_match is None


async def test_resolve_alerts_for_sensor(
    alert_repo: SQLAlchemyAlertRepository, registered_sensor: SensorModel
):
    sid = uuid.UUID(registered_sensor.id)
    await alert_repo.save(Alert(sensor_id=sid, level=AlertLevel.WARNING, title="A1", message=""))
    await alert_repo.save(Alert(sensor_id=sid, level=AlertLevel.CRITICAL, title="A2", message=""))

    now = datetime.now(timezone.utc)
    resolved = await alert_repo.resolve_alerts_for_sensor(sid, now)
    assert resolved == 2

    unacked = await alert_repo.get_unacknowledged()
    assert len(unacked) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AlertService — no violations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_normal_reading_no_alerts(
    service: AlertService, registered_sensor: SensorModel
):
    ctx = _make_context(registered_sensor.id, value=10.0, anomaly_score=0.1)
    alerts = await service.evaluate_reading(ctx)
    assert len(alerts) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AlertService — anomaly score alerts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_anomaly_warning_alert(
    service: AlertService, registered_sensor: SensorModel
):
    ctx = _make_context(registered_sensor.id, value=10.0, anomaly_score=0.65)
    alerts = await service.evaluate_reading(ctx)
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.WARNING
    assert alerts[0].title == "ANOMALY_SCORE"


async def test_anomaly_critical_alert(
    service: AlertService, registered_sensor: SensorModel
):
    ctx = _make_context(registered_sensor.id, value=10.0, anomaly_score=0.85)
    alerts = await service.evaluate_reading(ctx)
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.CRITICAL


async def test_anomaly_emergency_alert(
    service: AlertService, registered_sensor: SensorModel
):
    ctx = _make_context(registered_sensor.id, value=10.0, anomaly_score=0.96)
    alerts = await service.evaluate_reading(ctx)
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.EMERGENCY


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AlertService — value threshold alerts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_gas_warning_threshold(
    service: AlertService, registered_sensor: SensorModel
):
    ctx = _make_context(registered_sensor.id, sensor_type="GAS", value=60.0)
    alerts = await service.evaluate_reading(ctx)
    assert len(alerts) == 1
    assert alerts[0].title == "HIGH_GAS_CONCENTRATION"
    assert alerts[0].level == AlertLevel.WARNING


async def test_gas_critical_threshold(
    service: AlertService, registered_sensor: SensorModel
):
    ctx = _make_context(registered_sensor.id, sensor_type="GAS", value=150.0)
    alerts = await service.evaluate_reading(ctx)
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.CRITICAL


async def test_gas_emergency_threshold(
    service: AlertService, registered_sensor: SensorModel
):
    ctx = _make_context(registered_sensor.id, sensor_type="GAS", value=600.0)
    alerts = await service.evaluate_reading(ctx)
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.EMERGENCY


async def test_temperature_warning(
    service: AlertService, temp_sensor: SensorModel
):
    ctx = _make_context(
        temp_sensor.id, sensor_code="S-TEMP",
        sensor_type="TEMPERATURE", value=90.0,
    )
    alerts = await service.evaluate_reading(ctx)
    assert len(alerts) == 1
    assert alerts[0].title == "HIGH_TEMPERATURE"
    assert alerts[0].level == AlertLevel.WARNING


async def test_pressure_critical(
    service: AlertService, pressure_sensor: SensorModel
):
    ctx = _make_context(
        pressure_sensor.id, sensor_code="S-PRESS",
        sensor_type="PRESSURE", value=15.0,
    )
    alerts = await service.evaluate_reading(ctx)
    assert len(alerts) == 1
    assert alerts[0].title == "HIGH_PRESSURE"
    assert alerts[0].level == AlertLevel.CRITICAL


async def test_unknown_sensor_type_no_value_alert(
    service: AlertService, registered_sensor: SensorModel
):
    ctx = _make_context(registered_sensor.id, sensor_type="UNKNOWN_TYPE", value=999.0)
    alerts = await service.evaluate_reading(ctx)
    assert len(alerts) == 0  # No threshold configured


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AlertService — multi-violation escalation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_multi_violation_creates_escalated_alert(
    service: AlertService, registered_sensor: SensorModel
):
    """Both anomaly and value thresholds breached → multi-violation alert."""
    ctx = _make_context(
        registered_sensor.id,
        sensor_type="GAS",
        value=150.0,       # Critical gas
        anomaly_score=0.85, # Critical anomaly
    )
    alerts = await service.evaluate_reading(ctx)
    titles = [a.title for a in alerts]
    assert "ANOMALY_SCORE" in titles
    assert "HIGH_GAS_CONCENTRATION" in titles
    assert "MULTI_VIOLATION" in titles

    # Multi-violation should be escalated to EMERGENCY (one level above CRITICAL)
    multi = next(a for a in alerts if a.title == "MULTI_VIOLATION")
    assert multi.level == AlertLevel.EMERGENCY


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AlertService — duplicate prevention
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_duplicate_alert_suppressed(
    service: AlertService, registered_sensor: SensorModel
):
    ctx = _make_context(registered_sensor.id, sensor_type="GAS", value=60.0)
    alerts1 = await service.evaluate_reading(ctx)
    assert len(alerts1) == 1

    # Same condition again — should be suppressed
    alerts2 = await service.evaluate_reading(ctx)
    assert len(alerts2) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AlertService — auto-resolve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_auto_resolve_on_normal_reading(
    service: AlertService,
    alert_repo: SQLAlchemyAlertRepository,
    registered_sensor: SensorModel,
):
    # First: trigger an alert
    ctx_high = _make_context(registered_sensor.id, sensor_type="GAS", value=60.0)
    await service.evaluate_reading(ctx_high)
    unacked = await alert_repo.get_unacknowledged()
    assert len(unacked) == 1

    # Then: send a normal reading → auto-resolve
    ctx_normal = _make_context(registered_sensor.id, sensor_type="GAS", value=5.0)
    await service.evaluate_reading(ctx_normal)
    unacked = await alert_repo.get_unacknowledged()
    assert len(unacked) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AlertService — event publishing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_alert_publishes_event(
    service: AlertService,
    publisher: RecordingPublisher,
    registered_sensor: SensorModel,
):
    ctx = _make_context(registered_sensor.id, sensor_type="GAS", value=60.0)
    await service.evaluate_reading(ctx)

    assert len(publisher.events) == 1
    topic, event = publisher.events[0]
    assert topic == "alerts"
    assert event["sensor_id"] == "S001"
    assert event["alert_type"] == "HIGH_GAS_CONCENTRATION"
    assert event["severity"] == "WARNING"
    assert "zone_id" in event
    assert "equipment_id" in event


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AlertService — management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_acknowledge_alert_via_service(
    service: AlertService, registered_sensor: SensorModel
):
    ctx = _make_context(registered_sensor.id, sensor_type="GAS", value=60.0)
    alerts = await service.evaluate_reading(ctx)
    alert = alerts[0]

    acked = await service.acknowledge_alert(alert.id, "operator1")
    assert acked is not None
    assert acked.is_acknowledged is True
    assert acked.acknowledged_by == "operator1"


async def test_acknowledge_nonexistent_alert(service: AlertService):
    result = await service.acknowledge_alert(uuid.uuid4(), "operator1")
    assert result is None


async def test_get_active_alerts(
    service: AlertService, registered_sensor: SensorModel
):
    ctx = _make_context(registered_sensor.id, sensor_type="GAS", value=60.0)
    await service.evaluate_reading(ctx)

    active = await service.get_active_alerts()
    assert len(active) == 1


async def test_get_alert_summary(
    service: AlertService, registered_sensor: SensorModel
):
    ctx_warn = _make_context(registered_sensor.id, sensor_type="GAS", value=60.0)
    await service.evaluate_reading(ctx_warn)

    summary = await service.get_alert_summary()
    assert summary.warning >= 1
    assert summary.total >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AlertThresholdConfig — custom thresholds
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_custom_thresholds(
    alert_repo: SQLAlchemyAlertRepository,
    publisher: RecordingPublisher,
    registered_sensor: SensorModel,
):
    """Custom low thresholds should trigger alerts at lower values."""
    custom_config = AlertThresholdConfig(
        gas=SensorThreshold(warning=5.0, critical=10.0, emergency=20.0),
    )
    custom_service = AlertService(alert_repo, publisher, custom_config)

    ctx = _make_context(registered_sensor.id, sensor_type="GAS", value=6.0)
    alerts = await custom_service.evaluate_reading(ctx)
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.WARNING


async def test_boundary_value_at_threshold(
    service: AlertService, registered_sensor: SensorModel
):
    """Value exactly at warning threshold should trigger."""
    cfg = service._config
    ctx = _make_context(
        registered_sensor.id,
        sensor_type="GAS",
        value=cfg.gas.warning,  # Exactly at 50.0
    )
    alerts = await service.evaluate_reading(ctx)
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.WARNING


async def test_below_all_thresholds_no_alert(
    service: AlertService, registered_sensor: SensorModel
):
    ctx = _make_context(
        registered_sensor.id,
        sensor_type="GAS",
        value=49.9,
        anomaly_score=0.59,
    )
    alerts = await service.evaluate_reading(ctx)
    assert len(alerts) == 0
