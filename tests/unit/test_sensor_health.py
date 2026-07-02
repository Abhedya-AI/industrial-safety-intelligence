"""Comprehensive unit tests for the Sensor Health Monitoring module.

Covers:
  1. Pure scoring functions (health_scoring.py)
     - Calibration score
     - Anomaly score
     - Uptime score
     - Stability score
     - Missing readings score
     - Composite health score
     - Health classification

  2. SensorHealthService (integration with DB)
     - calculate_health (end-to-end)
     - update_sensor_health (create + update)
     - get_sensor_health
     - update_all_sensors
     - Auto-update after ingestion
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

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
from app.sensor_intelligence.services.health_scoring import (
    HealthFactors,
    HealthScoreResult,
    HealthStatus,
    HealthWeights,
    calculate_anomaly_score,
    calculate_calibration_score,
    calculate_health_score,
    calculate_missing_readings_score,
    calculate_stability_score,
    calculate_uptime_score,
    classify_health,
)
from app.sensor_intelligence.services.reading_service import ReadingService
from app.sensor_intelligence.services.sensor_health_service import (
    SensorHealthRepository,
    SensorHealthService,
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
        "max_value": 10000.0,
        "installation_date": date(2025, 1, 1),
        "last_calibration": date(2026, 6, 1),
        "next_calibration_due": date(2026, 12, 1),
    }
    defaults.update(overrides)
    return SensorModel(**defaults)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Pure scoring: classify_health
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_classify_excellent():
    assert classify_health(95) == HealthStatus.EXCELLENT
    assert classify_health(90) == HealthStatus.EXCELLENT


def test_classify_good():
    assert classify_health(89) == HealthStatus.GOOD
    assert classify_health(70) == HealthStatus.GOOD


def test_classify_fair():
    assert classify_health(69) == HealthStatus.FAIR
    assert classify_health(50) == HealthStatus.FAIR


def test_classify_poor():
    assert classify_health(49) == HealthStatus.POOR
    assert classify_health(30) == HealthStatus.POOR


def test_classify_critical():
    assert classify_health(29) == HealthStatus.CRITICAL
    assert classify_health(0) == HealthStatus.CRITICAL


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Pure scoring: calibration_score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_calibration_no_data():
    f = HealthFactors()
    assert calculate_calibration_score(f) == 50.0


def test_calibration_within_window():
    f = HealthFactors(
        last_calibration=date(2026, 6, 1),
        next_calibration_due=date(2026, 12, 1),
        today=date(2026, 7, 1),
    )
    assert calculate_calibration_score(f) == 100.0


def test_calibration_overdue():
    f = HealthFactors(
        last_calibration=date(2026, 1, 1),
        next_calibration_due=date(2026, 6, 1),
        today=date(2026, 6, 11),  # 10 days overdue → 100 - 20 = 80
    )
    assert calculate_calibration_score(f) == 80.0


def test_calibration_very_overdue():
    f = HealthFactors(
        last_calibration=date(2025, 1, 1),
        next_calibration_due=date(2025, 6, 1),
        today=date(2026, 7, 1),  # 395 days overdue → capped at 0
    )
    assert calculate_calibration_score(f) == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Pure scoring: anomaly_score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_anomaly_no_readings():
    f = HealthFactors(total_readings=0, anomaly_count=0)
    assert calculate_anomaly_score(f) == 100.0


def test_anomaly_zero_rate():
    f = HealthFactors(total_readings=100, anomaly_count=0)
    assert calculate_anomaly_score(f) == 100.0


def test_anomaly_5_percent():
    f = HealthFactors(total_readings=100, anomaly_count=5)
    assert calculate_anomaly_score(f) == 75.0


def test_anomaly_10_percent():
    f = HealthFactors(total_readings=100, anomaly_count=10)
    assert calculate_anomaly_score(f) == 50.0


def test_anomaly_20_percent_or_more():
    f = HealthFactors(total_readings=100, anomaly_count=20)
    assert calculate_anomaly_score(f) == 0.0
    f2 = HealthFactors(total_readings=100, anomaly_count=30)
    assert calculate_anomaly_score(f2) == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Pure scoring: uptime_score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_uptime_normal():
    f = HealthFactors(sensor_status="NORMAL", installation_date=date(2025, 1, 1))
    assert calculate_uptime_score(f) == 100.0


def test_uptime_warning():
    f = HealthFactors(sensor_status="WARNING", installation_date=date(2025, 1, 1))
    assert calculate_uptime_score(f) == 100.0


def test_uptime_critical():
    f = HealthFactors(sensor_status="CRITICAL")
    assert calculate_uptime_score(f) == 50.0


def test_uptime_offline():
    f = HealthFactors(sensor_status="OFFLINE")
    assert calculate_uptime_score(f) == 0.0


def test_uptime_no_installation_date():
    f = HealthFactors(sensor_status="NORMAL", installation_date=None)
    assert calculate_uptime_score(f) == 80.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Pure scoring: stability_score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_stability_no_data():
    f = HealthFactors(reading_mean=0.0, reading_std_dev=0.0)
    assert calculate_stability_score(f) == 100.0


def test_stability_very_stable():
    f = HealthFactors(reading_mean=100.0, reading_std_dev=3.0, total_readings=50)
    # CV = 0.03 < 0.05
    assert calculate_stability_score(f) == 100.0


def test_stability_moderate():
    f = HealthFactors(reading_mean=100.0, reading_std_dev=20.0, total_readings=50)
    # CV = 0.2, between 0.05 and 0.5 → interpolated
    score = calculate_stability_score(f)
    assert 50 < score < 100


def test_stability_high_variance():
    f = HealthFactors(reading_mean=100.0, reading_std_dev=80.0, total_readings=50)
    # CV = 0.8 > 0.5 → steep decay
    score = calculate_stability_score(f)
    assert score < 50


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Pure scoring: missing_readings_score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_missing_no_expected():
    f = HealthFactors(expected_readings=0, actual_readings=0)
    assert calculate_missing_readings_score(f) == 100.0


def test_missing_full_coverage():
    f = HealthFactors(expected_readings=100, actual_readings=100)
    assert calculate_missing_readings_score(f) == 100.0


def test_missing_90_percent():
    f = HealthFactors(expected_readings=100, actual_readings=90)
    score = calculate_missing_readings_score(f)
    assert score == 80.0


def test_missing_50_percent():
    f = HealthFactors(expected_readings=100, actual_readings=50)
    assert calculate_missing_readings_score(f) == 0.0


def test_missing_below_50_percent():
    f = HealthFactors(expected_readings=100, actual_readings=30)
    assert calculate_missing_readings_score(f) == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Pure scoring: composite health_score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_composite_perfect_health():
    f = HealthFactors(
        last_calibration=date(2026, 6, 1),
        next_calibration_due=date(2026, 12, 1),
        today=date(2026, 7, 1),
        total_readings=100,
        anomaly_count=0,
        installation_date=date(2025, 1, 1),
        sensor_status="NORMAL",
        reading_std_dev=2.0,
        reading_mean=100.0,
        expected_readings=100,
        actual_readings=100,
    )
    result = calculate_health_score(f)
    assert result.health_score == 100.0
    assert result.health_status == HealthStatus.EXCELLENT


def test_composite_poor_health():
    f = HealthFactors(
        last_calibration=date(2025, 1, 1),
        next_calibration_due=date(2025, 6, 1),
        today=date(2026, 7, 1),
        total_readings=100,
        anomaly_count=15,
        installation_date=date(2020, 1, 1),
        sensor_status="CRITICAL",
        reading_std_dev=90.0,
        reading_mean=100.0,
        expected_readings=100,
        actual_readings=60,
    )
    result = calculate_health_score(f)
    assert result.health_score < 50
    assert result.health_status in (HealthStatus.POOR, HealthStatus.CRITICAL)


def test_composite_custom_weights():
    f = HealthFactors(
        total_readings=100,
        anomaly_count=10,
        sensor_status="NORMAL",
        installation_date=date(2025, 1, 1),
        expected_readings=100,
        actual_readings=100,
        today=date(2026, 7, 1),
    )
    # Give anomaly 100% weight
    w = HealthWeights(calibration=0, anomaly=1.0, uptime=0, stability=0, missing_readings=0)
    result = calculate_health_score(f, w)
    assert result.health_score == 50.0  # 10% anomaly rate


def test_composite_result_has_details():
    f = HealthFactors(total_readings=200, anomaly_count=10, sensor_status="NORMAL")
    result = calculate_health_score(f)
    assert "total_readings" in result.details
    assert result.details["total_readings"] == 200
    assert result.details["anomaly_count"] == 10
    assert "anomaly_rate" in result.details


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. SensorHealthService (integration)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest_asyncio.fixture
async def sensor_repo(db_session: AsyncSession) -> SQLAlchemySensorRepository:
    return SQLAlchemySensorRepository(db_session)


@pytest_asyncio.fixture
async def reading_repo(db_session: AsyncSession) -> SQLAlchemyReadingRepository:
    return SQLAlchemyReadingRepository(db_session)


@pytest_asyncio.fixture
async def health_repo(db_session: AsyncSession) -> SensorHealthRepository:
    return SensorHealthRepository(db_session)


@pytest_asyncio.fixture
async def health_service(
    health_repo: SensorHealthRepository,
    sensor_repo: SQLAlchemySensorRepository,
    reading_repo: SQLAlchemyReadingRepository,
) -> SensorHealthService:
    return SensorHealthService(health_repo, sensor_repo, reading_repo)


@pytest_asyncio.fixture
async def reading_service(
    reading_repo: SQLAlchemyReadingRepository,
    sensor_repo: SQLAlchemySensorRepository,
) -> ReadingService:
    return ReadingService(reading_repo, sensor_repo)


@pytest_asyncio.fixture
async def healthy_sensor(
    sensor_repo: SQLAlchemySensorRepository,
) -> SensorModel:
    return await sensor_repo.create_sensor(
        _make_sensor(
            sensor_id="S-HEALTHY",
            status="NORMAL",
            installation_date=date(2025, 1, 1),
            last_calibration=date(2026, 6, 1),
            next_calibration_due=date(2026, 12, 1),
        )
    )


@pytest_asyncio.fixture
async def degraded_sensor(
    sensor_repo: SQLAlchemySensorRepository,
) -> SensorModel:
    return await sensor_repo.create_sensor(
        _make_sensor(
            sensor_id="S-DEGRADED",
            status="CRITICAL",
            installation_date=None,
            last_calibration=None,
            next_calibration_due=None,
        )
    )


async def test_calculate_health_healthy_sensor(
    health_service: SensorHealthService,
    healthy_sensor: SensorModel,
):
    result = await health_service.calculate_health(
        healthy_sensor, reference_date=date(2026, 7, 1)
    )
    assert isinstance(result, HealthScoreResult)
    assert result.health_score >= 50  # Healthy sensor, but no readings → penalised on missing
    assert result.calibration_score == 100.0
    assert result.uptime_score == 100.0


async def test_calculate_health_degraded_sensor(
    health_service: SensorHealthService,
    degraded_sensor: SensorModel,
):
    result = await health_service.calculate_health(degraded_sensor)
    assert result.uptime_score == 50.0  # CRITICAL status
    assert result.calibration_score == 50.0  # No calibration data


async def test_update_sensor_health_creates_record(
    health_service: SensorHealthService,
    health_repo: SensorHealthRepository,
    healthy_sensor: SensorModel,
):
    health = await health_service.update_sensor_health(healthy_sensor)
    assert health.sensor_id == healthy_sensor.id
    assert health.health_score > 0

    # Verify persisted
    fetched = await health_repo.get_by_sensor_id(healthy_sensor.id)
    assert fetched is not None
    assert fetched.health_score == health.health_score


async def test_update_sensor_health_updates_existing(
    health_service: SensorHealthService,
    health_repo: SensorHealthRepository,
    healthy_sensor: SensorModel,
):
    h1 = await health_service.update_sensor_health(healthy_sensor)
    h2 = await health_service.update_sensor_health(healthy_sensor)
    assert h1.id == h2.id  # Same record, not a duplicate


async def test_get_sensor_health(
    health_service: SensorHealthService,
    healthy_sensor: SensorModel,
):
    await health_service.update_sensor_health(healthy_sensor)
    result = await health_service.get_sensor_health("S-HEALTHY")
    assert result is not None
    assert result.health_status in ("EXCELLENT", "GOOD", "FAIR", "POOR", "CRITICAL")


async def test_get_sensor_health_nonexistent(
    health_service: SensorHealthService,
):
    result = await health_service.get_sensor_health("NONEXISTENT")
    assert result is None


async def test_update_all_sensors(
    health_service: SensorHealthService,
    healthy_sensor: SensorModel,
    degraded_sensor: SensorModel,
):
    results = await health_service.update_all_sensors()
    assert len(results) == 2


async def test_health_with_readings(
    health_service: SensorHealthService,
    reading_service: ReadingService,
    healthy_sensor: SensorModel,
):
    """Health score should improve with actual readings."""
    base = datetime.now(timezone.utc) - timedelta(hours=5)
    for i in range(10):
        await reading_service.ingest_reading(
            ReadingCreateRequest(
                sensor_id="S-HEALTHY",
                value=42.0 + i * 0.1,
                timestamp=(base + timedelta(hours=i * 0.5)).isoformat(),
                confidence=95.0,
            )
        )

    result = await health_service.calculate_health(healthy_sensor)
    # Should have higher missing_readings_score than without readings
    assert result.details["total_readings"] == 10
    assert result.anomaly_score == 100.0  # No anomalies
