"""End-to-end integration tests for the complete Sensor Intelligence pipeline.

Verifies the full data flow:

  Sensor Registration → Reading Ingestion → Validation → Repository →
  Statistics → Anomaly Detection → Alert Generation →
  Sensor Health Update → Baseline Learning → Database Persistence →
  API Response

Scenarios tested:
  1.  Normal readings — full happy-path pipeline
  2.  Anomalous readings — anomaly scoring and alert generation
  3.  Invalid readings — validation rejections
  4.  Duplicate readings — dedup enforcement
  5.  Sensor offline — ingestion blocked
  6.  Model loading failures — graceful degradation
  7.  Batch ingestion — multi-reading pipeline
  8.  Statistics and history — query pipeline
  9.  Sensor health — health scoring after ingestion
  10. Baseline learning — baseline computation after data
  11. Alert lifecycle — create → acknowledge → auto-resolve
  12. Model monitoring — inference tracking
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.sensor_intelligence.anomaly_detection.base import BaseAnomalyDetector
from app.sensor_intelligence.anomaly_detection.schemas import (
    AnomalyResult,
    AnomalyStatus,
)
from app.sensor_intelligence.models.reading_model import ReadingModel
from app.sensor_intelligence.models.sensor_model import SensorModel
from app.sensor_intelligence.repositories.sqlalchemy_alert_repo import (
    SQLAlchemyAlertRepository,
)
from app.sensor_intelligence.repositories.sqlalchemy_reading_repo import (
    SQLAlchemyReadingRepository,
)
from app.sensor_intelligence.repositories.sqlalchemy_sensor_repo import (
    SQLAlchemySensorRepository,
)
from app.sensor_intelligence.repositories.noop_publisher import NoOpPublisher
from app.sensor_intelligence.schemas.reading_schemas import (
    ReadingCreateRequest,
    BatchReadingCreateRequest,
)
from app.sensor_intelligence.services.alert_service import (
    AlertService,
    AlertThresholdConfig,
    ReadingContext,
)
from app.sensor_intelligence.services.baseline_service import (
    BaselineLearningService,
    BaselineRepository,
)
from app.sensor_intelligence.services.model_monitoring_service import (
    ModelMonitoringService,
    ModelMetadata,
)
from app.sensor_intelligence.services.reading_service import ReadingService
from app.sensor_intelligence.services.sensor_health_service import (
    SensorHealthRepository,
    SensorHealthService,
)
from app.sensor_intelligence.services.statistics_service import StatisticsService
from app.shared.exceptions.domain_exceptions import (
    BusinessRuleViolationError,
    InvalidReadingError,
    ResourceNotFoundError,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stub detectors for testing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class StubNormalDetector(BaseAnomalyDetector):
    """Detector that always returns NORMAL."""

    @property
    def name(self) -> str:
        return "stub_normal"

    @property
    def is_loaded(self) -> bool:
        return True

    def load_model(self, **kwargs) -> None:
        pass

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.ones(features.shape[0])

    def anomaly_score(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], 0.1)

    def classify(self, features: np.ndarray, sensor_ids: Sequence[str]) -> list[AnomalyResult]:
        return [
            AnomalyResult(
                sensor_id=sid, score=0.1, status=AnomalyStatus.NORMAL,
                detector_type=self.name,
            )
            for sid in sensor_ids
        ]


class StubAnomalyDetector(BaseAnomalyDetector):
    """Detector that always returns ANOMALY with a high score."""

    @property
    def name(self) -> str:
        return "stub_anomaly"

    @property
    def is_loaded(self) -> bool:
        return True

    def load_model(self, **kwargs) -> None:
        pass

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], -1)

    def anomaly_score(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], 0.92)

    def classify(self, features: np.ndarray, sensor_ids: Sequence[str]) -> list[AnomalyResult]:
        return [
            AnomalyResult(
                sensor_id=sid, score=0.92, status=AnomalyStatus.ANOMALY,
                detector_type=self.name, confidence=0.92,
            )
            for sid in sensor_ids
        ]


class StubFailingDetector(BaseAnomalyDetector):
    """Detector that raises on every call (simulates loading failure)."""

    @property
    def name(self) -> str:
        return "stub_failing"

    @property
    def is_loaded(self) -> bool:
        return False

    def load_model(self, **kwargs) -> None:
        raise FileNotFoundError("Model file not found: models/broken.pkl")

    def predict(self, features: np.ndarray) -> np.ndarray:
        raise RuntimeError("Model not loaded")

    def anomaly_score(self, features: np.ndarray) -> np.ndarray:
        raise RuntimeError("Model not loaded")

    def classify(self, features: np.ndarray, sensor_ids: Sequence[str]) -> list[AnomalyResult]:
        raise RuntimeError("Model not loaded")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _sensor_model(**overrides) -> SensorModel:
    defaults = {
        "sensor_id": f"S-{uuid.uuid4().hex[:6]}",
        "sensor_name": "Integration Test Sensor",
        "sensor_type": "TEMPERATURE",
        "status": "NORMAL",
        "location_zone": "ZONE_A",
        "equipment_id": "EQ-001",
        "unit": "°C",
        "min_value": -50.0,
        "max_value": 500.0,
        "installation_date": date(2025, 1, 1),
        "last_calibration": date(2026, 6, 1),
        "next_calibration_due": date(2026, 12, 1),
    }
    defaults.update(overrides)
    return SensorModel(**defaults)


def _reading_request(
    sensor_id: str, value: float, offset_hours: float = 1.0
) -> ReadingCreateRequest:
    ts = datetime.now(timezone.utc) - timedelta(hours=offset_hours)
    return ReadingCreateRequest(
        sensor_id=sensor_id,
        value=value,
        timestamp=ts.isoformat(),
        confidence=95.0,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest_asyncio.fixture
async def sensor_repo(db_session: AsyncSession) -> SQLAlchemySensorRepository:
    return SQLAlchemySensorRepository(db_session)


@pytest_asyncio.fixture
async def reading_repo(db_session: AsyncSession) -> SQLAlchemyReadingRepository:
    return SQLAlchemyReadingRepository(db_session)


@pytest_asyncio.fixture
async def alert_repo(db_session: AsyncSession) -> SQLAlchemyAlertRepository:
    return SQLAlchemyAlertRepository(db_session)


@pytest_asyncio.fixture
async def health_repo(db_session: AsyncSession) -> SensorHealthRepository:
    return SensorHealthRepository(db_session)


@pytest_asyncio.fixture
async def baseline_repo(db_session: AsyncSession) -> BaselineRepository:
    return BaselineRepository(db_session)


@pytest_asyncio.fixture
async def publisher() -> NoOpPublisher:
    return NoOpPublisher()


@pytest_asyncio.fixture
async def normal_detector() -> StubNormalDetector:
    return StubNormalDetector()


@pytest_asyncio.fixture
async def anomaly_detector() -> StubAnomalyDetector:
    return StubAnomalyDetector()


@pytest_asyncio.fixture
async def reading_service_normal(
    reading_repo, sensor_repo, normal_detector
) -> ReadingService:
    return ReadingService(reading_repo, sensor_repo, anomaly_detector=normal_detector)


@pytest_asyncio.fixture
async def reading_service_anomaly(
    reading_repo, sensor_repo, anomaly_detector
) -> ReadingService:
    return ReadingService(reading_repo, sensor_repo, anomaly_detector=anomaly_detector)


@pytest_asyncio.fixture
async def reading_service_no_detector(reading_repo, sensor_repo) -> ReadingService:
    return ReadingService(reading_repo, sensor_repo)


@pytest_asyncio.fixture
async def alert_service(alert_repo, publisher) -> AlertService:
    return AlertService(alert_repo, publisher)


@pytest_asyncio.fixture
async def health_service(health_repo, sensor_repo, reading_repo) -> SensorHealthService:
    return SensorHealthService(health_repo, sensor_repo, reading_repo)


@pytest_asyncio.fixture
async def baseline_service(baseline_repo, sensor_repo, reading_repo) -> BaselineLearningService:
    return BaselineLearningService(
        baseline_repo, sensor_repo, reading_repo, min_samples=5
    )


@pytest_asyncio.fixture
async def stats_service(reading_repo, sensor_repo) -> StatisticsService:
    return StatisticsService(reading_repo, sensor_repo)


@pytest_asyncio.fixture
async def model_monitor() -> ModelMonitoringService:
    svc = ModelMonitoringService()
    yield svc
    svc.reset()


@pytest_asyncio.fixture
async def registered_sensor(sensor_repo) -> SensorModel:
    return await sensor_repo.create_sensor(
        _sensor_model(sensor_id="S-INT-001")
    )


@pytest_asyncio.fixture
async def gas_sensor(sensor_repo) -> SensorModel:
    return await sensor_repo.create_sensor(
        _sensor_model(
            sensor_id="S-GAS-001", sensor_type="GAS", unit="ppm",
            min_value=0.0, max_value=10000.0,
        )
    )


@pytest_asyncio.fixture
async def offline_sensor(sensor_repo) -> SensorModel:
    return await sensor_repo.create_sensor(
        _sensor_model(sensor_id="S-OFFLINE", status="OFFLINE")
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Normal readings — full pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_e2e_normal_reading_pipeline(
    reading_service_normal: ReadingService,
    reading_repo: SQLAlchemyReadingRepository,
    registered_sensor: SensorModel,
):
    """Full pipeline: register → ingest → validate → persist → score (NORMAL)."""
    req = _reading_request("S-INT-001", 42.5)
    reading = await reading_service_normal.ingest_reading(req)

    # Persisted correctly
    assert reading.id is not None
    assert reading.value == 42.5
    assert reading.sensor_id == registered_sensor.id

    # Anomaly scored (normal detector)
    assert reading.anomaly_score == 0.1
    assert reading.anomaly_status == "NORMAL"

    # Retrievable from DB
    fetched = await reading_repo.get_reading_by_id(reading.id)
    assert fetched is not None
    assert fetched.value == 42.5


async def test_e2e_normal_reading_no_alert_generated(
    reading_service_normal: ReadingService,
    alert_service: AlertService,
    registered_sensor: SensorModel,
):
    """Normal reading should NOT trigger alerts."""
    req = _reading_request("S-INT-001", 42.5)
    reading = await reading_service_normal.ingest_reading(req)

    ctx = ReadingContext(
        sensor_id=uuid.UUID(registered_sensor.id),
        sensor_code="S-INT-001",
        sensor_type="TEMPERATURE",
        value=reading.value,
        anomaly_score=reading.anomaly_score or 0.0,
        anomaly_status=reading.anomaly_status or "NORMAL",
    )
    alerts = await alert_service.evaluate_reading(ctx)
    assert len(alerts) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Anomalous readings — scoring + alerts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_e2e_anomalous_reading_pipeline(
    reading_service_anomaly: ReadingService,
    alert_service: AlertService,
    registered_sensor: SensorModel,
):
    """Anomalous reading → high score → alert generated."""
    req = _reading_request("S-INT-001", 42.5)
    reading = await reading_service_anomaly.ingest_reading(req)

    assert reading.anomaly_score == 0.92
    assert reading.anomaly_status == "ANOMALY"

    # Run alert evaluation (score 0.92 → EMERGENCY)
    ctx = ReadingContext(
        sensor_id=uuid.UUID(registered_sensor.id),
        sensor_code="S-INT-001",
        sensor_type="TEMPERATURE",
        value=reading.value,
        anomaly_score=reading.anomaly_score,
        anomaly_status=reading.anomaly_status,
    )
    alerts = await alert_service.evaluate_reading(ctx)
    assert len(alerts) >= 1
    assert any(a.title == "ANOMALY_SCORE" for a in alerts)


async def test_e2e_high_value_triggers_threshold_alert(
    reading_service_normal: ReadingService,
    alert_service: AlertService,
    registered_sensor: SensorModel,
):
    """High temp value → threshold alert even with normal anomaly score."""
    req = _reading_request("S-INT-001", 150.0)  # Above TEMPERATURE critical=120
    reading = await reading_service_normal.ingest_reading(req)

    ctx = ReadingContext(
        sensor_id=uuid.UUID(registered_sensor.id),
        sensor_code="S-INT-001",
        sensor_type="TEMPERATURE",
        value=reading.value,
        anomaly_score=reading.anomaly_score or 0.0,
    )
    alerts = await alert_service.evaluate_reading(ctx)
    assert len(alerts) >= 1
    assert any(a.title == "HIGH_TEMPERATURE" for a in alerts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Invalid readings — validation rejections
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_e2e_reject_nonexistent_sensor(
    reading_service_normal: ReadingService,
):
    """Reject reading for a sensor that doesn't exist."""
    req = _reading_request("NONEXISTENT-SENSOR", 42.5)
    with pytest.raises(ResourceNotFoundError):
        await reading_service_normal.ingest_reading(req)


async def test_e2e_reject_nan_value(
    reading_service_normal: ReadingService,
    registered_sensor: SensorModel,
):
    """Reject reading with NaN value."""
    req = ReadingCreateRequest(
        sensor_id="S-INT-001",
        value=float("nan"),
        timestamp=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        confidence=95.0,
    )
    with pytest.raises(InvalidReadingError):
        await reading_service_normal.ingest_reading(req)


async def test_e2e_reject_inf_value(
    reading_service_normal: ReadingService,
    registered_sensor: SensorModel,
):
    """Reject reading with infinity value."""
    req = ReadingCreateRequest(
        sensor_id="S-INT-001",
        value=float("inf"),
        timestamp=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        confidence=95.0,
    )
    with pytest.raises(InvalidReadingError):
        await reading_service_normal.ingest_reading(req)


async def test_e2e_reject_out_of_range(
    reading_service_normal: ReadingService,
    registered_sensor: SensorModel,
):
    """Reject reading outside sensor min/max range."""
    req = _reading_request("S-INT-001", 999.0)  # max_value = 500
    with pytest.raises(InvalidReadingError):
        await reading_service_normal.ingest_reading(req)


async def test_e2e_reject_below_range(
    reading_service_normal: ReadingService,
    registered_sensor: SensorModel,
):
    """Reject reading below sensor minimum."""
    req = _reading_request("S-INT-001", -100.0)  # min_value = -50
    with pytest.raises(InvalidReadingError):
        await reading_service_normal.ingest_reading(req)


async def test_e2e_reject_future_timestamp(
    reading_service_normal: ReadingService,
    registered_sensor: SensorModel,
):
    """Reject reading with a future timestamp (>5 min tolerance)."""
    future_ts = datetime.now(timezone.utc) + timedelta(hours=1)
    req = ReadingCreateRequest(
        sensor_id="S-INT-001",
        value=42.5,
        timestamp=future_ts.isoformat(),
        confidence=95.0,
    )
    with pytest.raises(InvalidReadingError, match="future"):
        await reading_service_normal.ingest_reading(req)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Duplicate readings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_e2e_reject_duplicate_reading(
    reading_service_normal: ReadingService,
    registered_sensor: SensorModel,
):
    """Reject second reading with same sensor + timestamp."""
    ts = datetime.now(timezone.utc) - timedelta(hours=2)
    req = ReadingCreateRequest(
        sensor_id="S-INT-001", value=42.5,
        timestamp=ts.isoformat(), confidence=95.0,
    )
    await reading_service_normal.ingest_reading(req)

    with pytest.raises(BusinessRuleViolationError, match="Duplicate"):
        await reading_service_normal.ingest_reading(req)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Sensor offline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_e2e_reject_offline_sensor(
    reading_service_normal: ReadingService,
    offline_sensor: SensorModel,
):
    """Reject ingestion for OFFLINE sensors."""
    req = _reading_request("S-OFFLINE", 42.5)
    with pytest.raises(BusinessRuleViolationError, match="OFFLINE"):
        await reading_service_normal.ingest_reading(req)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Model loading failures — graceful degradation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_e2e_model_failure_graceful_fallback(
    reading_repo, sensor_repo,
    registered_sensor: SensorModel,
):
    """When detector fails, reading still ingests with NORMAL status."""
    failing = StubFailingDetector()
    service = ReadingService(reading_repo, sensor_repo, anomaly_detector=failing)

    req = _reading_request("S-INT-001", 42.5)
    reading = await service.ingest_reading(req)

    # Reading still persisted, defaults to NORMAL
    assert reading.id is not None
    assert reading.anomaly_score == 0.0
    assert reading.anomaly_status == "NORMAL"


async def test_e2e_model_failure_tracked_by_monitor(
    model_monitor: ModelMonitoringService,
):
    """Loading failures are tracked in the monitoring service."""
    model_monitor.record_loading_failure("isolation_forest", "FileNotFoundError")
    stats = model_monitor.get_inference_stats("isolation_forest")
    assert stats.loading_failure_count == 1
    report = model_monitor.get_model_health("isolation_forest")
    assert report.checks["no_loading_failures"] is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Batch ingestion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_e2e_batch_ingestion(
    reading_service_normal: ReadingService,
    reading_repo: SQLAlchemyReadingRepository,
    registered_sensor: SensorModel,
):
    """Batch ingestion: all readings validated and persisted."""
    base = datetime.now(timezone.utc) - timedelta(hours=10)
    requests = [
        ReadingCreateRequest(
            sensor_id="S-INT-001", value=40.0 + i,
            timestamp=(base + timedelta(hours=i)).isoformat(),
            confidence=95.0,
        )
        for i in range(5)
    ]
    readings = await reading_service_normal.ingest_batch(requests)
    assert len(readings) == 5

    # All persisted
    count = await reading_repo.count_for_sensor(registered_sensor.id)
    assert count == 5

    # All scored
    for r in readings:
        assert r.anomaly_status == "NORMAL"


async def test_e2e_batch_one_invalid_rejects_all(
    reading_service_normal: ReadingService,
    reading_repo: SQLAlchemyReadingRepository,
    registered_sensor: SensorModel,
):
    """If one reading in batch is invalid, entire batch rejected."""
    base = datetime.now(timezone.utc) - timedelta(hours=10)
    requests = [
        ReadingCreateRequest(
            sensor_id="S-INT-001", value=42.0,
            timestamp=(base + timedelta(hours=1)).isoformat(),
            confidence=95.0,
        ),
        ReadingCreateRequest(
            sensor_id="S-INT-001", value=999.0,  # Out of range
            timestamp=(base + timedelta(hours=2)).isoformat(),
            confidence=95.0,
        ),
    ]
    with pytest.raises(InvalidReadingError):
        await reading_service_normal.ingest_batch(requests)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Statistics pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_e2e_statistics_after_ingestion(
    reading_service_normal: ReadingService,
    stats_service: StatisticsService,
    registered_sensor: SensorModel,
):
    """Ingest readings, then compute statistics."""
    base = datetime.now(timezone.utc) - timedelta(hours=5)
    for i in range(10):
        await reading_service_normal.ingest_reading(
            ReadingCreateRequest(
                sensor_id="S-INT-001", value=40.0 + i,
                timestamp=(base + timedelta(minutes=i * 30)).isoformat(),
                confidence=95.0,
            )
        )

    result = await stats_service.compute_statistics("S-INT-001", time_range="24h")
    assert result.reading_count == 10
    assert result.descriptive is not None
    assert result.descriptive.mean == pytest.approx(44.5, abs=0.1)
    assert result.descriptive.count == 10


async def test_e2e_statistics_empty_sensor(
    stats_service: StatisticsService,
    registered_sensor: SensorModel,
):
    """Statistics for sensor with no readings returns None descriptive."""
    result = await stats_service.compute_statistics("S-INT-001", time_range="24h")
    assert result.reading_count == 0
    assert result.descriptive is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Sensor health pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_e2e_health_update_after_ingestion(
    reading_service_normal: ReadingService,
    health_service: SensorHealthService,
    registered_sensor: SensorModel,
):
    """Sensor health computed after ingestion."""
    base = datetime.now(timezone.utc) - timedelta(hours=5)
    for i in range(10):
        await reading_service_normal.ingest_reading(
            ReadingCreateRequest(
                sensor_id="S-INT-001", value=42.0 + i * 0.1,
                timestamp=(base + timedelta(minutes=i * 30)).isoformat(),
                confidence=95.0,
            )
        )

    health = await health_service.update_sensor_health(registered_sensor)
    assert health.health_score > 0
    assert health.health_status in ("EXCELLENT", "GOOD", "FAIR", "POOR", "CRITICAL")
    assert health.total_readings == 10
    assert health.anomaly_count == 0


async def test_e2e_health_degraded_sensor(
    health_service: SensorHealthService,
    offline_sensor: SensorModel,
):
    """Offline sensor gets degraded health score."""
    health = await health_service.update_sensor_health(offline_sensor)
    assert health.uptime_score == 0.0  # OFFLINE


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Baseline learning pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_e2e_baseline_after_ingestion(
    reading_service_normal: ReadingService,
    baseline_service: BaselineLearningService,
    registered_sensor: SensorModel,
):
    """Learn baseline after ingesting readings."""
    base = datetime.now(timezone.utc) - timedelta(hours=5)
    values = [42.0, 43.1, 41.5, 44.2, 42.8, 43.5, 41.9, 44.0, 42.3, 43.7]
    for i, v in enumerate(values):
        await reading_service_normal.ingest_reading(
            ReadingCreateRequest(
                sensor_id="S-INT-001", value=v,
                timestamp=(base + timedelta(minutes=i * 30)).isoformat(),
                confidence=95.0,
            )
        )

    baseline = await baseline_service.update_baseline(registered_sensor.id)
    assert baseline is not None
    assert baseline.sample_count == 10
    assert 41.0 < baseline.mean < 44.0
    assert baseline.std_dev > 0
    assert baseline.normal_range_low < baseline.mean
    assert baseline.normal_range_high > baseline.mean


async def test_e2e_baseline_normal_range_check(
    reading_service_normal: ReadingService,
    baseline_service: BaselineLearningService,
    registered_sensor: SensorModel,
):
    """After baseline is learned, check if values are within normal range."""
    base = datetime.now(timezone.utc) - timedelta(hours=5)
    for i in range(10):
        await reading_service_normal.ingest_reading(
            ReadingCreateRequest(
                sensor_id="S-INT-001", value=42.0 + i * 0.1,
                timestamp=(base + timedelta(minutes=i * 30)).isoformat(),
                confidence=95.0,
            )
        )

    await baseline_service.update_baseline(registered_sensor.id)
    assert await baseline_service.is_within_normal_range("S-INT-001", 42.5) is True
    assert await baseline_service.is_within_normal_range("S-INT-001", 999.0) is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. Alert lifecycle — create → acknowledge → auto-resolve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_e2e_alert_lifecycle(
    reading_service_anomaly: ReadingService,
    reading_service_normal: ReadingService,
    alert_service: AlertService,
    alert_repo: SQLAlchemyAlertRepository,
    registered_sensor: SensorModel,
):
    """Full alert lifecycle: anomaly → alert created → normal → auto-resolved."""
    # Step 1: Anomalous reading → generate alert
    req_anomaly = _reading_request("S-INT-001", 42.5, offset_hours=2.0)
    reading = await reading_service_anomaly.ingest_reading(req_anomaly)

    ctx = ReadingContext(
        sensor_id=uuid.UUID(registered_sensor.id),
        sensor_code="S-INT-001",
        sensor_type="TEMPERATURE",
        value=reading.value,
        anomaly_score=reading.anomaly_score,
    )
    alerts = await alert_service.evaluate_reading(ctx)
    assert len(alerts) >= 1

    # Step 2: Verify alert persisted
    active = await alert_service.get_active_alerts()
    assert len(active) >= 1

    # Step 3: Normal reading → auto-resolve
    ctx_normal = ReadingContext(
        sensor_id=uuid.UUID(registered_sensor.id),
        sensor_code="S-INT-001",
        sensor_type="TEMPERATURE",
        value=42.5,
        anomaly_score=0.1,
    )
    await alert_service.evaluate_reading(ctx_normal)

    # Step 4: All alerts should be resolved
    active_after = await alert_service.get_active_alerts()
    assert len(active_after) == 0


async def test_e2e_alert_acknowledge(
    reading_service_anomaly: ReadingService,
    alert_service: AlertService,
    registered_sensor: SensorModel,
):
    """Manual alert acknowledgment."""
    req = _reading_request("S-INT-001", 42.5, offset_hours=3.0)
    reading = await reading_service_anomaly.ingest_reading(req)

    ctx = ReadingContext(
        sensor_id=uuid.UUID(registered_sensor.id),
        sensor_code="S-INT-001",
        sensor_type="TEMPERATURE",
        value=reading.value,
        anomaly_score=reading.anomaly_score,
    )
    alerts = await alert_service.evaluate_reading(ctx)
    assert len(alerts) >= 1

    acked = await alert_service.acknowledge_alert(alerts[0].id, "operator1")
    assert acked is not None
    assert acked.is_acknowledged is True
    assert acked.acknowledged_by == "operator1"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. Model monitoring — inference tracking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_e2e_model_monitoring_records_inferences(
    model_monitor: ModelMonitoringService,
):
    """Full monitoring pipeline: register → record inferences → check health."""
    # Register model
    model_monitor.register_model(
        ModelMetadata(model_name="isolation_forest", model_version="2.1.0")
    )
    model_monitor.set_active_detector("isolation_forest")

    # Record inferences
    model_monitor.record_inference("isolation_forest", "S001", 0.1, False, 5.0)
    model_monitor.record_inference("isolation_forest", "S002", 0.92, True, 3.0)
    model_monitor.record_inference("isolation_forest", "S003", 0.15, False, 4.0)

    # Verify stats
    stats = model_monitor.get_inference_stats("isolation_forest")
    assert stats.prediction_count == 3
    assert stats.anomaly_count == 1
    assert stats.normal_count == 2
    assert abs(stats.anomaly_rate - 33.33) < 1.0

    # Verify log
    log = model_monitor.get_inference_log("isolation_forest")
    assert len(log) == 3

    # Verify system summary
    summary = model_monitor.get_system_summary()
    assert summary["total_predictions"] == 3
    assert summary["active_detector"] == "isolation_forest"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. Full pipeline end-to-end
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_e2e_full_pipeline(
    reading_service_normal: ReadingService,
    stats_service: StatisticsService,
    alert_service: AlertService,
    health_service: SensorHealthService,
    baseline_service: BaselineLearningService,
    model_monitor: ModelMonitoringService,
    registered_sensor: SensorModel,
):
    """Complete pipeline: register → ingest → stats → health → baseline → monitoring."""
    # 1. Ingest 15 readings
    base = datetime.now(timezone.utc) - timedelta(hours=10)
    for i in range(15):
        await reading_service_normal.ingest_reading(
            ReadingCreateRequest(
                sensor_id="S-INT-001",
                value=42.0 + i * 0.5,
                timestamp=(base + timedelta(minutes=i * 30)).isoformat(),
                confidence=95.0,
            )
        )

    # 2. Statistics
    result = await stats_service.compute_statistics("S-INT-001", time_range="24h")
    assert result.reading_count == 15
    assert result.descriptive is not None

    # 3. Alert evaluation (normal readings → no alerts)
    ctx = ReadingContext(
        sensor_id=uuid.UUID(registered_sensor.id),
        sensor_code="S-INT-001",
        sensor_type="TEMPERATURE",
        value=42.0,
        anomaly_score=0.1,
    )
    alerts = await alert_service.evaluate_reading(ctx)
    assert len(alerts) == 0

    # 4. Health update
    health = await health_service.update_sensor_health(registered_sensor)
    assert health.health_score > 0
    assert health.total_readings == 15

    # 5. Baseline learning
    baseline = await baseline_service.update_baseline(registered_sensor.id)
    assert baseline is not None
    assert baseline.sample_count == 15

    # 6. Model monitoring
    model_monitor.register_model(
        ModelMetadata(model_name="stub_normal", model_version="1.0.0")
    )
    for i in range(15):
        model_monitor.record_inference("stub_normal", "S-INT-001", 0.1, False, 2.0)
    summary = model_monitor.get_system_summary()
    assert summary["total_predictions"] == 15
    assert summary["total_anomalies"] == 0
