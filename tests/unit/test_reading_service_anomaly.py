"""Integration tests for anomaly detection in ReadingService.

Tests verify that:
  - Without a detector: readings get default NORMAL status
  - With a mock detector: readings get scored correctly
  - Detector failures don't break ingestion
  - Batch ingestion scores all readings
  - anomaly_score and anomaly_status are persisted
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Sequence
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
from app.sensor_intelligence.models.sensor_model import SensorModel
from app.sensor_intelligence.repositories.sqlalchemy_reading_repo import (
    SQLAlchemyReadingRepository,
)
from app.sensor_intelligence.repositories.sqlalchemy_sensor_repo import (
    SQLAlchemySensorRepository,
)
from app.sensor_intelligence.schemas.reading_schemas import ReadingCreateRequest
from app.sensor_intelligence.services.reading_service import ReadingService


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


class MockDetector(BaseAnomalyDetector):
    """A mock anomaly detector that returns configurable results."""

    def __init__(self, status: AnomalyStatus = AnomalyStatus.NORMAL, score: float = 0.1):
        self._status = status
        self._score = score
        self._loaded = True

    @property
    def name(self) -> str:
        return "mock"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self, **kwargs) -> None:
        self._loaded = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.zeros(len(features))

    def anomaly_score(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self._score)

    def classify(
        self, features: np.ndarray, sensor_ids: Sequence[str]
    ) -> list[AnomalyResult]:
        return [
            AnomalyResult(
                sensor_id=sid,
                score=self._score,
                status=self._status,
                detector_type=self.name,
                confidence=0.9,
            )
            for sid in sensor_ids
        ]


class FailingDetector(MockDetector):
    """A detector that always raises an exception."""

    def classify(self, features, sensor_ids):
        raise RuntimeError("Detector crashed!")


# ── Fixtures ──


@pytest_asyncio.fixture
async def sensor_repo(db_session: AsyncSession) -> SQLAlchemySensorRepository:
    return SQLAlchemySensorRepository(db_session)


@pytest_asyncio.fixture
async def reading_repo(db_session: AsyncSession) -> SQLAlchemyReadingRepository:
    return SQLAlchemyReadingRepository(db_session)


@pytest_asyncio.fixture
async def registered_sensor(sensor_repo: SQLAlchemySensorRepository) -> SensorModel:
    return await sensor_repo.create_sensor(
        _make_sensor(sensor_id="S001", min_value=0.0, max_value=10000.0)
    )


@pytest_asyncio.fixture
async def service_no_detector(
    reading_repo: SQLAlchemyReadingRepository,
    sensor_repo: SQLAlchemySensorRepository,
) -> ReadingService:
    """Service without an anomaly detector."""
    return ReadingService(reading_repo, sensor_repo)


@pytest_asyncio.fixture
async def service_with_normal_detector(
    reading_repo: SQLAlchemyReadingRepository,
    sensor_repo: SQLAlchemySensorRepository,
) -> ReadingService:
    """Service with a mock detector returning NORMAL."""
    return ReadingService(reading_repo, sensor_repo, MockDetector(AnomalyStatus.NORMAL, 0.15))


@pytest_asyncio.fixture
async def service_with_anomaly_detector(
    reading_repo: SQLAlchemyReadingRepository,
    sensor_repo: SQLAlchemySensorRepository,
) -> ReadingService:
    """Service with a mock detector returning ANOMALY."""
    return ReadingService(reading_repo, sensor_repo, MockDetector(AnomalyStatus.ANOMALY, 0.92))


@pytest_asyncio.fixture
async def service_with_failing_detector(
    reading_repo: SQLAlchemyReadingRepository,
    sensor_repo: SQLAlchemySensorRepository,
) -> ReadingService:
    """Service with a detector that always crashes."""
    return ReadingService(reading_repo, sensor_repo, FailingDetector())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# No detector (backward compatibility)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_no_detector_defaults_to_normal(
    service_no_detector: ReadingService, registered_sensor: SensorModel
):
    reading = await service_no_detector.ingest_reading(_reading_request())
    assert reading.anomaly_score == 0.0
    assert reading.anomaly_status == "NORMAL"


async def test_no_detector_batch_defaults_to_normal(
    service_no_detector: ReadingService, registered_sensor: SensorModel
):
    base = datetime.now(timezone.utc) - timedelta(hours=10)
    requests = [
        _reading_request(
            value=float(i * 10),
            timestamp=(base + timedelta(hours=i)).isoformat(),
        )
        for i in range(3)
    ]
    readings = await service_no_detector.ingest_batch(requests)
    for r in readings:
        assert r.anomaly_score == 0.0
        assert r.anomaly_status == "NORMAL"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# With detector — NORMAL classification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_detector_normal_sets_score_and_status(
    service_with_normal_detector: ReadingService, registered_sensor: SensorModel
):
    reading = await service_with_normal_detector.ingest_reading(_reading_request())
    assert reading.anomaly_score == 0.15
    assert reading.anomaly_status == "NORMAL"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# With detector — ANOMALY classification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_detector_anomaly_sets_score_and_status(
    service_with_anomaly_detector: ReadingService, registered_sensor: SensorModel
):
    reading = await service_with_anomaly_detector.ingest_reading(_reading_request())
    assert reading.anomaly_score == 0.92
    assert reading.anomaly_status == "ANOMALY"


async def test_detector_anomaly_batch(
    service_with_anomaly_detector: ReadingService, registered_sensor: SensorModel
):
    base = datetime.now(timezone.utc) - timedelta(hours=10)
    requests = [
        _reading_request(
            value=float(i * 100),
            timestamp=(base + timedelta(hours=i)).isoformat(),
        )
        for i in range(5)
    ]
    readings = await service_with_anomaly_detector.ingest_batch(requests)
    assert len(readings) == 5
    for r in readings:
        assert r.anomaly_score == 0.92
        assert r.anomaly_status == "ANOMALY"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Detector failure — graceful degradation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_detector_failure_defaults_to_normal(
    service_with_failing_detector: ReadingService, registered_sensor: SensorModel
):
    """Detector crash should NOT prevent ingestion."""
    reading = await service_with_failing_detector.ingest_reading(_reading_request())
    assert reading.id is not None  # Ingestion succeeded
    assert reading.anomaly_score == 0.0
    assert reading.anomaly_status == "NORMAL"


async def test_detector_failure_batch_still_ingests(
    service_with_failing_detector: ReadingService, registered_sensor: SensorModel
):
    base = datetime.now(timezone.utc) - timedelta(hours=10)
    requests = [
        _reading_request(
            value=float(i * 10),
            timestamp=(base + timedelta(hours=i)).isoformat(),
        )
        for i in range(3)
    ]
    readings = await service_with_failing_detector.ingest_batch(requests)
    assert len(readings) == 3
    for r in readings:
        assert r.anomaly_status == "NORMAL"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Persistence verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_anomaly_score_persisted_to_database(
    service_with_anomaly_detector: ReadingService,
    registered_sensor: SensorModel,
    reading_repo: SQLAlchemyReadingRepository,
):
    reading = await service_with_anomaly_detector.ingest_reading(_reading_request())
    fetched = await reading_repo.get_reading_by_id(reading.id)
    assert fetched is not None
    assert fetched.anomaly_score == 0.92
    assert fetched.anomaly_status == "ANOMALY"


async def test_normal_score_persisted_to_database(
    service_with_normal_detector: ReadingService,
    registered_sensor: SensorModel,
    reading_repo: SQLAlchemyReadingRepository,
):
    reading = await service_with_normal_detector.ingest_reading(_reading_request())
    fetched = await reading_repo.get_reading_by_id(reading.id)
    assert fetched is not None
    assert fetched.anomaly_score == 0.15
    assert fetched.anomaly_status == "NORMAL"
