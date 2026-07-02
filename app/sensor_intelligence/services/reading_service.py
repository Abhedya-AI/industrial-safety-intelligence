"""Reading service — business logic for sensor data ingestion.

Business rules enforced during ingestion:
  1. The referenced sensor must exist
  2. Reject readings for sensors with OFFLINE status
  3. Reading value must be a finite number (no NaN / Inf)
  4. Timestamp must not be in the future (>5 min tolerance)
  5. Reading value must be within sensor's configured min/max range
  6. Confidence must be 0-100
  7. Prevent duplicate readings (same sensor + same timestamp)
  8. Batch ingestion validates each reading; entire batch rejected on failure
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from app.sensor_intelligence.anomaly_detection.base import BaseAnomalyDetector
from app.sensor_intelligence.models.reading_model import ReadingModel
from app.sensor_intelligence.models.sensor_model import SensorModel
from app.sensor_intelligence.repositories.reading_repository import (
    ReadingRepository,
    ReadingStats,
)
from app.sensor_intelligence.repositories.sensor_repository import SensorRepository
from app.sensor_intelligence.schemas.reading_schemas import (
    ReadingCreateRequest,
)
from app.shared.exceptions.domain_exceptions import (
    BusinessRuleViolationError,
    InvalidReadingError,
    ResourceNotFoundError,
    ValidationError,
)

logger = logging.getLogger(__name__)

# Allow 5 minutes of clock drift for future timestamp detection
_FUTURE_TOLERANCE = timedelta(minutes=5)

# Sensor statuses that block ingestion
_INACTIVE_STATUSES = frozenset({"OFFLINE"})


class ReadingService:
    """Orchestrates sensor reading ingestion and retrieval."""

    def __init__(
        self,
        reading_repo: ReadingRepository,
        sensor_repo: SensorRepository,
        anomaly_detector: Optional[BaseAnomalyDetector] = None,
    ) -> None:
        self._reading_repo = reading_repo
        self._sensor_repo = sensor_repo
        self._detector = anomaly_detector

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Validation helpers (private)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _resolve_sensor(self, sensor_id: str) -> SensorModel:
        """Rule 1: referenced sensor must exist."""
        sensor = await self._sensor_repo.get_sensor_by_code(sensor_id)
        if sensor is None:
            raise ResourceNotFoundError(resource="Sensor", identifier=sensor_id)
        return sensor

    @staticmethod
    def _validate_sensor_active(sensor: SensorModel) -> None:
        """Rule 2: reject readings for OFFLINE sensors."""
        if sensor.status in _INACTIVE_STATUSES:
            raise BusinessRuleViolationError(
                f"Cannot ingest reading for sensor '{sensor.sensor_id}' "
                f"with status '{sensor.status}'. Sensor must be active."
            )

    @staticmethod
    def _validate_value(value: float) -> None:
        """Rule 3: value must be a finite number."""
        if math.isnan(value) or math.isinf(value):
            raise InvalidReadingError("value must be a finite number")

    @staticmethod
    def _validate_timestamp(timestamp: datetime) -> None:
        """Rule 4: timestamp must not be in the future (with tolerance)."""
        now = datetime.now(timezone.utc)
        # Make timestamp tz-aware if naive
        ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        if ts > now + _FUTURE_TOLERANCE:
            raise InvalidReadingError(
                f"timestamp {ts.isoformat()} is in the future"
            )

    @staticmethod
    def _validate_range(value: float, sensor: SensorModel) -> None:
        """Rule 5: reject values outside sensor's configured min/max."""
        if sensor.min_value is not None and value < sensor.min_value:
            raise InvalidReadingError(
                f"value {value} is below sensor minimum {sensor.min_value}"
            )
        if sensor.max_value is not None and value > sensor.max_value:
            raise InvalidReadingError(
                f"value {value} exceeds sensor maximum {sensor.max_value}"
            )

    @staticmethod
    def _validate_confidence(confidence: float) -> None:
        """Rule 6: confidence must be 0-100."""
        if confidence < 0 or confidence > 100:
            raise InvalidReadingError(
                f"confidence must be between 0 and 100, got {confidence}"
            )

    async def _check_duplicate(
        self, sensor_pk: str, timestamp: datetime
    ) -> None:
        """Rule 7: prevent duplicate readings (same sensor + timestamp)."""
        existing = await self._reading_repo.get_sensor_history(
            sensor_pk, timestamp, timestamp, limit=1
        )
        if existing:
            raise BusinessRuleViolationError(
                f"Duplicate reading: sensor already has a reading at "
                f"{timestamp.isoformat()}"
            )

    async def _validate_reading(
        self, request: ReadingCreateRequest, sensor: SensorModel
    ) -> None:
        """Run all validation rules for a single reading."""
        self._validate_sensor_active(sensor)
        self._validate_value(request.value)
        self._validate_timestamp(request.timestamp)
        self._validate_range(request.value, sensor)
        self._validate_confidence(request.confidence)
        await self._check_duplicate(sensor.id, request.timestamp)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Anomaly scoring (private)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _score_reading(self, reading: ReadingModel, sensor_id: str) -> None:
        """Score a reading using the configured anomaly detector.

        Sets anomaly_score and anomaly_status on the reading model.
        If no detector is configured, defaults to NORMAL with score 0.0.

        This is a synchronous, side-effect-only method — it mutates the
        reading in place. Detector failures are logged and swallowed to
        avoid blocking ingestion.
        """
        if self._detector is None:
            reading.anomaly_score = 0.0
            reading.anomaly_status = "NORMAL"
            return

        try:
            features = np.array([[reading.value]])
            results = self._detector.classify(features, [sensor_id])
            if results:
                result = results[0]
                reading.anomaly_score = result.score
                reading.anomaly_status = result.status.value
                logger.info(
                    "Anomaly score for sensor %s: %.4f (%s)",
                    sensor_id, result.score, result.status.value,
                )
            else:
                reading.anomaly_score = 0.0
                reading.anomaly_status = "NORMAL"
        except Exception:
            logger.exception("Anomaly detection failed for sensor %s — defaulting to NORMAL", sensor_id)
            reading.anomaly_score = 0.0
            reading.anomaly_status = "NORMAL"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Ingestion
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def ingest_reading(self, request: ReadingCreateRequest) -> ReadingModel:
        """Ingest a single sensor reading with full validation.

        If an anomaly detector is configured, the reading is scored
        after ingestion and anomaly_score / anomaly_status are set.
        """
        sensor = await self._resolve_sensor(request.sensor_id)
        await self._validate_reading(request, sensor)

        reading = ReadingModel(
            id=str(uuid.uuid4()),
            sensor_id=sensor.id,  # FK to sensors.id (UUID PK)
            value=request.value,
            timestamp=request.timestamp,
            confidence=request.confidence,
            raw_metadata=str(request.metadata) if request.metadata else None,
        )

        # Score reading if detector is available
        self._score_reading(reading, request.sensor_id)

        return await self._reading_repo.create_reading(reading)

    async def ingest_batch(
        self, requests: list[ReadingCreateRequest]
    ) -> list[ReadingModel]:
        """Ingest multiple readings. Validates each independently.

        Rule 8: each reading is validated; the entire batch is rejected
        if any single reading fails validation.
        """
        if not requests:
            raise ValidationError("Batch must contain at least one reading")

        # Pre-resolve all unique sensors in one pass
        sensor_cache: dict[str, SensorModel] = {}
        for req in requests:
            if req.sensor_id not in sensor_cache:
                sensor_cache[req.sensor_id] = await self._resolve_sensor(req.sensor_id)

        # Validate all readings
        for req in requests:
            await self._validate_reading(req, sensor_cache[req.sensor_id])

        # Build ORM objects
        models = []
        for req in requests:
            m = ReadingModel(
                id=str(uuid.uuid4()),
                sensor_id=sensor_cache[req.sensor_id].id,
                value=req.value,
                timestamp=req.timestamp,
                confidence=req.confidence,
                raw_metadata=str(req.metadata) if req.metadata else None,
            )
            self._score_reading(m, req.sensor_id)
            models.append(m)
        return await self._reading_repo.create_readings_batch(models)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Queries
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_latest_reading(self, sensor_id: str) -> Optional[ReadingModel]:
        """Get the most recent reading for a sensor."""
        sensor = await self._resolve_sensor(sensor_id)
        return await self._reading_repo.get_latest_reading(sensor.id)

    async def get_readings_range(
        self,
        sensor_id: str,
        from_dt: datetime,
        to_dt: datetime,
        limit: int = 1000,
    ) -> list[ReadingModel]:
        """Get historical readings for a sensor within a time range."""
        sensor = await self._resolve_sensor(sensor_id)
        return await self._reading_repo.get_sensor_history(sensor.id, from_dt, to_dt, limit)

    async def get_reading_stats(
        self,
        sensor_id: str,
        from_dt: datetime,
        to_dt: datetime,
    ) -> Optional[ReadingStats]:
        """Get aggregated statistics for a sensor's readings."""
        sensor = await self._resolve_sensor(sensor_id)
        return await self._reading_repo.get_stats(sensor.id, from_dt, to_dt)
