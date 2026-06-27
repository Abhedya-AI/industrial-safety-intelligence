"""Reading service — business logic for sensor data ingestion.

Business rules:
  1. The referenced sensor must exist
  2. Reading value must not be null/NaN
  3. Timestamp must not be in the future (>5 min tolerance)
  4. Reading value outside sensor min/max range logs a warning
  5. Confidence must be 0-100
  6. Batch ingestion validates each reading independently
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

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
    InvalidReadingError,
    ResourceNotFoundError,
    ValidationError,
)

logger = logging.getLogger(__name__)

# Allow 5 minutes of clock drift for future timestamp detection
_FUTURE_TOLERANCE = timedelta(minutes=5)


class ReadingService:
    """Orchestrates sensor reading ingestion and retrieval."""

    def __init__(
        self,
        reading_repo: ReadingRepository,
        sensor_repo: SensorRepository,
    ) -> None:
        self._reading_repo = reading_repo
        self._sensor_repo = sensor_repo

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
    def _validate_value(value: float) -> None:
        """Rule 2: value must be a finite number."""
        if math.isnan(value) or math.isinf(value):
            raise InvalidReadingError("value must be a finite number")

    @staticmethod
    def _validate_timestamp(timestamp: datetime) -> None:
        """Rule 3: timestamp must not be in the future (with tolerance)."""
        now = datetime.now(timezone.utc)
        # Make timestamp tz-aware if naive
        ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        if ts > now + _FUTURE_TOLERANCE:
            raise InvalidReadingError(
                f"timestamp {ts.isoformat()} is in the future"
            )

    @staticmethod
    def _check_range(value: float, sensor: SensorModel) -> None:
        """Rule 4: log warning if value outside sensor's calibrated range."""
        if sensor.min_value is not None and value < sensor.min_value:
            logger.warning(
                "Reading value %.2f below sensor %s min_value %.2f",
                value, sensor.sensor_id, sensor.min_value,
            )
        if sensor.max_value is not None and value > sensor.max_value:
            logger.warning(
                "Reading value %.2f above sensor %s max_value %.2f",
                value, sensor.sensor_id, sensor.max_value,
            )

    @staticmethod
    def _validate_confidence(confidence: float) -> None:
        """Rule 5: confidence must be 0-100."""
        if confidence < 0 or confidence > 100:
            raise InvalidReadingError(
                f"confidence must be between 0 and 100, got {confidence}"
            )

    def _validate_reading(self, request: ReadingCreateRequest, sensor: SensorModel) -> None:
        """Run all validation rules for a single reading."""
        self._validate_value(request.value)
        self._validate_timestamp(request.timestamp)
        self._validate_confidence(request.confidence)
        self._check_range(request.value, sensor)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Ingestion
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def ingest_reading(self, request: ReadingCreateRequest) -> ReadingModel:
        """Ingest a single sensor reading with full validation."""
        sensor = await self._resolve_sensor(request.sensor_id)
        self._validate_reading(request, sensor)

        reading = ReadingModel(
            id=str(uuid.uuid4()),
            sensor_id=sensor.id,  # FK to sensors.id (UUID PK)
            value=request.value,
            timestamp=request.timestamp,
            confidence=request.confidence,
            raw_metadata=str(request.metadata) if request.metadata else None,
        )
        return await self._reading_repo.save(reading)

    async def ingest_batch(
        self, requests: list[ReadingCreateRequest]
    ) -> list[ReadingModel]:
        """Ingest multiple readings. Validates each independently.

        Rule 6: each reading is validated; the entire batch is rejected
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
            self._validate_reading(req, sensor_cache[req.sensor_id])

        # Build ORM objects
        models = [
            ReadingModel(
                id=str(uuid.uuid4()),
                sensor_id=sensor_cache[req.sensor_id].id,
                value=req.value,
                timestamp=req.timestamp,
                confidence=req.confidence,
                raw_metadata=str(req.metadata) if req.metadata else None,
            )
            for req in requests
        ]
        return await self._reading_repo.save_batch(models)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Queries
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_latest_reading(self, sensor_id: str) -> Optional[ReadingModel]:
        """Get the most recent reading for a sensor."""
        sensor = await self._resolve_sensor(sensor_id)
        return await self._reading_repo.get_latest(sensor.id)

    async def get_readings_range(
        self,
        sensor_id: str,
        from_dt: datetime,
        to_dt: datetime,
        limit: int = 1000,
    ) -> list[ReadingModel]:
        """Get historical readings for a sensor within a time range."""
        sensor = await self._resolve_sensor(sensor_id)
        return await self._reading_repo.get_range(sensor.id, from_dt, to_dt, limit)

    async def get_reading_stats(
        self,
        sensor_id: str,
        from_dt: datetime,
        to_dt: datetime,
    ) -> Optional[ReadingStats]:
        """Get aggregated statistics for a sensor's readings."""
        sensor = await self._resolve_sensor(sensor_id)
        return await self._reading_repo.get_stats(sensor.id, from_dt, to_dt)
