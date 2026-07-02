"""Sensor Health Service — orchestrates health assessment for sensors.

Responsibilities:
  1. Gather data from repositories (readings, anomalies, sensor metadata)
  2. Compute health scores using the pure scoring module
  3. Persist health assessments to the sensor_health table
  4. Provide query methods for health data

Called by:
  - Post-ingestion pipeline (automatic health update after new readings)
  - Dashboard / API endpoints for sensor health status
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from app.sensor_intelligence.models.sensor_health_model import SensorHealthModel
from app.sensor_intelligence.models.sensor_model import SensorModel
from app.sensor_intelligence.repositories.reading_repository import ReadingRepository
from app.sensor_intelligence.repositories.sensor_repository import SensorRepository
from app.sensor_intelligence.services.health_scoring import (
    HealthFactors,
    HealthScoreResult,
    HealthWeights,
    calculate_health_score,
)

logger = logging.getLogger(__name__)

# Expected interval between readings (in minutes)
_DEFAULT_READING_INTERVAL_MINUTES = 15

# Default time window for health calculations (7 days)
_DEFAULT_HEALTH_WINDOW_DAYS = 7


class SensorHealthRepository:
    """Lightweight repository for SensorHealthModel.

    Uses the same async session pattern as other repositories.
    """

    def __init__(self, session) -> None:
        self._session = session

    async def get_by_sensor_id(self, sensor_pk: str) -> Optional[SensorHealthModel]:
        from sqlalchemy import select
        stmt = select(SensorHealthModel).where(
            SensorHealthModel.sensor_id == sensor_pk
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def save(self, health: SensorHealthModel) -> SensorHealthModel:
        self._session.add(health)
        await self._session.flush()
        await self._session.refresh(health)
        return health

    async def update(self, health: SensorHealthModel) -> SensorHealthModel:
        await self._session.flush()
        await self._session.refresh(health)
        return health

    async def list_all(
        self,
        health_status: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[SensorHealthModel]:
        from sqlalchemy import select
        stmt = select(SensorHealthModel)
        if health_status:
            stmt = stmt.where(SensorHealthModel.health_status == health_status)
        stmt = stmt.order_by(SensorHealthModel.health_score.asc()).offset(offset).limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


class SensorHealthService:
    """Service for computing and managing sensor health assessments.

    Uses the pure scoring module (health_scoring.py) for calculations
    and persists results via SensorHealthRepository.
    """

    def __init__(
        self,
        health_repo: SensorHealthRepository,
        sensor_repo: SensorRepository,
        reading_repo: ReadingRepository,
        weights: Optional[HealthWeights] = None,
        reading_interval_minutes: int = _DEFAULT_READING_INTERVAL_MINUTES,
        health_window_days: int = _DEFAULT_HEALTH_WINDOW_DAYS,
    ) -> None:
        self._health_repo = health_repo
        self._sensor_repo = sensor_repo
        self._reading_repo = reading_repo
        self._weights = weights
        self._reading_interval_minutes = reading_interval_minutes
        self._health_window_days = health_window_days

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Core: calculate health score
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def calculate_health(
        self,
        sensor: SensorModel,
        reference_date: Optional[date] = None,
    ) -> HealthScoreResult:
        """Calculate health score for a sensor.

        Gathers data from repositories and delegates to the pure scoring module.

        Args:
            sensor: The sensor to assess.
            reference_date: Date for calculations (default: today). Useful for testing.

        Returns:
            HealthScoreResult with composite and individual scores.
        """
        today = reference_date or date.today()
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(days=self._health_window_days)

        # Gather reading statistics
        stats = await self._reading_repo.get_stats(
            sensor.id, window_start, now
        )

        total_readings = stats.count if stats else 0
        reading_mean = stats.mean if stats else 0.0
        reading_std = stats.std_dev if stats else 0.0

        # Count anomalous readings
        anomaly_count = await self._count_anomalies(sensor.id, window_start, now)

        # Calculate expected readings in the window
        window_minutes = self._health_window_days * 24 * 60
        expected_readings = window_minutes // self._reading_interval_minutes

        factors = HealthFactors(
            last_calibration=sensor.last_calibration,
            next_calibration_due=sensor.next_calibration_due,
            today=today,
            total_readings=total_readings,
            anomaly_count=anomaly_count,
            installation_date=sensor.installation_date,
            sensor_status=sensor.status,
            reading_std_dev=reading_std,
            reading_mean=reading_mean,
            expected_readings=expected_readings,
            actual_readings=total_readings,
        )

        return calculate_health_score(factors, self._weights)

    async def _count_anomalies(
        self, sensor_pk: str, start: datetime, end: datetime
    ) -> int:
        """Count readings flagged as anomalies in the window."""
        readings = await self._reading_repo.get_sensor_history(
            sensor_pk, start, end, limit=10000
        )
        return sum(
            1 for r in readings
            if r.anomaly_status and r.anomaly_status == "ANOMALY"
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Persist / update health
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def update_sensor_health(
        self,
        sensor: SensorModel,
        reference_date: Optional[date] = None,
    ) -> SensorHealthModel:
        """Calculate and persist health score for a sensor.

        Creates a new record if none exists, otherwise updates in place.

        Args:
            sensor: The sensor to assess.
            reference_date: Date for calculations.

        Returns:
            Persisted SensorHealthModel.
        """
        result = await self.calculate_health(sensor, reference_date)

        existing = await self._health_repo.get_by_sensor_id(sensor.id)

        if existing:
            existing.health_score = result.health_score
            existing.health_status = result.health_status.value
            existing.calibration_score = result.calibration_score
            existing.anomaly_score = result.anomaly_score
            existing.uptime_score = result.uptime_score
            existing.stability_score = result.stability_score
            existing.missing_readings_score = result.missing_readings_score
            existing.total_readings = result.details.get("total_readings", 0)
            existing.anomaly_count = result.details.get("anomaly_count", 0)
            existing.details = json.dumps(result.details)
            existing.calculated_at = datetime.now(timezone.utc)
            return await self._health_repo.update(existing)

        health = SensorHealthModel(
            id=str(uuid.uuid4()),
            sensor_id=sensor.id,
            health_score=result.health_score,
            health_status=result.health_status.value,
            calibration_score=result.calibration_score,
            anomaly_score=result.anomaly_score,
            uptime_score=result.uptime_score,
            stability_score=result.stability_score,
            missing_readings_score=result.missing_readings_score,
            total_readings=result.details.get("total_readings", 0),
            anomaly_count=result.details.get("anomaly_count", 0),
            details=json.dumps(result.details),
        )
        return await self._health_repo.save(health)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Queries
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_sensor_health(
        self, sensor_id: str
    ) -> Optional[SensorHealthModel]:
        """Get the latest health assessment for a sensor by business ID.

        Args:
            sensor_id: Business sensor ID (e.g. "S001").

        Returns:
            SensorHealthModel or None if no assessment exists.
        """
        sensor = await self._sensor_repo.get_sensor_by_code(sensor_id)
        if sensor is None:
            return None
        return await self._health_repo.get_by_sensor_id(sensor.id)

    async def get_all_sensor_health(
        self,
        health_status: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[SensorHealthModel]:
        """List all sensor health records with optional status filter."""
        return await self._health_repo.list_all(health_status, offset, limit)

    async def update_all_sensors(
        self,
        reference_date: Optional[date] = None,
    ) -> list[SensorHealthModel]:
        """Recalculate health for all registered sensors.

        Useful for batch refresh (e.g. scheduled job).
        """
        sensors = await self._sensor_repo.list_sensors(limit=10000)
        results = []
        for sensor in sensors:
            try:
                health = await self.update_sensor_health(sensor, reference_date)
                results.append(health)
            except Exception:
                logger.exception(
                    "Failed to update health for sensor %s", sensor.sensor_id
                )
        return results
