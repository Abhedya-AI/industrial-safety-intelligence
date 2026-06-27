"""Sensor service — business logic for sensor CRUD and spec-aligned operations.

Business rules enforced:
  1. sensor_id (business code) must be unique
  2. last_calibration cannot be earlier than installation_date
  3. min_value must be less than max_value (when both provided)
  4. status defaults to NORMAL on creation
  5. Cannot delete sensors that have associated readings
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.sensor_intelligence.domain.value_objects.sensor_status import SensorStatus
from app.sensor_intelligence.domain.value_objects.sensor_type import SensorType
from app.sensor_intelligence.models.reading_model import ReadingModel
from app.sensor_intelligence.models.sensor_model import SensorModel
from app.sensor_intelligence.repositories.sensor_repository import SensorRepository
from app.sensor_intelligence.schemas.sensor_schemas import (
    CurrentSensorItem,
    CurrentSensorsResponse,
    SensorCreateRequest,
    SensorDetailInfo,
    SensorHistoryResponse,
    SensorSummary,
    SensorUpdateRequest,
)
from app.shared.exceptions.domain_exceptions import (
    BusinessRuleViolationError,
    DuplicateResourceError,
    ResourceNotFoundError,
    ValidationError,
)

logger = logging.getLogger(__name__)


class SensorService:
    """Orchestrates sensor domain operations with business rule enforcement."""

    def __init__(self, repo: SensorRepository, session: AsyncSession) -> None:
        self._repo = repo
        self._session = session

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Business Validation (private)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _validate_min_max(
        min_value: Optional[float], max_value: Optional[float]
    ) -> None:
        """Rule: min_value must be less than max_value when both are provided."""
        if min_value is not None and max_value is not None:
            if min_value >= max_value:
                raise ValidationError(
                    f"min_value ({min_value}) must be less than "
                    f"max_value ({max_value})"
                )

    @staticmethod
    def _validate_calibration_date(
        installation_date: Optional[date],
        calibration_date: Optional[date],
    ) -> None:
        """Rule: calibration_date cannot be earlier than installation_date."""
        if installation_date is not None and calibration_date is not None:
            if calibration_date < installation_date:
                raise ValidationError(
                    f"calibration_date ({calibration_date}) cannot be earlier "
                    f"than installation_date ({installation_date})"
                )

    async def _check_has_readings(self, sensor_pk: str) -> bool:
        """Check whether a sensor has associated reading records.

        Uses a lightweight EXISTS sub-query against the sensor_readings
        table via the raw session — no dependency on a ReadingRepository
        (which may not be implemented yet).
        """
        stmt = select(
            select(ReadingModel.id)
            .where(ReadingModel.sensor_id == sensor_pk)
            .limit(1)
            .exists()
        )
        result = await self._session.execute(stmt)
        return result.scalar_one()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CRUD
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def create_sensor(self, request: SensorCreateRequest) -> SensorModel:
        """Register a new sensor with full business validation.

        Validates:
          - sensor_id uniqueness
          - min_value < max_value
          - calibration_date >= installation_date
        """
        # Rule 1: sensor_id must be unique
        if await self._repo.sensor_exists(request.sensor_id):
            raise DuplicateResourceError(
                resource="Sensor",
                identifier=request.sensor_id,
            )

        # Rule 2: calibration_date >= installation_date
        self._validate_calibration_date(
            request.installation_date, request.last_calibration
        )

        # Rule 3: min_value < max_value
        self._validate_min_max(request.min_value, request.max_value)

        # Rule 4: status defaults to NORMAL (set by model default)
        sensor = SensorModel(
            sensor_id=request.sensor_id,
            sensor_name=request.sensor_name,
            sensor_type=request.sensor_type.value,
            location_zone=request.location_zone,
            equipment_id=request.equipment_id,
            manufacturer=request.manufacturer,
            model=request.model,
            unit=request.unit,
            min_value=request.min_value,
            max_value=request.max_value,
            accuracy_rating=request.accuracy_rating,
            installation_date=request.installation_date,
            last_calibration=request.last_calibration,
            next_calibration_due=request.next_calibration_due,
        )
        return await self._repo.create_sensor(sensor)

    async def get_sensor(self, sensor_id: str) -> SensorModel:
        """Fetch a sensor by business ID. Raises ResourceNotFoundError if missing."""
        sensor = await self._repo.get_sensor_by_code(sensor_id)
        if sensor is None:
            raise ResourceNotFoundError(resource="Sensor", identifier=sensor_id)
        return sensor

    async def update_sensor(
        self, sensor_id: str, request: SensorUpdateRequest
    ) -> SensorModel:
        """Partially update sensor fields with business validation.

        Validates:
          - min_value < max_value (using merged values)
          - calibration_date >= installation_date (using merged values)
        """
        sensor = await self.get_sensor(sensor_id)
        update_data = request.model_dump(exclude_unset=True)

        # Merge current + incoming values for cross-field validation
        new_min = update_data.get("min_value", sensor.min_value)
        new_max = update_data.get("max_value", sensor.max_value)
        self._validate_min_max(new_min, new_max)

        new_install = update_data.get("installation_date", sensor.installation_date)
        new_calib = update_data.get("last_calibration", sensor.last_calibration)
        self._validate_calibration_date(new_install, new_calib)

        for field, value in update_data.items():
            if isinstance(value, (SensorType, SensorStatus)):
                value = value.value
            setattr(sensor, field, value)

        return await self._repo.update_sensor(sensor)

    async def delete_sensor(self, sensor_id: str) -> None:
        """Delete a sensor by business ID.

        Rule 5: Cannot delete sensors that have associated readings.
        """
        sensor = await self.get_sensor(sensor_id)

        if await self._check_has_readings(sensor.id):
            raise BusinessRuleViolationError(
                f"Cannot delete sensor '{sensor_id}': "
                "it has associated readings. "
                "Delete or archive the readings first."
            )

        await self._repo.delete_sensor(sensor)

    async def list_sensors(
        self,
        sensor_type: Optional[SensorType] = None,
        status: Optional[SensorStatus] = None,
        location_zone: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[SensorModel], int]:
        """List sensors with pagination. Returns (items, total_count)."""
        type_val = sensor_type.value if sensor_type else None
        status_val = status.value if status else None
        items = await self._repo.list_sensors(
            sensor_type=type_val,
            status=status_val,
            location_zone=location_zone,
            offset=offset,
            limit=limit,
        )
        total = await self._repo.count(
            sensor_type=type_val,
            status=status_val,
            location_zone=location_zone,
        )
        return items, total

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Spec Endpoint 6: GET /sensors/current
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_current_sensors(
        self,
        sensor_type: Optional[SensorType] = None,
        status: Optional[SensorStatus] = None,
        location_zone: Optional[str] = None,
    ) -> CurrentSensorsResponse:
        """Build the spec-compliant GET /sensors/current response.

        NOTE: current_reading, threshold, trend, anomaly, and health sub-objects
        require the Reading / Threshold / Anomaly services which are not yet
        implemented. They are returned as None / defaults for now.
        """
        type_val = sensor_type.value if sensor_type else None
        status_val = status.value if status else None
        sensors = await self._repo.list_sensors(
            sensor_type=type_val,
            status=status_val,
            location_zone=location_zone,
            offset=0,
            limit=1000,
        )

        status_counts = await self._repo.count_by_status()

        sensor_items = [
            CurrentSensorItem(
                sensor_id=s.sensor_id,
                sensor_type=SensorType(s.sensor_type),
                location_zone=s.location_zone,
                status=SensorStatus(s.status),
                current_reading=None,
                threshold=None,
                trend=None,
                anomaly_detected=False,
                anomaly_score=None,
                anomaly_severity=None,
                health=None,
            )
            for s in sensors
        ]

        summary = SensorSummary(
            total_sensors=sum(status_counts.values()),
            sensors_normal=status_counts.get("NORMAL", 0),
            sensors_warning=status_counts.get("WARNING", 0),
            sensors_critical=status_counts.get("CRITICAL", 0),
            sensors_offline=status_counts.get("OFFLINE", 0),
            anomalies_detected=0,
        )

        return CurrentSensorsResponse(
            success=True,
            timestamp=datetime.now(timezone.utc),
            sensors=sensor_items,
            summary=summary,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Spec Endpoint 7: GET /sensors/{sensor_id}/history
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_sensor_history(
        self,
        sensor_id: str,
        time_range: str = "24h",
        granularity: str = "auto",
    ) -> SensorHistoryResponse:
        """Build the spec-compliant GET /sensors/{sensor_id}/history response.

        NOTE: readings, statistics, anomalies, and forecast are stubbed until
        the Reading and Anomaly services are implemented.
        """
        sensor = await self.get_sensor(sensor_id)

        detail = SensorDetailInfo(
            sensor_id=sensor.sensor_id,
            sensor_type=SensorType(sensor.sensor_type),
            location_zone=sensor.location_zone,
            equipment_id=sensor.equipment_id,
            manufacturer=sensor.manufacturer,
            model=sensor.model,
            installation_date=(
                sensor.installation_date.isoformat()
                if sensor.installation_date else None
            ),
            last_calibration=(
                sensor.last_calibration.isoformat()
                if sensor.last_calibration else None
            ),
            next_calibration_due=(
                sensor.next_calibration_due.isoformat()
                if sensor.next_calibration_due else None
            ),
            accuracy_rating=sensor.accuracy_rating,
        )

        return SensorHistoryResponse(
            success=True,
            sensor=detail,
            readings=[],
            statistics=None,
            anomalies_detected=[],
            forecast=None,
        )
