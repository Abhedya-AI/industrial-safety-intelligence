"""Sensor data transfer objects.

Framework-agnostic DTOs used by use cases to pass data between layers.
"""

from dataclasses import dataclass
from datetime import date, datetime
from uuid import UUID

from src.domain.value_objects.sensor_status import SensorStatus
from src.domain.value_objects.sensor_type import SensorType


@dataclass(frozen=True)
class SensorCreateDTO:
    """Input DTO for registering a new sensor."""

    sensor_id: str
    sensor_type: SensorType
    location_zone: str
    unit: str
    equipment_id: str | None = None
    model: str | None = None
    calibration_date: date | None = None
    accuracy_rating: float | None = None


@dataclass(frozen=True)
class SensorUpdateDTO:
    """Input DTO for updating a sensor's metadata."""

    location_zone: str | None = None
    status: SensorStatus | None = None
    equipment_id: str | None = None
    model: str | None = None
    calibration_date: date | None = None
    accuracy_rating: float | None = None


@dataclass(frozen=True)
class SensorResponseDTO:
    """Output DTO for sensor data."""

    id: UUID
    sensor_id: str
    sensor_type: SensorType
    location_zone: str
    unit: str
    status: SensorStatus
    equipment_id: str | None
    model: str | None
    calibration_date: date | None
    accuracy_rating: float | None
    created_at: datetime
    updated_at: datetime
