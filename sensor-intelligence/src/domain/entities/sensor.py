"""Sensor entity — represents a physical IoT sensor in the facility."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from uuid import UUID, uuid4

from src.domain.value_objects.sensor_status import SensorStatus
from src.domain.value_objects.sensor_type import SensorType


@dataclass
class Sensor:
    """Core domain entity for an IoT sensor.

    Attributes:
        id: Internal unique identifier (UUID).
        sensor_id: External business identifier (e.g. "S001").
        sensor_type: Classification of the sensor.
        location_zone: Physical zone in the facility.
        equipment_id: Equipment this sensor monitors.
        status: Current operational status.
        unit: Measurement unit (ppm, bar, °C, %, m/s²).
        model: Sensor hardware model name.
        calibration_date: Date of last calibration.
        accuracy_rating: Accuracy percentage (0.0–100.0).
        created_at: Record creation timestamp.
        updated_at: Record last-update timestamp.
    """

    sensor_id: str
    sensor_type: SensorType
    location_zone: str
    unit: str
    id: UUID = field(default_factory=uuid4)
    equipment_id: str | None = None
    status: SensorStatus = SensorStatus.NORMAL
    model: str | None = None
    calibration_date: date | None = None
    accuracy_rating: float | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
