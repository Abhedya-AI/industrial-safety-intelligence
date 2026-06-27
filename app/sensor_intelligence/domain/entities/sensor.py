"""Sensor entity — represents a physical IoT sensor."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from uuid import UUID, uuid4

from app.sensor_intelligence.domain.value_objects.sensor_status import SensorStatus
from app.sensor_intelligence.domain.value_objects.sensor_type import SensorType


@dataclass
class Sensor:
    """Core domain entity for an IoT sensor."""

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
