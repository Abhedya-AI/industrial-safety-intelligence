"""Threshold entity — configurable safety thresholds for sensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from app.sensor_intelligence.domain.value_objects.sensor_type import SensorType


@dataclass
class Threshold:
    """Configurable safety thresholds with warning/critical/emergency tiers."""

    sensor_type: SensorType
    warning_min: float
    warning_max: float
    critical_min: float
    critical_max: float
    emergency_min: float
    emergency_max: float
    id: UUID = field(default_factory=uuid4)
    sensor_id: UUID | None = None
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
