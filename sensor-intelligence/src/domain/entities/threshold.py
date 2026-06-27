"""Threshold entity — configurable safety thresholds for sensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from src.domain.value_objects.sensor_type import SensorType


@dataclass
class Threshold:
    """Configurable safety thresholds with three escalation tiers.

    Thresholds can be configured per-sensor (sensor_id set) or
    per-sensor-type (sensor_type set, sensor_id is None).

    Attributes:
        id: Internal unique identifier (UUID).
        sensor_id: Specific sensor this threshold applies to (nullable).
        sensor_type: Sensor type this threshold applies to.
        warning_min: Lower bound for warning level.
        warning_max: Upper bound for warning level.
        critical_min: Lower bound for critical level.
        critical_max: Upper bound for critical level.
        emergency_min: Lower bound for emergency level.
        emergency_max: Upper bound for emergency level.
        is_active: Whether this threshold config is currently active.
        created_at: Record creation timestamp.
        updated_at: Record last-update timestamp.
    """

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
