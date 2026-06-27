"""SensorReading entity — a single measurement from a sensor."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4


@dataclass
class SensorReading:
    """A timestamped measurement value from a sensor.

    Attributes:
        id: Internal unique identifier (UUID).
        sensor_id: FK to the sensor that produced this reading.
        value: The measured value.
        timestamp: When the reading was taken (sensor clock).
        confidence: Sensor-reported accuracy for this reading (0.0–100.0).
        raw_metadata: Original payload metadata as JSON string.
        received_at: Server receive time.
    """

    sensor_id: UUID
    value: float
    timestamp: datetime
    id: UUID = field(default_factory=uuid4)
    confidence: float = 100.0
    raw_metadata: str | None = None
    received_at: datetime = field(default_factory=datetime.utcnow)
