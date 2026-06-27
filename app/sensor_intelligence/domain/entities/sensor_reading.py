"""SensorReading entity — a single measurement from a sensor."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4


@dataclass
class SensorReading:
    """A timestamped measurement value from a sensor."""

    sensor_id: UUID
    value: float
    timestamp: datetime
    id: UUID = field(default_factory=uuid4)
    confidence: float = 100.0
    raw_metadata: str | None = None
    received_at: datetime = field(default_factory=datetime.utcnow)
