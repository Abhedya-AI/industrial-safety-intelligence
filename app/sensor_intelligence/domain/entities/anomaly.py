"""Anomaly entity — a detected abnormal pattern in sensor data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from app.sensor_intelligence.domain.value_objects.anomaly_type import AnomalyType


@dataclass
class Anomaly:
    """Represents an anomaly detected in sensor readings."""

    reading_id: UUID
    sensor_id: UUID
    anomaly_type: AnomalyType
    severity_score: float
    id: UUID = field(default_factory=uuid4)
    confidence: float = 0.0
    description: str = ""
    is_resolved: bool = False
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None
