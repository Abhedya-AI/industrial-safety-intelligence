"""Anomaly entity — a detected abnormal pattern in sensor data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from src.domain.value_objects.anomaly_type import AnomalyType


@dataclass
class Anomaly:
    """Represents an anomaly detected in sensor readings.

    Attributes:
        id: Internal unique identifier (UUID).
        reading_id: The reading that triggered this anomaly.
        sensor_id: The sensor that produced the anomalous reading.
        anomaly_type: Classification of the anomaly.
        severity_score: Severity on a 0.0–100.0 scale.
        confidence: Detection confidence (0.0–100.0).
        description: Human-readable description.
        is_resolved: Whether this anomaly has been resolved.
        detected_at: When the anomaly was detected.
        resolved_at: When the anomaly was resolved (if applicable).
    """

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
