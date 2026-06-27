"""Alert entity — a user-facing notification generated from anomalies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from app.sensor_intelligence.domain.value_objects.alert_level import AlertLevel


@dataclass
class Alert:
    """An actionable alert presented to operators."""

    sensor_id: UUID
    level: AlertLevel
    title: str
    message: str
    id: UUID = field(default_factory=uuid4)
    anomaly_id: UUID | None = None
    is_acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
