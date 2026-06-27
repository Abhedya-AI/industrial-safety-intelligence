"""Alert entity — a user-facing notification generated from anomalies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from src.domain.value_objects.alert_level import AlertLevel


@dataclass
class Alert:
    """An actionable alert presented to operators.

    Attributes:
        id: Internal unique identifier (UUID).
        anomaly_id: The anomaly that generated this alert (nullable for manual alerts).
        sensor_id: The sensor associated with this alert.
        level: Severity level of the alert.
        title: Short summary of the alert.
        message: Detailed alert message.
        is_acknowledged: Whether an operator has acknowledged the alert.
        acknowledged_by: Identifier of the acknowledging user.
        acknowledged_at: Timestamp of acknowledgment.
        created_at: When the alert was created.
    """

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
