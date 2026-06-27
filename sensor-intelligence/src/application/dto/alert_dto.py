"""Alert data transfer objects.

Framework-agnostic DTOs for alert data.
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from src.domain.value_objects.alert_level import AlertLevel


@dataclass(frozen=True)
class AlertResponseDTO:
    """Output DTO for alert data."""

    id: UUID
    sensor_id: UUID
    anomaly_id: UUID | None
    level: AlertLevel
    title: str
    message: str
    is_acknowledged: bool
    acknowledged_by: str | None
    acknowledged_at: datetime | None
    created_at: datetime


@dataclass(frozen=True)
class AlertAcknowledgeDTO:
    """Input DTO for acknowledging an alert."""

    acknowledged_by: str
