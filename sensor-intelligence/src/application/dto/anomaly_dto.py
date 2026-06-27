"""Anomaly data transfer objects.

Framework-agnostic DTOs for anomaly data.
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from src.domain.value_objects.anomaly_type import AnomalyType


@dataclass(frozen=True)
class AnomalyResponseDTO:
    """Output DTO for anomaly data."""

    id: UUID
    reading_id: UUID
    sensor_id: UUID
    anomaly_type: AnomalyType
    severity_score: float
    confidence: float
    description: str
    is_resolved: bool
    detected_at: datetime
    resolved_at: datetime | None


@dataclass(frozen=True)
class AnomalyResolveDTO:
    """Input DTO for resolving an anomaly."""

    resolved_by: str | None = None
    notes: str | None = None
