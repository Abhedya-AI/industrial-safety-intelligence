"""Pydantic schemas for alert endpoints."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.sensor_intelligence.domain.value_objects.alert_level import AlertLevel


class AlertResponse(BaseModel):
    """Response body for alert data."""

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

    model_config = {"from_attributes": True}


class AlertAcknowledgeRequest(BaseModel):
    """Request body for acknowledging an alert."""

    acknowledged_by: str = Field(
        ..., min_length=1, max_length=100, description="User acknowledging the alert"
    )


class AlertSummaryResponse(BaseModel):
    """Summary counts of alerts by level."""

    info: int = 0
    warning: int = 0
    critical: int = 0
    emergency: int = 0
    total: int = 0
    unacknowledged: int = 0
