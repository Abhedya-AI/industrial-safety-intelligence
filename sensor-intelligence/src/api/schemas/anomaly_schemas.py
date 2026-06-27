"""Pydantic schemas for anomaly endpoints."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from src.domain.value_objects.anomaly_type import AnomalyType


class AnomalyResponse(BaseModel):
    """Response body for anomaly data."""

    id: UUID
    reading_id: UUID
    sensor_id: UUID
    anomaly_type: AnomalyType
    severity_score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=100)
    description: str
    is_resolved: bool
    detected_at: datetime
    resolved_at: datetime | None

    model_config = {"from_attributes": True}


class AnomalyDetailResponse(AnomalyResponse):
    """Extended anomaly response with related data."""

    sensor_external_id: str | None = None
    reading_value: float | None = None


class AnomalyResolveRequest(BaseModel):
    """Request body for resolving an anomaly."""

    resolved_by: str | None = Field(None, description="User who resolved the anomaly")
    notes: str | None = Field(None, description="Resolution notes")
