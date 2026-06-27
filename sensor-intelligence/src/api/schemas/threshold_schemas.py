"""Pydantic schemas for threshold endpoints."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from src.domain.value_objects.sensor_type import SensorType


class ThresholdCreateRequest(BaseModel):
    """Request body for creating a threshold configuration."""

    sensor_id: UUID | None = Field(
        None, description="Specific sensor (null = applies to sensor_type)"
    )
    sensor_type: SensorType = Field(..., description="Sensor type this applies to")
    warning_min: float = Field(..., description="Warning lower bound")
    warning_max: float = Field(..., description="Warning upper bound")
    critical_min: float = Field(..., description="Critical lower bound")
    critical_max: float = Field(..., description="Critical upper bound")
    emergency_min: float = Field(..., description="Emergency lower bound")
    emergency_max: float = Field(..., description="Emergency upper bound")

    @model_validator(mode="after")
    def validate_threshold_ordering(self) -> "ThresholdCreateRequest":
        """Ensure thresholds are properly ordered: emergency ⊇ critical ⊇ warning."""
        if self.warning_min > self.warning_max:
            raise ValueError("warning_min must be ≤ warning_max")
        if self.critical_min > self.critical_max:
            raise ValueError("critical_min must be ≤ critical_max")
        if self.emergency_min > self.emergency_max:
            raise ValueError("emergency_min must be ≤ emergency_max")
        if self.critical_min > self.warning_min or self.critical_max < self.warning_max:
            raise ValueError("Critical range must contain warning range")
        if (
            self.emergency_min > self.critical_min
            or self.emergency_max < self.critical_max
        ):
            raise ValueError("Emergency range must contain critical range")
        return self


class ThresholdUpdateRequest(BaseModel):
    """Request body for updating a threshold configuration."""

    warning_min: float | None = None
    warning_max: float | None = None
    critical_min: float | None = None
    critical_max: float | None = None
    emergency_min: float | None = None
    emergency_max: float | None = None
    is_active: bool | None = None


class ThresholdResponse(BaseModel):
    """Response body for threshold data."""

    id: UUID
    sensor_id: UUID | None
    sensor_type: SensorType
    warning_min: float
    warning_max: float
    critical_min: float
    critical_max: float
    emergency_min: float
    emergency_max: float
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
