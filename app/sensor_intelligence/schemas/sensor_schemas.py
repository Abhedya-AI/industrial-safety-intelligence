"""Pydantic schemas for sensor endpoints."""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.sensor_intelligence.domain.value_objects.sensor_status import SensorStatus
from app.sensor_intelligence.domain.value_objects.sensor_type import SensorType


class SensorCreateRequest(BaseModel):
    """Request body for registering a new sensor."""

    sensor_id: str = Field(
        ..., min_length=1, max_length=50, description="Unique sensor identifier"
    )
    sensor_type: SensorType = Field(..., description="Type of sensor")
    location_zone: str = Field(
        ..., min_length=1, max_length=100, description="Facility zone"
    )
    unit: str = Field(..., min_length=1, max_length=20, description="Measurement unit")
    equipment_id: str | None = Field(
        None, max_length=100, description="Equipment this sensor monitors"
    )
    model: str | None = Field(None, max_length=100, description="Hardware model")
    calibration_date: date | None = Field(None, description="Last calibration date")
    accuracy_rating: float | None = Field(
        None, ge=0, le=100, description="Accuracy percentage"
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "sensor_id": "S001",
                "sensor_type": "GAS",
                "location_zone": "Zone A",
                "unit": "ppm",
                "equipment_id": "EQ-BOILER-001",
                "model": "MQ-135",
            }
        ]
    }}


class SensorUpdateRequest(BaseModel):
    """Request body for updating sensor metadata."""

    location_zone: str | None = Field(None, max_length=100)
    status: SensorStatus | None = None
    equipment_id: str | None = Field(None, max_length=100)
    model: str | None = Field(None, max_length=100)
    calibration_date: date | None = None
    accuracy_rating: float | None = Field(None, ge=0, le=100)


class SensorResponse(BaseModel):
    """Response body for sensor data."""

    id: UUID
    sensor_id: str
    sensor_type: SensorType
    location_zone: str
    unit: str
    status: SensorStatus
    equipment_id: str | None
    model: str | None
    calibration_date: date | None
    accuracy_rating: float | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SensorDetailResponse(SensorResponse):
    """Extended sensor response with latest reading info."""

    latest_value: float | None = None
    latest_reading_at: datetime | None = None


class HealthScoreResponse(BaseModel):
    """Response body for sensor health score."""

    sensor_id: str
    health_score: float = Field(..., ge=0, le=100)
    factors: dict[str, float] = Field(
        default_factory=dict, description="Individual factor scores"
    )
    computed_at: datetime = Field(default_factory=datetime.utcnow)
