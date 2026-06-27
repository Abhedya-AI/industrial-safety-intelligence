"""Pydantic schemas for reading endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ReadingCreateRequest(BaseModel):
    """Request body for ingesting a single sensor reading."""

    sensor_id: str = Field(
        ..., min_length=1, max_length=50, description="External sensor ID"
    )
    value: float = Field(..., description="Measured value")
    timestamp: datetime = Field(..., description="Measurement timestamp (ISO 8601)")
    confidence: float = Field(
        100.0, ge=0, le=100, description="Reading confidence percentage"
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Additional payload metadata"
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "sensor_id": "S001",
                "value": 125.7,
                "timestamp": "2026-06-27T08:30:00Z",
                "confidence": 98.5,
                "metadata": {"equipment_id": "EQ-BOILER-001"},
            }
        ]
    }}


class BatchReadingCreateRequest(BaseModel):
    """Request body for batch ingestion of readings."""

    readings: list[ReadingCreateRequest] = Field(
        ..., min_length=1, max_length=1000, description="Batch of readings"
    )


class ReadingResponse(BaseModel):
    """Response body for a sensor reading."""

    id: UUID
    sensor_id: UUID
    value: float
    timestamp: datetime
    confidence: float
    received_at: datetime

    model_config = {"from_attributes": True}


class ReadingStatsResponse(BaseModel):
    """Aggregated statistics for sensor readings."""

    sensor_id: UUID
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    count: int
    window_start: datetime
    window_end: datetime


class BatchReadingResponse(BaseModel):
    """Response body for batch ingestion."""

    ingested: int = Field(..., description="Number of readings ingested")
    readings: list[ReadingResponse]
