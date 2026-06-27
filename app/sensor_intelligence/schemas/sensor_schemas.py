"""Pydantic schemas for sensor endpoints.

Aligned to PS1_Detailed_API_Specifications_V2 (endpoints 6 & 7).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.sensor_intelligence.domain.value_objects.sensor_status import SensorStatus
from app.sensor_intelligence.domain.value_objects.sensor_type import SensorType


# ──────────────────────────────────────────────
# Request Schemas
# ──────────────────────────────────────────────


class SensorCreateRequest(BaseModel):
    """Request body for registering a new sensor."""

    sensor_id: str = Field(
        ..., min_length=1, max_length=50,
        description="Unique business identifier (e.g. S001)",
    )
    sensor_name: str = Field(
        ..., min_length=1, max_length=150,
        description="Human-readable sensor name",
    )
    sensor_type: SensorType = Field(..., description="Type of sensor")
    location_zone: Optional[str] = Field(
        None, max_length=100, description="Facility zone (e.g. ZONE_A)",
    )
    equipment_id: Optional[str] = Field(
        None, max_length=100, description="Equipment this sensor monitors",
    )
    manufacturer: Optional[str] = Field(
        None, max_length=150, description="Sensor manufacturer",
    )
    model: Optional[str] = Field(
        None, max_length=100, description="Hardware model",
    )
    unit: str = Field(
        ..., min_length=1, max_length=20,
        description="Measurement unit (e.g. ppm, °C, bar)",
    )
    min_value: Optional[float] = Field(None, description="Minimum measurable value")
    max_value: Optional[float] = Field(None, description="Maximum measurable value")
    accuracy_rating: Optional[float] = Field(
        None, ge=0, le=1, description="Accuracy rating (0.0 to 1.0)",
    )
    installation_date: Optional[date] = Field(None, description="Installation date")
    last_calibration: Optional[date] = Field(None, description="Last calibration date")
    next_calibration_due: Optional[date] = Field(None, description="Next calibration due")

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "examples": [
                {
                    "sensor_id": "S001",
                    "sensor_name": "Zone A Gas Detector",
                    "sensor_type": "GAS",
                    "location_zone": "ZONE_A",
                    "equipment_id": "EQ001",
                    "manufacturer": "Dräger",
                    "model": "POLYTRON 8700",
                    "unit": "ppm",
                    "min_value": 0.0,
                    "max_value": 10000.0,
                    "accuracy_rating": 0.99,
                    "installation_date": "2024-01-15",
                    "last_calibration": "2025-06-01",
                    "next_calibration_due": "2025-09-01",
                }
            ]
        },
    }


class SensorUpdateRequest(BaseModel):
    """Request body for updating sensor metadata."""

    sensor_name: Optional[str] = Field(None, max_length=150)
    sensor_type: Optional[SensorType] = None
    status: Optional[SensorStatus] = None
    location_zone: Optional[str] = Field(None, max_length=100)
    equipment_id: Optional[str] = Field(None, max_length=100)
    manufacturer: Optional[str] = Field(None, max_length=150)
    model: Optional[str] = Field(None, max_length=100)
    unit: Optional[str] = Field(None, max_length=20)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    accuracy_rating: Optional[float] = Field(None, ge=0, le=1)
    installation_date: Optional[date] = None
    last_calibration: Optional[date] = None
    next_calibration_due: Optional[date] = None

    model_config = {"protected_namespaces": ()}


# ──────────────────────────────────────────────
# Response Schemas — Internal CRUD
# ──────────────────────────────────────────────


class SensorResponse(BaseModel):
    """Full sensor record returned by CRUD operations."""

    id: UUID
    sensor_id: str
    sensor_name: str
    sensor_type: SensorType
    status: SensorStatus
    location_zone: Optional[str]
    equipment_id: Optional[str]
    manufacturer: Optional[str]
    model: Optional[str]
    unit: str
    min_value: Optional[float]
    max_value: Optional[float]
    accuracy_rating: Optional[float]
    installation_date: Optional[date]
    last_calibration: Optional[date]
    next_calibration_due: Optional[date]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class SensorListResponse(BaseModel):
    """Paginated list of sensors."""

    success: bool = True
    items: list[SensorResponse]
    total: int
    offset: int
    limit: int


# ──────────────────────────────────────────────
# Response Schemas — API Spec Endpoint 6:
# GET /sensors/current
# ──────────────────────────────────────────────


class CurrentReading(BaseModel):
    """Current measurement from a sensor."""

    value: float
    unit: str
    timestamp: datetime
    confidence: float


class ThresholdInfo(BaseModel):
    """Warning/critical thresholds for a sensor."""

    warning: float
    critical: float


class TrendInfo(BaseModel):
    """Trend direction for a sensor reading."""

    direction: str  # "increasing" | "decreasing" | "stable"
    rate_of_change: float
    time_unit: str


class SensorHealthInfo(BaseModel):
    """Hardware health status of a sensor."""

    battery_voltage: Optional[float] = None
    signal_strength: Optional[float] = None
    calibration_status: str = "valid"
    last_calibration: Optional[str] = None


class CurrentSensorItem(BaseModel):
    """A single sensor in the GET /sensors/current response."""

    sensor_id: str
    sensor_type: SensorType
    location_zone: Optional[str]
    current_reading: Optional[CurrentReading] = None
    status: SensorStatus
    threshold: Optional[ThresholdInfo] = None
    trend: Optional[TrendInfo] = None
    anomaly_detected: bool = False
    anomaly_score: Optional[float] = None
    anomaly_severity: Optional[str] = None
    health: Optional[SensorHealthInfo] = None

    model_config = {"protected_namespaces": ()}


class SensorSummary(BaseModel):
    """Summary counts for the current sensor fleet."""

    total_sensors: int
    sensors_normal: int
    sensors_warning: int
    sensors_critical: int
    sensors_offline: int
    anomalies_detected: int


class CurrentSensorsResponse(BaseModel):
    """Response for GET /sensors/current (spec endpoint 6)."""

    success: bool = True
    timestamp: datetime
    sensors: list[CurrentSensorItem]
    summary: SensorSummary


# ──────────────────────────────────────────────
# Response Schemas — API Spec Endpoint 7:
# GET /sensors/{sensor_id}/history
# ──────────────────────────────────────────────


class SensorDetailInfo(BaseModel):
    """Sensor metadata block in history response."""

    sensor_id: str
    sensor_type: SensorType
    location_zone: Optional[str]
    equipment_id: Optional[str]
    manufacturer: Optional[str]
    model: Optional[str]
    installation_date: Optional[str]
    last_calibration: Optional[str]
    next_calibration_due: Optional[str]
    accuracy_rating: Optional[float]

    model_config = {"protected_namespaces": ()}


class ReadingPoint(BaseModel):
    """A single historical reading data point."""

    timestamp: datetime
    value: float
    status: str


class ReadingStatistics(BaseModel):
    """Aggregated statistics over a time window."""

    average: float
    min: float
    max: float
    median: float
    std_dev: float


class DetectedAnomaly(BaseModel):
    """An anomaly detected in the reading history."""

    timestamp: datetime
    value: float
    anomaly_score: float
    severity: str
    anomaly_type: str


class ForecastInfo(BaseModel):
    """Simple forecast for the sensor."""

    next_hour_prediction: Optional[float] = None
    confidence: Optional[float] = None
    trend: Optional[str] = None


class SensorHistoryResponse(BaseModel):
    """Response for GET /sensors/{sensor_id}/history (spec endpoint 7)."""

    success: bool = True
    sensor: SensorDetailInfo
    readings: list[ReadingPoint]
    statistics: Optional[ReadingStatistics] = None
    anomalies_detected: list[DetectedAnomaly] = []
    forecast: Optional[ForecastInfo] = None

    model_config = {"protected_namespaces": ()}
