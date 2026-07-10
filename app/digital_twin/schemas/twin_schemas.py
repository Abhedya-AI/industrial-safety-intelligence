"""Pydantic schemas for the Digital Twin REST API.

Aligned with the API specification patterns and PS-1 v2.0
conventions. All responses use the ``success: bool`` wrapper
consistent with other modules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Nested schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SensorReadingSchema(BaseModel):
    """A single sensor's latest reading."""

    sensor_id: str
    sensor_type: str = ""
    value: float = 0.0
    unit: str = ""
    anomaly_score: float = 0.0
    is_anomalous: bool = False
    health_score: float = 100.0
    status: str = "ACTIVE"
    last_updated: str = ""


class EquipmentStateSchema(BaseModel):
    """Live state of a single piece of equipment."""

    equipment_id: str
    equipment_type: str = ""
    zone_id: str = ""
    operational_status: str = "ACTIVE"
    health_score: float = 100.0
    risk_score: float = 0.0
    sensor_count: int = 0
    last_updated: str = ""


class ActiveHazardSchema(BaseModel):
    """An active hazard in the facility."""

    hazard_id: str
    hazard_type: str
    severity: str = "HIGH"
    origin_zone: str = ""
    propagation_level: str = "CONTAINED"
    affected_zones: List[str] = Field(default_factory=list)
    detected_at: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Zone state response
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ZoneStateSchema(BaseModel):
    """Complete live state of a single zone."""

    zone_id: str
    zone_name: str = ""

    # Sensor Intelligence
    sensor_health: float = 100.0
    anomaly_count: int = 0
    sensor_count: int = 0
    latest_sensor_readings: Dict[str, SensorReadingSchema] = Field(
        default_factory=dict,
    )

    # Risk Prediction
    predicted_risk_score: float = 0.0
    risk_level: str = "LOW"
    accident_probability: float = 0.0

    # Compound Risk
    compound_risk_score: float = 0.0
    compound_risk_level: str = "LOW"
    compound_risk_confidence: float = 0.0
    contributing_factors: Dict[str, float] = Field(default_factory=dict)

    # Hazard Propagation
    active_hazards: List[ActiveHazardSchema] = Field(default_factory=list)
    affected_neighbors: List[str] = Field(default_factory=list)

    # Context
    workers_at_risk: int = 0
    worker_capacity: int = 0
    current_worker_count: int = 0
    equipment: List[EquipmentStateSchema] = Field(default_factory=list)
    connected_zones: List[str] = Field(default_factory=list)

    # Computed
    overall_risk_score: float = 0.0
    heatmap_color: str = "green"
    is_critical: bool = False

    # Metadata
    last_updated: str = ""
    event_count: int = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API response wrappers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class FacilityStateResponse(BaseModel):
    """Response for GET /twin/facility."""

    success: bool = True
    timestamp: str = ""
    facility_health: float = 100.0
    total_zones: int = 0
    active_hazards: int = 0
    critical_zones: int = 0
    workers_at_risk: int = 0
    total_workers: int = 0
    total_equipment: int = 0
    total_sensors: int = 0
    total_anomalies: int = 0
    average_risk_score: float = 0.0
    max_risk_score: float = 0.0
    events_processed: int = 0
    zone_ids: List[str] = Field(default_factory=list)


class ZonesListResponse(BaseModel):
    """Response for GET /twin/zones."""

    success: bool = True
    timestamp: str = ""
    total: int = 0
    zones: List[ZoneStateSchema] = Field(default_factory=list)


class ZoneDetailResponse(BaseModel):
    """Response for GET /twin/zones/{zone_id}."""

    success: bool = True
    timestamp: str = ""
    zone: Optional[ZoneStateSchema] = None


class HeatmapEntry(BaseModel):
    """A single zone's heatmap entry."""

    zone_id: str
    zone_name: str = ""
    risk_score: float = 0.0
    risk_level: str = "LOW"
    color: str = "green"


class HeatmapResponse(BaseModel):
    """Response for GET /twin/heatmap."""

    success: bool = True
    timestamp: str = ""
    total: int = 0
    heatmap: List[HeatmapEntry] = Field(default_factory=list)
