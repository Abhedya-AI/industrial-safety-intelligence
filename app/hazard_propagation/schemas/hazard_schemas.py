"""Pydantic schemas for the Hazard Propagation Engine.

Aligned with the API specification (§21, POST /hazard/propagate) and
the PS-1 SentinelAI Common Domain Names v2.0 conventions.

Request schemas:
  - HazardPropagationRequest — input for propagation simulation

Response schemas:
  - HazardPathResponse           — a single propagation path
  - ZoneRiskStateResponse        — zone impact summary
  - HazardPropagationResponse    — full simulation result
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Request
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class HazardPropagationRequest(BaseModel):
    """Request body for POST /hazard/propagate.

    Matches the API spec:
    ```json
    {
      "hazard_type": "GAS_LEAK",
      "origin_zone": "ZONE_A"
    }
    ```
    """

    hazard_type: str = Field(
        ...,
        description="Hazard type from PS-1 §4.6 (e.g. GAS_LEAK, FIRE, SMOKE)",
        examples=["GAS_LEAK", "FIRE", "CHEMICAL_SPILL"],
    )
    origin_zone: str = Field(
        ...,
        description="Zone ID where the hazard originated",
        examples=["ZONE_A", "ZONE_BOILER"],
    )
    severity: Optional[str] = Field(
        None,
        description="Optional override for hazard severity (LOW/MEDIUM/HIGH/CRITICAL)",
    )
    max_propagation_depth: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum number of zone hops to simulate",
    )
    include_paths: bool = Field(
        True,
        description="Include detailed propagation paths in response",
    )

    @field_validator("hazard_type")
    @classmethod
    def validate_hazard_type(cls, v: str) -> str:
        valid = {
            "GAS_LEAK", "FIRE", "SMOKE", "CHEMICAL_SPILL",
            "PPE_VIOLATION", "FALL_DETECTED", "ELECTRICAL_FAULT",
            "TEMPERATURE_ANOMALY", "PRESSURE_ANOMALY",
        }
        if v.upper() not in valid:
            raise ValueError(
                f"Invalid hazard_type '{v}'. Must be one of: {sorted(valid)}"
            )
        return v.upper()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "hazard_type": "GAS_LEAK",
                    "origin_zone": "ZONE_A",
                }
            ]
        }
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Response components
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class HazardPathResponse(BaseModel):
    """A single propagation path in the response."""

    from_zone: str = Field(..., description="Source zone ID")
    to_zone: str = Field(..., description="Destination zone ID")
    probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Propagation probability (0.0–1.0)",
    )
    estimated_time_minutes: float = Field(
        ..., ge=0.0,
        description="Estimated time for hazard to reach destination (minutes)",
    )
    path_type: str = Field(
        "CONNECTED_TO",
        description="Type of connection between zones",
    )
    distance_meters: Optional[float] = Field(
        None, description="Physical distance between zones (meters)",
    )
    is_blocked: bool = Field(
        False, description="True if the path is blocked",
    )


class ZoneRiskStateResponse(BaseModel):
    """Zone risk state in the propagation response."""

    zone_id: str = Field(..., description="Zone identifier")
    zone_name: str = Field("", description="Human-readable zone name")
    risk_level: str = Field(..., description="Risk classification")
    risk_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Numerical risk score (0–100)",
    )
    is_origin: bool = Field(
        False, description="True if this is the hazard origin zone",
    )
    is_affected: bool = Field(
        False, description="True if the hazard has reached this zone",
    )
    arrival_time_minutes: float = Field(
        0.0, ge=0.0,
        description="Estimated time for hazard arrival (0 if origin)",
    )
    propagation_probability: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Cumulative probability of being affected",
    )
    worker_count: int = Field(
        0, ge=0,
        description="Number of workers currently in the zone",
    )
    equipment_count: int = Field(
        0, ge=0,
        description="Number of equipment items in the zone",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full response
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class HazardPropagationResponse(BaseModel):
    """Response for POST /hazard/propagate.

    Matches the API spec response structure with extensions for
    detailed zone states and paths.
    """

    success: bool = Field(True, description="Operation success flag")
    propagation_id: str = Field(
        ..., description="Unique ID for this simulation",
    )
    hazard_type: str = Field(..., description="Type of hazard simulated")
    origin_zone: str = Field(..., description="Origin zone ID")
    propagation_level: str = Field(
        ...,
        description="Overall severity: CONTAINED/SPREADING/CRITICAL/EMERGENCY",
    )
    affected_zones: List[str] = Field(
        default_factory=list,
        description="List of affected zone IDs",
    )
    affected_workers: List[str] = Field(
        default_factory=list,
        description="Worker IDs in the impact area",
    )
    impact_radius_meters: float = Field(
        0.0, ge=0.0,
        description="Estimated radius of impact in meters",
    )
    time_to_critical_minutes: float = Field(
        0.0, ge=0.0,
        description="Estimated time until critical threshold (minutes)",
    )
    recommended_action: str = Field(
        "", description="Human-readable recommended response",
    )
    zone_details: List[ZoneRiskStateResponse] = Field(
        default_factory=list,
        description="Detailed risk state per zone",
    )
    propagation_paths: List[HazardPathResponse] = Field(
        default_factory=list,
        description="Detailed propagation paths (if requested)",
    )
    total_workers_at_risk: int = Field(
        0, ge=0,
        description="Total workers in affected zones",
    )
    created_at: Optional[datetime] = Field(
        None, description="When the simulation was run",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "propagation_id": "abc-123",
                    "hazard_type": "GAS_LEAK",
                    "origin_zone": "ZONE_A",
                    "propagation_level": "SPREADING",
                    "affected_zones": ["ZONE_A", "ZONE_B"],
                    "affected_workers": ["W001", "W002", "W010"],
                    "impact_radius_meters": 75,
                    "time_to_critical_minutes": 15,
                    "recommended_action": (
                        "Evacuate Zone A and restrict access to Zone B"
                    ),
                }
            ]
        }
    }
