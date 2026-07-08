"""Pydantic schemas for Compound Risk Intelligence endpoints.

Aligned with:
  - PS-1 API Specification: POST /risk/compound-analysis, GET /risk/history
  - PS-1 Common Domain Names v2.0 (enum values, naming conventions)

All enum values use EXACT values from the domain conventions document:
  RiskLevel: LOW | MEDIUM | HIGH | CRITICAL
  ShiftType: MORNING | AFTERNOON | NIGHT
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.compound_risk.domain.value_objects import (
    CompoundRiskStatus,
    RiskLevel,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Nested schemas (matching API spec structure)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ScenarioInput(BaseModel):
    """Scenario input for compound risk analysis.

    Matches the ``scenario`` object in POST /risk/compound-analysis.
    Field names use snake_case with units as per PS-1 conventions.
    """

    gas_level_ppm: float = Field(
        0.0, ge=0, description="Gas level in parts per million",
    )
    temperature_celsius: float = Field(
        0.0, description="Temperature in degrees Celsius",
    )
    pressure_bar: float = Field(
        0.0, ge=0, description="Pressure in bar",
    )
    humidity_percent: float = Field(
        0.0, ge=0, le=100, description="Humidity percentage",
    )
    vibration_level: float = Field(
        0.0, ge=0, description="Vibration level in m/s²",
    )
    maintenance_active: bool = Field(
        False, description="Whether maintenance activity is currently active",
    )
    worker_count: int = Field(
        0, ge=0, description="Number of workers in the zone",
    )
    permit_type: Optional[str] = Field(
        None, description="Active permit type (e.g. HOT_WORK, CONFINED_SPACE)",
    )
    permit_active: bool = Field(
        False, description="Whether a work permit is currently active",
    )
    shift_type: Optional[str] = Field(
        None, description="Current shift: MORNING | AFTERNOON | NIGHT",
    )
    equipment_health: float = Field(
        1.0, ge=0, le=1, description="Equipment health score (0.0–1.0)",
    )

    # ── Enrichment from upstream modules ──
    anomaly_score: float = Field(
        0.0, ge=0, le=1, description="Anomaly score from anomaly detection",
    )
    accident_probability: float = Field(
        0.0, ge=0, le=1, description="Accident probability from risk prediction",
    )
    risk_score: float = Field(
        0.0, ge=0, le=100, description="Risk score from risk prediction (0–100)",
    )
    sensor_health_score: float = Field(
        100.0, ge=0, le=100, description="Sensor health score (0–100)",
    )


class ContributingFactor(BaseModel):
    """A contributing factor to the compound risk score."""

    factor: str = Field(..., description="Factor name")
    weight: float = Field(..., ge=0, le=1, description="Factor weight")
    current_value: str = Field(..., description="Current value as string")
    contribution: float = Field(
        ..., ge=0, le=1, description="Contribution to overall risk",
    )


class DangerousCombination(BaseModel):
    """A detected dangerous combination of conditions.

    Matches the ``dangerous_combinations`` array in the API response.
    """

    condition_1: str = Field(..., description="First condition")
    condition_2: str = Field(..., description="Second condition")
    condition_3: Optional[str] = Field(None, description="Third condition")
    risk_score: float = Field(
        ..., ge=0, le=1, description="Combined risk score",
    )
    severity: RiskLevel = Field(..., description="Severity level")
    historical_incidents: List[str] = Field(
        default_factory=list,
        description="Related incident IDs (e.g. INC-2024-005)",
    )
    probability_of_incident: float = Field(
        0.0, ge=0, le=1, description="Estimated probability of incident",
    )


class RecommendedAction(BaseModel):
    """A recommended action from the compound risk analysis.

    Matches the ``recommended_actions`` array in the API response.
    """

    priority: int = Field(..., ge=1, description="Action priority (1 = highest)")
    action: str = Field(..., description="Recommended action text")
    rationale: str = Field(..., description="Why this action is recommended")
    estimated_effect: str = Field(
        ..., description="Estimated risk reduction (e.g. 'Reduces risk by 45%')",
    )


class HistoricalContext(BaseModel):
    """Historical context for compound risk analysis."""

    similar_incidents_count: int = Field(
        0, ge=0, description="Number of similar past incidents",
    )
    most_severe_incident: Optional[str] = Field(
        None, description="Most severe incident ID (e.g. INC-2024-005)",
    )
    most_severe_outcome: Optional[str] = Field(
        None, description="Outcome description of most severe incident",
    )
    pattern: Optional[str] = Field(
        None, description="Identified pattern or trend",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Request schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CompoundRiskRequest(BaseModel):
    """Request body for POST /risk/compound-analysis.

    Matches the API specification exactly.
    """

    zone_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Zone ID (e.g. ZONE_A, ZONE_BOILER)",
    )
    scenario: ScenarioInput = Field(
        ..., description="Scenario parameters for analysis",
    )
    equipment_id: Optional[str] = Field(
        None, max_length=100,
        description="Optional equipment context (e.g. EQ001)",
    )
    include_historical: bool = Field(
        False, description="Include historical context in response",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Response schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CompoundRiskResponse(BaseModel):
    """Response for a compound risk analysis.

    Maps 1-to-1 with the CompoundRiskModel ORM fields.
    """

    id: str = Field(..., description="Analysis UUID")
    equipment_id: Optional[str] = Field(None, description="Equipment ID")
    zone_id: Optional[str] = Field(None, description="Zone ID")
    anomaly_score: float = Field(
        ..., ge=0, le=1, description="Input anomaly score",
    )
    accident_probability: float = Field(
        ..., ge=0, le=1, description="Input accident probability",
    )
    risk_score: float = Field(
        ..., ge=0, le=100, description="Input risk score (0–100)",
    )
    sensor_health_score: float = Field(
        ..., ge=0, le=100, description="Input sensor health score",
    )
    compound_risk_score: float = Field(
        ..., ge=0, le=1, description="Computed compound risk score (0.0–1.0)",
    )
    risk_level: RiskLevel = Field(
        ..., description="Classified risk level: LOW | MEDIUM | HIGH | CRITICAL",
    )
    confidence_score: float = Field(
        ..., ge=0, le=1, description="Analysis confidence",
    )
    contributing_factors: Optional[List[ContributingFactor]] = Field(
        None, description="Contributing factors",
    )
    recommendation: Optional[str] = Field(
        None, description="Text recommendation or JSON actions",
    )
    created_at: datetime = Field(
        ..., description="Record creation timestamp (ISO 8601 UTC)",
    )

    @field_validator("contributing_factors", mode="before")
    @classmethod
    def _parse_contributing_factors(cls, v):
        """Deserialise JSON string stored in ORM Text column."""
        if isinstance(v, str):
            return json.loads(v)
        return v

    model_config = {"from_attributes": True}


class CompoundRiskHistoryResponse(BaseModel):
    """Paginated response for compound risk analysis history."""

    success: bool = True
    predictions: List[CompoundRiskResponse] = Field(
        default_factory=list, description="List of compound risk analyses",
    )
    total: int = Field(0, ge=0, description="Total matching records")
    offset: int = Field(0, ge=0, description="Pagination offset")
    limit: int = Field(50, ge=1, description="Page size")
