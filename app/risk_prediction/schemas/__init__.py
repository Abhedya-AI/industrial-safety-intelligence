"""Pydantic schemas for Risk Prediction endpoints.

Aligned with:
  - API Specification: GET /risk/current, POST /risk/compound-analysis
  - Architecture: RiskPredictionService output format
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.risk_prediction.domain.value_objects import PredictionStatus, RiskLevel


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Request schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RiskPredictionRequest(BaseModel):
    """Request body for computing a risk prediction.

    The caller provides sensor/equipment context and the feature values
    needed by the ensemble model.
    """

    sensor_id: Optional[str] = Field(
        None, min_length=1, max_length=50,
        description="Sensor ID to compute risk for",
    )
    equipment_id: Optional[str] = Field(
        None, max_length=100,
        description="Equipment ID to scope the prediction",
    )
    zone_id: Optional[str] = Field(
        None, max_length=100,
        description="Zone ID for zone-level risk prediction",
    )
    features: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature name→value map for the model",
    )
    include_breakdown: bool = Field(
        False,
        description="Include per-factor risk breakdown in response",
    )
    include_explanation: bool = Field(
        False,
        description="Include human-readable risk explanation",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "sensor_id": "S001",
                "zone_id": "ZONE_A",
                "features": {
                    "gas_level": 120.0,
                    "pressure": 180.5,
                    "temperature": 95.0,
                    "humidity": 62.0,
                    "vibration": 3.2,
                    "maintenance_active": 1.0,
                    "worker_count": 12.0,
                },
                "include_breakdown": True,
                "include_explanation": True,
            }
        ]
    }}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Response schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RiskFactorBreakdown(BaseModel):
    """Per-factor risk scores (from API spec risk_breakdown)."""

    gas_risk: float = Field(0.0, ge=0, le=1, description="Gas concentration risk")
    temperature_risk: float = Field(0.0, ge=0, le=1, description="Temperature risk")
    pressure_risk: float = Field(0.0, ge=0, le=1, description="Pressure risk")
    worker_density_risk: float = Field(0.0, ge=0, le=1, description="Worker density risk")
    equipment_health_risk: float = Field(0.0, ge=0, le=1, description="Equipment health risk")
    permit_risk: float = Field(0.0, ge=0, le=1, description="Active permit risk")
    maintenance_risk: float = Field(0.0, ge=0, le=1, description="Maintenance status risk")


class ContributingFactor(BaseModel):
    """A top contributing factor to the risk score (from API spec)."""

    factor: str = Field(..., description="Factor name")
    weight: float = Field(..., ge=0, le=1, description="Feature weight in model")
    current_value: str = Field(..., description="Current observed value (stringified)")
    contribution: float = Field(..., ge=0, le=1, description="Contribution to risk score")


class ForecastPoint(BaseModel):
    """A single point in the risk forecast timeline (from API spec)."""

    time: datetime = Field(..., description="Forecast timestamp")
    predicted_risk: int = Field(..., ge=0, le=100, description="Predicted risk score")
    confidence: float = Field(..., ge=0, le=1, description="Forecast confidence")


class RiskPredictionResponse(BaseModel):
    """Response for a single risk prediction computation."""

    id: str = Field(..., description="Prediction UUID")
    sensor_id: Optional[str] = Field(None, description="Source sensor ID")
    equipment_id: Optional[str] = Field(None, description="Equipment ID")
    zone_id: Optional[str] = Field(None, description="Zone ID")
    prediction_timestamp: datetime = Field(..., description="When prediction was computed")
    accident_probability: float = Field(
        ..., ge=0, le=1, description="Raw accident probability",
    )
    predicted_risk_score: int = Field(
        ..., ge=0, le=100, description="Normalised risk score",
    )
    risk_level: RiskLevel = Field(..., description="Classified risk level")
    confidence_score: float = Field(
        ..., ge=0, le=1, description="Model confidence",
    )
    model_name: str = Field(..., description="Model that produced this prediction")
    model_version: str = Field(..., description="Model version")
    status: PredictionStatus = Field(
        PredictionStatus.COMPLETED, description="Prediction status",
    )
    risk_breakdown: Optional[RiskFactorBreakdown] = Field(
        None, description="Per-factor risk scores (if requested)",
    )
    top_contributing_factors: Optional[List[ContributingFactor]] = Field(
        None, description="Top features contributing to risk",
    )
    explanation: Optional[str] = Field(
        None, description="Human-readable risk explanation",
    )
    created_at: datetime = Field(..., description="Record creation timestamp")

    @field_validator("top_contributing_factors", mode="before")
    @classmethod
    def _parse_contributing_factors(cls, v):
        """Deserialise JSON string stored in ORM Text column."""
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    @field_validator("risk_breakdown", mode="before")
    @classmethod
    def _parse_risk_breakdown(cls, v):
        """Deserialise JSON string stored in ORM Text column."""
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class SingleRiskPredictionResponse(BaseModel):
    """Wrapper for a single prediction response."""

    success: bool = True
    prediction: RiskPredictionResponse


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Current Risk Status (API Spec Endpoint 3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ZoneRiskSummary(BaseModel):
    """Per-zone risk summary (from API spec by_zone array)."""

    zone_id: str
    zone_name: str = ""
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    workers_present: int = 0
    equipment_count: int = 0
    active_hazards: List[str] = Field(default_factory=list)
    active_permits: List[str] = Field(default_factory=list)


class CurrentRiskSummary(BaseModel):
    """Top-level current risk (from API spec current_risk object)."""

    overall_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    trend: str = Field("stable", description="stable | increasing | decreasing")
    trend_direction: str = Field("flat", description="up | down | flat")
    last_update: datetime


class CurrentRiskStatusResponse(BaseModel):
    """Full response for GET /risk/current (API Spec Endpoint 3)."""

    success: bool = True
    timestamp: datetime
    current_risk: CurrentRiskSummary
    by_zone: List[ZoneRiskSummary] = Field(default_factory=list)
    risk_breakdown: Optional[RiskFactorBreakdown] = None
    forecast: Optional[Dict[str, List[ForecastPoint]]] = None
    top_contributing_factors: Optional[List[ContributingFactor]] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# History
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RiskPredictionHistoryResponse(BaseModel):
    """Paginated risk prediction history."""

    success: bool = True
    predictions: List[RiskPredictionResponse]
    total: int = Field(..., description="Total matching predictions")
    offset: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=200)
