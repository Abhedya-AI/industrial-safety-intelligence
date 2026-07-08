"""Compound Risk Intelligence API endpoints.

Implements:
  - POST /risk/compound-analysis       — compute compound risk analysis
  - GET  /risk/compound-analysis/latest — latest compound risk result
  - GET  /risk/compound-analysis/history — paginated analysis history

Follows the same patterns as the Risk Prediction API.
All business logic is delegated to CompoundRiskService.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.dependencies import get_compound_risk_service
from app.compound_risk.domain.exceptions import (
    CompoundRiskAnalysisFailedError,
    CompoundRiskError,
    InsufficientScenarioDataError,
)
from app.compound_risk.schemas import (
    CompoundRiskHistoryResponse,
    CompoundRiskRequest,
    CompoundRiskResponse,
)
from app.compound_risk.services.compound_risk_facade import CompoundRiskService
from app.compound_risk.services.compound_risk_service import CompoundRiskInput

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["Compound Risk Intelligence"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Response wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from pydantic import BaseModel, Field


class SingleCompoundRiskResponse(BaseModel):
    """Wrapper for a single compound risk analysis result."""

    success: bool = True
    compound_risk_analysis: CompoundRiskResponse = Field(
        ..., description="The compound risk analysis result",
    )


# ──────────────────────────────────────────────
# POST /risk/compound-analysis
# ──────────────────────────────────────────────


@router.post(
    "/compound-analysis",
    response_model=SingleCompoundRiskResponse,
    status_code=201,
    summary="Calculate compound risk",
    description=(
        "Analyze a specific compound risk scenario by combining anomaly "
        "detection, risk prediction, sensor health, and operational context. "
        "Returns a compound risk score, risk level, contributing factors, "
        "and recommendations."
    ),
)
async def calculate_compound_risk(
    request: CompoundRiskRequest,
    service: CompoundRiskService = Depends(get_compound_risk_service),
) -> SingleCompoundRiskResponse:
    """Calculate compound risk from a scenario input."""
    try:
        # Map scenario to CompoundRiskInput
        scenario = request.scenario
        inp = CompoundRiskInput(
            isolation_forest_score=scenario.anomaly_score,
            autoencoder_score=scenario.anomaly_score * 0.8,
            accident_probability=scenario.accident_probability,
            risk_score=scenario.risk_score,
            sensor_health_score=scenario.sensor_health_score,
            active_alert_count=1 if scenario.maintenance_active else 0,
            alert_severity_max=1.0 - scenario.equipment_health,
            threshold_violation_count=(
                (1 if scenario.gas_level_ppm > 100 else 0)
                + (1 if scenario.temperature_celsius > 60 else 0)
                + (1 if scenario.pressure_bar > 5 else 0)
            ),
            equipment_id=request.equipment_id,
            zone_id=request.zone_id,
        )

        # Build sensor facts for rule engine
        sensor_facts = {
            "gas_level_ppm": scenario.gas_level_ppm,
            "temperature_celsius": scenario.temperature_celsius,
            "pressure_bar": scenario.pressure_bar,
            "humidity_percent": scenario.humidity_percent,
            "vibration_level": scenario.vibration_level,
            "sensor_health_score": scenario.sensor_health_score,
            "equipment_health": scenario.equipment_health,
            "maintenance_active": scenario.maintenance_active,
            "worker_count": scenario.worker_count,
        }

        result = await service.analyze(
            inp, sensor_facts=sensor_facts,
        )

        return SingleCompoundRiskResponse(
            success=True,
            compound_risk_analysis=CompoundRiskResponse.model_validate(
                result.model,
            ),
        )

    except InsufficientScenarioDataError as exc:
        raise HTTPException(status_code=422, detail=exc.message) from exc
    except CompoundRiskAnalysisFailedError as exc:
        raise HTTPException(status_code=500, detail=exc.message) from exc
    except CompoundRiskError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc


# ──────────────────────────────────────────────
# GET /risk/compound-analysis/latest
# ──────────────────────────────────────────────


@router.get(
    "/compound-analysis/latest",
    response_model=SingleCompoundRiskResponse,
    summary="Get latest compound risk analysis",
    description=(
        "Returns the most recent compound risk analysis. "
        "Optionally scoped by zone or equipment."
    ),
)
async def get_latest_compound_risk(
    zone_id: Optional[str] = Query(None, description="Filter by zone ID"),
    equipment_id: Optional[str] = Query(None, description="Filter by equipment ID"),
    service: CompoundRiskService = Depends(get_compound_risk_service),
) -> SingleCompoundRiskResponse:
    """Get the most recent compound risk analysis."""
    model = await service.get_latest(
        zone_id=zone_id, equipment_id=equipment_id,
    )
    if model is None:
        raise HTTPException(
            status_code=404,
            detail="No compound risk analyses found for the given filters.",
        )
    return SingleCompoundRiskResponse(
        success=True,
        compound_risk_analysis=CompoundRiskResponse.model_validate(model),
    )


# ──────────────────────────────────────────────
# GET /risk/compound-analysis/history
# ──────────────────────────────────────────────


@router.get(
    "/compound-analysis/history",
    response_model=CompoundRiskHistoryResponse,
    summary="Get compound risk analysis history",
    description=(
        "Returns paginated history of compound risk analyses. "
        "Supports filtering by zone, equipment, and risk level."
    ),
)
async def get_compound_risk_history(
    zone_id: Optional[str] = Query(None, description="Filter by zone ID"),
    equipment_id: Optional[str] = Query(None, description="Filter by equipment ID"),
    risk_level: Optional[str] = Query(
        None,
        description="Filter by risk level (LOW, MEDIUM, HIGH, CRITICAL)",
    ),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, ge=1, le=200, description="Page size"),
    service: CompoundRiskService = Depends(get_compound_risk_service),
) -> CompoundRiskHistoryResponse:
    """Get paginated compound risk analysis history."""
    analyses = await service.get_history(
        zone_id=zone_id,
        equipment_id=equipment_id,
        risk_level=risk_level,
        offset=offset,
        limit=limit,
    )
    total = await service.count(
        zone_id=zone_id,
        equipment_id=equipment_id,
        risk_level=risk_level,
    )
    return CompoundRiskHistoryResponse(
        success=True,
        predictions=[
            CompoundRiskResponse.model_validate(a) for a in analyses
        ],
        total=total,
        offset=offset,
        limit=limit,
    )
