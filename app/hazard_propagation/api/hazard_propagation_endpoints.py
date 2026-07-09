"""Hazard Propagation API endpoints.

Implements:
  - POST /hazard/propagate               — trigger hazard propagation simulation
  - GET  /hazard/propagation/{id}         — get propagation result by ID
  - GET  /hazard/propagation/history      — propagation history
  - GET  /hazard/affected-zones/{zone_id} — get affected zones from a zone
  - GET  /hazard/paths/{zone_id}          — get propagation paths from a zone

Follows the API specification §21 and the same patterns as
CompoundRiskEndpoints and RiskPredictionEndpoints.
All business logic is delegated to HazardPropagationService.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.dependencies import get_hazard_propagation_service
from app.hazard_propagation.domain.exceptions import (
    HazardPropagationError,
    InvalidHazardError,
    PropagationSimulationError,
    ZoneNotFoundError,
)
from app.hazard_propagation.schemas.hazard_schemas import (
    HazardPathResponse,
    HazardPropagationRequest,
    HazardPropagationResponse,
    ZoneRiskStateResponse,
)
from app.hazard_propagation.services.hazard_propagation_service import (
    HazardPropagationService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hazard", tags=["Hazard Propagation Engine"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Additional response schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PropagationHistoryResponse(BaseModel):
    """Paginated propagation history response."""

    success: bool = True
    propagations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of propagation simulation results",
    )
    total: int = Field(0, description="Total matching results")
    offset: int = Field(0, description="Pagination offset")
    limit: int = Field(50, description="Page size")


class AffectedZonesResponse(BaseModel):
    """Response for affected zones query."""

    success: bool = True
    origin_zone: str = Field(..., description="Starting zone")
    affected_zones: Dict[str, int] = Field(
        default_factory=dict,
        description="Reachable zones with hop distances",
    )
    max_hops: int = Field(2, description="Max hops queried")


class PropagationPathsResponse(BaseModel):
    """Response for propagation paths query."""

    success: bool = True
    origin_zone: str = Field(..., description="Starting zone")
    paths: List[List[str]] = Field(
        default_factory=list,
        description="All propagation paths as lists of zone IDs",
    )
    max_depth: int = Field(3, description="Max depth queried")


class ZoneRiskAssessmentResponse(BaseModel):
    """Response for zone risk assessment."""

    success: bool = True
    assessment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Zone risk assessment details",
    )


class GraphStatsResponse(BaseModel):
    """Response for graph statistics."""

    success: bool = True
    stats: Dict[str, int] = Field(
        default_factory=dict,
        description="Graph node and edge counts",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST /hazard/propagate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post(
    "/propagate",
    response_model=HazardPropagationResponse,
    status_code=201,
    summary="Trigger hazard propagation simulation",
    description=(
        "Simulate hazard spread through the facility graph. "
        "Calculates affected zones, equipment, workers at risk, "
        "impact radius, and recommended actions."
    ),
)
async def trigger_propagation(
    request: HazardPropagationRequest,
    service: HazardPropagationService = Depends(
        get_hazard_propagation_service,
    ),
) -> HazardPropagationResponse:
    """Trigger a hazard propagation simulation."""
    try:
        # Map severity to compound_risk_score
        severity_scores = {
            "LOW": 25.0, "MEDIUM": 50.0,
            "HIGH": 75.0, "CRITICAL": 95.0,
        }
        compound_score = severity_scores.get(
            (request.severity or "HIGH").upper(), 75.0,
        )

        result = await service.propagate_hazard(
            hazard_type=request.hazard_type,
            origin_zone=request.origin_zone,
            compound_risk_score=compound_score,
            max_depth=request.max_propagation_depth,
        )

        pr = result.propagation_result
        return HazardPropagationResponse(
            success=True,
            propagation_id=pr.propagation_id,
            hazard_type=pr.hazard_type,
            origin_zone=pr.origin_zone,
            propagation_level=pr.propagation_level.value,
            affected_zones=pr.affected_zone_ids,
            affected_workers=pr.affected_workers,
            impact_radius_meters=pr.impact_radius_meters,
            time_to_critical_minutes=pr.time_to_critical_minutes,
            recommended_action=pr.recommended_action,
            zone_details=[
                ZoneRiskStateResponse(
                    zone_id=z.zone_id,
                    zone_name=z.zone_name or "",
                    risk_level=z.risk_level.value,
                    risk_score=z.risk_score,
                    is_origin=z.is_origin,
                    is_affected=z.is_affected,
                    arrival_time_minutes=z.arrival_time_minutes,
                    propagation_probability=z.propagation_probability,
                    worker_count=z.worker_count,
                    equipment_count=z.equipment_count,
                )
                for z in pr.affected_zones
            ] if request.include_paths else [],
            propagation_paths=[
                HazardPathResponse(
                    from_zone=p.from_zone,
                    to_zone=p.to_zone,
                    probability=p.probability,
                    estimated_time_minutes=p.estimated_time_minutes,
                    path_type=p.path_type or "CONNECTED_TO",
                    is_blocked=p.blocked,
                )
                for p in pr.propagation_paths
            ] if request.include_paths else [],
            total_workers_at_risk=pr.total_workers_at_risk,
            created_at=datetime.now(timezone.utc),
        )

    except InvalidHazardError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ZoneNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PropagationSimulationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HazardPropagationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST /hazard/simulate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post(
    "/simulate",
    response_model=HazardPropagationResponse,
    status_code=200,
    summary="Simulate hazard propagation (dry-run)",
    description=(
        "Run a propagation simulation without persisting results "
        "or publishing Kafka events. Useful for what-if analysis."
    ),
)
async def simulate_propagation(
    request: HazardPropagationRequest,
    service: HazardPropagationService = Depends(
        get_hazard_propagation_service,
    ),
) -> HazardPropagationResponse:
    """Simulate propagation without side effects."""
    try:
        severity_scores = {
            "LOW": 25.0, "MEDIUM": 50.0,
            "HIGH": 75.0, "CRITICAL": 95.0,
        }
        compound_score = severity_scores.get(
            (request.severity or "HIGH").upper(), 75.0,
        )

        result_dict = await service.simulate(
            hazard_type=request.hazard_type,
            origin_zone=request.origin_zone,
            compound_risk_score=compound_score,
            max_depth=request.max_propagation_depth,
        )

        return HazardPropagationResponse(
            success=True,
            propagation_id=result_dict["propagation_id"],
            hazard_type=result_dict["hazard_type"],
            origin_zone=result_dict["origin_zone"],
            propagation_level=result_dict["propagation_level"],
            affected_zones=result_dict["affected_zones"],
            impact_radius_meters=result_dict["impact_radius_meters"],
            time_to_critical_minutes=result_dict["time_to_critical_minutes"],
            recommended_action=result_dict["recommended_action"],
            total_workers_at_risk=result_dict["total_workers_at_risk"],
        )

    except InvalidHazardError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ZoneNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PropagationSimulationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HazardPropagationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /hazard/affected-zones/{zone_id}
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/affected-zones/{zone_id}",
    response_model=AffectedZonesResponse,
    summary="Get potentially affected zones",
    description=(
        "Get all zones reachable from the specified zone within a "
        "given number of hops. Useful for assessing blast radius."
    ),
)
async def get_affected_zones(
    zone_id: str,
    max_hops: int = Query(2, ge=1, le=10, description="Max hops"),
    service: HazardPropagationService = Depends(
        get_hazard_propagation_service,
    ),
) -> AffectedZonesResponse:
    """Get zones reachable from origin within max_hops."""
    try:
        neighbors = await service.get_zone_neighbors(zone_id, max_hops)
        return AffectedZonesResponse(
            success=True,
            origin_zone=zone_id,
            affected_zones=neighbors,
            max_hops=max_hops,
        )
    except ZoneNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /hazard/paths/{zone_id}
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/paths/{zone_id}",
    response_model=PropagationPathsResponse,
    summary="Get propagation paths from a zone",
    description=(
        "Find all possible hazard propagation paths originating "
        "from the specified zone up to the given depth."
    ),
)
async def get_propagation_paths(
    zone_id: str,
    max_depth: int = Query(3, ge=1, le=10, description="Max depth"),
    service: HazardPropagationService = Depends(
        get_hazard_propagation_service,
    ),
) -> PropagationPathsResponse:
    """Get all propagation paths from origin zone."""
    try:
        paths = await service.get_hazard_paths(zone_id, max_depth)
        return PropagationPathsResponse(
            success=True,
            origin_zone=zone_id,
            paths=paths,
            max_depth=max_depth,
        )
    except ZoneNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /hazard/zone/{zone_id}/risk
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/zone/{zone_id}/risk",
    response_model=ZoneRiskAssessmentResponse,
    summary="Get zone risk assessment",
    description=(
        "Get the current risk assessment for a specific zone, "
        "including equipment count, sensor count, and connectivity."
    ),
)
async def get_zone_risk(
    zone_id: str,
    service: HazardPropagationService = Depends(
        get_hazard_propagation_service,
    ),
) -> ZoneRiskAssessmentResponse:
    """Get risk assessment for a zone."""
    try:
        assessment = await service.get_zone_risk_assessment(zone_id)
        return ZoneRiskAssessmentResponse(
            success=True,
            assessment=assessment,
        )
    except ZoneNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /hazard/graph/stats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/graph/stats",
    response_model=GraphStatsResponse,
    summary="Get facility graph statistics",
    description=(
        "Return counts of zones, equipment, sensors, and edges "
        "in the facility graph."
    ),
)
async def get_graph_stats(
    service: HazardPropagationService = Depends(
        get_hazard_propagation_service,
    ),
) -> GraphStatsResponse:
    """Get graph statistics."""
    stats = await service.get_graph_stats()
    return GraphStatsResponse(success=True, stats=stats)
