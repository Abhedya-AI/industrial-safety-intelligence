"""Digital Twin API endpoints.

Implements:
  - GET /twin/facility          — facility-wide aggregated state
  - GET /twin/zones             — all zone states
  - GET /twin/zones/{zone_id}   — single zone detail
  - GET /twin/heatmap           — risk heatmap for all zones

Follows the same patterns as CompoundRiskEndpoints and
HazardPropagationEndpoints. All business logic is delegated
to TwinStateManager.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_digital_twin_service
from app.digital_twin.domain.exceptions import ZoneNotFoundInTwinError
from app.digital_twin.schemas.twin_schemas import (
    ActiveHazardSchema,
    EquipmentStateSchema,
    FacilityStateResponse,
    HeatmapEntry,
    HeatmapResponse,
    SensorReadingSchema,
    ZoneDetailResponse,
    ZoneStateSchema,
    ZonesListResponse,
)
from app.digital_twin.services.twin_state_manager import TwinStateManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/twin", tags=["Digital Twin"])


def _zone_to_schema(zone) -> ZoneStateSchema:
    """Convert a ZoneState domain entity to a Pydantic schema."""
    return ZoneStateSchema(
        zone_id=zone.zone_id,
        zone_name=zone.zone_name,
        sensor_health=round(zone.sensor_health, 2),
        anomaly_count=zone.anomaly_count,
        sensor_count=zone.sensor_count,
        latest_sensor_readings={
            sid: SensorReadingSchema(
                sensor_id=r.sensor_id,
                sensor_type=r.sensor_type,
                value=r.value,
                unit=r.unit,
                anomaly_score=r.anomaly_score,
                is_anomalous=r.is_anomalous,
                health_score=r.health_score,
                status=r.status,
                last_updated=r.last_updated,
            )
            for sid, r in zone.latest_sensor_readings.items()
        },
        predicted_risk_score=round(zone.predicted_risk_score, 2),
        risk_level=zone.risk_level,
        accident_probability=round(zone.accident_probability, 4),
        compound_risk_score=round(zone.compound_risk_score, 2),
        compound_risk_level=zone.compound_risk_level,
        compound_risk_confidence=round(zone.compound_risk_confidence, 4),
        contributing_factors=zone.contributing_factors,
        active_hazards=[
            ActiveHazardSchema(
                hazard_id=h.hazard_id,
                hazard_type=h.hazard_type,
                severity=h.severity,
                origin_zone=h.origin_zone,
                propagation_level=h.propagation_level,
                affected_zones=h.affected_zones,
                detected_at=h.detected_at,
            )
            for h in zone.active_hazards
        ],
        affected_neighbors=zone.affected_neighbors,
        workers_at_risk=zone.workers_at_risk,
        worker_capacity=zone.worker_capacity,
        current_worker_count=zone.current_worker_count,
        equipment=[
            EquipmentStateSchema(
                equipment_id=eq.equipment_id,
                equipment_type=eq.equipment_type,
                zone_id=eq.zone_id,
                operational_status=eq.operational_status,
                health_score=eq.health_score,
                risk_score=eq.risk_score,
                sensor_count=eq.sensor_count,
                last_updated=eq.last_updated,
            )
            for eq in zone.equipment
        ],
        connected_zones=zone.connected_zones,
        overall_risk_score=round(zone.overall_risk_score, 2),
        heatmap_color=zone.heatmap_color,
        is_critical=zone.is_critical,
        last_updated=zone.last_updated,
        event_count=zone.event_count,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /twin/facility
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/facility",
    response_model=FacilityStateResponse,
    summary="Get facility-wide state",
    description="Returns aggregated facility health, zone counts, "
    "hazard status, and worker risk summary.",
)
async def get_facility_state(
    twin: TwinStateManager = Depends(get_digital_twin_service),
) -> FacilityStateResponse:
    """Get the complete facility-wide aggregated state."""
    state = twin.get_facility_state()
    return FacilityStateResponse(
        success=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        facility_health=state.facility_health,
        total_zones=state.total_zones,
        active_hazards=state.active_hazards,
        critical_zones=state.critical_zones,
        workers_at_risk=state.workers_at_risk,
        total_workers=state.total_workers,
        total_equipment=state.total_equipment,
        total_sensors=state.total_sensors,
        total_anomalies=state.total_anomalies,
        average_risk_score=state.average_risk_score,
        max_risk_score=state.max_risk_score,
        events_processed=state.events_processed,
        zone_ids=state.zone_ids,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /twin/zones
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/zones",
    response_model=ZonesListResponse,
    summary="Get all zone states",
    description="Returns live state for every zone in the facility.",
)
async def get_all_zones(
    twin: TwinStateManager = Depends(get_digital_twin_service),
) -> ZonesListResponse:
    """Get all zone states."""
    zones = twin.get_all_zones()
    zone_schemas = [_zone_to_schema(z) for z in zones]
    return ZonesListResponse(
        success=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total=len(zone_schemas),
        zones=zone_schemas,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /twin/zones/{zone_id}
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/zones/{zone_id}",
    response_model=ZoneDetailResponse,
    summary="Get zone detail",
    description="Returns the complete live state for a single zone.",
)
async def get_zone_detail(
    zone_id: str,
    twin: TwinStateManager = Depends(get_digital_twin_service),
) -> ZoneDetailResponse:
    """Get detailed state for a single zone."""
    try:
        zone = twin.get_zone(zone_id)
    except ZoneNotFoundInTwinError:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": "RESOURCE_NOT_FOUND",
                "message": f"Zone '{zone_id}' not found in digital twin",
            },
        )

    return ZoneDetailResponse(
        success=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        zone=_zone_to_schema(zone),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /twin/heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/heatmap",
    response_model=HeatmapResponse,
    summary="Get risk heatmap",
    description="Returns color-coded risk data for all zones. "
    "Colors: 0-25=green, 26-50=yellow, 51-75=orange, 76-100=red.",
)
async def get_heatmap(
    twin: TwinStateManager = Depends(get_digital_twin_service),
) -> HeatmapResponse:
    """Get risk heatmap data for all zones."""
    heatmap_data = twin.get_heatmap()
    entries = [
        HeatmapEntry(
            zone_id=h["zone_id"],
            zone_name=h["zone_name"],
            risk_score=h["risk_score"],
            risk_level=h["risk_level"],
            color=h["color"],
        )
        for h in heatmap_data
    ]
    return HeatmapResponse(
        success=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total=len(entries),
        heatmap=entries,
    )
