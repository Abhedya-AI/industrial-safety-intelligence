"""Sensor endpoints — aligned to API specification.

Implements:
  - CRUD: POST/GET/PUT/DELETE /sensors
  - Spec Endpoint 6: GET /sensors/current
  - Spec Endpoint 7: GET /sensors/{sensor_id}/history
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.core.dependencies import get_sensor_service
from app.sensor_intelligence.domain.value_objects.sensor_status import SensorStatus
from app.sensor_intelligence.domain.value_objects.sensor_type import SensorType
from app.sensor_intelligence.schemas.sensor_schemas import (
    CurrentSensorsResponse,
    SensorCreateRequest,
    SensorHistoryResponse,
    SensorListResponse,
    SensorResponse,
    SensorUpdateRequest,
)
from app.sensor_intelligence.services.sensor_service import SensorService

router = APIRouter(prefix="/sensors", tags=["Sensors"])


# ──────────────────────────────────────────────
# Spec Endpoint 6: GET /sensors/current
# ──────────────────────────────────────────────


@router.get(
    "/current",
    response_model=CurrentSensorsResponse,
    summary="Get current sensor readings",
    description="Returns all sensors with their current status, readings, "
    "thresholds, trends, anomalies, and health information.",
)
async def get_current_sensors(
    sensor_type: Optional[SensorType] = Query(
        None, description="Filter by sensor type"
    ),
    zone_id: Optional[str] = Query(None, alias="zone_id", description="Filter by zone"),
    status: Optional[SensorStatus] = Query(None, description="Filter by status"),
    service: SensorService = Depends(get_sensor_service),
) -> CurrentSensorsResponse:
    return await service.get_current_sensors(
        sensor_type=sensor_type,
        status=status,
        location_zone=zone_id,
    )


# ──────────────────────────────────────────────
# Spec Endpoint 7: GET /sensors/{sensor_id}/history
# ──────────────────────────────────────────────


@router.get(
    "/{sensor_id}/history",
    response_model=SensorHistoryResponse,
    summary="Get sensor details and history",
    description="Returns sensor metadata, historical readings, statistics, "
    "detected anomalies, and forecast data.",
)
async def get_sensor_history(
    sensor_id: str,
    time_range: str = Query("24h", description="1h | 6h | 24h | 7d"),
    granularity: str = Query("auto", description="1m | 5m | 15m | 1h | auto"),
    service: SensorService = Depends(get_sensor_service),
) -> SensorHistoryResponse:
    return await service.get_sensor_history(
        sensor_id=sensor_id,
        time_range=time_range,
        granularity=granularity,
    )


# ──────────────────────────────────────────────
# CRUD Endpoints
# ──────────────────────────────────────────────


@router.post(
    "",
    response_model=SensorResponse,
    status_code=201,
    summary="Register a new sensor",
)
async def create_sensor(
    request: SensorCreateRequest,
    service: SensorService = Depends(get_sensor_service),
) -> SensorResponse:
    sensor = await service.create_sensor(request)
    return SensorResponse.model_validate(sensor)


@router.get(
    "",
    response_model=SensorListResponse,
    summary="List all sensors",
)
async def list_sensors(
    sensor_type: Optional[SensorType] = Query(None),
    status: Optional[SensorStatus] = Query(None),
    zone_id: Optional[str] = Query(None, alias="zone_id"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    service: SensorService = Depends(get_sensor_service),
) -> SensorListResponse:
    items, total = await service.list_sensors(
        sensor_type=sensor_type,
        status=status,
        location_zone=zone_id,
        offset=offset,
        limit=limit,
    )
    return SensorListResponse(
        success=True,
        items=[SensorResponse.model_validate(s) for s in items],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get(
    "/{sensor_id}",
    response_model=SensorResponse,
    summary="Get sensor by ID",
)
async def get_sensor(
    sensor_id: str,
    service: SensorService = Depends(get_sensor_service),
) -> SensorResponse:
    sensor = await service.get_sensor(sensor_id)
    return SensorResponse.model_validate(sensor)


@router.put(
    "/{sensor_id}",
    response_model=SensorResponse,
    summary="Update sensor metadata",
)
async def update_sensor(
    sensor_id: str,
    request: SensorUpdateRequest,
    service: SensorService = Depends(get_sensor_service),
) -> SensorResponse:
    sensor = await service.update_sensor(sensor_id, request)
    return SensorResponse.model_validate(sensor)


@router.delete(
    "/{sensor_id}",
    status_code=204,
    response_model=None,
    summary="Delete a sensor",
)
async def delete_sensor(
    sensor_id: str,
    service: SensorService = Depends(get_sensor_service),
):
    await service.delete_sensor(sensor_id)
