"""Reading endpoints — sensor data ingestion and retrieval.

Implements:
  - POST /readings/ingest              — single reading ingestion
  - POST /readings/ingest/batch        — batch ingestion (all-or-nothing)
  - GET  /readings/latest/{sensor_id}  — latest reading for a sensor
  - GET  /readings/{sensor_id}         — historical readings for a sensor
  - GET  /readings/{sensor_id}/stats   — aggregated statistics
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.core.dependencies import get_reading_service
from app.sensor_intelligence.schemas.reading_schemas import (
    BatchReadingCreateRequest,
    BatchReadingResponse,
    ReadingCreateRequest,
    ReadingResponse,
    ReadingStatsResponse,
    SingleReadingResponse,
)
from app.sensor_intelligence.services.reading_service import ReadingService
from app.shared.exceptions.domain_exceptions import ResourceNotFoundError

router = APIRouter(prefix="/readings", tags=["Readings"])


# ──────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=SingleReadingResponse,
    status_code=201,
    summary="Ingest a single sensor reading",
    description="Validates the reading against all business rules "
    "(sensor existence, OFFLINE status, value range, timestamp, "
    "duplicates) and persists it.",
)
async def ingest_reading(
    request: ReadingCreateRequest,
    service: ReadingService = Depends(get_reading_service),
) -> SingleReadingResponse:
    reading = await service.ingest_reading(request)
    return SingleReadingResponse(
        success=True,
        reading=ReadingResponse.model_validate(reading),
    )


@router.post(
    "/ingest/batch",
    response_model=BatchReadingResponse,
    status_code=201,
    summary="Ingest a batch of sensor readings",
    description="Validates all readings and persists them atomically. "
    "The entire batch is rejected if any single reading fails validation.",
)
async def ingest_batch(
    request: BatchReadingCreateRequest,
    service: ReadingService = Depends(get_reading_service),
) -> BatchReadingResponse:
    readings = await service.ingest_batch(request.readings)
    return BatchReadingResponse(
        success=True,
        ingested=len(readings),
        readings=[ReadingResponse.model_validate(r) for r in readings],
    )


# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────


@router.get(
    "/latest/{sensor_id}",
    response_model=SingleReadingResponse,
    summary="Get the latest reading for a sensor",
    description="Returns the most recent reading for the given sensor ID. "
    "Returns 404 if no readings exist.",
)
async def get_latest_reading(
    sensor_id: str,
    service: ReadingService = Depends(get_reading_service),
) -> SingleReadingResponse:
    reading = await service.get_latest_reading(sensor_id)
    if reading is None:
        raise ResourceNotFoundError(
            resource="Reading", identifier=f"latest for {sensor_id}"
        )
    return SingleReadingResponse(
        success=True,
        reading=ReadingResponse.model_validate(reading),
    )


@router.get(
    "/{sensor_id}",
    response_model=dict,
    summary="Get sensor readings history",
    description="Returns historical readings for a sensor within the "
    "specified time range.",
)
async def get_sensor_readings(
    sensor_id: str,
    start: Optional[datetime] = Query(
        None,
        description="Start of time range (ISO 8601). Defaults to 24h ago.",
    ),
    end: Optional[datetime] = Query(
        None,
        description="End of time range (ISO 8601). Defaults to now.",
    ),
    limit: int = Query(1000, ge=1, le=5000, description="Max readings returned"),
    service: ReadingService = Depends(get_reading_service),
) -> dict:
    now = datetime.now(timezone.utc)
    from_dt = start if start else now - timedelta(hours=24)
    to_dt = end if end else now

    readings = await service.get_readings_range(sensor_id, from_dt, to_dt, limit)
    return {
        "success": True,
        "sensor_id": sensor_id,
        "count": len(readings),
        "readings": [ReadingResponse.model_validate(r) for r in readings],
    }


@router.get(
    "/{sensor_id}/stats",
    response_model=dict,
    summary="Get aggregated reading statistics",
    description="Returns aggregated statistics (mean, std dev, min, max, count) "
    "for a sensor's readings within the specified time range.",
)
async def get_reading_stats(
    sensor_id: str,
    start: Optional[datetime] = Query(
        None,
        description="Start of time range (ISO 8601). Defaults to 24h ago.",
    ),
    end: Optional[datetime] = Query(
        None,
        description="End of time range (ISO 8601). Defaults to now.",
    ),
    service: ReadingService = Depends(get_reading_service),
) -> dict:
    now = datetime.now(timezone.utc)
    from_dt = start if start else now - timedelta(hours=24)
    to_dt = end if end else now

    stats = await service.get_reading_stats(sensor_id, from_dt, to_dt)
    if stats is None:
        return {
            "success": True,
            "sensor_id": sensor_id,
            "statistics": None,
            "message": "No readings found in the specified time range",
        }
    return {
        "success": True,
        "sensor_id": sensor_id,
        "statistics": ReadingStatsResponse(
            sensor_id=stats.sensor_id,
            mean=stats.mean,
            std_dev=stats.std_dev,
            min_value=stats.min_value,
            max_value=stats.max_value,
            count=stats.count,
            window_start=stats.window_start,
            window_end=stats.window_end,
        ).model_dump(),
    }
