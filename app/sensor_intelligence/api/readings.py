"""Reading endpoints — sensor data ingestion and retrieval.

Implements:
  - POST /readings/ingest       — single reading ingestion
  - POST /readings/ingest/batch — batch ingestion
  - GET  /readings/latest/{sensor_id}  — latest reading
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.dependencies import get_reading_service
from app.sensor_intelligence.schemas.reading_schemas import (
    BatchReadingCreateRequest,
    BatchReadingResponse,
    ReadingCreateRequest,
    ReadingResponse,
    SingleReadingResponse,
)
from app.sensor_intelligence.services.reading_service import ReadingService

router = APIRouter(prefix="/readings", tags=["Readings"])


@router.post(
    "/ingest",
    response_model=SingleReadingResponse,
    status_code=201,
    summary="Ingest a single sensor reading",
    description="Validates the reading against business rules and persists it.",
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
    "The entire batch is rejected if any single reading fails.",
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


@router.get(
    "/latest/{sensor_id}",
    response_model=SingleReadingResponse,
    summary="Get the latest reading for a sensor",
)
async def get_latest_reading(
    sensor_id: str,
    service: ReadingService = Depends(get_reading_service),
) -> SingleReadingResponse:
    reading = await service.get_latest_reading(sensor_id)
    if reading is None:
        from app.shared.exceptions.domain_exceptions import ResourceNotFoundError
        raise ResourceNotFoundError(
            resource="Reading", identifier=f"latest for {sensor_id}"
        )
    return SingleReadingResponse(
        success=True,
        reading=ReadingResponse.model_validate(reading),
    )
