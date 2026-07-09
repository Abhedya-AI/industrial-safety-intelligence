"""Risk Prediction API endpoints.

Implements:
  - POST /risk/predict           — compute a risk prediction
  - GET  /risk/predictions       — prediction history (paginated)
  - GET  /risk/predictions/latest — latest prediction

Kafka events published after successful operations:
  - risk.assessment.generated   — after a prediction is computed
  - risk.score.updated          — when a risk score is persisted
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.dependencies import (
    get_risk_prediction_publisher,
    get_risk_prediction_service,
)
from app.risk_prediction.domain.exceptions import (
    InsufficientFeaturesError,
    RiskModelNotLoadedError,
    RiskPredictionFailedError,
)
from app.risk_prediction.messaging.publisher import RiskPredictionPublisher
from app.risk_prediction.schemas import (
    RiskPredictionHistoryResponse,
    RiskPredictionRequest,
    RiskPredictionResponse,
    SingleRiskPredictionResponse,
)
from app.risk_prediction.services.risk_prediction_service import (
    RiskPredictionService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["Risk Prediction"])


# ──────────────────────────────────────────────
# POST /risk/predict
# ──────────────────────────────────────────────


@router.post(
    "/predict",
    response_model=SingleRiskPredictionResponse,
    status_code=201,
    summary="Compute a risk prediction",
    description=(
        "Accepts sensor features and returns an accident probability, "
        "risk score, risk level, and confidence score. Optionally includes "
        "a per-factor breakdown and human-readable explanation."
    ),
)
async def predict_risk(
    request: RiskPredictionRequest,
    service: RiskPredictionService = Depends(get_risk_prediction_service),
    publisher: RiskPredictionPublisher = Depends(get_risk_prediction_publisher),
) -> SingleRiskPredictionResponse:
    try:
        prediction = await service.predict_from_features(
            request.features,
            sensor_id=request.sensor_id,
            equipment_id=request.equipment_id,
            zone_id=request.zone_id,
            include_breakdown=request.include_breakdown,
            include_explanation=request.include_explanation,
        )
    except RiskModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InsufficientFeaturesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RiskPredictionFailedError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Publish events AFTER successful prediction (outside business logic)
    publisher.publish_prediction_events(prediction)

    return SingleRiskPredictionResponse(
        success=True,
        prediction=RiskPredictionResponse.model_validate(prediction),
    )


# ──────────────────────────────────────────────
# GET /risk/predictions (history)
# ──────────────────────────────────────────────


@router.get(
    "/predictions",
    response_model=RiskPredictionHistoryResponse,
    summary="Get prediction history",
    description=(
        "Returns paginated risk prediction history. "
        "Supports filtering by sensor, zone, and risk level."
    ),
)
async def get_prediction_history(
    sensor_id: Optional[str] = Query(None, description="Filter by sensor ID"),
    zone_id: Optional[str] = Query(None, description="Filter by zone ID"),
    risk_level: Optional[str] = Query(
        None, description="Filter by risk level (LOW, MEDIUM, HIGH, CRITICAL)",
    ),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, ge=1, le=200, description="Page size"),
    service: RiskPredictionService = Depends(get_risk_prediction_service),
) -> RiskPredictionHistoryResponse:
    predictions, total = await service.get_prediction_history(
        sensor_id=sensor_id,
        zone_id=zone_id,
        risk_level=risk_level,
        offset=offset,
        limit=limit,
    )
    return RiskPredictionHistoryResponse(
        success=True,
        predictions=[
            RiskPredictionResponse.model_validate(p) for p in predictions
        ],
        total=total,
        offset=offset,
        limit=limit,
    )


# ──────────────────────────────────────────────
# GET /risk/predictions/latest
# ──────────────────────────────────────────────


@router.get(
    "/predictions/latest",
    response_model=SingleRiskPredictionResponse,
    summary="Get the latest risk prediction",
    description=(
        "Returns the most recent risk prediction. "
        "Optionally scoped by sensor or zone."
    ),
)
async def get_latest_prediction(
    sensor_id: Optional[str] = Query(None, description="Filter by sensor ID"),
    zone_id: Optional[str] = Query(None, description="Filter by zone ID"),
    service: RiskPredictionService = Depends(get_risk_prediction_service),
) -> SingleRiskPredictionResponse:
    prediction = await service.get_latest_prediction(
        sensor_id=sensor_id, zone_id=zone_id,
    )
    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="No risk predictions found for the given filters.",
        )
    return SingleRiskPredictionResponse(
        success=True,
        prediction=RiskPredictionResponse.model_validate(prediction),
    )
