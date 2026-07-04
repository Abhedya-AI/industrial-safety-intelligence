"""Aggregated API router for the Risk Prediction module."""

from fastapi import APIRouter

from app.risk_prediction.api.risk_predictions import router as predictions_router

risk_prediction_router = APIRouter()

risk_prediction_router.include_router(predictions_router)
