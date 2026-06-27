"""Aggregated v1 API router.

Collects all v1 sub-routers into a single router
that gets mounted on the FastAPI app.
"""

from fastapi import APIRouter

from src.api.v1.alerts import router as alerts_router
from src.api.v1.anomalies import router as anomalies_router
from src.api.v1.health import router as health_router
from src.api.v1.readings import router as readings_router
from src.api.v1.sensors import router as sensors_router
from src.api.v1.thresholds import router as thresholds_router

v1_router = APIRouter()

v1_router.include_router(health_router)
v1_router.include_router(sensors_router)
v1_router.include_router(readings_router)
v1_router.include_router(anomalies_router)
v1_router.include_router(alerts_router)
v1_router.include_router(thresholds_router)
