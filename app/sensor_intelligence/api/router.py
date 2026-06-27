"""Aggregated v1 API router for the Sensor Intelligence module."""

from fastapi import APIRouter

from app.sensor_intelligence.api.alerts import router as alerts_router
from app.sensor_intelligence.api.anomalies import router as anomalies_router
from app.sensor_intelligence.api.health import router as health_router
from app.sensor_intelligence.api.readings import router as readings_router
from app.sensor_intelligence.api.sensors import router as sensors_router
from app.sensor_intelligence.api.thresholds import router as thresholds_router

sensor_intelligence_router = APIRouter()

sensor_intelligence_router.include_router(health_router)
sensor_intelligence_router.include_router(sensors_router)
sensor_intelligence_router.include_router(readings_router)
sensor_intelligence_router.include_router(anomalies_router)
sensor_intelligence_router.include_router(alerts_router)
sensor_intelligence_router.include_router(thresholds_router)
