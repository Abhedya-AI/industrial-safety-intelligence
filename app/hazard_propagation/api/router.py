"""Aggregated API router for the Hazard Propagation Engine module."""

from fastapi import APIRouter

from app.hazard_propagation.api.hazard_propagation_endpoints import (
    router as endpoints_router,
)

hazard_propagation_router = APIRouter()

hazard_propagation_router.include_router(endpoints_router)
