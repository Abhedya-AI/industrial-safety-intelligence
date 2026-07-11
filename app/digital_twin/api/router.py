"""Aggregated API router for the Digital Twin module."""

from fastapi import APIRouter

from app.digital_twin.api.twin_endpoints import router as endpoints_router
from app.digital_twin.api.websocket_endpoints import ws_router

digital_twin_router = APIRouter()

digital_twin_router.include_router(endpoints_router)
digital_twin_router.include_router(ws_router)
