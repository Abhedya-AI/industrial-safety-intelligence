"""Aggregated API router for the Compound Risk Intelligence module."""

from fastapi import APIRouter

from app.compound_risk.api.compound_risk_endpoints import router as endpoints_router

compound_risk_router = APIRouter()

compound_risk_router.include_router(endpoints_router)
