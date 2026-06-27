"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.common import HealthResponse, ReadinessResponse
from src.infrastructure.config.settings import Settings, get_settings
from src.infrastructure.database.connection import get_async_session

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    description="Returns 200 if the service is running.",
)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.app_env,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Returns 200 if the service can connect to the database.",
)
async def readiness_check(
    session: AsyncSession = Depends(get_async_session),
) -> ReadinessResponse:
    try:
        await session.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return ReadinessResponse(
        status="ready" if db_status == "connected" else "not_ready",
        database=db_status,
    )
