"""FastAPI Middleware configuration.

Sets up request logging, CORS, and global domain error mapping.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.settings import get_settings
from app.shared.exceptions.domain_exceptions import (
    AlertNotFoundError,
    AnomalyNotFoundError,
    DomainError,
    DuplicateSensorError,
    InvalidReadingError,
    SensorNotFoundError,
    ThresholdNotFoundError,
)

logger = logging.getLogger(__name__)

# Map domain exceptions to HTTP status codes
_EXCEPTION_STATUS_MAP: dict[type[DomainError], int] = {
    SensorNotFoundError: 404,
    AlertNotFoundError: 404,
    AnomalyNotFoundError: 404,
    ThresholdNotFoundError: 404,
    DuplicateSensorError: 409,
    InvalidReadingError: 422,
}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs every HTTP request with timing information."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "%s %s → %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


def setup_middleware(app: FastAPI) -> None:
    """Setup middlewares (CORS, Logging) and global error handlers."""
    settings = get_settings()

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Logging
    app.add_middleware(RequestLoggingMiddleware)

    # Global Exception Handlers
    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        status_code = _EXCEPTION_STATUS_MAP.get(type(exc), 400)
        logger.warning(
            "Domain error | status=%d | error=%s | path=%s",
            status_code,
            exc.message,
            request.url.path,
        )
        return JSONResponse(
            status_code=status_code,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "path": request.url.path,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception(
            "Unhandled error | path=%s | error=%s",
            request.url.path,
            str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "path": request.url.path,
            },
        )
