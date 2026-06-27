"""Global error handler middleware.

Maps domain exceptions to proper HTTP responses so use cases
can throw domain-specific errors without knowing about HTTP.
"""

import logging
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.domain.exceptions import (
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


def register_error_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app."""

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


def _create_domain_handler(
    status_code: int,
) -> Callable[[Request, DomainError], JSONResponse]:
    """Factory for domain exception handlers (unused but kept for extensibility)."""

    async def handler(request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={"error": type(exc).__name__, "message": exc.message},
        )

    return handler
