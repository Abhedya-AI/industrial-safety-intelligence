"""FastAPI Middleware configuration.

Sets up request logging, CORS, and global domain error mapping.
Error response format matches the API specification:
  {"success": false, "error": "ERROR_CODE", "message": "...",
   "details": {...}, "request_id": "...", "timestamp": "..."}
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.settings import get_settings
from app.shared.exceptions.domain_exceptions import (
    BusinessRuleViolationError,
    DomainError,
    DuplicateResourceError,
    InvalidReadingError,
    ResourceNotFoundError,
    ValidationError,
)

logger = logging.getLogger(__name__)

# Map domain exception types to HTTP status codes
_EXCEPTION_STATUS_MAP: dict[type[DomainError], int] = {
    ResourceNotFoundError: 404,
    DuplicateResourceError: 409,
    InvalidReadingError: 422,
    ValidationError: 422,
    BusinessRuleViolationError: 409,
}


def _error_response(
    status_code: int,
    error_code: str,
    message: str,
    request: Request,
    details: dict | None = None,
) -> JSONResponse:
    """Build a spec-compliant error response."""
    body: dict = {
        "success": False,
        "error": error_code,
        "message": message,
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if details:
        body["details"] = details
    return JSONResponse(status_code=status_code, content=body)


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

    # ── Domain Error Handler ──
    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        status_code = _EXCEPTION_STATUS_MAP.get(type(exc), 400)
        logger.warning(
            "Domain error | status=%d | code=%s | message=%s | path=%s",
            status_code,
            exc.error_code,
            exc.message,
            request.url.path,
        )
        return _error_response(
            status_code=status_code,
            error_code=exc.error_code,
            message=exc.message,
            request=request,
        )

    # ── Pydantic / FastAPI Validation Error Handler ──
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning(
            "Validation error | path=%s | errors=%s",
            request.url.path,
            exc.errors(),
        )
        return _error_response(
            status_code=422,
            error_code="VALIDATION_ERROR",
            message="Input validation failed",
            request=request,
            details={"validation_errors": exc.errors()},
        )

    # ── Catch-all Handler ──
    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception(
            "Unhandled error | path=%s | error=%s",
            request.url.path,
            str(exc),
        )
        return _error_response(
            status_code=500,
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            request=request,
        )
