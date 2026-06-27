"""Shared Pydantic schemas used across multiple endpoints.

Includes pagination, error responses, and timestamp mixins.
"""

from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(..., description="Error type name")
    message: str = Field(..., description="Human-readable error message")
    path: str | None = Field(None, description="Request path that caused the error")


class PaginationParams(BaseModel):
    """Pagination query parameters."""

    offset: int = Field(0, ge=0, description="Number of items to skip")
    limit: int = Field(50, ge=1, le=200, description="Max items to return")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""

    items: list[T]
    total: int = Field(..., description="Total number of matching items")
    offset: int = Field(..., description="Current offset")
    limit: int = Field(..., description="Current page size")

    @property
    def has_more(self) -> bool:
        """Whether there are more items beyond this page."""
        return self.offset + self.limit < self.total


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Current environment")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Current server time"
    )


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    status: str = Field(..., description="Readiness status")
    database: str = Field(..., description="Database connection status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
