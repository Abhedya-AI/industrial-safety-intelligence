"""Alert repository interface (port)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from app.sensor_intelligence.domain.entities.alert import Alert
from app.sensor_intelligence.domain.value_objects.alert_level import AlertLevel


@dataclass
class AlertSummary:
    """Summary counts of alerts by level."""

    info: int = 0
    warning: int = 0
    critical: int = 0
    emergency: int = 0
    total: int = 0
    unacknowledged: int = 0


class AlertRepository(ABC):
    """Abstract interface for alert persistence operations."""

    @abstractmethod
    async def save(self, alert: Alert) -> Alert:
        """Persist a new alert."""
        ...

    @abstractmethod
    async def get_by_id(self, id: UUID) -> Alert | None:
        """Retrieve an alert by its internal UUID."""
        ...

    @abstractmethod
    async def list_all(
        self,
        level: AlertLevel | None = None,
        is_acknowledged: bool | None = None,
        sensor_id: UUID | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[Alert]:
        """List alerts with optional filters and pagination."""
        ...

    @abstractmethod
    async def count(
        self,
        level: AlertLevel | None = None,
        is_acknowledged: bool | None = None,
        sensor_id: UUID | None = None,
    ) -> int:
        """Count alerts matching the given filters."""
        ...

    @abstractmethod
    async def acknowledge(
        self, id: UUID, acknowledged_by: str, acknowledged_at: datetime
    ) -> Alert:
        """Mark an alert as acknowledged."""
        ...

    @abstractmethod
    async def get_unacknowledged(self) -> list[Alert]:
        """Get all unacknowledged alerts."""
        ...

    @abstractmethod
    async def get_summary(self) -> AlertSummary:
        """Get a summary count of alerts by level."""
        ...
