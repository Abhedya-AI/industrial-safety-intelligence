"""Anomaly repository interface (port)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from uuid import UUID

from app.sensor_intelligence.domain.entities.anomaly import Anomaly
from app.sensor_intelligence.domain.value_objects.anomaly_type import AnomalyType


class AnomalyRepository(ABC):
    """Abstract interface for anomaly persistence operations."""

    @abstractmethod
    async def save(self, anomaly: Anomaly) -> Anomaly:
        """Persist a new anomaly."""
        ...

    @abstractmethod
    async def get_by_id(self, id: UUID) -> Anomaly | None:
        """Retrieve an anomaly by its internal UUID."""
        ...

    @abstractmethod
    async def list_all(
        self,
        sensor_id: UUID | None = None,
        anomaly_type: AnomalyType | None = None,
        is_resolved: bool | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[Anomaly]:
        """List anomalies with optional filters and pagination."""
        ...

    @abstractmethod
    async def count(
        self,
        sensor_id: UUID | None = None,
        anomaly_type: AnomalyType | None = None,
        is_resolved: bool | None = None,
    ) -> int:
        """Count anomalies matching the given filters."""
        ...

    @abstractmethod
    async def get_unresolved(self, sensor_id: UUID) -> list[Anomaly]:
        """Get all unresolved anomalies for a sensor."""
        ...

    @abstractmethod
    async def resolve(self, id: UUID, resolved_at: datetime) -> Anomaly:
        """Mark an anomaly as resolved."""
        ...

    @abstractmethod
    async def count_in_window(
        self,
        sensor_id: UUID,
        window: timedelta,
    ) -> int:
        """Count anomalies detected in a recent time window."""
        ...
