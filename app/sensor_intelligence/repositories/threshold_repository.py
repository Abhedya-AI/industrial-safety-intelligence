"""Threshold repository interface (port)."""

from abc import ABC, abstractmethod
from uuid import UUID

from app.sensor_intelligence.domain.entities.threshold import Threshold
from app.sensor_intelligence.domain.value_objects.sensor_type import SensorType


class ThresholdRepository(ABC):
    """Abstract interface for threshold persistence operations."""

    @abstractmethod
    async def save(self, threshold: Threshold) -> Threshold:
        """Persist a new threshold configuration."""
        ...

    @abstractmethod
    async def get_by_id(self, id: UUID) -> Threshold | None:
        """Retrieve a threshold by its internal UUID."""
        ...

    @abstractmethod
    async def get_for_sensor(self, sensor_id: UUID) -> Threshold | None:
        """Get the active threshold for a specific sensor."""
        ...

    @abstractmethod
    async def get_for_type(self, sensor_type: SensorType) -> Threshold | None:
        """Get the active threshold for a sensor type (fallback)."""
        ...

    @abstractmethod
    async def list_all(
        self,
        sensor_type: SensorType | None = None,
        is_active: bool | None = None,
    ) -> list[Threshold]:
        """List thresholds with optional filters."""
        ...

    @abstractmethod
    async def update(self, threshold: Threshold) -> Threshold:
        """Update an existing threshold."""
        ...

    @abstractmethod
    async def deactivate(self, id: UUID) -> None:
        """Deactivate a threshold (soft delete)."""
        ...
