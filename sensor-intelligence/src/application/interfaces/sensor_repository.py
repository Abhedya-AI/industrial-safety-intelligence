"""Sensor repository interface (port).

Defines the contract that any sensor persistence adapter must implement.
"""

from abc import ABC, abstractmethod
from uuid import UUID

from src.domain.entities.sensor import Sensor
from src.domain.value_objects.sensor_status import SensorStatus
from src.domain.value_objects.sensor_type import SensorType


class SensorRepository(ABC):
    """Abstract interface for sensor persistence operations."""

    @abstractmethod
    async def get_by_id(self, id: UUID) -> Sensor | None:
        """Retrieve a sensor by its internal UUID."""
        ...

    @abstractmethod
    async def get_by_sensor_id(self, sensor_id: str) -> Sensor | None:
        """Retrieve a sensor by its external business identifier."""
        ...

    @abstractmethod
    async def list_all(
        self,
        zone: str | None = None,
        sensor_type: SensorType | None = None,
        status: SensorStatus | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[Sensor]:
        """List sensors with optional filters and pagination."""
        ...

    @abstractmethod
    async def count(
        self,
        zone: str | None = None,
        sensor_type: SensorType | None = None,
        status: SensorStatus | None = None,
    ) -> int:
        """Count sensors matching the given filters."""
        ...

    @abstractmethod
    async def save(self, sensor: Sensor) -> Sensor:
        """Persist a new sensor."""
        ...

    @abstractmethod
    async def update(self, sensor: Sensor) -> Sensor:
        """Update an existing sensor."""
        ...

    @abstractmethod
    async def delete(self, id: UUID) -> None:
        """Delete a sensor by its internal UUID."""
        ...
