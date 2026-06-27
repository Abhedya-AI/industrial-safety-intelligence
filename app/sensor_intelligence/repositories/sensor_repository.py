"""Sensor repository interface (port).

Defines the abstract contract for sensor persistence operations.
Concrete implementations (e.g. SQLAlchemy) must implement every method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from app.sensor_intelligence.models.sensor_model import SensorModel


class SensorRepository(ABC):
    """Abstract interface for sensor persistence operations.

    Returns SensorModel ORM objects. Business validation is NOT performed
    here — that responsibility belongs to the service layer.
    """

    # ── Queries ──

    @abstractmethod
    async def get_sensor_by_id(self, id: UUID) -> Optional[SensorModel]:
        """Retrieve a sensor by its internal UUID primary key.

        Returns None if no sensor matches.
        """
        ...

    @abstractmethod
    async def get_sensor_by_code(self, sensor_id: str) -> Optional[SensorModel]:
        """Retrieve a sensor by its external business code (e.g. 'S001').

        Maps to the ``sensor_id`` column in the database.
        Returns None if no sensor matches.
        """
        ...

    @abstractmethod
    async def list_sensors(
        self,
        sensor_type: Optional[str] = None,
        status: Optional[str] = None,
        location_zone: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[SensorModel]:
        """List sensors with optional filters and pagination.

        All filter parameters accept raw string values matching
        their respective enum values.
        """
        ...

    @abstractmethod
    async def count(
        self,
        sensor_type: Optional[str] = None,
        status: Optional[str] = None,
        location_zone: Optional[str] = None,
    ) -> int:
        """Count sensors matching the given filters."""
        ...

    @abstractmethod
    async def count_by_status(self) -> dict[str, int]:
        """Count sensors grouped by status."""
        ...

    @abstractmethod
    async def sensor_exists(self, sensor_id: str) -> bool:
        """Check whether a sensor with the given business code exists.

        More efficient than ``get_sensor_by_code`` when only existence
        is needed (avoids full row hydration).
        """
        ...

    # ── Mutations ──

    @abstractmethod
    async def create_sensor(self, sensor: SensorModel) -> SensorModel:
        """Persist a new sensor. Returns the persisted instance with
        server-generated defaults (id, created_at, updated_at) populated.
        """
        ...

    @abstractmethod
    async def update_sensor(self, sensor: SensorModel) -> SensorModel:
        """Flush pending attribute changes on an already-tracked sensor.

        The caller is responsible for mutating attributes before calling
        this method; the repository simply persists the changes.
        """
        ...

    @abstractmethod
    async def delete_sensor(self, sensor: SensorModel) -> None:
        """Delete a sensor from persistent storage."""
        ...
