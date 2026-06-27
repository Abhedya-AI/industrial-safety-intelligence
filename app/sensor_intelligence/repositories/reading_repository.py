"""Reading repository interface (port).

Defines the abstract contract for sensor reading persistence.
Concrete implementations (e.g. SQLAlchemy) must implement every method.

Method naming follows the SensorRepository convention:
  create_reading, get_reading_by_id, get_latest_reading, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.sensor_intelligence.models.reading_model import ReadingModel


@dataclass
class ReadingStats:
    """Aggregated statistics for sensor readings over a time window."""

    sensor_id: str
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    count: int
    window_start: datetime
    window_end: datetime


class ReadingRepository(ABC):
    """Abstract interface for sensor reading persistence.

    Returns ReadingModel ORM objects. Business validation is NOT performed
    here — that responsibility belongs to the service layer.
    """

    # ── Queries ──

    @abstractmethod
    async def get_reading_by_id(self, id: str) -> Optional[ReadingModel]:
        """Retrieve a reading by its UUID primary key.

        Returns None if no reading matches.
        """
        ...

    @abstractmethod
    async def get_latest_reading(self, sensor_pk: str) -> Optional[ReadingModel]:
        """Get the most recent reading for a sensor (by sensor PK).

        Returns None if the sensor has no readings.
        """
        ...

    @abstractmethod
    async def get_sensor_history(
        self,
        sensor_pk: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
    ) -> list[ReadingModel]:
        """Get readings for a sensor within a time range.

        Results are ordered by timestamp ascending (oldest first).
        """
        ...

    @abstractmethod
    async def list_readings(
        self,
        sensor_pk: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[ReadingModel]:
        """List readings with optional sensor filter and pagination.

        Results are ordered by timestamp descending (newest first).
        """
        ...

    @abstractmethod
    async def count_for_sensor(self, sensor_pk: str) -> int:
        """Count all readings for a sensor."""
        ...

    @abstractmethod
    async def reading_exists(self, id: str) -> bool:
        """Check whether a reading with the given UUID exists.

        More efficient than ``get_reading_by_id`` when only existence
        is needed (avoids full row hydration).
        """
        ...

    @abstractmethod
    async def get_stats(
        self,
        sensor_pk: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Optional[ReadingStats]:
        """Compute aggregate statistics over a time window.

        Returns None if no readings exist in the window.
        """
        ...

    # ── Mutations ──

    @abstractmethod
    async def create_reading(self, reading: ReadingModel) -> ReadingModel:
        """Persist a new reading. Returns the persisted instance with
        server-generated defaults (received_at) populated.
        """
        ...

    @abstractmethod
    async def create_readings_batch(
        self, readings: list[ReadingModel]
    ) -> list[ReadingModel]:
        """Persist multiple readings in a single flush."""
        ...
