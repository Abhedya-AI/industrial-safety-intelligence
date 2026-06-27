"""Reading repository interface (port).

Defines the contract for sensor reading persistence.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from src.domain.entities.sensor_reading import SensorReading


@dataclass
class ReadingStats:
    """Aggregated statistics for sensor readings over a time window."""

    sensor_id: UUID
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    count: int
    window_start: datetime
    window_end: datetime


class ReadingRepository(ABC):
    """Abstract interface for sensor reading persistence operations."""

    @abstractmethod
    async def save(self, reading: SensorReading) -> SensorReading:
        """Persist a single sensor reading."""
        ...

    @abstractmethod
    async def save_batch(self, readings: list[SensorReading]) -> list[SensorReading]:
        """Persist multiple sensor readings in a single transaction."""
        ...

    @abstractmethod
    async def get_latest(self, sensor_id: UUID) -> SensorReading | None:
        """Get the most recent reading for a sensor."""
        ...

    @abstractmethod
    async def get_range(
        self,
        sensor_id: UUID,
        from_dt: datetime,
        to_dt: datetime,
        limit: int = 1000,
    ) -> list[SensorReading]:
        """Get readings for a sensor within a time range."""
        ...

    @abstractmethod
    async def get_recent(
        self,
        sensor_id: UUID,
        window: timedelta,
    ) -> list[SensorReading]:
        """Get readings within a recent time window from now."""
        ...

    @abstractmethod
    async def get_stats(
        self,
        sensor_id: UUID,
        window: timedelta,
    ) -> ReadingStats | None:
        """Compute aggregate statistics over a time window."""
        ...

    @abstractmethod
    async def count_in_window(
        self,
        sensor_id: UUID,
        window: timedelta,
    ) -> int:
        """Count readings in a recent time window."""
        ...
