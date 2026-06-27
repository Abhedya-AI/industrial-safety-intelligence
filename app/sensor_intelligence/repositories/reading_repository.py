"""Reading repository interface (port).

Defines the abstract contract for sensor reading persistence.
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
    """Abstract interface for sensor reading persistence."""

    # ── Queries ──

    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[ReadingModel]:
        """Retrieve a reading by its UUID primary key."""
        ...

    @abstractmethod
    async def get_latest(self, sensor_pk: str) -> Optional[ReadingModel]:
        """Get the most recent reading for a sensor (by sensor PK)."""
        ...

    @abstractmethod
    async def get_range(
        self,
        sensor_pk: str,
        from_dt: datetime,
        to_dt: datetime,
        limit: int = 1000,
    ) -> list[ReadingModel]:
        """Get readings for a sensor within a time range."""
        ...

    @abstractmethod
    async def count_for_sensor(self, sensor_pk: str) -> int:
        """Count all readings for a sensor."""
        ...

    @abstractmethod
    async def get_stats(
        self,
        sensor_pk: str,
        from_dt: datetime,
        to_dt: datetime,
    ) -> Optional[ReadingStats]:
        """Compute aggregate statistics over a time window."""
        ...

    # ── Mutations ──

    @abstractmethod
    async def save(self, reading: ReadingModel) -> ReadingModel:
        """Persist a single reading."""
        ...

    @abstractmethod
    async def save_batch(self, readings: list[ReadingModel]) -> list[ReadingModel]:
        """Persist multiple readings in one flush."""
        ...
