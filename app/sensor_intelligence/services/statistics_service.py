"""Statistics service — computes statistical analysis for sensor readings.

Connects the pure-function statistics module to the reading repository,
providing configurable time-range queries and structured result DTOs.

This service is consumed by:
  - Sensor History endpoint (spec endpoint 7)
  - Future anomaly detection modules (Isolation Forest, Autoencoder)
  - Baseline learning
  - Risk prediction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.sensor_intelligence.analysis.statistics import (
    DescriptiveStats,
    TrendResult,
    WindowStats,
    describe,
    rate_of_change,
    rolling_average,
    rolling_std_dev,
    trend,
)
from app.sensor_intelligence.repositories.reading_repository import ReadingRepository
from app.sensor_intelligence.repositories.sensor_repository import SensorRepository
from app.shared.exceptions.domain_exceptions import ResourceNotFoundError

logger = logging.getLogger(__name__)

# Default time windows
_DEFAULT_WINDOW = timedelta(hours=24)

# Named time ranges (used by spec endpoint 7)
TIME_RANGE_MAP: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "6h": timedelta(hours=6),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}


@dataclass(frozen=True)
class SensorStatisticsResult:
    """Full statistical analysis result for a sensor over a time range."""

    sensor_id: str
    descriptive: Optional[DescriptiveStats]
    trend: Optional[TrendResult]
    rate_of_change: list[float]
    rolling_avg: Optional[WindowStats]
    rolling_std: Optional[WindowStats]
    reading_count: int
    window_start: datetime
    window_end: datetime


class StatisticsService:
    """Computes statistical analysis over sensor reading history."""

    def __init__(
        self,
        reading_repo: ReadingRepository,
        sensor_repo: SensorRepository,
    ) -> None:
        self._reading_repo = reading_repo
        self._sensor_repo = sensor_repo

    async def _resolve_sensor_pk(self, sensor_id: str) -> str:
        """Resolve business sensor_id → internal PK. Raises 404 if missing."""
        sensor = await self._sensor_repo.get_sensor_by_code(sensor_id)
        if sensor is None:
            raise ResourceNotFoundError(resource="Sensor", identifier=sensor_id)
        return sensor.id

    async def compute_statistics(
        self,
        sensor_id: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        time_range: Optional[str] = None,
        rolling_window: int = 10,
    ) -> SensorStatisticsResult:
        """Compute full statistical analysis for a sensor.

        Args:
            sensor_id: External business sensor ID.
            start: Explicit start of analysis window (overrides time_range).
            end: Explicit end of analysis window (defaults to now).
            time_range: Named time range ("1h", "6h", "24h", "7d", "30d").
            rolling_window: Window size for rolling average / rolling std dev.
        """
        sensor_pk = await self._resolve_sensor_pk(sensor_id)

        # Resolve time window
        now = datetime.now(timezone.utc)
        window_end = end if end else now

        if start:
            window_start = start
        elif time_range and time_range in TIME_RANGE_MAP:
            window_start = window_end - TIME_RANGE_MAP[time_range]
        else:
            window_start = window_end - _DEFAULT_WINDOW

        # Fetch readings from repository
        readings = await self._reading_repo.get_sensor_history(
            sensor_pk, window_start, window_end, limit=10000
        )

        # Extract ordered values
        values = [r.value for r in readings]

        # Compute all statistics using pure functions
        descriptive_stats = describe(values)
        trend_result = trend(values) if len(values) >= 2 else None
        roc = rate_of_change(values)

        # Rolling window (only if enough data)
        r_avg = (
            rolling_average(values, rolling_window)
            if len(values) >= rolling_window
            else None
        )
        r_std = (
            rolling_std_dev(values, rolling_window)
            if len(values) >= rolling_window
            else None
        )

        logger.debug(
            "Statistics for %s: %d readings, window=%s→%s",
            sensor_id,
            len(values),
            window_start.isoformat(),
            window_end.isoformat(),
        )

        return SensorStatisticsResult(
            sensor_id=sensor_id,
            descriptive=descriptive_stats,
            trend=trend_result,
            rate_of_change=roc,
            rolling_avg=r_avg,
            rolling_std=r_std,
            reading_count=len(values),
            window_start=window_start,
            window_end=window_end,
        )

    async def compute_descriptive(
        self,
        sensor_id: str,
        start: datetime,
        end: datetime,
    ) -> Optional[DescriptiveStats]:
        """Compute only descriptive statistics (lightweight).

        Returns None if no readings exist in the window.
        """
        sensor_pk = await self._resolve_sensor_pk(sensor_id)
        readings = await self._reading_repo.get_sensor_history(
            sensor_pk, start, end, limit=10000
        )
        values = [r.value for r in readings]
        return describe(values)

    async def compute_trend(
        self,
        sensor_id: str,
        start: datetime,
        end: datetime,
    ) -> Optional[TrendResult]:
        """Compute trend direction and slope.

        Returns None if fewer than 2 readings.
        """
        sensor_pk = await self._resolve_sensor_pk(sensor_id)
        readings = await self._reading_repo.get_sensor_history(
            sensor_pk, start, end, limit=10000
        )
        values = [r.value for r in readings]
        return trend(values)
