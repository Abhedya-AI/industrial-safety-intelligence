"""Baseline Learning Service — learns and persists normal operating statistics.

For every sensor, learns:
  - Mean, median, standard deviation, variance
  - Normal operating range (mean ± n·σ)
  - Rolling averages (window 5 and 10)
  - Seasonal hourly patterns (per-hour-of-day mean)
  - Trend direction and slope

Baselines are persisted in the sensor_baselines table and can be
recomputed periodically. The anomaly detection module uses these
baselines for comparison via get_baseline().

Architecture:
  - Reuses existing pure statistics functions from analysis.statistics
  - Reads from ReadingRepository (no direct SQL)
  - Persists via BaselineRepository (lightweight SQLAlchemy wrapper)
  - No API modifications
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional

from app.sensor_intelligence.analysis.statistics import (
    describe,
    rolling_average,
    trend,
    DescriptiveStats,
    TrendResult,
)
from app.sensor_intelligence.models.sensor_baseline_model import SensorBaselineModel
from app.sensor_intelligence.repositories.reading_repository import ReadingRepository
from app.sensor_intelligence.repositories.sensor_repository import SensorRepository

logger = logging.getLogger(__name__)

# Default configuration
_DEFAULT_WINDOW_DAYS = 7
_DEFAULT_SIGMA_MULTIPLIER = 2.0
_DEFAULT_MIN_SAMPLES = 10


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Baseline result (returned by learn_baseline)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class BaselineResult:
    """Computed baseline statistics for a sensor."""

    sensor_id: str  # Internal PK

    # Core statistics
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    variance: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0

    # Normal operating range
    normal_range_low: float = 0.0
    normal_range_high: float = 0.0

    # Rolling averages
    rolling_avg_5: list[float] = field(default_factory=list)
    rolling_avg_10: list[float] = field(default_factory=list)

    # Seasonal pattern (24 hourly means)
    hourly_pattern: dict[int, float] = field(default_factory=dict)

    # Trend
    trend_direction: Optional[str] = None
    trend_slope: Optional[float] = None

    # Metadata
    sample_count: int = 0
    window_days: int = _DEFAULT_WINDOW_DAYS
    sigma_multiplier: float = _DEFAULT_SIGMA_MULTIPLIER


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Baseline repository
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BaselineRepository:
    """Lightweight repository for SensorBaselineModel."""

    def __init__(self, session) -> None:
        self._session = session

    async def get_by_sensor_id(self, sensor_pk: str) -> Optional[SensorBaselineModel]:
        from sqlalchemy import select
        stmt = select(SensorBaselineModel).where(
            SensorBaselineModel.sensor_id == sensor_pk
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def save(self, baseline: SensorBaselineModel) -> SensorBaselineModel:
        self._session.add(baseline)
        await self._session.flush()
        await self._session.refresh(baseline)
        return baseline

    async def update(self, baseline: SensorBaselineModel) -> SensorBaselineModel:
        await self._session.flush()
        await self._session.refresh(baseline)
        return baseline

    async def list_all(
        self, offset: int = 0, limit: int = 50
    ) -> list[SensorBaselineModel]:
        from sqlalchemy import select
        stmt = (
            select(SensorBaselineModel)
            .order_by(SensorBaselineModel.sensor_id)
            .offset(offset)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Baseline Learning Service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BaselineLearningService:
    """Service for learning and managing sensor baselines.

    Gathers historical readings, computes statistics using the pure
    analysis module, and persists results for anomaly detection use.
    """

    def __init__(
        self,
        baseline_repo: BaselineRepository,
        sensor_repo: SensorRepository,
        reading_repo: ReadingRepository,
        window_days: int = _DEFAULT_WINDOW_DAYS,
        sigma_multiplier: float = _DEFAULT_SIGMA_MULTIPLIER,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
    ) -> None:
        self._baseline_repo = baseline_repo
        self._sensor_repo = sensor_repo
        self._reading_repo = reading_repo
        self._window_days = window_days
        self._sigma_multiplier = sigma_multiplier
        self._min_samples = min_samples

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Core: learn baseline
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def learn_baseline(
        self,
        sensor_pk: str,
        window_days: Optional[int] = None,
        sigma_multiplier: Optional[float] = None,
    ) -> Optional[BaselineResult]:
        """Learn normal operating statistics for a sensor.

        Fetches readings from the configured window, computes
        descriptive statistics, rolling averages, seasonal patterns,
        and trend analysis.

        Args:
            sensor_pk: Internal sensor UUID (primary key).
            window_days: Override the default analysis window.
            sigma_multiplier: Override the default sigma for normal range.

        Returns:
            BaselineResult or None if insufficient data.
        """
        days = window_days or self._window_days
        sigma = sigma_multiplier or self._sigma_multiplier

        now = datetime.now(timezone.utc)
        window_start = now - timedelta(days=days)

        readings = await self._reading_repo.get_sensor_history(
            sensor_pk, window_start, now, limit=50000
        )

        if len(readings) < self._min_samples:
            logger.info(
                "Insufficient data for baseline (sensor=%s, readings=%d, min=%d)",
                sensor_pk, len(readings), self._min_samples,
            )
            return None

        values = [r.value for r in readings]

        # 1. Descriptive statistics (reusing pure functions)
        stats = describe(values)

        # 2. Normal operating range
        normal_low = stats.mean - sigma * stats.std_dev
        normal_high = stats.mean + sigma * stats.std_dev

        # 3. Rolling averages
        ra_5 = rolling_average(values, 5) if len(values) >= 5 else None
        ra_10 = rolling_average(values, 10) if len(values) >= 10 else None

        # 4. Seasonal hourly patterns
        hourly_pattern = self._compute_hourly_pattern(readings)

        # 5. Trend analysis
        trend_result = trend(values) if len(values) >= 2 else None

        return BaselineResult(
            sensor_id=sensor_pk,
            mean=stats.mean,
            median=stats.median,
            std_dev=stats.std_dev,
            variance=stats.variance,
            min_value=stats.minimum,
            max_value=stats.maximum,
            normal_range_low=round(normal_low, 6),
            normal_range_high=round(normal_high, 6),
            rolling_avg_5=ra_5.values if ra_5 else [],
            rolling_avg_10=ra_10.values if ra_10 else [],
            hourly_pattern=hourly_pattern,
            trend_direction=trend_result.direction if trend_result else None,
            trend_slope=trend_result.slope if trend_result else None,
            sample_count=len(values),
            window_days=days,
            sigma_multiplier=sigma,
        )

    @staticmethod
    def _compute_hourly_pattern(readings) -> dict[int, float]:
        """Compute per-hour-of-day mean values (0-23).

        Groups readings by their timestamp hour and computes the
        average value for each hour. This captures daily seasonal
        patterns in sensor data.
        """
        hourly_sums: dict[int, float] = defaultdict(float)
        hourly_counts: dict[int, int] = defaultdict(int)

        for r in readings:
            ts = r.timestamp
            if ts is None:
                continue
            hour = ts.hour
            hourly_sums[hour] += r.value
            hourly_counts[hour] += 1

        return {
            h: round(hourly_sums[h] / hourly_counts[h], 6)
            for h in sorted(hourly_counts.keys())
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Persist baseline
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def update_baseline(
        self,
        sensor_pk: str,
        window_days: Optional[int] = None,
        sigma_multiplier: Optional[float] = None,
    ) -> Optional[SensorBaselineModel]:
        """Learn and persist a baseline for a sensor.

        Creates a new record if none exists, otherwise updates in place.

        Returns:
            Persisted SensorBaselineModel, or None if insufficient data.
        """
        result = await self.learn_baseline(sensor_pk, window_days, sigma_multiplier)
        if result is None:
            return None

        existing = await self._baseline_repo.get_by_sensor_id(sensor_pk)

        if existing:
            self._apply_result(existing, result)
            return await self._baseline_repo.update(existing)

        model = SensorBaselineModel(
            id=str(uuid.uuid4()),
            sensor_id=sensor_pk,
        )
        self._apply_result(model, result)
        return await self._baseline_repo.save(model)

    @staticmethod
    def _apply_result(model: SensorBaselineModel, result: BaselineResult) -> None:
        """Apply computed result fields to the ORM model."""
        model.mean = result.mean
        model.median = result.median
        model.std_dev = result.std_dev
        model.variance = result.variance
        model.min_value = result.min_value
        model.max_value = result.max_value
        model.normal_range_low = result.normal_range_low
        model.normal_range_high = result.normal_range_high
        model.rolling_avg_5 = json.dumps(result.rolling_avg_5[-20:])  # Keep last 20
        model.rolling_avg_10 = json.dumps(result.rolling_avg_10[-20:])
        model.hourly_pattern = json.dumps(result.hourly_pattern)
        model.trend_direction = result.trend_direction
        model.trend_slope = result.trend_slope
        model.sample_count = result.sample_count
        model.window_days = result.window_days
        model.sigma_multiplier = result.sigma_multiplier

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Queries
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_baseline(self, sensor_id: str) -> Optional[SensorBaselineModel]:
        """Get the learned baseline for a sensor by business ID.

        This is the primary method used by anomaly detection for comparison.

        Args:
            sensor_id: Business sensor ID (e.g. "S001").

        Returns:
            SensorBaselineModel or None.
        """
        sensor = await self._sensor_repo.get_sensor_by_code(sensor_id)
        if sensor is None:
            return None
        return await self._baseline_repo.get_by_sensor_id(sensor.id)

    async def get_baseline_by_pk(self, sensor_pk: str) -> Optional[SensorBaselineModel]:
        """Get baseline by internal sensor PK."""
        return await self._baseline_repo.get_by_sensor_id(sensor_pk)

    async def is_within_normal_range(
        self, sensor_id: str, value: float
    ) -> Optional[bool]:
        """Check if a value is within the learned normal range.

        Returns:
            True if within range, False if outside, None if no baseline.
        """
        baseline = await self.get_baseline(sensor_id)
        if baseline is None:
            return None
        return baseline.normal_range_low <= value <= baseline.normal_range_high

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Batch operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def update_all_baselines(
        self,
        window_days: Optional[int] = None,
        sigma_multiplier: Optional[float] = None,
    ) -> list[SensorBaselineModel]:
        """Recompute baselines for all registered sensors.

        Suitable for scheduled periodic recomputation.
        Skips sensors with insufficient data.

        Returns:
            List of updated baseline models.
        """
        sensors = await self._sensor_repo.list_sensors(limit=10000)
        results = []
        for sensor in sensors:
            try:
                baseline = await self.update_baseline(
                    sensor.id, window_days, sigma_multiplier
                )
                if baseline:
                    results.append(baseline)
                    logger.info(
                        "Baseline updated: sensor=%s mean=%.4f std=%.4f range=[%.4f, %.4f]",
                        sensor.sensor_id, baseline.mean, baseline.std_dev,
                        baseline.normal_range_low, baseline.normal_range_high,
                    )
            except Exception:
                logger.exception(
                    "Failed to update baseline for sensor %s", sensor.sensor_id
                )
        return results
