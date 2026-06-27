"""Pure statistical analysis functions for sensor time-series data.

All functions operate on plain Python sequences (list[float]) and are
completely independent of FastAPI, SQLAlchemy, and ML frameworks.

Design goals:
  - Reusable by anomaly detection (Isolation Forest, Autoencoder),
    baseline learning, and risk prediction modules.
  - Handle edge cases gracefully (empty lists, single values, constant data).
  - No side effects or I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result data classes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class DescriptiveStats:
    """Result of descriptive statistical analysis."""

    mean: float
    median: float
    minimum: float
    maximum: float
    variance: float
    std_dev: float
    count: int


@dataclass(frozen=True)
class TrendResult:
    """Result of linear trend analysis on a time series."""

    direction: str  # "increasing" | "decreasing" | "stable"
    slope: float  # units per sample step
    rate_of_change: float  # percentage change from first to last value


@dataclass(frozen=True)
class WindowStats:
    """Statistics computed over a rolling / moving window."""

    values: list[float]  # one stat value per window position
    window_size: int


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core descriptive statistics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def mean(values: Sequence[float]) -> Optional[float]:
    """Arithmetic mean. Returns None for empty sequences."""
    if not values:
        return None
    return sum(values) / len(values)


def median(values: Sequence[float]) -> Optional[float]:
    """Median value. Returns None for empty sequences."""
    if not values:
        return None
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_v[mid - 1] + sorted_v[mid]) / 2.0
    return sorted_v[mid]


def minimum(values: Sequence[float]) -> Optional[float]:
    """Minimum value. Returns None for empty sequences."""
    return min(values) if values else None


def maximum(values: Sequence[float]) -> Optional[float]:
    """Maximum value. Returns None for empty sequences."""
    return max(values) if values else None


def variance(values: Sequence[float], *, population: bool = True) -> Optional[float]:
    """Variance (population by default). Returns None for empty sequences.

    Args:
        population: If True, uses N denominator (population variance).
                    If False, uses N-1 denominator (sample variance).
    """
    if not values:
        return None
    n = len(values)
    if not population and n < 2:
        return 0.0
    avg = sum(values) / n
    ss = sum((v - avg) ** 2 for v in values)
    denom = n if population else (n - 1)
    return ss / denom


def std_dev(values: Sequence[float], *, population: bool = True) -> Optional[float]:
    """Standard deviation. Returns None for empty sequences."""
    var = variance(values, population=population)
    return math.sqrt(var) if var is not None else None


def describe(values: Sequence[float]) -> Optional[DescriptiveStats]:
    """Compute all descriptive statistics in one pass.

    Returns None for empty sequences.
    """
    if not values:
        return None
    n = len(values)
    avg = sum(values) / n
    sorted_v = sorted(values)
    mid = n // 2
    med = sorted_v[mid] if n % 2 != 0 else (sorted_v[mid - 1] + sorted_v[mid]) / 2.0
    ss = sum((v - avg) ** 2 for v in values)
    var = ss / n
    return DescriptiveStats(
        mean=round(avg, 6),
        median=round(med, 6),
        minimum=sorted_v[0],
        maximum=sorted_v[-1],
        variance=round(var, 6),
        std_dev=round(math.sqrt(var), 6),
        count=n,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rolling / Moving window functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def rolling_average(
    values: Sequence[float], window_size: int
) -> WindowStats:
    """Compute rolling (moving) average.

    Returns a WindowStats with len(values) - window_size + 1 entries.
    Empty result for insufficient data.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if len(values) < window_size:
        return WindowStats(values=[], window_size=window_size)

    result: list[float] = []
    window_sum = sum(values[:window_size])
    result.append(round(window_sum / window_size, 6))

    for i in range(window_size, len(values)):
        window_sum += values[i] - values[i - window_size]
        result.append(round(window_sum / window_size, 6))

    return WindowStats(values=result, window_size=window_size)


def rolling_std_dev(
    values: Sequence[float], window_size: int
) -> WindowStats:
    """Compute rolling (moving) standard deviation.

    Uses population std dev within each window.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if len(values) < window_size:
        return WindowStats(values=[], window_size=window_size)

    result: list[float] = []
    for i in range(len(values) - window_size + 1):
        window = values[i : i + window_size]
        sd = std_dev(window)
        result.append(round(sd, 6) if sd is not None else 0.0)

    return WindowStats(values=result, window_size=window_size)


def moving_window_stats(
    values: Sequence[float], window_size: int
) -> list[DescriptiveStats]:
    """Compute full descriptive statistics for each rolling window position.

    Returns a list of DescriptiveStats with len(values) - window_size + 1 entries.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if len(values) < window_size:
        return []

    return [
        describe(values[i : i + window_size])
        for i in range(len(values) - window_size + 1)
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trend and rate of change
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def trend(values: Sequence[float], *, threshold: float = 1e-6) -> Optional[TrendResult]:
    """Compute linear trend using least-squares regression.

    Args:
        threshold: slope magnitude below which the trend is "stable".

    Returns None for sequences with fewer than 2 points.
    """
    n = len(values)
    if n < 2:
        return None

    # Least-squares linear regression: y = slope * x + intercept
    # x = 0, 1, 2, ..., n-1
    sum_x = n * (n - 1) / 2.0
    sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0
    sum_y = sum(values)
    sum_xy = sum(i * v for i, v in enumerate(values))

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-12:
        slope = 0.0
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denom

    # Rate of change: percentage change from first to last
    first, last = values[0], values[-1]
    if abs(first) < 1e-12:
        roc = 0.0 if abs(last - first) < 1e-12 else float("inf")
    else:
        roc = ((last - first) / abs(first)) * 100.0

    if abs(slope) < threshold:
        direction = "stable"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    return TrendResult(
        direction=direction,
        slope=round(slope, 6),
        rate_of_change=round(roc, 4),
    )


def rate_of_change(values: Sequence[float]) -> list[float]:
    """Point-to-point rate of change (first differences).

    Returns a list of (n-1) values. Empty for sequences with < 2 points.
    """
    if len(values) < 2:
        return []
    return [round(values[i + 1] - values[i], 6) for i in range(len(values) - 1)]


def percentage_rate_of_change(values: Sequence[float]) -> list[Optional[float]]:
    """Point-to-point percentage rate of change.

    Returns None for transitions from zero.
    """
    if len(values) < 2:
        return []
    result: list[Optional[float]] = []
    for i in range(len(values) - 1):
        if abs(values[i]) < 1e-12:
            result.append(None)
        else:
            pct = ((values[i + 1] - values[i]) / abs(values[i])) * 100.0
            result.append(round(pct, 4))
    return result
