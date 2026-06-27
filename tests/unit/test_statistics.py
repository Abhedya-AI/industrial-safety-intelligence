"""Comprehensive unit tests for the statistics pure functions.

Coverage matrix:
  - Empty datasets
  - Single reading
  - Multiple readings
  - Constant values
  - Increasing values
  - Decreasing values
  - Large datasets
  - Rolling window edge cases
  - Trend detection
  - Rate of change
"""

from __future__ import annotations

import math

import pytest

from app.sensor_intelligence.analysis.statistics import (
    DescriptiveStats,
    TrendResult,
    WindowStats,
    describe,
    maximum,
    mean,
    median,
    minimum,
    moving_window_stats,
    percentage_rate_of_change,
    rate_of_change,
    rolling_average,
    rolling_std_dev,
    std_dev,
    trend,
    variance,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# mean
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_mean_empty():
    assert mean([]) is None


def test_mean_single():
    assert mean([42.0]) == 42.0


def test_mean_multiple():
    assert mean([10.0, 20.0, 30.0]) == 20.0


def test_mean_constant():
    assert mean([5.0, 5.0, 5.0, 5.0]) == 5.0


def test_mean_negative():
    assert mean([-10.0, 10.0]) == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# median
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_median_empty():
    assert median([]) is None


def test_median_single():
    assert median([99.0]) == 99.0


def test_median_odd():
    assert median([1.0, 3.0, 2.0]) == 2.0


def test_median_even():
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_median_constant():
    assert median([7.0, 7.0, 7.0]) == 7.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# minimum / maximum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_minimum_empty():
    assert minimum([]) is None


def test_minimum_multiple():
    assert minimum([30.0, 10.0, 20.0]) == 10.0


def test_maximum_empty():
    assert maximum([]) is None


def test_maximum_multiple():
    assert maximum([30.0, 10.0, 20.0]) == 30.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# variance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_variance_empty():
    assert variance([]) is None


def test_variance_single():
    assert variance([42.0]) == 0.0


def test_variance_constant():
    assert variance([5.0, 5.0, 5.0]) == 0.0


def test_variance_population():
    # [2, 4, 6] → mean=4, var = ((4+0+4)/3) = 2.666...
    v = variance([2.0, 4.0, 6.0], population=True)
    assert v is not None
    assert abs(v - 8 / 3) < 1e-9


def test_variance_sample():
    # [2, 4, 6] → sample var = 8/(3-1) = 4
    v = variance([2.0, 4.0, 6.0], population=False)
    assert v is not None
    assert abs(v - 4.0) < 1e-9


def test_variance_sample_single_value():
    """Sample variance with 1 value should return 0."""
    assert variance([42.0], population=False) == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# std_dev
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_std_dev_empty():
    assert std_dev([]) is None


def test_std_dev_constant():
    assert std_dev([5.0, 5.0, 5.0]) == 0.0


def test_std_dev_known():
    sd = std_dev([2.0, 4.0, 6.0])
    assert sd is not None
    assert abs(sd - math.sqrt(8 / 3)) < 1e-9


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# describe (all-in-one)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_describe_empty():
    assert describe([]) is None


def test_describe_single():
    d = describe([42.0])
    assert d is not None
    assert d.mean == 42.0
    assert d.median == 42.0
    assert d.minimum == 42.0
    assert d.maximum == 42.0
    assert d.variance == 0.0
    assert d.std_dev == 0.0
    assert d.count == 1


def test_describe_increasing():
    d = describe([10.0, 20.0, 30.0, 40.0, 50.0])
    assert d is not None
    assert d.mean == 30.0
    assert d.median == 30.0
    assert d.minimum == 10.0
    assert d.maximum == 50.0
    assert d.count == 5
    assert d.std_dev > 0


def test_describe_constant():
    d = describe([7.0, 7.0, 7.0, 7.0])
    assert d is not None
    assert d.mean == 7.0
    assert d.variance == 0.0
    assert d.std_dev == 0.0


def test_describe_large_dataset():
    values = list(range(1000))
    d = describe(values)
    assert d is not None
    assert d.count == 1000
    assert d.minimum == 0
    assert d.maximum == 999
    assert abs(d.mean - 499.5) < 1e-3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# rolling_average
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_rolling_average_insufficient_data():
    result = rolling_average([1.0, 2.0], window_size=5)
    assert result.values == []


def test_rolling_average_window_1():
    result = rolling_average([1.0, 2.0, 3.0], window_size=1)
    assert result.values == [1.0, 2.0, 3.0]


def test_rolling_average_window_3():
    result = rolling_average([1.0, 2.0, 3.0, 4.0, 5.0], window_size=3)
    assert len(result.values) == 3
    assert result.values[0] == 2.0  # (1+2+3)/3
    assert result.values[1] == 3.0  # (2+3+4)/3
    assert result.values[2] == 4.0  # (3+4+5)/3


def test_rolling_average_constant():
    result = rolling_average([5.0] * 10, window_size=3)
    assert all(v == 5.0 for v in result.values)


def test_rolling_average_invalid_window():
    with pytest.raises(ValueError):
        rolling_average([1.0, 2.0], window_size=0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# rolling_std_dev
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_rolling_std_dev_insufficient_data():
    result = rolling_std_dev([1.0], window_size=3)
    assert result.values == []


def test_rolling_std_dev_constant():
    result = rolling_std_dev([5.0] * 10, window_size=3)
    assert all(v == 0.0 for v in result.values)


def test_rolling_std_dev_increasing():
    result = rolling_std_dev([1.0, 2.0, 3.0, 4.0, 5.0], window_size=3)
    assert len(result.values) == 3
    assert all(v > 0 for v in result.values)


def test_rolling_std_dev_invalid_window():
    with pytest.raises(ValueError):
        rolling_std_dev([1.0], window_size=0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# moving_window_stats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_moving_window_stats_insufficient():
    assert moving_window_stats([1.0], window_size=3) == []


def test_moving_window_stats_full():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = moving_window_stats(values, window_size=3)
    assert len(result) == 3
    assert result[0].mean == 20.0
    assert result[0].count == 3
    assert result[2].mean == 40.0


def test_moving_window_stats_invalid_window():
    with pytest.raises(ValueError):
        moving_window_stats([1.0], window_size=0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# trend
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_trend_empty():
    assert trend([]) is None


def test_trend_single():
    assert trend([42.0]) is None


def test_trend_increasing():
    t = trend([1.0, 2.0, 3.0, 4.0, 5.0])
    assert t is not None
    assert t.direction == "increasing"
    assert t.slope == 1.0
    assert t.rate_of_change == 400.0  # (5-1)/1 * 100


def test_trend_decreasing():
    t = trend([50.0, 40.0, 30.0, 20.0, 10.0])
    assert t is not None
    assert t.direction == "decreasing"
    assert t.slope == -10.0
    assert t.rate_of_change == -80.0  # (10-50)/50 * 100


def test_trend_constant():
    t = trend([7.0, 7.0, 7.0, 7.0])
    assert t is not None
    assert t.direction == "stable"
    assert t.slope == 0.0
    assert t.rate_of_change == 0.0


def test_trend_two_points():
    t = trend([10.0, 20.0])
    assert t is not None
    assert t.direction == "increasing"
    assert t.slope == 10.0


def test_trend_noisy_increasing():
    """Even with noise, overall trend should be increasing."""
    values = [10.0, 12.0, 11.0, 15.0, 14.0, 18.0, 17.0, 20.0]
    t = trend(values)
    assert t is not None
    assert t.direction == "increasing"
    assert t.slope > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# rate_of_change
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_roc_empty():
    assert rate_of_change([]) == []


def test_roc_single():
    assert rate_of_change([42.0]) == []


def test_roc_increasing():
    roc = rate_of_change([10.0, 20.0, 30.0])
    assert roc == [10.0, 10.0]


def test_roc_decreasing():
    roc = rate_of_change([30.0, 20.0, 10.0])
    assert roc == [-10.0, -10.0]


def test_roc_constant():
    roc = rate_of_change([5.0, 5.0, 5.0])
    assert roc == [0.0, 0.0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# percentage_rate_of_change
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_pct_roc_empty():
    assert percentage_rate_of_change([]) == []


def test_pct_roc_single():
    assert percentage_rate_of_change([42.0]) == []


def test_pct_roc_double():
    result = percentage_rate_of_change([50.0, 100.0])
    assert result == [100.0]  # 100% increase


def test_pct_roc_from_zero():
    result = percentage_rate_of_change([0.0, 10.0])
    assert result == [None]  # Division by zero → None


def test_pct_roc_decreasing():
    result = percentage_rate_of_change([100.0, 50.0])
    assert result == [-50.0]
