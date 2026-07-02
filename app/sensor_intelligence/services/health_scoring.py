"""Health scoring domain service.

Pure business logic for computing sensor health scores.
No side effects, no database access — takes data in, returns scores.

Health score is a weighted composite of individual factors (0-100):
  - Calibration age:     25% — how overdue is calibration?
  - Anomaly frequency:   25% — ratio of anomalous readings
  - Uptime:              20% — time since installation vs expected lifetime
  - Reading stability:   15% — coefficient of variation of recent readings
  - Missing readings:    15% — gap ratio in expected reading intervals

Classification:
  90-100  EXCELLENT
  70-89   GOOD
  50-69   FAIR
  30-49   POOR
  0-29    CRITICAL
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Optional


class HealthStatus(str, Enum):
    """Sensor health classification tiers."""

    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    CRITICAL = "CRITICAL"


@dataclass
class HealthFactors:
    """Raw inputs for health score calculation."""

    # Calibration
    last_calibration: Optional[date] = None
    next_calibration_due: Optional[date] = None
    today: Optional[date] = None  # Injected for testability

    # Anomaly frequency
    total_readings: int = 0
    anomaly_count: int = 0

    # Uptime
    installation_date: Optional[date] = None
    sensor_status: str = "NORMAL"

    # Reading stability
    reading_std_dev: float = 0.0
    reading_mean: float = 0.0

    # Missing readings
    expected_readings: int = 0
    actual_readings: int = 0


@dataclass
class HealthWeights:
    """Configurable weights for each health factor.

    All weights must sum to 1.0.
    """

    calibration: float = 0.25
    anomaly: float = 0.25
    uptime: float = 0.20
    stability: float = 0.15
    missing_readings: float = 0.15


@dataclass
class HealthScoreResult:
    """Complete health assessment output."""

    health_score: float
    health_status: HealthStatus
    calibration_score: float
    anomaly_score: float
    uptime_score: float
    stability_score: float
    missing_readings_score: float
    details: dict = field(default_factory=dict)


def classify_health(score: float) -> HealthStatus:
    """Classify a numeric health score into a status tier."""
    if score >= 90:
        return HealthStatus.EXCELLENT
    if score >= 70:
        return HealthStatus.GOOD
    if score >= 50:
        return HealthStatus.FAIR
    if score >= 30:
        return HealthStatus.POOR
    return HealthStatus.CRITICAL


def calculate_calibration_score(factors: HealthFactors) -> float:
    """Score calibration freshness (0-100).

    - No calibration data → 50 (unknown, penalised)
    - Within calibration window → 100
    - Overdue: decays by 2 points per day overdue (min 0)
    """
    today = factors.today or date.today()

    if factors.last_calibration is None or factors.next_calibration_due is None:
        return 50.0  # Unknown calibration status

    if today <= factors.next_calibration_due:
        return 100.0  # Within window

    overdue_days = (today - factors.next_calibration_due).days
    # Decay: 2 points per overdue day
    return max(0.0, 100.0 - overdue_days * 2.0)


def calculate_anomaly_score(factors: HealthFactors) -> float:
    """Score anomaly frequency (0-100).

    - No readings → 100 (no evidence of problems)
    - 0% anomaly rate → 100
    - Decays linearly: 5% anomaly rate → 75, 10% → 50, 20% → 0
    """
    if factors.total_readings == 0:
        return 100.0

    anomaly_rate = factors.anomaly_count / factors.total_readings
    # Linear decay: 0% → 100, 20%+ → 0
    return max(0.0, 100.0 - anomaly_rate * 500.0)


def calculate_uptime_score(factors: HealthFactors) -> float:
    """Score sensor uptime/availability (0-100).

    - NORMAL/WARNING → 100
    - CRITICAL → 50
    - OFFLINE → 0
    - No installation date → 80 (mild penalty)
    """
    status = factors.sensor_status.upper()

    if status == "OFFLINE":
        return 0.0
    if status == "CRITICAL":
        return 50.0

    if factors.installation_date is None:
        return 80.0

    return 100.0


def calculate_stability_score(factors: HealthFactors) -> float:
    """Score reading stability via coefficient of variation (0-100).

    CV = std_dev / |mean|
    - CV < 0.05  → 100 (very stable)
    - CV 0.05-0.5 → linear decay to 50
    - CV > 0.5    → steeper decay to 0
    """
    if factors.reading_mean == 0.0 or factors.total_readings < 2:
        return 100.0  # Not enough data or zero mean

    cv = factors.reading_std_dev / abs(factors.reading_mean)

    if cv < 0.05:
        return 100.0
    if cv <= 0.5:
        # Linear from 100 @ cv=0.05 to 50 @ cv=0.5
        return 100.0 - (cv - 0.05) / 0.45 * 50.0
    # Steeper decay for high CV
    return max(0.0, 50.0 - (cv - 0.5) * 100.0)


def calculate_missing_readings_score(factors: HealthFactors) -> float:
    """Score reading completeness (0-100).

    - No expected readings → 100
    - 100% coverage → 100
    - Linear decay: 90% → 80, 80% → 60, 50% → 0
    """
    if factors.expected_readings == 0:
        return 100.0

    coverage = factors.actual_readings / factors.expected_readings
    coverage = min(1.0, coverage)  # Cap at 100%

    if coverage >= 1.0:
        return 100.0
    # Linear from 100 @ 100% to 0 @ 50%
    return max(0.0, (coverage - 0.5) / 0.5 * 100.0)


def calculate_health_score(
    factors: HealthFactors,
    weights: Optional[HealthWeights] = None,
) -> HealthScoreResult:
    """Calculate composite sensor health score.

    Args:
        factors: Raw health factor inputs.
        weights: Configurable weights (defaults to standard weights).

    Returns:
        HealthScoreResult with composite score and individual factors.
    """
    w = weights or HealthWeights()

    cal = calculate_calibration_score(factors)
    anom = calculate_anomaly_score(factors)
    up = calculate_uptime_score(factors)
    stab = calculate_stability_score(factors)
    miss = calculate_missing_readings_score(factors)

    composite = (
        cal * w.calibration
        + anom * w.anomaly
        + up * w.uptime
        + stab * w.stability
        + miss * w.missing_readings
    )
    composite = round(max(0.0, min(100.0, composite)), 2)

    return HealthScoreResult(
        health_score=composite,
        health_status=classify_health(composite),
        calibration_score=round(cal, 2),
        anomaly_score=round(anom, 2),
        uptime_score=round(up, 2),
        stability_score=round(stab, 2),
        missing_readings_score=round(miss, 2),
        details={
            "total_readings": factors.total_readings,
            "anomaly_count": factors.anomaly_count,
            "anomaly_rate": round(
                factors.anomaly_count / factors.total_readings, 4
            ) if factors.total_readings > 0 else 0.0,
            "sensor_status": factors.sensor_status,
            "cv": round(
                factors.reading_std_dev / abs(factors.reading_mean), 4
            ) if factors.reading_mean != 0 else 0.0,
        },
    )
