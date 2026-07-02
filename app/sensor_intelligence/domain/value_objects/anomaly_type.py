"""Anomaly classification types."""

from __future__ import annotations

from enum import Enum


class AnomalyType(str, Enum):
    """Types of anomalies detectable in sensor readings."""

    SPIKE = "SPIKE"
    DRIFT = "DRIFT"
    FLATLINE = "FLATLINE"
    OUT_OF_RANGE = "OUT_OF_RANGE"
