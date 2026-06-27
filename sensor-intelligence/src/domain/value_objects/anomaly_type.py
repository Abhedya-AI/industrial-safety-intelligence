"""Anomaly classification types."""

from enum import StrEnum


class AnomalyType(StrEnum):
    """Types of anomalies detectable in sensor readings."""

    SPIKE = "SPIKE"
    DRIFT = "DRIFT"
    FLATLINE = "FLATLINE"
    OUT_OF_RANGE = "OUT_OF_RANGE"
