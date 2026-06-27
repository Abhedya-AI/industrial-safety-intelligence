"""Sensor operational status."""

from __future__ import annotations

from enum import StrEnum


class SensorStatus(StrEnum):
    """Current operational status of a sensor."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    OFFLINE = "OFFLINE"
