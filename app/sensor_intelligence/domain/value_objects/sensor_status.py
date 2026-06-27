"""Sensor operational status."""

from __future__ import annotations

from enum import Enum


class SensorStatus(str, Enum):
    """Current operational status of a sensor.

    Values aligned to API specification (PS1_Detailed_API_Specifications_V2).
    """

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    OFFLINE = "OFFLINE"
