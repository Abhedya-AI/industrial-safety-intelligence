"""Sensor type classification."""

from __future__ import annotations

from enum import Enum


class SensorType(str, Enum):
    """Types of IoT sensors deployed in the industrial facility.

    Values aligned to API specification (PS1_Detailed_API_Specifications_V2).
    """

    GAS = "GAS"
    PRESSURE = "PRESSURE"
    TEMPERATURE = "TEMPERATURE"
    HUMIDITY = "HUMIDITY"
    VIBRATION = "VIBRATION"
