"""Sensor type classification."""

from __future__ import annotations

from enum import StrEnum


class SensorType(StrEnum):
    """Types of IoT sensors deployed in the industrial facility."""

    GAS = "GAS"
    PRESSURE = "PRESSURE"
    TEMPERATURE = "TEMPERATURE"
    HUMIDITY = "HUMIDITY"
    VIBRATION = "VIBRATION"
