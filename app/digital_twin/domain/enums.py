"""Digital Twin domain enumerations.

Aligned with PS-1 Common Domain Names v2.0 §4.
Reuses existing enums where available; defines twin-specific types only.
"""

from __future__ import annotations

from enum import Enum


class TwinUpdateType(str, Enum):
    """Type of update applied to the twin state."""

    SENSOR_READING = "SENSOR_READING"
    SENSOR_STATUS = "SENSOR_STATUS"
    SENSOR_HEALTH = "SENSOR_HEALTH"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    RISK_SCORE = "RISK_SCORE"
    RISK_THRESHOLD = "RISK_THRESHOLD"
    COMPOUND_RISK = "COMPOUND_RISK"
    HAZARD_DETECTED = "HAZARD_DETECTED"
    HAZARD_PROPAGATED = "HAZARD_PROPAGATED"


class RiskLevel(str, Enum):
    """Risk level enum from PS-1 §4.1."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class HeatmapColor(str, Enum):
    """Heatmap color buckets for zone risk visualization."""

    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"

    @classmethod
    def from_score(cls, score: float) -> "HeatmapColor":
        """Map a 0-100 risk score to a heatmap color."""
        if score <= 25:
            return cls.GREEN
        elif score <= 50:
            return cls.YELLOW
        elif score <= 75:
            return cls.ORANGE
        else:
            return cls.RED


class WsChannel(str, Enum):
    """WebSocket subscription channels for Phase 4."""

    RISK_UPDATES = "risk-updates"
    SENSOR_DATA = "sensor-data"
    HAZARD_EVENTS = "hazard-events"
    TWIN_STATE = "twin-state"

