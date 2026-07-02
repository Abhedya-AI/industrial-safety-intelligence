"""Alert severity levels."""

from __future__ import annotations

from enum import Enum


class AlertLevel(str, Enum):
    """Graduated alert severity levels for operational response."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

