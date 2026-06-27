"""Alert severity levels."""

from __future__ import annotations

from enum import StrEnum


class AlertLevel(StrEnum):
    """Graduated alert severity levels for operational response."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"
