"""Value objects for the Risk Prediction module.

Enumerations aligned to the API specification and architecture document.
"""

from __future__ import annotations

from enum import Enum


class RiskLevel(str, Enum):
    """Classified risk level based on prediction score.

    Thresholds (from architecture doc):
      LOW:      probability < 0.25  →  score 0–24
      MEDIUM:   probability < 0.50  →  score 25–49
      HIGH:     probability < 0.75  →  score 50–74
      CRITICAL: probability >= 0.75 →  score 75–100
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PredictionStatus(str, Enum):
    """Status of a risk prediction computation."""

    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STALE = "STALE"
