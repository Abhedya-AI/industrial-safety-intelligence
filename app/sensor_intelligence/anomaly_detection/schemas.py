"""Structured output schemas for anomaly detection results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AnomalyStatus(str, Enum):
    """Classification status of an anomaly detection result."""

    NORMAL = "NORMAL"
    ANOMALY = "ANOMALY"


@dataclass(frozen=True)
class AnomalyResult:
    """Structured output from an anomaly detector.

    Attributes:
        sensor_id: The sensor that was evaluated.
        score: Raw anomaly score (model-specific, higher = more anomalous).
        status: Binary classification — NORMAL or ANOMALY.
        detector_type: Name of the detector that produced this result.
        confidence: Optional confidence level (0.0–1.0).
        threshold: The threshold used for classification.
        details: Optional extra metadata from the detector.
    """

    sensor_id: str
    score: float
    status: AnomalyStatus
    detector_type: str
    confidence: Optional[float] = None
    threshold: Optional[float] = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "sensor_id": self.sensor_id,
            "score": round(self.score, 6),
            "status": self.status.value,
            "detector_type": self.detector_type,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "threshold": round(self.threshold, 6) if self.threshold is not None else None,
            "details": self.details,
        }
