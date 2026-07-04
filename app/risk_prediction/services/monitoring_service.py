"""Risk Prediction Monitoring Service.

Tracks runtime behaviour of the risk prediction model:
  - Prediction counts and throughput
  - Accident probability distribution (LOW / MEDIUM / HIGH / CRITICAL)
  - Confidence score statistics
  - Prediction latency histogram
  - Feature drift (running mean/std per input feature)
  - Model version tracking

Integrates with the existing ``ModelMonitoringService`` from Sensor
Intelligence for unified health reporting.

Architecture:
  - In-memory, thread-safe (Lock)
  - No database dependency — pure counters and rolling statistics
  - Does NOT retrain models
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

from app.risk_prediction.domain.value_objects import RiskLevel
from app.sensor_intelligence.services.model_monitoring_service import (
    ModelMetadata,
    ModelMonitoringService,
    create_model_metadata_from_path,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL_NAME = "xgboost_risk_prediction"
DEFAULT_MODEL_PATH = "models/risk_prediction_xgboost.pkl"


@dataclass
class PredictionDistribution:
    """Counts of predictions by risk level."""

    low: int = 0
    medium: int = 0
    high: int = 0
    critical: int = 0

    @property
    def total(self) -> int:
        return self.low + self.medium + self.high + self.critical

    def percentages(self) -> Dict[str, float]:
        t = self.total
        if t == 0:
            return {"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 0.0, "CRITICAL": 0.0}
        return {
            "LOW": round(self.low / t * 100, 2),
            "MEDIUM": round(self.medium / t * 100, 2),
            "HIGH": round(self.high / t * 100, 2),
            "CRITICAL": round(self.critical / t * 100, 2),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counts": asdict(self),
            "total": self.total,
            "percentages": self.percentages(),
        }


@dataclass
class RunningStats:
    """Welford's online algorithm for running mean/variance.

    Memory-efficient: tracks only count, mean, and M2 — not all values.
    """

    count: int = 0
    mean: float = 0.0
    _m2: float = 0.0  # Sum of squares of differences from the current mean
    min_value: float = float("inf")
    max_value: float = float("-inf")

    def update(self, value: float) -> None:
        """Add a new observation."""
        if math.isnan(value) or math.isinf(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._m2 += delta * delta2
        if value < self.min_value:
            self.min_value = value
        if value > self.max_value:
            self.max_value = value

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)

    @property
    def std_dev(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean": round(self.mean, 6),
            "std_dev": round(self.std_dev, 6),
            "min": round(self.min_value, 6) if self.count > 0 else None,
            "max": round(self.max_value, 6) if self.count > 0 else None,
        }


@dataclass
class LatencyHistogram:
    """Simple histogram for prediction latency tracking."""

    _buckets: Dict[str, int] = field(default_factory=lambda: {
        "<1ms": 0, "1-5ms": 0, "5-10ms": 0, "10-50ms": 0,
        "50-100ms": 0, "100-500ms": 0, ">500ms": 0,
    })
    total_ms: float = 0.0
    count: int = 0
    max_ms: float = 0.0

    def record(self, latency_ms: float) -> None:
        self.total_ms += latency_ms
        self.count += 1
        if latency_ms > self.max_ms:
            self.max_ms = latency_ms

        if latency_ms < 1:
            self._buckets["<1ms"] += 1
        elif latency_ms < 5:
            self._buckets["1-5ms"] += 1
        elif latency_ms < 10:
            self._buckets["5-10ms"] += 1
        elif latency_ms < 50:
            self._buckets["10-50ms"] += 1
        elif latency_ms < 100:
            self._buckets["50-100ms"] += 1
        elif latency_ms < 500:
            self._buckets["100-500ms"] += 1
        else:
            self._buckets[">500ms"] += 1

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buckets": dict(self._buckets),
            "count": self.count,
            "avg_ms": round(self.avg_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


@dataclass
class FeatureDriftTracker:
    """Tracks per-feature running statistics for basic drift detection.

    Compares running stats against a baseline (training-time stats)
    to detect distribution shifts.
    """

    feature_stats: Dict[str, RunningStats] = field(default_factory=dict)
    baseline_means: Dict[str, float] = field(default_factory=dict)
    baseline_stds: Dict[str, float] = field(default_factory=dict)

    def record(self, features: Dict[str, float]) -> None:
        """Update running stats for each feature."""
        for name, value in features.items():
            if name not in self.feature_stats:
                self.feature_stats[name] = RunningStats()
            self.feature_stats[name].update(value)

    def set_baseline(
        self, means: Dict[str, float], stds: Dict[str, float],
    ) -> None:
        """Set training-time baseline statistics for drift comparison."""
        self.baseline_means = dict(means)
        self.baseline_stds = dict(stds)

    def get_drift_report(self, top_n: int = 10) -> Dict[str, Any]:
        """Compute per-feature drift scores (z-score of mean shift).

        Returns top-N features with highest drift.
        """
        drift_scores: list[tuple[str, float]] = []
        for name, stats in self.feature_stats.items():
            if stats.count < 10:
                continue
            baseline_mean = self.baseline_means.get(name)
            baseline_std = self.baseline_stds.get(name)
            if baseline_mean is None or baseline_std is None or baseline_std == 0:
                continue
            z = abs(stats.mean - baseline_mean) / baseline_std
            drift_scores.append((name, round(z, 4)))

        drift_scores.sort(key=lambda x: x[1], reverse=True)

        return {
            "tracked_features": len(self.feature_stats),
            "baseline_features": len(self.baseline_means),
            "top_drift": [
                {"feature": name, "z_score": z, "current_mean": round(self.feature_stats[name].mean, 4)}
                for name, z in drift_scores[:top_n]
            ],
            "has_significant_drift": any(z > 2.0 for _, z in drift_scores),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            name: stats.to_dict()
            for name, stats in sorted(self.feature_stats.items())
        }


@dataclass
class RiskMonitoringSnapshot:
    """Complete snapshot of risk prediction monitoring state."""

    model_name: str
    model_version: str
    prediction_count: int
    distribution: Dict[str, Any]
    confidence_stats: Dict[str, Any]
    probability_stats: Dict[str, Any]
    latency: Dict[str, Any]
    drift_report: Dict[str, Any]
    uptime_seconds: float
    last_prediction_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RiskPredictionMonitoringService
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RiskPredictionMonitoringService:
    """Monitors runtime behaviour of the risk prediction model.

    Thread-safe, in-memory service.  Integrates with the shared
    ``ModelMonitoringService`` for unified health reporting.

    Does NOT retrain models.
    """

    def __init__(
        self,
        model_monitoring: Optional[ModelMonitoringService] = None,
        model_name: str = MODEL_NAME,
        model_version: str = "1.0.0",
    ) -> None:
        self._model_name = model_name
        self._model_version = model_version
        self._lock = Lock()
        self._start_time = time.monotonic()

        # Trackers
        self._distribution = PredictionDistribution()
        self._confidence_stats = RunningStats()
        self._probability_stats = RunningStats()
        self._latency = LatencyHistogram()
        self._drift_tracker = FeatureDriftTracker()
        self._last_prediction_at: Optional[str] = None
        self._prediction_count = 0

        # Integrate with existing monitoring framework
        self._model_monitoring = model_monitoring
        if model_monitoring:
            self._register_with_framework()

    def _register_with_framework(self) -> None:
        """Register the risk model with the shared ModelMonitoringService."""
        try:
            metadata = create_model_metadata_from_path(
                model_name=self._model_name,
                model_path=DEFAULT_MODEL_PATH,
                algorithm="XGBoost",
                version=self._model_version,
            )
            self._model_monitoring.register_model(metadata)
            logger.info(
                "Risk model registered with monitoring framework: %s v%s",
                self._model_name, self._model_version,
            )
        except Exception:
            logger.warning(
                "Could not register risk model with monitoring framework",
                exc_info=True,
            )

    # ── Recording ──

    def record_prediction(
        self,
        probability: float,
        risk_level: RiskLevel,
        confidence: float,
        latency_ms: float,
        features: Optional[Dict[str, float]] = None,
        sensor_id: Optional[str] = None,
    ) -> None:
        """Record a single prediction for monitoring.

        Should be called after every successful inference.

        Args:
            probability: Accident probability (0.0–1.0).
            risk_level: Classified risk level.
            confidence: Model confidence score.
            latency_ms: Inference time in milliseconds.
            features: Input feature dict (for drift tracking).
            sensor_id: Sensor that was evaluated.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._prediction_count += 1
            self._last_prediction_at = now

            # Distribution
            level_attr = risk_level.value.lower()
            if hasattr(self._distribution, level_attr):
                setattr(
                    self._distribution, level_attr,
                    getattr(self._distribution, level_attr) + 1,
                )

            # Confidence
            self._confidence_stats.update(confidence)

            # Probability
            self._probability_stats.update(probability)

            # Latency
            self._latency.record(latency_ms)

            # Feature drift
            if features:
                self._drift_tracker.record(features)

        # Forward to shared monitoring framework
        if self._model_monitoring and sensor_id:
            is_anomaly = risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
            self._model_monitoring.record_inference(
                model_name=self._model_name,
                sensor_id=sensor_id,
                anomaly_score=probability,
                is_anomaly=is_anomaly,
                latency_ms=latency_ms,
            )

    def record_error(self, error: str) -> None:
        """Record a prediction failure."""
        if self._model_monitoring:
            self._model_monitoring.record_loading_failure(
                self._model_name, error,
            )
        logger.error("Risk prediction error recorded: %s", error)

    # ── Drift baseline ──

    def set_drift_baseline(
        self, means: Dict[str, float], stds: Dict[str, float],
    ) -> None:
        """Set training-time feature statistics for drift detection.

        Should be called once at startup using the training dataset stats.
        """
        with self._lock:
            self._drift_tracker.set_baseline(means, stds)
        logger.info(
            "Feature drift baseline set: %d features", len(means),
        )

    # ── Queries ──

    @property
    def prediction_count(self) -> int:
        return self._prediction_count

    @property
    def model_version(self) -> str:
        return self._model_version

    def get_distribution(self) -> Dict[str, Any]:
        """Get prediction distribution by risk level."""
        with self._lock:
            return self._distribution.to_dict()

    def get_confidence_stats(self) -> Dict[str, Any]:
        """Get confidence score statistics."""
        with self._lock:
            return self._confidence_stats.to_dict()

    def get_probability_stats(self) -> Dict[str, Any]:
        """Get probability statistics."""
        with self._lock:
            return self._probability_stats.to_dict()

    def get_latency_stats(self) -> Dict[str, Any]:
        """Get latency histogram and stats."""
        with self._lock:
            return self._latency.to_dict()

    def get_drift_report(self, top_n: int = 10) -> Dict[str, Any]:
        """Get feature drift report."""
        with self._lock:
            return self._drift_tracker.get_drift_report(top_n)

    def get_feature_stats(self) -> Dict[str, Any]:
        """Get per-feature running statistics."""
        with self._lock:
            return self._drift_tracker.to_dict()

    def get_snapshot(self) -> RiskMonitoringSnapshot:
        """Get a complete monitoring snapshot."""
        with self._lock:
            return RiskMonitoringSnapshot(
                model_name=self._model_name,
                model_version=self._model_version,
                prediction_count=self._prediction_count,
                distribution=self._distribution.to_dict(),
                confidence_stats=self._confidence_stats.to_dict(),
                probability_stats=self._probability_stats.to_dict(),
                latency=self._latency.to_dict(),
                drift_report=self._drift_tracker.get_drift_report(),
                uptime_seconds=round(time.monotonic() - self._start_time, 2),
                last_prediction_at=self._last_prediction_at,
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get a concise monitoring summary (for health endpoints)."""
        with self._lock:
            return {
                "model_name": self._model_name,
                "model_version": self._model_version,
                "prediction_count": self._prediction_count,
                "avg_confidence": round(self._confidence_stats.mean, 4),
                "avg_probability": round(self._probability_stats.mean, 4),
                "avg_latency_ms": round(self._latency.avg_ms, 3),
                "distribution": self._distribution.percentages(),
                "has_drift": self._drift_tracker.get_drift_report().get(
                    "has_significant_drift", False,
                ),
                "last_prediction_at": self._last_prediction_at,
            }

    # ── Reset ──

    def reset(self) -> None:
        """Reset all monitoring state. Primarily for testing."""
        with self._lock:
            self._distribution = PredictionDistribution()
            self._confidence_stats = RunningStats()
            self._probability_stats = RunningStats()
            self._latency = LatencyHistogram()
            self._drift_tracker = FeatureDriftTracker()
            self._last_prediction_at = None
            self._prediction_count = 0
            self._start_time = time.monotonic()
