"""Model Monitoring Service — tracks ML model health, metadata, and inference stats.

Responsibilities:
  1. Track model metadata (version, training date, dataset version)
  2. Record inference statistics (prediction counts, avg score, anomaly counts)
  3. Monitor model health (loading failures, validation checks)
  4. Log inference activity for audit and debugging

Architecture:
  - Operates in-memory with thread-safe counters (no DB dependency)
  - Queries detector instances via DetectorFactory / BaseAnomalyDetector
  - Uses existing logging infrastructure
  - Does NOT retrain models — inference and monitoring only
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ModelHealthStatus(str, Enum):
    """Health status of a deployed model."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    NOT_LOADED = "NOT_LOADED"


@dataclass
class ModelMetadata:
    """Metadata about a trained and deployed model."""

    model_name: str
    model_version: str = "1.0.0"
    training_date: Optional[str] = None
    dataset_version: Optional[str] = None
    model_path: Optional[str] = None
    model_size_bytes: int = 0
    algorithm: str = ""
    feature_count: int = 0


@dataclass
class InferenceStats:
    """Running statistics for model inference."""

    prediction_count: int = 0
    anomaly_count: int = 0
    normal_count: int = 0
    total_anomaly_score: float = 0.0
    min_anomaly_score: float = float("inf")
    max_anomaly_score: float = 0.0
    loading_failure_count: int = 0
    last_inference_at: Optional[str] = None
    last_error: Optional[str] = None
    last_error_at: Optional[str] = None

    @property
    def avg_anomaly_score(self) -> float:
        """Average anomaly score across all predictions."""
        if self.prediction_count == 0:
            return 0.0
        return self.total_anomaly_score / self.prediction_count

    @property
    def anomaly_rate(self) -> float:
        """Percentage of predictions classified as anomalies."""
        if self.prediction_count == 0:
            return 0.0
        return (self.anomaly_count / self.prediction_count) * 100.0


@dataclass
class ModelHealthReport:
    """Complete health report for a model."""

    model_name: str
    health_status: ModelHealthStatus
    is_loaded: bool
    metadata: ModelMetadata
    inference_stats: InferenceStats
    uptime_seconds: float
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceLogEntry:
    """Single inference log entry for audit trail."""

    timestamp: str
    model_name: str
    sensor_id: str
    anomaly_score: float
    status: str  # "NORMAL" | "ANOMALY"
    latency_ms: float


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model Monitoring Service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ModelMonitoringService:
    """Service for tracking ML model health, metadata, and inference statistics.

    Thread-safe singleton that maintains in-memory counters and can be
    queried by health endpoints and dashboards.
    """

    def __init__(self) -> None:
        self._metadata: dict[str, ModelMetadata] = {}
        self._stats: dict[str, InferenceStats] = {}
        self._active_detector: Optional[str] = None
        self._start_time = time.monotonic()
        self._started_at = datetime.now(timezone.utc)
        self._lock = Lock()
        self._inference_log: list[InferenceLogEntry] = []
        self._max_log_entries = 1000

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Model registration
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def register_model(self, metadata: ModelMetadata) -> None:
        """Register a model for monitoring.

        Should be called when a model is loaded at startup.

        Args:
            metadata: ModelMetadata describing the model.
        """
        with self._lock:
            self._metadata[metadata.model_name] = metadata
            if metadata.model_name not in self._stats:
                self._stats[metadata.model_name] = InferenceStats()
            logger.info(
                "Model registered for monitoring: %s (v%s)",
                metadata.model_name, metadata.model_version,
            )

    def set_active_detector(self, detector_name: str) -> None:
        """Set the currently active detector.

        Args:
            detector_name: Name of the active detector (e.g. "isolation_forest").
        """
        with self._lock:
            self._active_detector = detector_name
            logger.info("Active detector set to: %s", detector_name)

    def get_active_detector(self) -> Optional[str]:
        """Get the name of the currently active detector."""
        return self._active_detector

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Inference recording
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def record_inference(
        self,
        model_name: str,
        sensor_id: str,
        anomaly_score: float,
        is_anomaly: bool,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a single inference result.

        Called after every prediction to update running statistics.

        Args:
            model_name: Name of the model that produced the result.
            sensor_id: Sensor that was evaluated.
            anomaly_score: Raw anomaly score.
            is_anomaly: Whether the result was classified as anomaly.
            latency_ms: Inference latency in milliseconds.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            stats = self._stats.setdefault(model_name, InferenceStats())
            stats.prediction_count += 1
            stats.total_anomaly_score += anomaly_score
            stats.last_inference_at = now

            if anomaly_score < stats.min_anomaly_score:
                stats.min_anomaly_score = anomaly_score
            if anomaly_score > stats.max_anomaly_score:
                stats.max_anomaly_score = anomaly_score

            if is_anomaly:
                stats.anomaly_count += 1
            else:
                stats.normal_count += 1

            # Log entry
            entry = InferenceLogEntry(
                timestamp=now,
                model_name=model_name,
                sensor_id=sensor_id,
                anomaly_score=anomaly_score,
                status="ANOMALY" if is_anomaly else "NORMAL",
                latency_ms=latency_ms,
            )
            self._inference_log.append(entry)
            if len(self._inference_log) > self._max_log_entries:
                self._inference_log = self._inference_log[-self._max_log_entries:]

        logger.debug(
            "Inference recorded: model=%s sensor=%s score=%.4f anomaly=%s latency=%.1fms",
            model_name, sensor_id, anomaly_score, is_anomaly, latency_ms,
        )

    def record_loading_failure(self, model_name: str, error: str) -> None:
        """Record a model loading failure.

        Args:
            model_name: Name of the model that failed to load.
            error: Error message.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            stats = self._stats.setdefault(model_name, InferenceStats())
            stats.loading_failure_count += 1
            stats.last_error = error
            stats.last_error_at = now

        logger.error(
            "Model loading failure: %s — %s", model_name, error
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Health checks
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_model_health(self, model_name: str) -> ModelHealthReport:
        """Run health checks for a specific model.

        Checks:
          - model_registered: Is the model registered?
          - model_file_exists: Does the model file exist on disk?
          - no_loading_failures: Have there been no loading failures?
          - recent_inference: Has inference been performed recently?
          - anomaly_rate_normal: Is the anomaly rate below 50%?

        Args:
            model_name: Name of the model to check.

        Returns:
            ModelHealthReport with check results.
        """
        with self._lock:
            metadata = self._metadata.get(model_name, ModelMetadata(model_name=model_name))
            stats = self._stats.get(model_name, InferenceStats())
            is_registered = model_name in self._metadata

        checks = {}

        # Check 1: Model registered
        checks["model_registered"] = is_registered

        # Check 2: Model file exists
        if metadata.model_path:
            checks["model_file_exists"] = Path(metadata.model_path).exists()
        else:
            checks["model_file_exists"] = False

        # Check 3: No loading failures
        checks["no_loading_failures"] = stats.loading_failure_count == 0

        # Check 4: Recent inference (within last hour)
        if stats.last_inference_at:
            last = datetime.fromisoformat(stats.last_inference_at)
            age_seconds = (datetime.now(timezone.utc) - last).total_seconds()
            checks["recent_inference"] = age_seconds < 3600
        else:
            checks["recent_inference"] = False

        # Check 5: Anomaly rate is reasonable (< 50%)
        checks["anomaly_rate_normal"] = stats.anomaly_rate < 50.0

        # Determine overall health
        failed_checks = [k for k, v in checks.items() if not v]
        critical_checks = {"model_registered", "no_loading_failures"}
        critical_failures = critical_checks.intersection(failed_checks)

        if critical_failures:
            health = ModelHealthStatus.UNHEALTHY
        elif len(failed_checks) > 1:
            health = ModelHealthStatus.DEGRADED
        elif not is_registered:
            health = ModelHealthStatus.NOT_LOADED
        else:
            health = ModelHealthStatus.HEALTHY

        uptime = time.monotonic() - self._start_time

        return ModelHealthReport(
            model_name=model_name,
            health_status=health,
            is_loaded=is_registered,
            metadata=metadata,
            inference_stats=stats,
            uptime_seconds=round(uptime, 2),
            checks=checks,
            details={
                "active_detector": self._active_detector,
                "started_at": self._started_at.isoformat(),
            },
        )

    def validate_model(self, model_name: str) -> dict[str, Any]:
        """Validate a model's integrity and readiness.

        Returns a validation result dict with pass/fail for each check.

        Args:
            model_name: Name of the model to validate.
        """
        with self._lock:
            metadata = self._metadata.get(model_name)
            stats = self._stats.get(model_name, InferenceStats())

        result = {
            "model_name": model_name,
            "valid": True,
            "checks": {},
        }

        # Check: registered
        result["checks"]["registered"] = metadata is not None
        if not metadata:
            result["valid"] = False
            return result

        # Check: model file exists
        if metadata.model_path:
            exists = Path(metadata.model_path).exists()
            result["checks"]["file_exists"] = exists
            if not exists:
                result["valid"] = False
        else:
            result["checks"]["file_exists"] = False
            result["valid"] = False

        # Check: no recent loading failures
        result["checks"]["no_recent_failures"] = stats.loading_failure_count == 0
        if stats.loading_failure_count > 0:
            result["valid"] = False

        # Check: model version present
        result["checks"]["version_present"] = bool(metadata.model_version)

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Metadata / queries
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Get metadata for a registered model."""
        return self._metadata.get(model_name)

    def get_inference_stats(self, model_name: str) -> InferenceStats:
        """Get inference statistics for a model."""
        return self._stats.get(model_name, InferenceStats())

    def get_all_model_names(self) -> list[str]:
        """Get names of all registered models."""
        return list(self._metadata.keys())

    def get_inference_log(
        self,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[InferenceLogEntry]:
        """Get recent inference log entries.

        Args:
            model_name: Filter by model name (None = all).
            limit: Max entries to return.

        Returns:
            List of InferenceLogEntry (newest first).
        """
        with self._lock:
            entries = self._inference_log.copy()

        if model_name:
            entries = [e for e in entries if e.model_name == model_name]

        return list(reversed(entries[-limit:]))

    def get_system_summary(self) -> dict[str, Any]:
        """Get a summary of the entire monitoring system.

        Returns:
            Dict with model counts, active detector, uptime, etc.
        """
        with self._lock:
            models = list(self._metadata.keys())
            total_predictions = sum(s.prediction_count for s in self._stats.values())
            total_anomalies = sum(s.anomaly_count for s in self._stats.values())
            total_failures = sum(s.loading_failure_count for s in self._stats.values())

        uptime = time.monotonic() - self._start_time

        return {
            "registered_models": len(models),
            "model_names": models,
            "active_detector": self._active_detector,
            "total_predictions": total_predictions,
            "total_anomalies": total_anomalies,
            "total_loading_failures": total_failures,
            "uptime_seconds": round(uptime, 2),
            "started_at": self._started_at.isoformat(),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Reset (for testing)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def reset(self) -> None:
        """Reset all monitoring state. Primarily for testing."""
        with self._lock:
            self._metadata.clear()
            self._stats.clear()
            self._active_detector = None
            self._inference_log.clear()
            self._start_time = time.monotonic()
            self._started_at = datetime.now(timezone.utc)


def create_model_metadata_from_path(
    model_name: str,
    model_path: str,
    algorithm: str = "",
    version: str = "1.0.0",
    training_date: Optional[str] = None,
    dataset_version: Optional[str] = None,
) -> ModelMetadata:
    """Helper to create ModelMetadata from a model file path.

    Automatically reads file size. Useful when registering models at startup.
    """
    path = Path(model_path)
    size = path.stat().st_size if path.exists() else 0

    return ModelMetadata(
        model_name=model_name,
        model_version=version,
        training_date=training_date or (
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            if path.exists() else None
        ),
        dataset_version=dataset_version,
        model_path=model_path,
        model_size_bytes=size,
        algorithm=algorithm,
    )
