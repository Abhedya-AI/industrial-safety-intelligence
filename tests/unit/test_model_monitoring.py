"""Comprehensive unit tests for the Model Monitoring Service.

Covers:
  1. Model registration and metadata
  2. Inference recording and statistics
  3. Loading failure tracking
  4. Health checks
  5. Model validation
  6. Inference logging
  7. System summary
  8. Active detector tracking
  9. Edge cases
  10. create_model_metadata_from_path helper
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

from app.sensor_intelligence.services.model_monitoring_service import (
    InferenceLogEntry,
    InferenceStats,
    ModelHealthReport,
    ModelHealthStatus,
    ModelMetadata,
    ModelMonitoringService,
    create_model_metadata_from_path,
)


# ── Fixtures ──


@pytest.fixture
def service() -> ModelMonitoringService:
    svc = ModelMonitoringService()
    yield svc
    svc.reset()


@pytest.fixture
def registered_service(service: ModelMonitoringService, tmp_path: Path) -> ModelMonitoringService:
    """Service with a registered model backed by a real file."""
    model_file = tmp_path / "test_model.pkl"
    model_file.write_bytes(b"fake model data for testing")

    meta = ModelMetadata(
        model_name="isolation_forest",
        model_version="2.1.0",
        training_date="2026-06-15T10:00:00Z",
        dataset_version="features_v3",
        model_path=str(model_file),
        model_size_bytes=model_file.stat().st_size,
        algorithm="IsolationForest",
        feature_count=42,
    )
    service.register_model(meta)
    service.set_active_detector("isolation_forest")
    return service


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Model registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_register_model(service: ModelMonitoringService):
    meta = ModelMetadata(model_name="test_model", model_version="1.0.0")
    service.register_model(meta)
    assert "test_model" in service.get_all_model_names()


def test_get_model_metadata(service: ModelMonitoringService):
    meta = ModelMetadata(
        model_name="test_model",
        model_version="2.0.0",
        training_date="2026-06-01",
        dataset_version="v3",
        algorithm="IsolationForest",
        feature_count=42,
    )
    service.register_model(meta)
    retrieved = service.get_model_metadata("test_model")
    assert retrieved is not None
    assert retrieved.model_version == "2.0.0"
    assert retrieved.algorithm == "IsolationForest"
    assert retrieved.feature_count == 42


def test_get_metadata_unregistered(service: ModelMonitoringService):
    assert service.get_model_metadata("nonexistent") is None


def test_register_multiple_models(service: ModelMonitoringService):
    service.register_model(ModelMetadata(model_name="model_a"))
    service.register_model(ModelMetadata(model_name="model_b"))
    names = service.get_all_model_names()
    assert "model_a" in names
    assert "model_b" in names


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Active detector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_set_active_detector(service: ModelMonitoringService):
    service.set_active_detector("autoencoder")
    assert service.get_active_detector() == "autoencoder"


def test_active_detector_initially_none(service: ModelMonitoringService):
    assert service.get_active_detector() is None


def test_change_active_detector(service: ModelMonitoringService):
    service.set_active_detector("isolation_forest")
    service.set_active_detector("autoencoder")
    assert service.get_active_detector() == "autoencoder"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Inference recording
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_record_inference_normal(service: ModelMonitoringService):
    service.register_model(ModelMetadata(model_name="m1"))
    service.record_inference("m1", "S001", 0.15, False, 5.0)
    stats = service.get_inference_stats("m1")
    assert stats.prediction_count == 1
    assert stats.normal_count == 1
    assert stats.anomaly_count == 0
    assert stats.total_anomaly_score == 0.15


def test_record_inference_anomaly(service: ModelMonitoringService):
    service.register_model(ModelMetadata(model_name="m1"))
    service.record_inference("m1", "S001", 0.85, True, 3.0)
    stats = service.get_inference_stats("m1")
    assert stats.anomaly_count == 1
    assert stats.normal_count == 0


def test_record_multiple_inferences(service: ModelMonitoringService):
    service.register_model(ModelMetadata(model_name="m1"))
    service.record_inference("m1", "S001", 0.1, False)
    service.record_inference("m1", "S002", 0.9, True)
    service.record_inference("m1", "S003", 0.5, False)

    stats = service.get_inference_stats("m1")
    assert stats.prediction_count == 3
    assert stats.anomaly_count == 1
    assert stats.normal_count == 2
    assert abs(stats.avg_anomaly_score - 0.5) < 0.01
    assert stats.min_anomaly_score == 0.1
    assert stats.max_anomaly_score == 0.9


def test_avg_anomaly_score(service: ModelMonitoringService):
    service.register_model(ModelMetadata(model_name="m1"))
    service.record_inference("m1", "S1", 0.2, False)
    service.record_inference("m1", "S2", 0.8, True)
    stats = service.get_inference_stats("m1")
    assert abs(stats.avg_anomaly_score - 0.5) < 0.01


def test_anomaly_rate(service: ModelMonitoringService):
    service.register_model(ModelMetadata(model_name="m1"))
    service.record_inference("m1", "S1", 0.1, False)
    service.record_inference("m1", "S2", 0.9, True)
    service.record_inference("m1", "S3", 0.1, False)
    service.record_inference("m1", "S4", 0.9, True)
    stats = service.get_inference_stats("m1")
    assert abs(stats.anomaly_rate - 50.0) < 0.01


def test_inference_stats_empty_model(service: ModelMonitoringService):
    stats = service.get_inference_stats("nonexistent")
    assert stats.prediction_count == 0
    assert stats.avg_anomaly_score == 0.0
    assert stats.anomaly_rate == 0.0


def test_last_inference_timestamp(service: ModelMonitoringService):
    service.register_model(ModelMetadata(model_name="m1"))
    service.record_inference("m1", "S1", 0.3, False)
    stats = service.get_inference_stats("m1")
    assert stats.last_inference_at is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Loading failure tracking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_record_loading_failure(service: ModelMonitoringService):
    service.record_loading_failure("m1", "FileNotFoundError: model.pkl")
    stats = service.get_inference_stats("m1")
    assert stats.loading_failure_count == 1
    assert stats.last_error == "FileNotFoundError: model.pkl"
    assert stats.last_error_at is not None


def test_multiple_loading_failures(service: ModelMonitoringService):
    service.record_loading_failure("m1", "error 1")
    service.record_loading_failure("m1", "error 2")
    stats = service.get_inference_stats("m1")
    assert stats.loading_failure_count == 2
    assert stats.last_error == "error 2"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Health checks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_health_check_healthy(registered_service: ModelMonitoringService):
    # Add a recent inference
    registered_service.record_inference("isolation_forest", "S1", 0.1, False)
    report = registered_service.get_model_health("isolation_forest")
    assert report.health_status == ModelHealthStatus.HEALTHY
    assert report.is_loaded is True
    assert report.checks["model_registered"] is True
    assert report.checks["model_file_exists"] is True
    assert report.checks["no_loading_failures"] is True
    assert report.checks["recent_inference"] is True


def test_health_check_unregistered(service: ModelMonitoringService):
    report = service.get_model_health("nonexistent")
    assert report.health_status == ModelHealthStatus.UNHEALTHY
    assert report.is_loaded is False
    assert report.checks["model_registered"] is False


def test_health_check_with_loading_failure(
    registered_service: ModelMonitoringService,
):
    registered_service.record_loading_failure("isolation_forest", "corrupt file")
    report = registered_service.get_model_health("isolation_forest")
    assert report.health_status == ModelHealthStatus.UNHEALTHY
    assert report.checks["no_loading_failures"] is False


def test_health_check_no_recent_inference(
    registered_service: ModelMonitoringService,
):
    """No inferences recorded → recent_inference check fails."""
    report = registered_service.get_model_health("isolation_forest")
    assert report.checks["recent_inference"] is False


def test_health_report_has_uptime(registered_service: ModelMonitoringService):
    report = registered_service.get_model_health("isolation_forest")
    assert report.uptime_seconds >= 0


def test_health_report_shows_active_detector(
    registered_service: ModelMonitoringService,
):
    report = registered_service.get_model_health("isolation_forest")
    assert report.details["active_detector"] == "isolation_forest"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Model validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_validate_model_valid(registered_service: ModelMonitoringService):
    result = registered_service.validate_model("isolation_forest")
    assert result["valid"] is True
    assert result["checks"]["registered"] is True
    assert result["checks"]["file_exists"] is True
    assert result["checks"]["no_recent_failures"] is True
    assert result["checks"]["version_present"] is True


def test_validate_model_unregistered(service: ModelMonitoringService):
    result = service.validate_model("nonexistent")
    assert result["valid"] is False
    assert result["checks"]["registered"] is False


def test_validate_model_missing_file(service: ModelMonitoringService):
    meta = ModelMetadata(
        model_name="broken",
        model_path="/nonexistent/path/model.pkl",
    )
    service.register_model(meta)
    result = service.validate_model("broken")
    assert result["valid"] is False
    assert result["checks"]["file_exists"] is False


def test_validate_model_with_failures(registered_service: ModelMonitoringService):
    registered_service.record_loading_failure("isolation_forest", "err")
    result = registered_service.validate_model("isolation_forest")
    assert result["valid"] is False
    assert result["checks"]["no_recent_failures"] is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Inference logging
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_inference_log_recorded(service: ModelMonitoringService):
    service.register_model(ModelMetadata(model_name="m1"))
    service.record_inference("m1", "S001", 0.5, True, 2.5)
    entries = service.get_inference_log("m1")
    assert len(entries) == 1
    assert entries[0].sensor_id == "S001"
    assert entries[0].anomaly_score == 0.5
    assert entries[0].status == "ANOMALY"
    assert entries[0].latency_ms == 2.5


def test_inference_log_limit(service: ModelMonitoringService):
    service.register_model(ModelMetadata(model_name="m1"))
    for i in range(20):
        service.record_inference("m1", f"S{i:03d}", 0.1 * (i % 10), False)

    entries = service.get_inference_log("m1", limit=5)
    assert len(entries) == 5


def test_inference_log_filter_by_model(service: ModelMonitoringService):
    service.register_model(ModelMetadata(model_name="m1"))
    service.register_model(ModelMetadata(model_name="m2"))
    service.record_inference("m1", "S1", 0.1, False)
    service.record_inference("m2", "S2", 0.2, False)

    entries_m1 = service.get_inference_log("m1")
    entries_m2 = service.get_inference_log("m2")
    assert len(entries_m1) == 1
    assert len(entries_m2) == 1
    assert entries_m1[0].model_name == "m1"


def test_inference_log_all(service: ModelMonitoringService):
    service.register_model(ModelMetadata(model_name="m1"))
    service.register_model(ModelMetadata(model_name="m2"))
    service.record_inference("m1", "S1", 0.1, False)
    service.record_inference("m2", "S2", 0.2, False)

    all_entries = service.get_inference_log()
    assert len(all_entries) == 2


def test_inference_log_bounded(service: ModelMonitoringService):
    """Log should not exceed max_log_entries."""
    service._max_log_entries = 10
    service.register_model(ModelMetadata(model_name="m1"))
    for i in range(50):
        service.record_inference("m1", f"S{i}", 0.1, False)

    all_entries = service.get_inference_log()
    assert len(all_entries) <= 10


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. System summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_system_summary(registered_service: ModelMonitoringService):
    registered_service.record_inference("isolation_forest", "S1", 0.2, False)
    registered_service.record_inference("isolation_forest", "S2", 0.9, True)

    summary = registered_service.get_system_summary()
    assert summary["registered_models"] == 1
    assert "isolation_forest" in summary["model_names"]
    assert summary["active_detector"] == "isolation_forest"
    assert summary["total_predictions"] == 2
    assert summary["total_anomalies"] == 1
    assert summary["total_loading_failures"] == 0
    assert summary["uptime_seconds"] >= 0


def test_system_summary_empty(service: ModelMonitoringService):
    summary = service.get_system_summary()
    assert summary["registered_models"] == 0
    assert summary["total_predictions"] == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Reset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_reset(service: ModelMonitoringService):
    service.register_model(ModelMetadata(model_name="m1"))
    service.record_inference("m1", "S1", 0.5, True)
    service.set_active_detector("m1")
    service.reset()

    assert service.get_all_model_names() == []
    assert service.get_active_detector() is None
    assert service.get_inference_log() == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. create_model_metadata_from_path helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_create_metadata_from_path(tmp_path: Path):
    model_file = tmp_path / "model.pkl"
    model_file.write_bytes(b"x" * 1024)

    meta = create_model_metadata_from_path(
        model_name="test_model",
        model_path=str(model_file),
        algorithm="IsolationForest",
        version="3.0.0",
        dataset_version="features_v5",
    )
    assert meta.model_name == "test_model"
    assert meta.model_version == "3.0.0"
    assert meta.model_size_bytes == 1024
    assert meta.algorithm == "IsolationForest"
    assert meta.dataset_version == "features_v5"
    assert meta.training_date is not None  # Auto-read from mtime


def test_create_metadata_nonexistent_file():
    meta = create_model_metadata_from_path(
        model_name="missing",
        model_path="/nonexistent/model.pkl",
    )
    assert meta.model_size_bytes == 0
    assert meta.training_date is None
