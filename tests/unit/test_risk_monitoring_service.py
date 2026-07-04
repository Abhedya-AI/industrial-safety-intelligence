"""Unit tests for the RiskPredictionMonitoringService.

All tests are in-memory — no DB, no model loading.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from app.risk_prediction.domain.value_objects import RiskLevel
from app.risk_prediction.services.monitoring_service import (
    FeatureDriftTracker,
    LatencyHistogram,
    PredictionDistribution,
    RiskMonitoringSnapshot,
    RiskPredictionMonitoringService,
    RunningStats,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. RunningStats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRunningStats:
    def test_empty(self):
        s = RunningStats()
        assert s.count == 0
        assert s.mean == 0.0
        assert s.std_dev == 0.0

    def test_single_value(self):
        s = RunningStats()
        s.update(5.0)
        assert s.count == 1
        assert s.mean == 5.0
        assert s.std_dev == 0.0
        assert s.min_value == 5.0
        assert s.max_value == 5.0

    def test_multiple_values(self):
        s = RunningStats()
        for v in [2, 4, 4, 4, 5, 5, 7, 9]:
            s.update(v)
        assert s.count == 8
        assert s.mean == pytest.approx(5.0)
        assert s.std_dev == pytest.approx(2.138, abs=0.01)
        assert s.min_value == 2
        assert s.max_value == 9

    def test_nan_ignored(self):
        s = RunningStats()
        s.update(3.0)
        s.update(float("nan"))
        assert s.count == 1
        assert s.mean == 3.0

    def test_inf_ignored(self):
        s = RunningStats()
        s.update(3.0)
        s.update(float("inf"))
        assert s.count == 1

    def test_to_dict(self):
        s = RunningStats()
        s.update(10.0)
        d = s.to_dict()
        assert d["count"] == 1
        assert d["mean"] == 10.0
        assert d["min"] == 10.0
        assert d["max"] == 10.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. PredictionDistribution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPredictionDistribution:
    def test_empty(self):
        d = PredictionDistribution()
        assert d.total == 0
        pct = d.percentages()
        assert pct["LOW"] == 0.0

    def test_counts(self):
        d = PredictionDistribution(low=70, medium=20, high=8, critical=2)
        assert d.total == 100

    def test_percentages(self):
        d = PredictionDistribution(low=50, medium=30, high=15, critical=5)
        pct = d.percentages()
        assert pct["LOW"] == 50.0
        assert pct["CRITICAL"] == 5.0
        assert sum(pct.values()) == pytest.approx(100.0)

    def test_to_dict(self):
        d = PredictionDistribution(low=1, medium=2, high=3, critical=4)
        result = d.to_dict()
        assert result["total"] == 10
        assert "counts" in result
        assert "percentages" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. LatencyHistogram
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLatencyHistogram:
    def test_empty(self):
        h = LatencyHistogram()
        assert h.count == 0
        assert h.avg_ms == 0.0

    def test_buckets(self):
        h = LatencyHistogram()
        h.record(0.5)   # <1ms
        h.record(3.0)   # 1-5ms
        h.record(7.0)   # 5-10ms
        h.record(25.0)  # 10-50ms
        h.record(75.0)  # 50-100ms
        h.record(200.0) # 100-500ms
        h.record(600.0) # >500ms
        assert h.count == 7
        d = h.to_dict()
        assert d["buckets"]["<1ms"] == 1
        assert d["buckets"]["1-5ms"] == 1
        assert d["buckets"]["10-50ms"] == 1
        assert d["buckets"][">500ms"] == 1

    def test_avg(self):
        h = LatencyHistogram()
        h.record(10.0)
        h.record(20.0)
        assert h.avg_ms == 15.0

    def test_max(self):
        h = LatencyHistogram()
        h.record(5.0)
        h.record(100.0)
        h.record(50.0)
        assert h.max_ms == 100.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. FeatureDriftTracker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFeatureDriftTracker:
    def test_record_updates_stats(self):
        t = FeatureDriftTracker()
        for _ in range(20):
            t.record({"Temp": 85.0, "Pressure": 15.0})
        assert "Temp" in t.feature_stats
        assert t.feature_stats["Temp"].count == 20
        assert t.feature_stats["Temp"].mean == pytest.approx(85.0)

    def test_drift_without_baseline(self):
        t = FeatureDriftTracker()
        for i in range(20):
            t.record({"Temp": 50.0 + i})
        report = t.get_drift_report()
        assert report["tracked_features"] == 1
        assert report["baseline_features"] == 0
        assert report["top_drift"] == []

    def test_drift_with_baseline(self):
        t = FeatureDriftTracker()
        t.set_baseline({"Temp": 50.0}, {"Temp": 5.0})
        # Record values that are significantly shifted
        for _ in range(20):
            t.record({"Temp": 80.0})  # 6 std devs away
        report = t.get_drift_report()
        assert report["has_significant_drift"] is True
        assert len(report["top_drift"]) == 1
        assert report["top_drift"][0]["feature"] == "Temp"
        assert report["top_drift"][0]["z_score"] > 2.0

    def test_no_drift(self):
        t = FeatureDriftTracker()
        t.set_baseline({"Temp": 50.0}, {"Temp": 5.0})
        for _ in range(20):
            t.record({"Temp": 50.5})
        report = t.get_drift_report()
        assert report["has_significant_drift"] is False

    def test_to_dict(self):
        t = FeatureDriftTracker()
        t.record({"A": 1.0, "B": 2.0})
        d = t.to_dict()
        assert "A" in d
        assert "B" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. RiskPredictionMonitoringService — recording
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMonitoringServiceRecording:
    def test_record_prediction(self):
        svc = RiskPredictionMonitoringService()
        svc.record_prediction(
            probability=0.65,
            risk_level=RiskLevel.HIGH,
            confidence=0.85,
            latency_ms=12.5,
            features={"Temp": 80.0},
        )
        assert svc.prediction_count == 1

    def test_distribution_updated(self):
        svc = RiskPredictionMonitoringService()
        svc.record_prediction(0.1, RiskLevel.LOW, 0.9, 5.0)
        svc.record_prediction(0.3, RiskLevel.MEDIUM, 0.8, 5.0)
        svc.record_prediction(0.6, RiskLevel.HIGH, 0.7, 5.0)
        svc.record_prediction(0.9, RiskLevel.CRITICAL, 0.95, 5.0)

        dist = svc.get_distribution()
        assert dist["counts"]["low"] == 1
        assert dist["counts"]["medium"] == 1
        assert dist["counts"]["high"] == 1
        assert dist["counts"]["critical"] == 1
        assert dist["total"] == 4

    def test_confidence_tracked(self):
        svc = RiskPredictionMonitoringService()
        svc.record_prediction(0.5, RiskLevel.HIGH, 0.80, 5.0)
        svc.record_prediction(0.5, RiskLevel.HIGH, 0.90, 5.0)
        stats = svc.get_confidence_stats()
        assert stats["count"] == 2
        assert stats["mean"] == pytest.approx(0.85)

    def test_probability_tracked(self):
        svc = RiskPredictionMonitoringService()
        svc.record_prediction(0.2, RiskLevel.LOW, 0.9, 5.0)
        svc.record_prediction(0.8, RiskLevel.CRITICAL, 0.9, 5.0)
        stats = svc.get_probability_stats()
        assert stats["count"] == 2
        assert stats["mean"] == pytest.approx(0.5)

    def test_latency_tracked(self):
        svc = RiskPredictionMonitoringService()
        svc.record_prediction(0.5, RiskLevel.HIGH, 0.9, 10.0)
        svc.record_prediction(0.5, RiskLevel.HIGH, 0.9, 20.0)
        lat = svc.get_latency_stats()
        assert lat["count"] == 2
        assert lat["avg_ms"] == 15.0

    def test_features_tracked_for_drift(self):
        svc = RiskPredictionMonitoringService()
        svc.record_prediction(
            0.5, RiskLevel.HIGH, 0.9, 5.0,
            features={"Temp": 85.0, "Pressure": 22.0},
        )
        fstats = svc.get_feature_stats()
        assert "Temp" in fstats
        assert fstats["Temp"]["count"] == 1

    def test_last_prediction_timestamp(self):
        svc = RiskPredictionMonitoringService()
        assert svc.get_snapshot().last_prediction_at is None
        svc.record_prediction(0.5, RiskLevel.HIGH, 0.9, 5.0)
        assert svc.get_snapshot().last_prediction_at is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Integration with ModelMonitoringService
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMonitoringIntegration:
    def test_integrates_with_model_monitoring(self):
        from app.sensor_intelligence.services.model_monitoring_service import (
            ModelMonitoringService,
        )
        mm = ModelMonitoringService()
        svc = RiskPredictionMonitoringService(model_monitoring=mm)

        # Verify registration
        assert "xgboost_risk_prediction" in mm.get_all_model_names()

    def test_forwards_inference_to_framework(self):
        from app.sensor_intelligence.services.model_monitoring_service import (
            ModelMonitoringService,
        )
        mm = ModelMonitoringService()
        svc = RiskPredictionMonitoringService(model_monitoring=mm)

        svc.record_prediction(
            0.7, RiskLevel.HIGH, 0.85, 12.0,
            sensor_id="S001",
        )
        stats = mm.get_inference_stats("xgboost_risk_prediction")
        assert stats.prediction_count == 1
        assert stats.anomaly_count == 1  # HIGH → anomaly

    def test_forwards_low_as_normal(self):
        from app.sensor_intelligence.services.model_monitoring_service import (
            ModelMonitoringService,
        )
        mm = ModelMonitoringService()
        svc = RiskPredictionMonitoringService(model_monitoring=mm)

        svc.record_prediction(
            0.1, RiskLevel.LOW, 0.95, 5.0,
            sensor_id="S001",
        )
        stats = mm.get_inference_stats("xgboost_risk_prediction")
        assert stats.normal_count == 1
        assert stats.anomaly_count == 0

    def test_error_forwarded(self):
        from app.sensor_intelligence.services.model_monitoring_service import (
            ModelMonitoringService,
        )
        mm = ModelMonitoringService()
        svc = RiskPredictionMonitoringService(model_monitoring=mm)

        svc.record_error("Test error")
        stats = mm.get_inference_stats("xgboost_risk_prediction")
        assert stats.loading_failure_count == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Queries and snapshots
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMonitoringQueries:
    def test_get_snapshot(self):
        svc = RiskPredictionMonitoringService()
        svc.record_prediction(0.5, RiskLevel.HIGH, 0.85, 10.0)
        snap = svc.get_snapshot()
        assert isinstance(snap, RiskMonitoringSnapshot)
        assert snap.model_name == "xgboost_risk_prediction"
        assert snap.prediction_count == 1

    def test_snapshot_to_dict(self):
        svc = RiskPredictionMonitoringService()
        svc.record_prediction(0.5, RiskLevel.HIGH, 0.85, 10.0)
        d = svc.get_snapshot().to_dict()
        assert "distribution" in d
        assert "confidence_stats" in d
        assert "latency" in d
        assert "drift_report" in d

    def test_get_summary(self):
        svc = RiskPredictionMonitoringService()
        svc.record_prediction(0.7, RiskLevel.HIGH, 0.85, 10.0)
        svc.record_prediction(0.3, RiskLevel.MEDIUM, 0.75, 8.0)
        summary = svc.get_summary()
        assert summary["prediction_count"] == 2
        assert summary["avg_confidence"] == pytest.approx(0.8)
        assert summary["model_version"] == "1.0.0"

    def test_model_version_property(self):
        svc = RiskPredictionMonitoringService(model_version="2.0.0")
        assert svc.model_version == "2.0.0"

    def test_drift_baseline_and_report(self):
        svc = RiskPredictionMonitoringService()
        svc.set_drift_baseline(
            {"Temp": 50.0, "Pressure": 15.0},
            {"Temp": 5.0, "Pressure": 2.0},
        )
        for _ in range(20):
            svc.record_prediction(
                0.5, RiskLevel.HIGH, 0.85, 5.0,
                features={"Temp": 80.0, "Pressure": 15.5},
            )
        report = svc.get_drift_report()
        assert report["has_significant_drift"] is True
        assert report["top_drift"][0]["feature"] == "Temp"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Reset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMonitoringReset:
    def test_reset_clears_all(self):
        svc = RiskPredictionMonitoringService()
        svc.record_prediction(0.5, RiskLevel.HIGH, 0.85, 10.0)
        assert svc.prediction_count == 1
        svc.reset()
        assert svc.prediction_count == 0
        assert svc.get_distribution()["total"] == 0
        assert svc.get_confidence_stats()["count"] == 0
        assert svc.get_latency_stats()["count"] == 0
