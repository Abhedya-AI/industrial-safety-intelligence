"""Unit tests for the runtime anomaly detection module.

Covers:
  - AnomalyResult schema (serialization)
  - BaseAnomalyDetector (_ensure_loaded guard)
  - IsolationForestDetector (load, predict, anomaly_score, classify)
  - AutoencoderDetector (load, predict, anomaly_score, classify)
  - DetectorFactory (create, available, errors, register)
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.sensor_intelligence.anomaly_detection.base import BaseAnomalyDetector
from app.sensor_intelligence.anomaly_detection.schemas import (
    AnomalyResult,
    AnomalyStatus,
)
from app.sensor_intelligence.anomaly_detection.isolation_forest_detector import (
    IsolationForestDetector,
)
from app.sensor_intelligence.anomaly_detection.factory import DetectorFactory


# ── Helpers ──


def _train_and_save_if(tmpdir: Path, n: int = 200) -> tuple[Path, Path]:
    """Train a minimal IF model, save to tmpdir, return (model_path, scaler_path)."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (n, 5))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=10, contamination=0.05, random_state=42)
    model.fit(X_scaled)

    model_path = tmpdir / "if_model.pkl"
    scaler_path = tmpdir / "if_scaler.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    return model_path, scaler_path


def _train_and_save_ae(tmpdir: Path, n: int = 200, n_features: int = 5) -> tuple[Path, Path, Path]:
    """Train a minimal AE, save to tmpdir. Returns (model, threshold, scaler) paths."""
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorflow import keras

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (n, n_features))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Tiny autoencoder
    inputs = keras.Input(shape=(n_features,))
    x = keras.layers.Dense(4, activation="relu")(inputs)
    x = keras.layers.Dense(2, activation="relu")(x)
    x = keras.layers.Dense(4, activation="relu")(x)
    outputs = keras.layers.Dense(n_features, activation="linear")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_scaled, X_scaled, epochs=5, batch_size=32, verbose=0)

    # Compute threshold
    reconstructed = model.predict(X_scaled, verbose=0)
    errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
    threshold = float(np.percentile(errors, 95))

    model_path = tmpdir / "ae_model.keras"
    threshold_path = tmpdir / "ae_threshold.json"
    scaler_path = tmpdir / "ae_scaler.pkl"

    model.save(str(model_path))
    with open(threshold_path, "w") as f:
        json.dump({
            "threshold": threshold,
            "percentile": 95.0,
            "feature_count": n_features,
            "feature_columns": [f"feat_{i}" for i in range(n_features)],
        }, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    return model_path, threshold_path, scaler_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AnomalyResult schema
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_anomaly_result_to_dict():
    result = AnomalyResult(
        sensor_id="S001",
        score=0.85,
        status=AnomalyStatus.ANOMALY,
        detector_type="isolation_forest",
        confidence=0.9,
        threshold=0.5,
    )
    d = result.to_dict()
    assert d["sensor_id"] == "S001"
    assert d["score"] == 0.85
    assert d["status"] == "ANOMALY"
    assert d["detector_type"] == "isolation_forest"


def test_anomaly_result_normal():
    result = AnomalyResult(
        sensor_id="S002",
        score=0.1,
        status=AnomalyStatus.NORMAL,
        detector_type="autoencoder",
    )
    assert result.status == AnomalyStatus.NORMAL
    assert result.to_dict()["status"] == "NORMAL"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IsolationForestDetector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_if_not_loaded_raises():
    detector = IsolationForestDetector()
    assert not detector.is_loaded
    with pytest.raises(RuntimeError, match="not loaded"):
        detector.predict(np.zeros((1, 5)))


def test_if_load_file_not_found():
    detector = IsolationForestDetector()
    with pytest.raises(FileNotFoundError):
        detector.load_model(model_path="/nonexistent.pkl", scaler_path="/nonexistent.pkl")


def test_if_load_and_predict():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, scaler_path = _train_and_save_if(Path(tmpdir))
        detector = IsolationForestDetector()
        detector.load_model(model_path=model_path, scaler_path=scaler_path)

        assert detector.is_loaded
        assert detector.name == "isolation_forest"

        X = np.random.default_rng(42).normal(0, 1, (10, 5))
        preds = detector.predict(X)
        assert preds.shape == (10,)
        assert set(np.unique(preds)).issubset({-1, 1})


def test_if_anomaly_score_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, scaler_path = _train_and_save_if(Path(tmpdir))
        detector = IsolationForestDetector()
        detector.load_model(model_path=model_path, scaler_path=scaler_path)

        X = np.random.default_rng(42).normal(0, 1, (50, 5))
        scores = detector.anomaly_score(X)
        assert scores.shape == (50,)
        assert (scores >= 0).all()
        assert (scores <= 1).all()


def test_if_classify_returns_anomaly_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, scaler_path = _train_and_save_if(Path(tmpdir))
        detector = IsolationForestDetector()
        detector.load_model(model_path=model_path, scaler_path=scaler_path)

        X = np.random.default_rng(42).normal(0, 1, (5, 5))
        sensor_ids = [f"S{i:03d}" for i in range(5)]
        results = detector.classify(X, sensor_ids)

        assert len(results) == 5
        for r in results:
            assert isinstance(r, AnomalyResult)
            assert r.detector_type == "isolation_forest"
            assert r.status in (AnomalyStatus.NORMAL, AnomalyStatus.ANOMALY)


def test_if_classify_detects_outliers():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, scaler_path = _train_and_save_if(Path(tmpdir), n=500)
        detector = IsolationForestDetector()
        detector.load_model(model_path=model_path, scaler_path=scaler_path)

        # Normal data
        rng = np.random.default_rng(42)
        X_normal = rng.normal(0, 1, (5, 5))
        # Extreme outlier
        X_outlier = np.full((1, 5), 100.0)
        X = np.vstack([X_normal, X_outlier])
        sensor_ids = [f"S{i:03d}" for i in range(6)]

        results = detector.classify(X, sensor_ids)
        # The outlier should be classified as ANOMALY
        assert results[-1].status == AnomalyStatus.ANOMALY


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AutoencoderDetector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_ae_not_loaded_raises():
    from app.sensor_intelligence.anomaly_detection.autoencoder_detector import (
        AutoencoderDetector,
    )
    detector = AutoencoderDetector()
    assert not detector.is_loaded
    with pytest.raises(RuntimeError, match="not loaded"):
        detector.predict(np.zeros((1, 5)))


def test_ae_load_and_predict():
    from app.sensor_intelligence.anomaly_detection.autoencoder_detector import (
        AutoencoderDetector,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, threshold_path, scaler_path = _train_and_save_ae(Path(tmpdir))
        detector = AutoencoderDetector()
        detector.load_model(
            model_path=model_path,
            threshold_path=threshold_path,
            scaler_path=scaler_path,
        )

        assert detector.is_loaded
        assert detector.name == "autoencoder"

        X = np.random.default_rng(42).normal(0, 1, (10, 5))
        preds = detector.predict(X)
        assert preds.shape == (10,)
        assert set(np.unique(preds)).issubset({0, 1})


def test_ae_anomaly_score_positive():
    from app.sensor_intelligence.anomaly_detection.autoencoder_detector import (
        AutoencoderDetector,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, threshold_path, scaler_path = _train_and_save_ae(Path(tmpdir))
        detector = AutoencoderDetector()
        detector.load_model(
            model_path=model_path,
            threshold_path=threshold_path,
            scaler_path=scaler_path,
        )

        X = np.random.default_rng(42).normal(0, 1, (20, 5))
        scores = detector.anomaly_score(X)
        assert scores.shape == (20,)
        assert (scores >= 0).all()


def test_ae_classify_returns_anomaly_results():
    from app.sensor_intelligence.anomaly_detection.autoencoder_detector import (
        AutoencoderDetector,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, threshold_path, scaler_path = _train_and_save_ae(Path(tmpdir))
        detector = AutoencoderDetector()
        detector.load_model(
            model_path=model_path,
            threshold_path=threshold_path,
            scaler_path=scaler_path,
        )

        X = np.random.default_rng(42).normal(0, 1, (5, 5))
        sensor_ids = [f"S{i:03d}" for i in range(5)]
        results = detector.classify(X, sensor_ids)

        assert len(results) == 5
        for r in results:
            assert isinstance(r, AnomalyResult)
            assert r.detector_type == "autoencoder"
            assert "reconstruction_error" in r.details


def test_ae_classify_detects_outliers():
    from app.sensor_intelligence.anomaly_detection.autoencoder_detector import (
        AutoencoderDetector,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, threshold_path, scaler_path = _train_and_save_ae(Path(tmpdir), n=500)
        detector = AutoencoderDetector()
        detector.load_model(
            model_path=model_path,
            threshold_path=threshold_path,
            scaler_path=scaler_path,
        )

        rng = np.random.default_rng(42)
        X_normal = rng.normal(0, 1, (5, 5))
        X_outlier = np.full((1, 5), 50.0)  # Far from training distribution
        X = np.vstack([X_normal, X_outlier])
        sensor_ids = [f"S{i:03d}" for i in range(6)]

        results = detector.classify(X, sensor_ids)
        assert results[-1].status == AnomalyStatus.ANOMALY
        assert results[-1].score > 1.0  # Score > 1.0 = above threshold


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DetectorFactory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_factory_available_detectors():
    available = DetectorFactory.available_detectors()
    assert "isolation_forest" in available
    assert "autoencoder" in available


def test_factory_create_if():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, scaler_path = _train_and_save_if(Path(tmpdir))
        detector = DetectorFactory.create(
            "isolation_forest",
            model_path=model_path,
            scaler_path=scaler_path,
        )
        assert isinstance(detector, IsolationForestDetector)
        assert detector.is_loaded


def test_factory_create_unknown_raises():
    with pytest.raises(ValueError, match="Unknown detector"):
        DetectorFactory.create("nonexistent_model", auto_load=False)


def test_factory_create_no_auto_load():
    detector = DetectorFactory.create("isolation_forest", auto_load=False)
    assert isinstance(detector, IsolationForestDetector)
    assert not detector.is_loaded


def test_factory_register_custom():
    class DummyDetector(BaseAnomalyDetector):
        @property
        def name(self): return "dummy"
        @property
        def is_loaded(self): return True
        def load_model(self, **kw): pass
        def predict(self, features): return np.zeros(len(features))
        def anomaly_score(self, features): return np.zeros(len(features))
        def classify(self, features, sensor_ids):
            return [
                AnomalyResult(sid, 0.0, AnomalyStatus.NORMAL, "dummy")
                for sid in sensor_ids
            ]

    DetectorFactory.register("dummy", DummyDetector)
    assert "dummy" in DetectorFactory.available_detectors()

    detector = DetectorFactory.create("dummy")
    assert detector.name == "dummy"
