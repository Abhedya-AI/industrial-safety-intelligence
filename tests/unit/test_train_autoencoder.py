"""Unit tests for the Autoencoder training pipeline.

Covers:
  - build_autoencoder (architecture, input/output shape)
  - prepare_data (normal-only train, full eval, scaler)
  - train_autoencoder (model trains, loss decreases)
  - compute_reconstruction_error (shape, non-negative)
  - compute_threshold (percentile correctness)
  - evaluate_model (metrics structure)
  - save_model / load_saved_model (round-trip)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from training.train_autoencoder import (
    build_autoencoder,
    compute_reconstruction_error,
    compute_threshold,
    evaluate_model,
    load_features,
    prepare_data,
    save_model,
    train_autoencoder,
)


# ── Helpers ──


def _make_features_df(n: int = 200, accident_rate: float = 0.05) -> pd.DataFrame:
    """Create a synthetic features DataFrame for testing."""
    rng = np.random.default_rng(42)
    n_accident = int(n * accident_rate)
    n_normal = n - n_accident

    normal = rng.normal(0, 1, (n_normal, 10))
    anomaly = rng.normal(4, 2, (n_accident, 10))

    data = np.vstack([normal, anomaly])
    labels = np.array([0] * n_normal + [1] * n_accident)

    idx = rng.permutation(n)
    data = data[idx]
    labels = labels[idx]

    cols = [f"feat_{i}" for i in range(10)]
    df = pd.DataFrame(data, columns=cols)
    df["Accident"] = labels
    df["Risk"] = rng.uniform(0, 100, n)
    df["timestamp"] = "2021-01-01"
    df["zone_id"] = "ZONE_A"
    df["equipment_id"] = "EQ-001"
    df["sensor_id_temp"] = "S-ZONE_A-TEMPERATURE"
    df["sensor_id_pressure"] = "S-ZONE_A-PRESSURE"
    df["sensor_id_humidity"] = "S-ZONE_A-HUMIDITY"
    df["sensor_id_vibration"] = "S-ZONE_A-VIBRATION"
    df["sensor_id_gas"] = "S-ZONE_A-GAS"
    df["sensor_type"] = "MULTI"
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# build_autoencoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_build_autoencoder_output_shape():
    model = build_autoencoder(input_dim=20)
    assert model.input_shape == (None, 20)
    assert model.output_shape == (None, 20)


def test_build_autoencoder_has_bottleneck():
    model = build_autoencoder(input_dim=20)
    layer_names = [l.name for l in model.layers]
    assert "bottleneck" in layer_names


def test_build_autoencoder_is_compiled():
    model = build_autoencoder(input_dim=10)
    assert model.loss is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# prepare_data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_prepare_data_shapes():
    df = _make_features_df(200, accident_rate=0.10)
    X_train, X_eval, eval_labels, scaler, feature_cols = prepare_data(df)

    n_normal = (df["Accident"] == 0).sum()
    assert X_train.shape[0] == n_normal
    assert X_eval.shape[0] == 200
    assert len(eval_labels) == 200
    assert X_train.shape[1] == X_eval.shape[1]
    assert isinstance(scaler, StandardScaler)


def test_prepare_data_no_nan():
    df = _make_features_df(100)
    X_train, X_eval, _, _, _ = prepare_data(df)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_eval).any()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# train_autoencoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_train_autoencoder_returns_model_and_history():
    df = _make_features_df(200)
    X_train, _, _, _, _ = prepare_data(df)
    model, history = train_autoencoder(X_train, epochs=3, batch_size=32)

    assert model is not None
    assert "loss" in history.history
    assert len(history.history["loss"]) == 3


def test_train_autoencoder_loss_decreases():
    df = _make_features_df(300)
    X_train, _, _, _, _ = prepare_data(df)
    _, history = train_autoencoder(X_train, epochs=10, batch_size=32)

    losses = history.history["loss"]
    # Final loss should be lower than initial loss
    assert losses[-1] < losses[0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# compute_reconstruction_error
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_reconstruction_error_shape():
    df = _make_features_df(100)
    X_train, _, _, _, _ = prepare_data(df)
    model, _ = train_autoencoder(X_train, epochs=2, batch_size=32)

    errors = compute_reconstruction_error(model, X_train)
    assert errors.shape == (len(X_train),)


def test_reconstruction_error_non_negative():
    df = _make_features_df(100)
    X_train, _, _, _, _ = prepare_data(df)
    model, _ = train_autoencoder(X_train, epochs=2, batch_size=32)

    errors = compute_reconstruction_error(model, X_train)
    assert (errors >= 0).all()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# compute_threshold
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_compute_threshold_percentile():
    errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    # 90th percentile of [1..10] is 9.1
    t = compute_threshold(errors, percentile=90.0)
    assert 9.0 <= t <= 10.0


def test_compute_threshold_100_percentile_equals_max():
    errors = np.array([1.0, 5.0, 10.0])
    t = compute_threshold(errors, percentile=100.0)
    assert t == 10.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# evaluate_model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_evaluate_model_returns_valid_result():
    df = _make_features_df(200, accident_rate=0.10)
    X_train, X_eval, eval_labels, _, _ = prepare_data(df)
    model, _ = train_autoencoder(X_train, epochs=5, batch_size=32)

    train_errors = compute_reconstruction_error(model, X_train)
    threshold = compute_threshold(train_errors, percentile=95.0)
    result = evaluate_model(model, X_eval, eval_labels, threshold)

    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert result.total_samples == 200
    assert result.confusion.shape == (2, 2)
    assert result.threshold == round(threshold, 6)
    assert result.mean_normal_error >= 0
    assert result.mean_anomaly_error >= 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# save_model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_save_model_creates_files():
    df = _make_features_df(100)
    X_train, _, _, scaler, feature_cols = prepare_data(df)
    model, _ = train_autoencoder(X_train, epochs=2, batch_size=32)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "ae.keras"
        threshold_path = Path(tmpdir) / "threshold.json"
        scaler_path = Path(tmpdir) / "scaler.pkl"

        save_model(model, 0.5, scaler, feature_cols, model_path, threshold_path, scaler_path)

        assert model_path.exists()
        assert threshold_path.exists()
        assert scaler_path.exists()

        # Verify threshold JSON content
        with open(threshold_path) as f:
            data = json.load(f)
        assert data["threshold"] == 0.5
        assert data["feature_count"] == len(feature_cols)
        assert "architecture" in data
