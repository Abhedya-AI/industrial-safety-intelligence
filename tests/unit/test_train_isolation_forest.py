"""Unit tests for the Isolation Forest training pipeline.

Covers:
  - load_features (file not found, success)
  - prepare_data (normal-only train, full eval, label split)
  - train_model (returns model + scaler, handles contamination)
  - evaluate_model (metrics structure, prediction mapping)
  - save_model / load_model (round-trip persistence)
  - run_pipeline (end-to-end on synthetic data)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from training.train_isolation_forest import (
    evaluate_model,
    load_features,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


# ── Helpers ──


def _make_features_df(n: int = 200, accident_rate: float = 0.05) -> pd.DataFrame:
    """Create a synthetic features DataFrame for testing."""
    rng = np.random.default_rng(42)
    n_accident = int(n * accident_rate)
    n_normal = n - n_accident

    # Normal data: clustered around 0
    normal = rng.normal(0, 1, (n_normal, 10))
    # Anomaly data: shifted distribution
    anomaly = rng.normal(3, 1.5, (n_accident, 10))

    data = np.vstack([normal, anomaly])
    labels = np.array([0] * n_normal + [1] * n_accident)

    # Shuffle
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


def _save_temp_csv(df: pd.DataFrame) -> str:
    """Save a DataFrame to a temporary CSV and return the path."""
    f = tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False)
    df.to_csv(f, index=False)
    f.close()
    return f.name


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# load_features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_load_features_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_features("/nonexistent/features.csv")


def test_load_features_success():
    df = _make_features_df(50)
    path = _save_temp_csv(df)
    loaded = load_features(path)
    assert len(loaded) == 50
    assert "Accident" in loaded.columns
    Path(path).unlink()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# prepare_data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_prepare_data_splits_correctly():
    df = _make_features_df(200, accident_rate=0.10)
    train_df, eval_df, eval_labels, feature_cols = prepare_data(df)

    # Train should be normal-only
    n_normal = (df["Accident"] == 0).sum()
    assert len(train_df) == n_normal

    # Eval should be full dataset
    assert len(eval_df) == 200
    assert len(eval_labels) == 200

    # Feature columns should not include Accident, Risk, or string cols
    assert "Accident" not in feature_cols
    assert "Risk" not in feature_cols
    assert "timestamp" not in feature_cols


def test_prepare_data_feature_columns_are_numeric():
    df = _make_features_df(100)
    train_df, _, _, _ = prepare_data(df)
    for col in train_df.columns:
        assert train_df[col].dtype in (np.float64, np.int64, np.float32, np.int32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# train_model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_train_model_returns_model_and_scaler():
    df = _make_features_df(200)
    train_df, _, _, _ = prepare_data(df)
    model, scaler = train_model(train_df, contamination=0.05)

    assert isinstance(model, IsolationForest)
    assert isinstance(scaler, StandardScaler)
    assert hasattr(scaler, "mean_")


def test_train_model_default_contamination():
    df = _make_features_df(200)
    train_df, _, _, _ = prepare_data(df)
    model, scaler = train_model(train_df)

    # Default contamination should be set
    assert model.contamination == 0.04


def test_train_model_custom_contamination():
    df = _make_features_df(200)
    train_df, _, _, _ = prepare_data(df)
    model, _ = train_model(train_df, contamination=0.10)
    assert model.contamination == 0.10


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# evaluate_model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_evaluate_model_returns_valid_result():
    df = _make_features_df(200, accident_rate=0.10)
    train_df, eval_df, eval_labels, _ = prepare_data(df)
    model, scaler = train_model(train_df, contamination=0.10)
    result = evaluate_model(model, scaler, eval_df, eval_labels)

    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert result.total_samples == 200
    assert result.actual_accidents == eval_labels.sum()
    assert result.confusion.shape == (2, 2)
    assert len(result.report) > 0


def test_evaluate_model_predicted_anomalies_count():
    df = _make_features_df(200, accident_rate=0.10)
    train_df, eval_df, eval_labels, _ = prepare_data(df)
    model, scaler = train_model(train_df, contamination=0.10)
    result = evaluate_model(model, scaler, eval_df, eval_labels)

    # Predicted anomalies should be > 0
    assert result.predicted_anomalies > 0
    # True positives should not exceed actual accidents
    assert result.true_positives <= result.actual_accidents


def test_evaluate_model_confusion_matrix_sums():
    df = _make_features_df(300, accident_rate=0.05)
    train_df, eval_df, eval_labels, _ = prepare_data(df)
    model, scaler = train_model(train_df, contamination=0.05)
    result = evaluate_model(model, scaler, eval_df, eval_labels)

    # Confusion matrix should sum to total samples
    assert result.confusion.sum() == result.total_samples


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# save_model / load_model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_save_and_load_model_roundtrip():
    df = _make_features_df(100)
    train_df, eval_df, eval_labels, _ = prepare_data(df)
    model, scaler = train_model(train_df, contamination=0.05)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pkl"
        scaler_path = Path(tmpdir) / "scaler.pkl"

        save_model(model, scaler, model_path, scaler_path)
        assert model_path.exists()
        assert scaler_path.exists()

        loaded_model, loaded_scaler = load_model(model_path, scaler_path)
        assert isinstance(loaded_model, IsolationForest)
        assert isinstance(loaded_scaler, StandardScaler)

        # Verify loaded model produces same predictions
        X_eval = scaler.transform(eval_df.values)
        orig_preds = model.predict(X_eval)
        loaded_preds = loaded_model.predict(loaded_scaler.transform(eval_df.values))
        np.testing.assert_array_equal(orig_preds, loaded_preds)


def test_save_model_creates_directories():
    df = _make_features_df(50)
    train_df, _, _, _ = prepare_data(df)
    model, scaler = train_model(train_df, contamination=0.05)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "nested" / "dir" / "model.pkl"
        scaler_path = Path(tmpdir) / "nested" / "dir" / "scaler.pkl"

        save_model(model, scaler, model_path, scaler_path)
        assert model_path.exists()
        assert scaler_path.exists()
