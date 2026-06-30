"""Unit tests for the feature engineering module.

Covers:
  - load_processed (file check, sorting, timestamp parsing)
  - add_rolling_features (mean, std, max, min for multiple windows)
  - add_interaction_features (product correctness, missing columns)
  - add_rate_features (first differences, group isolation)
  - add_sensor_aggregate_features (mean, std, min, max per group)
  - get_feature_columns (excludes identifiers / targets)
  - get_isolation_forest_features (excludes sensor aggregates)
  - get_autoencoder_features (excludes interactions + aggregates)
  - get_risk_prediction_features (full feature set)
  - build_features (integrated pipeline)
  - save_features (file creation)
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.feature_engineering import (
    INTERACTIONS,
    ROLLING_WINDOWS,
    SENSOR_GROUP_KEY,
    TELEMETRY_COLS,
    add_interaction_features,
    add_rate_features,
    add_rolling_features,
    add_sensor_aggregate_features,
    build_features,
    get_autoencoder_features,
    get_feature_columns,
    get_isolation_forest_features,
    get_risk_prediction_features,
    load_processed,
    save_features,
)


# ── Helpers ──


def _make_df(n: int = 20, zones: int = 2) -> pd.DataFrame:
    """Build a minimal processed-style DataFrame for testing."""
    rng = np.random.default_rng(42)
    zone_labels = [f"ZONE_{chr(65 + i)}" for i in range(zones)]

    rows = []
    base = datetime(2021, 1, 1)
    for i in range(n):
        rows.append({
            "timestamp": base + timedelta(minutes=15 * i),
            "zone_id": zone_labels[i % zones],
            "equipment_id": f"EQ-{(i % zones) + 1:03d}",
            "sensor_id_temp": f"S-{zone_labels[i % zones]}-TEMPERATURE",
            "sensor_id_pressure": f"S-{zone_labels[i % zones]}-PRESSURE",
            "sensor_id_humidity": f"S-{zone_labels[i % zones]}-HUMIDITY",
            "sensor_id_vibration": f"S-{zone_labels[i % zones]}-VIBRATION",
            "sensor_id_gas": f"S-{zone_labels[i % zones]}-GAS",
            "sensor_type": "MULTI",
            "Temp": rng.uniform(20, 40),
            "Pressure": rng.uniform(15, 35),
            "Humidity": rng.uniform(30, 80),
            "Vibration": rng.uniform(0, 5),
            "Speed": rng.uniform(1000, 4000),
            "Gas": rng.uniform(0, 10),
            "Sparks": float(rng.integers(0, 5)),
            "Age": rng.integers(1, 20),
            "Service_Days": rng.integers(1, 365),
            "Workers": float(rng.integers(5, 50)),
            "Risk": rng.uniform(0, 100),
            "Accident": rng.choice([0, 1], p=[0.96, 0.04]),
            # One-hot encoded columns
            "Factory_Factory_A": int(i % zones == 0),
            "Factory_Factory_B": int(i % zones == 1),
            "Shift_Day": rng.integers(0, 2),
            "Shift_Night": 0,
            "Alarm_On": rng.integers(0, 2),
            "Alarm_Off": 0,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# load_processed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_load_processed_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_processed("/nonexistent/data.csv")


def test_load_processed_parses_timestamp_and_sorts():
    df = _make_df(10)
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f, index=False)
        path = f.name

    loaded = load_processed(path)
    assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])
    # Verify sorted by zone_id then timestamp
    for zone in loaded["zone_id"].unique():
        subset = loaded[loaded["zone_id"] == zone]
        assert subset["timestamp"].is_monotonic_increasing
    Path(path).unlink()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# add_rolling_features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_rolling_features_columns_created():
    df = _make_df(20)
    result = add_rolling_features(df, columns=["Temp"], windows=[5])
    assert "rolling_mean_Temp_w5" in result.columns
    assert "rolling_std_Temp_w5" in result.columns
    assert "rolling_max_Temp_w5" in result.columns
    assert "rolling_min_Temp_w5" in result.columns


def test_rolling_features_multiple_windows():
    df = _make_df(20)
    result = add_rolling_features(df, columns=["Temp", "Gas"], windows=[5, 10])
    for w in [5, 10]:
        for col in ["Temp", "Gas"]:
            assert f"rolling_mean_{col}_w{w}" in result.columns
            assert f"rolling_std_{col}_w{w}" in result.columns


def test_rolling_mean_correctness():
    """Verify rolling mean for a simple known series within a single group."""
    df = pd.DataFrame({
        "zone_id": ["Z"] * 5,
        "Temp": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    result = add_rolling_features(df, columns=["Temp"], windows=[3])
    # Row 0: mean([10]) = 10 (min_periods=1)
    # Row 1: mean([10,20]) = 15
    # Row 2: mean([10,20,30]) = 20
    # Row 3: mean([20,30,40]) = 30
    # Row 4: mean([30,40,50]) = 40
    expected = [10.0, 15.0, 20.0, 30.0, 40.0]
    np.testing.assert_array_almost_equal(
        result["rolling_mean_Temp_w3"].values, expected
    )


def test_rolling_features_per_group_isolation():
    """Rolling windows should not cross zone boundaries."""
    df = pd.DataFrame({
        "zone_id": ["A", "A", "A", "B", "B", "B"],
        "Temp": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
    })
    result = add_rolling_features(df, columns=["Temp"], windows=[3])
    # Group B should have its own rolling mean
    b_means = result[result["zone_id"] == "B"]["rolling_mean_Temp_w3"].values
    expected_b = [100.0, 150.0, 200.0]
    np.testing.assert_array_almost_equal(b_means, expected_b)


def test_rolling_std_zero_for_constant():
    df = pd.DataFrame({
        "zone_id": ["Z"] * 5,
        "Temp": [42.0] * 5,
    })
    result = add_rolling_features(df, columns=["Temp"], windows=[3])
    assert (result["rolling_std_Temp_w3"].fillna(0.0) == 0.0).all()


def test_rolling_max_min_correctness():
    df = pd.DataFrame({
        "zone_id": ["Z"] * 5,
        "Temp": [10.0, 50.0, 30.0, 20.0, 40.0],
    })
    result = add_rolling_features(df, columns=["Temp"], windows=[3])
    # Row 2: max([10,50,30])=50, min=10
    assert result.loc[2, "rolling_max_Temp_w3"] == 50.0
    assert result.loc[2, "rolling_min_Temp_w3"] == 10.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# add_interaction_features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_interaction_columns_created():
    df = _make_df(10)
    result = add_interaction_features(df)
    for _, _, name in INTERACTIONS:
        assert name in result.columns


def test_interaction_product_correctness():
    df = pd.DataFrame({"Temp": [2.0, 3.0], "Gas": [5.0, 4.0]})
    result = add_interaction_features(df, interactions=[("Temp", "Gas", "T_x_G")])
    assert list(result["T_x_G"]) == [10.0, 12.0]


def test_interaction_missing_column_skipped():
    df = pd.DataFrame({"Temp": [1.0]})
    result = add_interaction_features(
        df, interactions=[("Temp", "NonExistent", "skip_me")]
    )
    assert "skip_me" not in result.columns


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# add_rate_features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_rate_features_columns_created():
    df = _make_df(10)
    result = add_rate_features(df)
    assert "rate_Temp" in result.columns
    assert "rate_Pressure" in result.columns
    assert "rate_Gas" in result.columns


def test_rate_first_row_is_zero():
    df = pd.DataFrame({
        "zone_id": ["Z"] * 4,
        "Temp": [10.0, 20.0, 25.0, 30.0],
    })
    result = add_rate_features(df, columns=["Temp"])
    assert result.loc[0, "rate_Temp"] == 0.0


def test_rate_differences_correct():
    df = pd.DataFrame({
        "zone_id": ["Z"] * 4,
        "Temp": [10.0, 20.0, 25.0, 30.0],
    })
    result = add_rate_features(df, columns=["Temp"])
    expected = [0.0, 10.0, 5.0, 5.0]
    np.testing.assert_array_almost_equal(result["rate_Temp"].values, expected)


def test_rate_per_group_isolation():
    df = pd.DataFrame({
        "zone_id": ["A", "A", "B", "B"],
        "Temp": [10.0, 30.0, 100.0, 200.0],
    })
    result = add_rate_features(df, columns=["Temp"])
    # Group B first row should be 0 (not diff from group A)
    b_rates = result[result["zone_id"] == "B"]["rate_Temp"].values
    np.testing.assert_array_almost_equal(b_rates, [0.0, 100.0])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# add_sensor_aggregate_features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_sensor_agg_columns_created():
    df = _make_df(20)
    result = add_sensor_aggregate_features(df, columns=["Temp"])
    assert "sensor_agg_mean_Temp" in result.columns
    assert "sensor_agg_std_Temp" in result.columns
    assert "sensor_agg_min_Temp" in result.columns
    assert "sensor_agg_max_Temp" in result.columns


def test_sensor_agg_values_correct():
    df = pd.DataFrame({
        "zone_id": ["A", "A", "A", "B", "B"],
        "Temp": [10.0, 20.0, 30.0, 100.0, 200.0],
    })
    result = add_sensor_aggregate_features(df, columns=["Temp"])
    # Group A mean = 20, min = 10, max = 30
    a_rows = result[result["zone_id"] == "A"]
    assert (a_rows["sensor_agg_mean_Temp"] == 20.0).all()
    assert (a_rows["sensor_agg_min_Temp"] == 10.0).all()
    assert (a_rows["sensor_agg_max_Temp"] == 30.0).all()

    # Group B mean = 150
    b_rows = result[result["zone_id"] == "B"]
    assert (b_rows["sensor_agg_mean_Temp"] == 150.0).all()


def test_sensor_agg_single_value_std_is_zero():
    df = pd.DataFrame({
        "zone_id": ["A"],
        "Temp": [42.0],
    })
    result = add_sensor_aggregate_features(df, columns=["Temp"])
    assert result.loc[0, "sensor_agg_std_Temp"] == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Feature selectors
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_get_feature_columns_excludes_targets():
    df = _make_df(10)
    df = build_features(df)
    cols = get_feature_columns(df)
    assert "Risk" not in cols
    assert "Accident" not in cols
    assert "timestamp" not in cols
    assert "zone_id" not in cols


def test_get_feature_columns_excludes_strings():
    df = _make_df(10)
    df = build_features(df)
    cols = get_feature_columns(df)
    for c in cols:
        assert df[c].dtype in (np.float64, np.int64, np.float32, np.int32)


def test_isolation_forest_excludes_sensor_agg():
    df = _make_df(20)
    df = build_features(df)
    cols = get_isolation_forest_features(df)
    assert not any(c.startswith("sensor_agg_") for c in cols)


def test_autoencoder_excludes_interactions_and_agg():
    df = _make_df(20)
    df = build_features(df)
    cols = get_autoencoder_features(df)
    assert not any(c.startswith("sensor_agg_") for c in cols)
    assert "Temp_x_Gas" not in cols


def test_risk_prediction_includes_all():
    df = _make_df(20)
    df = build_features(df)
    all_cols = get_feature_columns(df)
    risk_cols = get_risk_prediction_features(df)
    assert set(risk_cols) == set(all_cols)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# build_features (integrated)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_build_features_does_not_mutate_input():
    df = _make_df(10)
    original_cols = list(df.columns)
    _ = build_features(df)
    assert list(df.columns) == original_cols


def test_build_features_adds_expected_column_families():
    df = _make_df(20)
    result = build_features(df)
    # Rolling
    assert any(c.startswith("rolling_") for c in result.columns)
    # Interactions
    assert "Temp_x_Gas" in result.columns
    assert "Pressure_x_Vibration" in result.columns
    # Rate
    assert "rate_Temp" in result.columns
    # Sensor agg
    assert any(c.startswith("sensor_agg_") for c in result.columns)


def test_build_features_no_nan_in_key_features():
    """Rolling features with min_periods=1 and rate filled → no NaN."""
    df = _make_df(20)
    result = build_features(df)
    for col in ["rate_Temp", "rate_Pressure", "rate_Gas"]:
        assert result[col].isna().sum() == 0
    for col in [c for c in result.columns if c.startswith("rolling_mean_")]:
        assert result[col].isna().sum() == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# save_features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_save_features_creates_file():
    df = _make_df(5)
    df = build_features(df)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "sub" / "features.csv"
        result_path = save_features(df, out)
        assert result_path.exists()
        loaded = pd.read_csv(result_path)
        assert len(loaded) == 5
        assert set(loaded.columns) == set(df.columns)
