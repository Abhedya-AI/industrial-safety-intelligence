"""Unit tests for the Risk Prediction feature builder.

Validates:
  1. RiskFeatureInput dataclass + to_flat_dict()
  2. build_risk_features_from_dict — single-row path
  3. build_risk_features_from_input — convenience wrapper
  4. build_risk_feature_vector — numpy vector output
  5. build_risk_features_batch — DataFrame path (delegates to SI)
  6. get_risk_feature_columns — column list
  7. extract_feature_matrix — numpy matrix output
  8. validate_features — validation helper
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.risk_prediction.preprocessing.feature_builder import (
    CONTEXT_FEATURES,
    RISK_EXTRA_FEATURES,
    SENSOR_FEATURES,
    RiskFeatureInput,
    build_risk_feature_vector,
    build_risk_features_batch,
    build_risk_features_from_dict,
    build_risk_features_from_input,
    extract_feature_matrix,
    get_risk_feature_columns,
    validate_features,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. RiskFeatureInput
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRiskFeatureInput:
    def test_defaults(self):
        inp = RiskFeatureInput()
        assert inp.temperature == 0.0
        assert inp.sensor_health_score == 100.0
        assert inp.workers == 0
        assert inp.training is False
        assert inp.experience == "Junior"

    def test_custom_values(self):
        inp = RiskFeatureInput(
            temperature=95.0, pressure=180.0, gas=120.0,
            anomaly_score_if=0.85, sensor_health_score=45.0,
            workers=12, training=True, experience="Senior",
        )
        assert inp.temperature == 95.0
        assert inp.anomaly_score_if == 0.85
        assert inp.workers == 12
        assert inp.training is True

    def test_to_flat_dict_contains_telemetry(self):
        inp = RiskFeatureInput(temperature=50.0, pressure=20.0)
        d = inp.to_flat_dict()
        assert d["Temp"] == 50.0
        assert d["Pressure"] == 20.0
        assert d["Humidity"] == 0.0

    def test_to_flat_dict_contains_anomaly_scores(self):
        inp = RiskFeatureInput(anomaly_score_if=0.9, anomaly_score_ae=0.7)
        d = inp.to_flat_dict()
        assert d["anomaly_score_if"] == 0.9
        assert d["anomaly_score_ae"] == 0.7

    def test_to_flat_dict_onehot_training(self):
        inp_yes = RiskFeatureInput(training=True)
        d = inp_yes.to_flat_dict()
        assert d["Training_Yes"] == 1.0
        assert d["Training_No"] == 0.0

        inp_no = RiskFeatureInput(training=False)
        d = inp_no.to_flat_dict()
        assert d["Training_Yes"] == 0.0
        assert d["Training_No"] == 1.0

    def test_to_flat_dict_onehot_experience(self):
        d = RiskFeatureInput(experience="Senior").to_flat_dict()
        assert d["Exp_Senior"] == 1.0
        assert d["Exp_Junior"] == 0.0

        d = RiskFeatureInput(experience="Junior").to_flat_dict()
        assert d["Exp_Senior"] == 0.0
        assert d["Exp_Junior"] == 1.0

    def test_to_flat_dict_onehot_zone(self):
        d = RiskFeatureInput(zone_id="ZONE_B").to_flat_dict()
        assert d["Factory_Factory_B"] == 1.0
        assert d["Factory_Factory_A"] == 0.0
        assert d["Factory_Factory_C"] == 0.0

    def test_to_flat_dict_onehot_shift(self):
        d = RiskFeatureInput(shift="Night").to_flat_dict()
        assert d["Shift_Night"] == 1.0
        assert d["Shift_Day"] == 0.0

    def test_to_flat_dict_onehot_region(self):
        d = RiskFeatureInput(region="Urban").to_flat_dict()
        assert d["Region_Urban"] == 1.0
        assert d["Region_Industrial_Zone"] == 0.0

    def test_to_flat_dict_onehot_alarm(self):
        d = RiskFeatureInput(alarm="On").to_flat_dict()
        assert d["Alarm_On"] == 1.0
        assert d["Alarm_Off"] == 0.0

    def test_to_flat_dict_all_numeric(self):
        d = RiskFeatureInput().to_flat_dict()
        for k, v in d.items():
            assert isinstance(v, (int, float)), f"{k} is {type(v)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. build_risk_features_from_dict
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBuildFromDict:
    def test_fills_defaults(self):
        result = build_risk_features_from_dict({"Temp": 80.0})
        assert result["Temp"] == 80.0
        assert result["anomaly_score_if"] == 0.0
        assert result["sensor_health_score"] == 100.0
        assert result["Workers"] == 0.0

    def test_preserves_existing_values(self):
        result = build_risk_features_from_dict({
            "Temp": 95.0,
            "anomaly_score_if": 0.9,
        })
        assert result["Temp"] == 95.0
        assert result["anomaly_score_if"] == 0.9

    def test_no_fill_defaults(self):
        result = build_risk_features_from_dict(
            {"Temp": 80.0}, fill_defaults=False,
        )
        assert "anomaly_score_if" not in result
        assert result["Temp"] == 80.0

    def test_empty_dict_gets_all_defaults(self):
        result = build_risk_features_from_dict({})
        for col in SENSOR_FEATURES:
            assert col in result
        assert "anomaly_score_if" in result
        assert "anomaly_score_ae" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. build_risk_features_from_input
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBuildFromInput:
    def test_roundtrip(self):
        inp = RiskFeatureInput(
            temperature=95.0, pressure=180.0, gas=120.0,
            anomaly_score_if=0.85, sensor_health_score=45.0,
            workers=12,
        )
        result = build_risk_features_from_input(inp)
        assert result["Temp"] == 95.0
        assert result["Gas"] == 120.0
        assert result["anomaly_score_if"] == 0.85
        assert result["sensor_health_score"] == 45.0
        assert result["Workers"] == 12.0

    def test_default_input(self):
        result = build_risk_features_from_input(RiskFeatureInput())
        assert result["Temp"] == 0.0
        assert result["sensor_health_score"] == 100.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. build_risk_feature_vector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBuildFeatureVector:
    def test_correct_order(self):
        features = {"A": 1.0, "B": 2.0, "C": 3.0}
        vec = build_risk_feature_vector(features, ["C", "A", "B"])
        np.testing.assert_array_equal(vec, [3.0, 1.0, 2.0])

    def test_missing_keys_default_zero(self):
        features = {"A": 1.0}
        vec = build_risk_feature_vector(features, ["A", "B", "C"])
        np.testing.assert_array_equal(vec, [1.0, 0.0, 0.0])

    def test_returns_float64(self):
        vec = build_risk_feature_vector({"x": 5.0}, ["x"])
        assert vec.dtype == np.float64

    def test_1d_shape(self):
        vec = build_risk_feature_vector({"x": 1.0, "y": 2.0}, ["x", "y"])
        assert vec.ndim == 1
        assert vec.shape == (2,)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. build_risk_features_batch (DataFrame)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_sample_df(n: int = 20) -> pd.DataFrame:
    """Create a minimal preprocessed DataFrame mimicking preprocessing output."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "Temp": rng.normal(35, 5, n),
        "Pressure": rng.normal(15, 2, n),
        "Humidity": rng.normal(50, 10, n),
        "Vibration": rng.normal(2, 0.5, n),
        "Speed": rng.normal(100, 20, n),
        "Gas": rng.normal(50, 15, n),
        "Sparks": rng.integers(0, 2, n).astype(float),
        "Workers": rng.integers(1, 15, n).astype(float),
        "Age": rng.normal(40, 10, n),
        "Service_Days": rng.normal(500, 200, n),
        "Risk": rng.integers(0, 2, n),
        "Accident": rng.integers(0, 2, n),
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
        "zone_id": ["ZONE_A"] * n,
        "equipment_id": ["EQ-001"] * n,
        "sensor_type": ["TEMPERATURE"] * n,
        "sensor_id_temp": ["S001"] * n,
        "sensor_id_pressure": ["S002"] * n,
        "sensor_id_humidity": ["S003"] * n,
        "sensor_id_vibration": ["S004"] * n,
        "sensor_id_gas": ["S005"] * n,
        # One-hot categoricals
        "Factory_Factory_A": [1.0] * n,
        "Factory_Factory_B": [0.0] * n,
        "Factory_Factory_C": [0.0] * n,
        "Factory_Factory_D": [0.0] * n,
        "Region_Industrial_Zone": [1.0] * n,
        "Region_Urban": [0.0] * n,
        "Region_Rural": [0.0] * n,
        "Shift_Day": [1.0] * n,
        "Shift_Night": [0.0] * n,
        "Exp_Junior": [1.0] * n,
        "Exp_Senior": [0.0] * n,
        "Training_Yes": [1.0] * n,
        "Training_No": [0.0] * n,
        "Alarm_On": [0.0] * n,
        "Alarm_Off": [1.0] * n,
    })
    return df


class TestBuildRiskFeaturesBatch:
    def test_adds_risk_columns(self):
        df = _make_sample_df()
        result = build_risk_features_batch(df)
        assert "anomaly_score_if" in result.columns
        assert "anomaly_score_ae" in result.columns
        assert "sensor_health_score" in result.columns

    def test_default_enrichment_values(self):
        df = _make_sample_df()
        result = build_risk_features_batch(df)
        assert (result["anomaly_score_if"] == 0.0).all()
        assert (result["anomaly_score_ae"] == 0.0).all()
        assert (result["sensor_health_score"] == 100.0).all()

    def test_custom_enrichment(self):
        df = _make_sample_df(n=5)
        if_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        ae_scores = [0.5, 0.4, 0.3, 0.2, 0.1]
        health = [90.0, 80.0, 70.0, 60.0, 50.0]
        result = build_risk_features_batch(
            df,
            anomaly_scores_if=if_scores,
            anomaly_scores_ae=ae_scores,
            health_scores=health,
        )
        np.testing.assert_array_almost_equal(
            result["anomaly_score_if"].values, if_scores,
        )
        np.testing.assert_array_almost_equal(
            result["sensor_health_score"].values, health,
        )

    def test_includes_rolling_features(self):
        df = _make_sample_df()
        result = build_risk_features_batch(df)
        rolling_cols = [c for c in result.columns if c.startswith("rolling_")]
        assert len(rolling_cols) > 0

    def test_includes_interaction_features(self):
        df = _make_sample_df()
        result = build_risk_features_batch(df)
        assert "Temp_x_Gas" in result.columns
        assert "Pressure_x_Vibration" in result.columns

    def test_includes_rate_features(self):
        df = _make_sample_df()
        result = build_risk_features_batch(df)
        assert "rate_Temp" in result.columns
        assert "rate_Pressure" in result.columns

    def test_row_count_preserved(self):
        df = _make_sample_df(n=15)
        result = build_risk_features_batch(df)
        assert len(result) == 15


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. get_risk_feature_columns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGetRiskFeatureColumns:
    def test_includes_extra_risk_cols(self):
        df = _make_sample_df()
        df = build_risk_features_batch(df)
        cols = get_risk_feature_columns(df)
        for extra in RISK_EXTRA_FEATURES:
            assert extra in cols

    def test_excludes_identifiers(self):
        df = _make_sample_df()
        df = build_risk_features_batch(df)
        cols = get_risk_feature_columns(df)
        for excluded in ["timestamp", "zone_id", "equipment_id", "sensor_type"]:
            assert excluded not in cols

    def test_returns_list(self):
        df = _make_sample_df()
        df = build_risk_features_batch(df)
        cols = get_risk_feature_columns(df)
        assert isinstance(cols, list)
        assert len(cols) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. extract_feature_matrix
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestExtractFeatureMatrix:
    def test_shape(self):
        df = _make_sample_df(n=10)
        df = build_risk_features_batch(df)
        cols = get_risk_feature_columns(df)
        matrix = extract_feature_matrix(df, cols)
        assert matrix.shape == (10, len(cols))

    def test_dtype_float64(self):
        df = _make_sample_df(n=5)
        df = build_risk_features_batch(df)
        matrix = extract_feature_matrix(df)
        assert matrix.dtype == np.float64

    def test_no_nan(self):
        df = _make_sample_df(n=5)
        df = build_risk_features_batch(df)
        matrix = extract_feature_matrix(df)
        assert not np.isnan(matrix).any()

    def test_explicit_column_order(self):
        df = _make_sample_df(n=5)
        df = build_risk_features_batch(df)
        matrix = extract_feature_matrix(df, ["Temp", "Gas"])
        assert matrix.shape == (5, 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. validate_features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestValidateFeatures:
    def test_valid_features(self):
        features = {col: 1.0 for col in SENSOR_FEATURES}
        issues = validate_features(features)
        assert issues == []

    def test_missing_features(self):
        issues = validate_features({"Temp": 1.0})
        # Should report all missing sensor features except Temp
        missing = [f for f in SENSOR_FEATURES if f != "Temp"]
        assert set(issues) == set(missing)

    def test_nan_feature(self):
        features = {col: 1.0 for col in SENSOR_FEATURES}
        features["Gas"] = float("nan")
        issues = validate_features(features)
        assert "Gas" in issues

    def test_inf_feature(self):
        features = {col: 1.0 for col in SENSOR_FEATURES}
        features["Pressure"] = float("inf")
        issues = validate_features(features)
        assert "Pressure" in issues

    def test_custom_required(self):
        issues = validate_features({"x": 1.0}, required=["x", "y"])
        assert issues == ["y"]

    def test_empty_dict(self):
        issues = validate_features({})
        assert len(issues) == len(SENSOR_FEATURES)
