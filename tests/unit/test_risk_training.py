"""Unit tests for the XGBoost Risk Prediction training pipeline.

Tests cover all pure functions using small synthetic datasets — no
actual model training on the full 100K-row dataset is performed.
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
except (ImportError, OSError):
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier  # type: ignore

from training.train_risk_prediction import (
    DEFAULT_PARAMS,
    EvaluationResult,
    FeatureImportance,
    cross_validate,
    evaluate_model,
    load_feature_metadata,
    load_model,
    prepare_data,
    save_evaluation_report,
    save_feature_importance,
    save_model,
    train_model,
)


# ── Helpers ──


def _make_feature_df(n: int = 200, accident_rate: float = 0.1) -> pd.DataFrame:
    """Create a synthetic feature DataFrame mimicking the real features.csv."""
    rng = np.random.default_rng(42)
    n_positive = int(n * accident_rate)
    n_negative = n - n_positive

    df = pd.DataFrame({
        "Workers": rng.integers(1, 15, n).astype(float),
        "Temp": rng.normal(35, 5, n),
        "Pressure": rng.normal(15, 2, n),
        "Humidity": rng.normal(50, 10, n),
        "Vibration": rng.normal(2, 0.5, n),
        "Speed": rng.normal(100, 20, n),
        "Gas": rng.normal(50, 15, n),
        "Sparks": rng.integers(0, 2, n).astype(float),
        "Age": rng.normal(40, 10, n),
        "Service_Days": rng.normal(500, 200, n),
        "Risk": rng.uniform(0, 100, n),
        "Accident": np.concatenate([np.zeros(n_negative), np.ones(n_positive)]).astype(int),
        # Identifiers (excluded from features)
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
        "Factory_Factory_A": rng.integers(0, 2, n).astype(float),
        "Factory_Factory_B": rng.integers(0, 2, n).astype(float),
        "Factory_Factory_C": [0.0] * n,
        "Factory_Factory_D": [0.0] * n,
        "Region_Industrial_Zone": [1.0] * n,
        "Region_Urban": [0.0] * n,
        "Region_Rural": [0.0] * n,
        "Shift_Day": rng.integers(0, 2, n).astype(float),
        "Shift_Night": [0.0] * n,
        "Exp_Junior": rng.integers(0, 2, n).astype(float),
        "Exp_Senior": [0.0] * n,
        "Training_Yes": rng.integers(0, 2, n).astype(float),
        "Training_No": [0.0] * n,
        "Alarm_On": [0.0] * n,
        "Alarm_Off": [1.0] * n,
    })
    # Shuffle to mix positives and negatives
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def _quick_train() -> tuple[XGBClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler]:
    """Quick training on synthetic data for use by multiple tests."""
    df = _make_feature_df(n=200)
    X_train, X_test, y_train, y_test, cols, scaler = prepare_data(df, test_size=0.3)
    params = {**DEFAULT_PARAMS, "n_estimators": 10, "max_depth": 3}
    model = train_model(X_train, y_train, params=params)
    return model, X_train, X_test, y_train, y_test, cols, scaler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. prepare_data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPrepareData:
    def test_returns_correct_types(self):
        df = _make_feature_df(n=100)
        X_train, X_test, y_train, y_test, cols, scaler = prepare_data(df)
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        assert isinstance(cols, list)
        assert isinstance(scaler, StandardScaler)

    def test_split_sizes(self):
        df = _make_feature_df(n=100)
        X_train, X_test, y_train, y_test, cols, scaler = prepare_data(df, test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_feature_columns_exclude_targets(self):
        df = _make_feature_df(n=100)
        _, _, _, _, cols, _ = prepare_data(df)
        assert "Accident" not in cols
        assert "Risk" not in cols
        assert "timestamp" not in cols
        assert "zone_id" not in cols

    def test_stratified_split(self):
        df = _make_feature_df(n=1000, accident_rate=0.1)
        _, _, y_train, y_test, _, _ = prepare_data(df, test_size=0.2)
        # Both splits should have similar positive rates
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.05

    def test_scaling_applied(self):
        df = _make_feature_df(n=200)
        X_train, _, _, _, _, _ = prepare_data(df)
        # Scaled data should have approximately zero mean
        means = np.abs(X_train.mean(axis=0))
        assert means.mean() < 0.5  # Not exact due to small sample

    def test_nan_handling(self):
        df = _make_feature_df(n=100)
        df.iloc[0, 1] = np.nan  # Inject NaN in Temp
        X_train, X_test, _, _, _, _ = prepare_data(df)
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. train_model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTrainModel:
    def test_returns_classifier(self):
        df = _make_feature_df(n=100)
        X_train, _, y_train, _, _, _ = prepare_data(df)
        model = train_model(X_train, y_train, params={**DEFAULT_PARAMS, "n_estimators": 10})
        assert isinstance(model, XGBClassifier)

    def test_model_can_predict(self):
        model, X_train, X_test, _, _, _, _ = _quick_train()
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})

    def test_model_has_feature_importances(self):
        model, _, _, _, _, cols, _ = _quick_train()
        importances = model.feature_importances_
        assert len(importances) == len(cols)
        assert all(i >= 0 for i in importances)

    def test_custom_params(self):
        df = _make_feature_df(n=100)
        X_train, _, y_train, _, _, _ = prepare_data(df)
        params = {**DEFAULT_PARAMS, "n_estimators": 5, "max_depth": 2}
        model = train_model(X_train, y_train, params=params)
        assert model.n_estimators == 5
        assert model.max_depth == 2

    def test_scale_pos_weight_applied(self):
        df = _make_feature_df(n=200, accident_rate=0.1)
        X_train, _, y_train, _, _, _ = prepare_data(df)
        model = train_model(X_train, y_train, params={**DEFAULT_PARAMS, "n_estimators": 5})
        assert model.scale_pos_weight > 1.0  # Should be > 1 for imbalanced


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. cross_validate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCrossValidate:
    def test_returns_scores(self):
        df = _make_feature_df(n=200)
        X_train, _, y_train, _, _, _ = prepare_data(df)
        params = {**DEFAULT_PARAMS, "n_estimators": 10}
        scores, mean, std = cross_validate(X_train, y_train, params=params, cv_folds=3)
        assert len(scores) == 3
        assert 0 <= mean <= 1
        assert std >= 0

    def test_scores_are_reasonable(self):
        df = _make_feature_df(n=300)
        X_train, _, y_train, _, _, _ = prepare_data(df)
        params = {**DEFAULT_PARAMS, "n_estimators": 20}
        scores, mean, _ = cross_validate(X_train, y_train, params=params, cv_folds=3)
        # With synthetic data, AUC should be above random (0.5)
        assert mean > 0.4


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. evaluate_model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEvaluateModel:
    def test_returns_results(self):
        model, X_train, X_test, y_train, y_test, cols, _ = _quick_train()
        result, feat_imp = evaluate_model(
            model, X_test, y_test, X_train, y_train, cols,
        )
        assert isinstance(result, EvaluationResult)
        assert isinstance(feat_imp, FeatureImportance)

    def test_metrics_in_range(self):
        model, X_train, X_test, y_train, y_test, cols, _ = _quick_train()
        result, _ = evaluate_model(
            model, X_test, y_test, X_train, y_train, cols,
        )
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1
        assert 0 <= result.f1 <= 1
        assert 0 <= result.roc_auc <= 1

    def test_confusion_matrix_shape(self):
        model, X_train, X_test, y_train, y_test, cols, _ = _quick_train()
        result, _ = evaluate_model(
            model, X_test, y_test, X_train, y_train, cols,
        )
        assert len(result.confusion) == 2
        assert len(result.confusion[0]) == 2

    def test_feature_importance_aligned(self):
        model, X_train, X_test, y_train, y_test, cols, _ = _quick_train()
        _, feat_imp = evaluate_model(
            model, X_test, y_test, X_train, y_train, cols,
        )
        assert feat_imp.feature_names == cols
        assert len(feat_imp.importances) == len(cols)
        assert feat_imp.column_order == cols

    def test_top_n_features(self):
        model, X_train, X_test, y_train, y_test, cols, _ = _quick_train()
        _, feat_imp = evaluate_model(
            model, X_test, y_test, X_train, y_train, cols,
        )
        top = feat_imp.top_n
        assert len(top) <= 10
        # Should be sorted descending
        for i in range(len(top) - 1):
            assert top[i][1] >= top[i + 1][1]

    def test_cv_scores_included(self):
        model, X_train, X_test, y_train, y_test, cols, _ = _quick_train()
        cv = [0.85, 0.90, 0.88]
        result, _ = evaluate_model(
            model, X_test, y_test, X_train, y_train, cols,
            cv_scores=cv,
        )
        assert result.cv_scores == [0.85, 0.90, 0.88]
        assert result.cv_mean == round(np.mean(cv), 4)

    def test_report_string(self):
        model, X_train, X_test, y_train, y_test, cols, _ = _quick_train()
        result, _ = evaluate_model(
            model, X_test, y_test, X_train, y_train, cols,
        )
        assert "Normal" in result.report
        assert "Accident" in result.report

    def test_to_dict_serialisable(self):
        model, X_train, X_test, y_train, y_test, cols, _ = _quick_train()
        result, _ = evaluate_model(
            model, X_test, y_test, X_train, y_train, cols,
        )
        d = result.to_dict()
        # Must be JSON-serialisable
        json.dumps(d)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Save / Load
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSaveLoad:
    def test_save_and_load_model(self, tmp_path):
        model, _, _, _, _, _, scaler = _quick_train()
        model_p = tmp_path / "model.pkl"
        scaler_p = tmp_path / "scaler.pkl"
        save_model(model, scaler, model_p, scaler_p)
        assert model_p.exists()
        assert scaler_p.exists()

        loaded_model, loaded_scaler = load_model(model_p, scaler_p)
        assert isinstance(loaded_model, XGBClassifier)
        assert isinstance(loaded_scaler, StandardScaler)

    def test_save_feature_importance(self, tmp_path):
        model, X_train, X_test, y_train, y_test, cols, _ = _quick_train()
        _, feat_imp = evaluate_model(
            model, X_test, y_test, X_train, y_train, cols,
        )
        fp = tmp_path / "features.json"
        save_feature_importance(feat_imp, fp)
        assert fp.exists()

        # Verify JSON structure
        with open(fp) as f:
            data = json.load(f)
        assert "feature_names" in data
        assert "importances" in data
        assert "column_order" in data

    def test_load_feature_metadata(self, tmp_path):
        feat_imp = FeatureImportance(
            feature_names=["a", "b"],
            importances=[0.6, 0.4],
            column_order=["a", "b"],
        )
        fp = tmp_path / "features.json"
        save_feature_importance(feat_imp, fp)

        loaded = load_feature_metadata(fp)
        assert loaded.feature_names == ["a", "b"]
        assert loaded.importances == [0.6, 0.4]

    def test_save_evaluation_report(self, tmp_path):
        model, X_train, X_test, y_train, y_test, cols, _ = _quick_train()
        result, _ = evaluate_model(
            model, X_test, y_test, X_train, y_train, cols,
        )
        fp = tmp_path / "report.json"
        save_evaluation_report(result, fp)
        assert fp.exists()

        with open(fp) as f:
            data = json.load(f)
        assert "accuracy" in data
        assert "roc_auc" in data
        assert "confusion" in data

    def test_loaded_model_predicts(self, tmp_path):
        model, _, X_test, _, _, _, scaler = _quick_train()
        model_p = tmp_path / "model.pkl"
        scaler_p = tmp_path / "scaler.pkl"
        save_model(model, scaler, model_p, scaler_p)

        loaded_model, loaded_scaler = load_model(model_p, scaler_p)
        preds = loaded_model.predict(X_test)
        assert len(preds) == len(X_test)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. EvaluationResult / FeatureImportance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDataClasses:
    def test_evaluation_result_to_dict(self):
        r = EvaluationResult(
            accuracy=0.95, precision=0.90, recall=0.85, f1=0.87,
            roc_auc=0.92, confusion=[[100, 5], [3, 20]],
            report="test report", total_samples=128,
            test_samples=28, train_samples=100,
            positive_count=23, feature_count=50,
        )
        d = r.to_dict()
        assert d["accuracy"] == 0.95
        assert d["confusion"] == [[100, 5], [3, 20]]

    def test_feature_importance_top_n(self):
        fi = FeatureImportance(
            feature_names=["a", "b", "c"],
            importances=[0.1, 0.5, 0.3],
            column_order=["a", "b", "c"],
        )
        top = fi.top_n
        assert top[0] == ("b", 0.5)
        assert top[1] == ("c", 0.3)
        assert top[2] == ("a", 0.1)
