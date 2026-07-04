"""Unit tests for the RiskPredictionService.

Tests use a mock model, scaler, and repository — no real model loading
or database access.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

from app.risk_prediction.domain.exceptions import (
    InsufficientFeaturesError,
    RiskModelNotLoadedError,
    RiskPredictionFailedError,
)
from app.risk_prediction.domain.value_objects import PredictionStatus, RiskLevel
from app.risk_prediction.models.risk_prediction_model import RiskPredictionModel
from app.risk_prediction.preprocessing.feature_builder import (
    SENSOR_FEATURES,
    RiskFeatureInput,
)
from app.risk_prediction.services.risk_prediction_service import (
    MODEL_NAME,
    MODEL_VERSION,
    RiskPredictionService,
    _ModelHolder,
    build_explanation,
    calculate_confidence,
    classify_risk_level,
    get_top_contributing_factors,
    probability_to_risk_score,
)


# ── Test fixtures ──


def _make_mock_model(proba: float = 0.65):
    """Create a mock XGBoost model that returns a fixed probability."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1.0 - proba, proba]])
    model.predict.return_value = np.array([1 if proba >= 0.5 else 0])
    model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1])
    return model


def _make_mock_scaler(n_features: int = 8):
    """Create a mock scaler that returns input unchanged."""
    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, n_features))
    return scaler


def _make_holder(proba: float = 0.65, n_features: int = 8) -> _ModelHolder:
    """Create a pre-loaded _ModelHolder with mocks."""
    holder = _ModelHolder()
    holder.model = _make_mock_model(proba)
    holder.scaler = _make_mock_scaler(n_features)
    holder.column_order = [f"feature_{i}" for i in range(n_features)]
    holder.feature_importances = {
        "Temp": 0.30, "Service_Days": 0.12, "Pressure": 0.08,
        "Age": 0.03, "Gas": 0.02, "Humidity": 0.01,
    }
    holder._loaded = True
    return holder


def _make_mock_repo() -> AsyncMock:
    """Create a mock repository that returns the prediction as-is."""
    repo = AsyncMock()
    repo.create_prediction = AsyncMock(side_effect=lambda p: p)
    repo.get_prediction = AsyncMock(return_value=None)
    repo.get_latest_prediction = AsyncMock(return_value=None)
    repo.get_prediction_history = AsyncMock(return_value=[])
    repo.count_predictions = AsyncMock(return_value=0)
    return repo


def _make_features() -> dict[str, float]:
    """Create a minimal feature dict with all required sensor features."""
    return {
        "Temp": 85.0,
        "Pressure": 22.0,
        "Humidity": 55.0,
        "Gas": 110.0,
        "Vibration": 3.5,
        "Speed": 120.0,
        "Sparks": 1.0,
        "Workers": 8.0,
        "Age": 45.0,
        "Service_Days": 600.0,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. classify_risk_level
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestClassifyRiskLevel:
    def test_critical(self):
        assert classify_risk_level(0.95) == RiskLevel.CRITICAL
        assert classify_risk_level(0.75) == RiskLevel.CRITICAL

    def test_high(self):
        assert classify_risk_level(0.74) == RiskLevel.HIGH
        assert classify_risk_level(0.50) == RiskLevel.HIGH

    def test_medium(self):
        assert classify_risk_level(0.49) == RiskLevel.MEDIUM
        assert classify_risk_level(0.25) == RiskLevel.MEDIUM

    def test_low(self):
        assert classify_risk_level(0.24) == RiskLevel.LOW
        assert classify_risk_level(0.0) == RiskLevel.LOW

    def test_boundary_values(self):
        assert classify_risk_level(1.0) == RiskLevel.CRITICAL
        assert classify_risk_level(0.0) == RiskLevel.LOW


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. probability_to_risk_score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestProbabilityToRiskScore:
    def test_zero(self):
        assert probability_to_risk_score(0.0) == 0

    def test_one(self):
        assert probability_to_risk_score(1.0) == 100

    def test_midpoint(self):
        assert probability_to_risk_score(0.5) == 50

    def test_typical_values(self):
        assert probability_to_risk_score(0.65) == 65
        assert probability_to_risk_score(0.123) == 12

    def test_clamped_high(self):
        assert probability_to_risk_score(1.5) == 100

    def test_clamped_low(self):
        assert probability_to_risk_score(-0.1) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. calculate_confidence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCalculateConfidence:
    def test_high_probability(self):
        conf = calculate_confidence(0.95)
        assert 0.9 <= conf <= 1.0

    def test_low_probability(self):
        conf = calculate_confidence(0.05)
        assert 0.9 <= conf <= 1.0  # Also confident (far from boundary)

    def test_uncertain(self):
        conf = calculate_confidence(0.5)
        assert conf == 0.5  # Maximum uncertainty at decision boundary

    def test_range(self):
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            conf = calculate_confidence(p)
            assert 0.0 <= conf <= 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. build_explanation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBuildExplanation:
    def test_returns_string(self):
        result = build_explanation(RiskLevel.HIGH, 0.65, [("Temp", 0.3)])
        assert isinstance(result, str)
        assert "high risk" in result.lower()

    def test_includes_probability(self):
        result = build_explanation(RiskLevel.LOW, 0.1, [])
        assert "10.0%" in result

    def test_includes_factors(self):
        factors = [("Temp", 0.3), ("Gas", 0.2)]
        result = build_explanation(RiskLevel.CRITICAL, 0.9, factors)
        assert "Temp" in result
        assert "Gas" in result

    def test_critical_message(self):
        result = build_explanation(RiskLevel.CRITICAL, 0.95, [])
        assert "immediate action" in result.lower()

    def test_low_message(self):
        result = build_explanation(RiskLevel.LOW, 0.05, [])
        assert "normal" in result.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. get_top_contributing_factors
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGetTopContributingFactors:
    def test_returns_top_n(self):
        importances = {"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.05, "E": 0.05}
        values = {"A": 80.0, "B": 20.0, "C": 5.0}
        factors = get_top_contributing_factors(importances, values, n=3)
        assert len(factors) == 3
        assert factors[0]["factor"] == "A"
        assert factors[1]["factor"] == "B"

    def test_factor_structure(self):
        importances = {"Temp": 0.3}
        values = {"Temp": 85.0}
        factors = get_top_contributing_factors(importances, values, n=1)
        f = factors[0]
        assert f["factor"] == "Temp"
        assert f["weight"] == 0.3
        assert f["current_value"] == "85.0"
        assert f["contribution"] == 0.3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. _ModelHolder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestModelHolder:
    def test_initially_not_loaded(self):
        holder = _ModelHolder()
        assert holder.is_loaded is False
        assert holder.model is None
        assert holder.column_order == []

    def test_reset(self):
        holder = _make_holder()
        assert holder.is_loaded is True
        holder.reset()
        assert holder.is_loaded is False
        assert holder.model is None

    def test_load_missing_model(self, tmp_path):
        holder = _ModelHolder()
        with pytest.raises(RiskModelNotLoadedError):
            holder.load(
                model_path=tmp_path / "nonexistent.pkl",
                scaler_path=tmp_path / "scaler.pkl",
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. RiskPredictionService — predict_from_features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPredictFromFeatures:
    async def test_basic_prediction(self):
        holder = _make_holder(proba=0.65)
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)

        result = await service.predict_from_features(
            _make_features(), sensor_id="S001",
        )

        assert isinstance(result, RiskPredictionModel)
        assert result.accident_probability == pytest.approx(0.65, abs=0.01)
        assert result.risk_level == RiskLevel.HIGH.value
        assert result.predicted_risk_score == 65
        assert result.model_name == MODEL_NAME
        assert result.status == PredictionStatus.COMPLETED.value
        repo.create_prediction.assert_awaited_once()

    async def test_critical_prediction(self):
        holder = _make_holder(proba=0.92)
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)

        result = await service.predict_from_features(_make_features())
        assert result.risk_level == RiskLevel.CRITICAL.value
        assert result.predicted_risk_score == 92

    async def test_low_prediction(self):
        holder = _make_holder(proba=0.05)
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)

        result = await service.predict_from_features(_make_features())
        assert result.risk_level == RiskLevel.LOW.value
        assert result.predicted_risk_score == 5

    async def test_with_context_fields(self):
        holder = _make_holder()
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)

        result = await service.predict_from_features(
            _make_features(),
            sensor_id="S001",
            equipment_id="EQ-001",
            zone_id="ZONE_A",
        )
        assert result.sensor_id == "S001"
        assert result.equipment_id == "EQ-001"
        assert result.zone_id == "ZONE_A"

    async def test_no_persist(self):
        holder = _make_holder()
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)

        result = await service.predict_from_features(
            _make_features(), persist=False,
        )
        assert isinstance(result, RiskPredictionModel)
        repo.create_prediction.assert_not_awaited()

    async def test_with_explanation(self):
        holder = _make_holder(proba=0.80)
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)

        result = await service.predict_from_features(
            _make_features(), include_explanation=True,
        )
        assert result.explanation is not None
        assert "Temp" in result.explanation

    async def test_with_breakdown(self):
        holder = _make_holder()
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)

        result = await service.predict_from_features(
            _make_features(), include_breakdown=True,
        )
        assert result.top_contributing_factors is not None
        factors = json.loads(result.top_contributing_factors)
        assert len(factors) > 0
        assert factors[0]["factor"] == "Temp"

    async def test_insufficient_features_raises(self):
        holder = _make_holder()
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)

        # All sensor features missing + disable defaults
        with pytest.raises(InsufficientFeaturesError):
            await service.predict_from_features(
                {"Temp": float("nan")},  # NaN = invalid
            )

    async def test_model_not_loaded_raises(self):
        holder = _ModelHolder()  # Not loaded
        repo = _make_mock_repo()
        service = RiskPredictionService(
            repo,
            model_holder=holder,
            model_path="/nonexistent/model.pkl",
        )

        with pytest.raises(RiskModelNotLoadedError):
            await service.predict_from_features(_make_features())

    async def test_scaler_failure_raises(self):
        holder = _make_holder()
        holder.scaler.transform.side_effect = ValueError("Scaler error")
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)

        with pytest.raises(RiskPredictionFailedError, match="scaling"):
            await service.predict_from_features(_make_features())

    async def test_model_inference_failure_raises(self):
        holder = _make_holder()
        holder.model.predict_proba.side_effect = RuntimeError("inference error")
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)

        with pytest.raises(RiskPredictionFailedError, match="inference"):
            await service.predict_from_features(_make_features())

    async def test_confidence_populated(self):
        holder = _make_holder(proba=0.95)
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)

        result = await service.predict_from_features(_make_features())
        assert 0 < result.confidence_score <= 1.0

    async def test_prediction_id_is_uuid(self):
        holder = _make_holder()
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)

        result = await service.predict_from_features(_make_features())
        uuid.UUID(result.id)  # Validates it's a valid UUID


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. RiskPredictionService — predict_from_input
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPredictFromInput:
    async def test_basic(self):
        holder = _make_holder()
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)

        inp = RiskFeatureInput(
            temperature=85.0, pressure=22.0, humidity=55.0,
            gas=110.0, vibration=3.5, speed=120.0,
            workers=8, training=True, experience="Senior",
        )
        result = await service.predict_from_input(
            inp, sensor_id="S001", zone_id="ZONE_A",
        )
        assert isinstance(result, RiskPredictionModel)
        assert result.zone_id == "ZONE_A"

    async def test_uses_input_zone_id(self):
        holder = _make_holder()
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)

        inp = RiskFeatureInput(zone_id="ZONE_B")
        result = await service.predict_from_input(inp)
        assert result.zone_id == "ZONE_B"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Query helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestQueryHelpers:
    async def test_get_prediction(self):
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=_make_holder())
        await service.get_prediction("pred-123")
        repo.get_prediction.assert_awaited_once_with("pred-123")

    async def test_get_latest(self):
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=_make_holder())
        await service.get_latest_prediction(sensor_id="S001")
        repo.get_latest_prediction.assert_awaited_once_with(
            sensor_id="S001", zone_id=None,
        )

    async def test_get_history(self):
        repo = _make_mock_repo()
        repo.get_prediction_history.return_value = []
        repo.count_predictions.return_value = 0
        service = RiskPredictionService(repo, model_holder=_make_holder())

        preds, total = await service.get_prediction_history(
            sensor_id="S001", offset=5, limit=10,
        )
        assert preds == []
        assert total == 0
        repo.get_prediction_history.assert_awaited_once()
        repo.count_predictions.assert_awaited_once()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Model info
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestModelInfo:
    def test_model_info(self):
        holder = _make_holder()
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)
        info = service.get_model_info()
        assert info["model_name"] == MODEL_NAME
        assert info["model_version"] == MODEL_VERSION
        assert info["is_loaded"] is True
        assert info["feature_count"] == 8

    def test_model_info_not_loaded(self):
        holder = _ModelHolder()
        repo = _make_mock_repo()
        service = RiskPredictionService(repo, model_holder=holder)
        info = service.get_model_info()
        assert info["is_loaded"] is False
        assert info["feature_count"] == 0
