"""Unit tests for the Risk Prediction domain layer.

Covers:
  1. Value objects (RiskLevel, PredictionStatus enums)
  2. Domain exceptions
  3. ORM model (RiskPredictionModel)
  4. Pydantic schemas (request, response, validation)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest

from app.risk_prediction.domain.exceptions import (
    InsufficientFeaturesError,
    InvalidRiskScoreError,
    RiskModelNotLoadedError,
    RiskPredictionError,
    RiskPredictionFailedError,
    StaleDataError,
)
from app.risk_prediction.domain.value_objects import PredictionStatus, RiskLevel
from app.risk_prediction.models.risk_prediction_model import RiskPredictionModel
from app.risk_prediction.schemas import (
    ContributingFactor,
    CurrentRiskStatusResponse,
    CurrentRiskSummary,
    ForecastPoint,
    RiskFactorBreakdown,
    RiskPredictionHistoryResponse,
    RiskPredictionRequest,
    RiskPredictionResponse,
    SingleRiskPredictionResponse,
    ZoneRiskSummary,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Value Objects
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRiskLevel:
    def test_all_values(self):
        assert RiskLevel.LOW == "LOW"
        assert RiskLevel.MEDIUM == "MEDIUM"
        assert RiskLevel.HIGH == "HIGH"
        assert RiskLevel.CRITICAL == "CRITICAL"

    def test_from_string(self):
        assert RiskLevel("LOW") is RiskLevel.LOW
        assert RiskLevel("CRITICAL") is RiskLevel.CRITICAL

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            RiskLevel("EXTREME")

    def test_member_count(self):
        assert len(RiskLevel) == 4

    def test_is_str(self):
        assert isinstance(RiskLevel.HIGH, str)


class TestPredictionStatus:
    def test_all_values(self):
        assert PredictionStatus.PENDING == "PENDING"
        assert PredictionStatus.COMPLETED == "COMPLETED"
        assert PredictionStatus.FAILED == "FAILED"
        assert PredictionStatus.STALE == "STALE"

    def test_from_string(self):
        assert PredictionStatus("COMPLETED") is PredictionStatus.COMPLETED

    def test_member_count(self):
        assert len(PredictionStatus) == 4


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Domain Exceptions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestExceptions:
    def test_base_error(self):
        err = RiskPredictionError("something broke")
        assert str(err) == "something broke"
        assert err.error_code == "RISK_PREDICTION_ERROR"

    def test_model_not_loaded(self):
        err = RiskModelNotLoadedError("xgboost_v2")
        assert "xgboost_v2" in str(err)
        assert err.model_name == "xgboost_v2"
        assert err.error_code == "RISK_MODEL_NOT_LOADED"

    def test_prediction_failed(self):
        err = RiskPredictionFailedError("timeout")
        assert "timeout" in str(err)
        assert err.error_code == "RISK_PREDICTION_FAILED"

    def test_insufficient_features_with_list(self):
        err = InsufficientFeaturesError(["gas_level", "pressure"])
        assert "gas_level" in str(err)
        assert "pressure" in str(err)
        assert err.missing_features == ["gas_level", "pressure"]

    def test_insufficient_features_empty(self):
        err = InsufficientFeaturesError()
        assert "Insufficient" in str(err)
        assert err.missing_features == []

    def test_invalid_risk_score(self):
        err = InvalidRiskScoreError(105.0)
        assert "105" in str(err)
        assert err.score == 105.0

    def test_stale_data(self):
        err = StaleDataError("S001", 3600.0)
        assert "S001" in str(err)
        assert "3600" in str(err)
        assert err.sensor_id == "S001"
        assert err.age_seconds == 3600.0

    def test_inherits_from_domain_error(self):
        from app.shared.exceptions.domain_exceptions import DomainError

        err = RiskModelNotLoadedError("test")
        assert isinstance(err, DomainError)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. ORM Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRiskPredictionModel:
    def test_create_minimal(self):
        now = datetime.now(timezone.utc)
        model = RiskPredictionModel(
            id=str(uuid.uuid4()),
            prediction_timestamp=now,
            accident_probability=0.65,
            predicted_risk_score=72,
            risk_level="HIGH",
            confidence_score=0.88,
            model_name="xgboost_ensemble",
        )
        assert model.risk_level == "HIGH"
        assert model.predicted_risk_score == 72
        assert model.accident_probability == 0.65
        assert model.model_version == "1.0.0"  # default
        assert model.status == "COMPLETED"  # default

    def test_create_full(self):
        now = datetime.now(timezone.utc)
        model = RiskPredictionModel(
            id=str(uuid.uuid4()),
            sensor_id="S001",
            equipment_id="EQ-BOILER-001",
            zone_id="ZONE_A",
            prediction_timestamp=now,
            accident_probability=0.82,
            predicted_risk_score=89,
            risk_level="CRITICAL",
            confidence_score=0.91,
            model_name="xgboost_ensemble",
            model_version="2.1.0",
            risk_factors=json.dumps({"gas_risk": 0.65, "temperature_risk": 0.45}),
            top_contributing_factors=json.dumps([
                {"factor": "Gas Level", "weight": 0.4, "contribution": 0.65},
            ]),
            explanation="Risk Level: CRITICAL. Gas levels elevated.",
            status="COMPLETED",
        )
        assert model.sensor_id == "S001"
        assert model.equipment_id == "EQ-BOILER-001"
        assert model.zone_id == "ZONE_A"
        assert model.model_version == "2.1.0"
        assert "gas_risk" in model.risk_factors

    def test_repr(self):
        model = RiskPredictionModel(
            id="abc-123",
            prediction_timestamp=datetime.now(timezone.utc),
            accident_probability=0.5,
            predicted_risk_score=50,
            risk_level="MEDIUM",
            confidence_score=0.8,
            model_name="test_model",
        )
        r = repr(model)
        assert "RiskPredictionModel" in r
        assert "MEDIUM" in r
        assert "50" in r

    def test_tablename(self):
        assert RiskPredictionModel.__tablename__ == "risk_predictions"

    def test_nullable_context_fields(self):
        model = RiskPredictionModel(
            id=str(uuid.uuid4()),
            prediction_timestamp=datetime.now(timezone.utc),
            accident_probability=0.1,
            predicted_risk_score=5,
            risk_level="LOW",
            confidence_score=0.95,
            model_name="xgboost",
        )
        assert model.sensor_id is None
        assert model.equipment_id is None
        assert model.zone_id is None
        assert model.risk_factors is None
        assert model.explanation is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Pydantic Schemas — Request
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRiskPredictionRequest:
    def test_minimal_request(self):
        req = RiskPredictionRequest()
        assert req.sensor_id is None
        assert req.features == {}
        assert req.include_breakdown is False

    def test_full_request(self):
        req = RiskPredictionRequest(
            sensor_id="S001",
            equipment_id="EQ-001",
            zone_id="ZONE_A",
            features={
                "gas_level": 120.0,
                "pressure": 180.5,
                "temperature": 95.0,
            },
            include_breakdown=True,
            include_explanation=True,
        )
        assert req.sensor_id == "S001"
        assert req.features["gas_level"] == 120.0
        assert req.include_breakdown is True

    def test_json_roundtrip(self):
        req = RiskPredictionRequest(
            sensor_id="S001",
            features={"gas_level": 120.0},
        )
        data = req.model_dump()
        restored = RiskPredictionRequest(**data)
        assert restored.sensor_id == "S001"
        assert restored.features["gas_level"] == 120.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Pydantic Schemas — Response
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRiskPredictionResponse:
    def _sample_response(self, **overrides) -> dict:
        defaults = {
            "id": str(uuid.uuid4()),
            "prediction_timestamp": datetime.now(timezone.utc),
            "accident_probability": 0.65,
            "predicted_risk_score": 72,
            "risk_level": "HIGH",
            "confidence_score": 0.88,
            "model_name": "xgboost_ensemble",
            "model_version": "2.1.0",
            "status": "COMPLETED",
            "created_at": datetime.now(timezone.utc),
        }
        defaults.update(overrides)
        return defaults

    def test_valid_response(self):
        resp = RiskPredictionResponse(**self._sample_response())
        assert resp.risk_level == RiskLevel.HIGH
        assert resp.predicted_risk_score == 72

    def test_with_breakdown(self):
        breakdown = RiskFactorBreakdown(
            gas_risk=0.65,
            temperature_risk=0.45,
            pressure_risk=0.32,
        )
        resp = RiskPredictionResponse(
            **self._sample_response(risk_breakdown=breakdown)
        )
        assert resp.risk_breakdown is not None
        assert resp.risk_breakdown.gas_risk == 0.65

    def test_with_contributing_factors(self):
        factors = [
            ContributingFactor(
                factor="Gas Level", weight=0.4,
                current_value="120 ppm", contribution=0.65,
            ),
        ]
        resp = RiskPredictionResponse(
            **self._sample_response(top_contributing_factors=factors)
        )
        assert len(resp.top_contributing_factors) == 1
        assert resp.top_contributing_factors[0].factor == "Gas Level"

    def test_with_explanation(self):
        resp = RiskPredictionResponse(
            **self._sample_response(explanation="Risk is high due to gas levels.")
        )
        assert "gas" in resp.explanation.lower()

    def test_all_risk_levels(self):
        for level in RiskLevel:
            resp = RiskPredictionResponse(
                **self._sample_response(risk_level=level.value)
            )
            assert resp.risk_level == level

    def test_invalid_probability_too_high(self):
        with pytest.raises(Exception):
            RiskPredictionResponse(
                **self._sample_response(accident_probability=1.5)
            )

    def test_invalid_score_too_high(self):
        with pytest.raises(Exception):
            RiskPredictionResponse(
                **self._sample_response(predicted_risk_score=150)
            )


class TestSingleRiskPredictionResponse:
    def test_wrapper(self):
        inner = RiskPredictionResponse(
            id=str(uuid.uuid4()),
            prediction_timestamp=datetime.now(timezone.utc),
            accident_probability=0.3,
            predicted_risk_score=30,
            risk_level="MEDIUM",
            confidence_score=0.85,
            model_name="test",
            model_version="1.0.0",
            status="COMPLETED",
            created_at=datetime.now(timezone.utc),
        )
        resp = SingleRiskPredictionResponse(prediction=inner)
        assert resp.success is True
        assert resp.prediction.risk_level == RiskLevel.MEDIUM


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Pydantic Schemas — Current Risk Status
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCurrentRiskStatusResponse:
    def test_full_response(self):
        now = datetime.now(timezone.utc)
        resp = CurrentRiskStatusResponse(
            timestamp=now,
            current_risk=CurrentRiskSummary(
                overall_score=62,
                risk_level="HIGH",
                trend="increasing",
                trend_direction="up",
                last_update=now,
            ),
            by_zone=[
                ZoneRiskSummary(
                    zone_id="ZONE_A",
                    zone_name="Production Area 1",
                    risk_score=78,
                    risk_level="CRITICAL",
                    workers_present=12,
                    equipment_count=5,
                    active_hazards=["gas_leak", "high_temperature"],
                    active_permits=["hot_work_permit_001"],
                ),
            ],
            risk_breakdown=RiskFactorBreakdown(
                gas_risk=0.65,
                temperature_risk=0.45,
                pressure_risk=0.32,
                worker_density_risk=0.55,
            ),
        )
        assert resp.current_risk.overall_score == 62
        assert resp.current_risk.risk_level == RiskLevel.HIGH
        assert len(resp.by_zone) == 1
        assert resp.by_zone[0].zone_id == "ZONE_A"
        assert resp.by_zone[0].risk_level == RiskLevel.CRITICAL

    def test_minimal_response(self):
        now = datetime.now(timezone.utc)
        resp = CurrentRiskStatusResponse(
            timestamp=now,
            current_risk=CurrentRiskSummary(
                overall_score=10,
                risk_level="LOW",
                last_update=now,
            ),
        )
        assert resp.current_risk.overall_score == 10
        assert resp.by_zone == []
        assert resp.risk_breakdown is None
        assert resp.forecast is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Pydantic Schemas — History
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRiskPredictionHistoryResponse:
    def test_empty_history(self):
        resp = RiskPredictionHistoryResponse(
            predictions=[], total=0, offset=0, limit=50,
        )
        assert resp.success is True
        assert len(resp.predictions) == 0
        assert resp.total == 0

    def test_with_predictions(self):
        now = datetime.now(timezone.utc)
        pred = RiskPredictionResponse(
            id=str(uuid.uuid4()),
            prediction_timestamp=now,
            accident_probability=0.5,
            predicted_risk_score=50,
            risk_level="HIGH",
            confidence_score=0.8,
            model_name="xgboost",
            model_version="1.0.0",
            status="COMPLETED",
            created_at=now,
        )
        resp = RiskPredictionHistoryResponse(
            predictions=[pred, pred],
            total=100,
            offset=0,
            limit=50,
        )
        assert len(resp.predictions) == 2
        assert resp.total == 100


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Supporting Schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRiskFactorBreakdown:
    def test_defaults(self):
        bd = RiskFactorBreakdown()
        assert bd.gas_risk == 0.0
        assert bd.maintenance_risk == 0.0

    def test_full(self):
        bd = RiskFactorBreakdown(
            gas_risk=0.65, temperature_risk=0.45, pressure_risk=0.32,
            worker_density_risk=0.55, equipment_health_risk=0.28,
            permit_risk=0.72, maintenance_risk=0.40,
        )
        assert bd.permit_risk == 0.72

    def test_invalid_range(self):
        with pytest.raises(Exception):
            RiskFactorBreakdown(gas_risk=1.5)


class TestContributingFactor:
    def test_valid(self):
        cf = ContributingFactor(
            factor="Gas Level", weight=0.4,
            current_value="120 ppm", contribution=0.65,
        )
        assert cf.factor == "Gas Level"

    def test_invalid_weight(self):
        with pytest.raises(Exception):
            ContributingFactor(
                factor="X", weight=2.0,
                current_value="Y", contribution=0.5,
            )


class TestForecastPoint:
    def test_valid(self):
        fp = ForecastPoint(
            time=datetime.now(timezone.utc),
            predicted_risk=72,
            confidence=0.85,
        )
        assert fp.predicted_risk == 72

    def test_invalid_risk(self):
        with pytest.raises(Exception):
            ForecastPoint(
                time=datetime.now(timezone.utc),
                predicted_risk=150,
                confidence=0.5,
            )


class TestZoneRiskSummary:
    def test_full(self):
        zrs = ZoneRiskSummary(
            zone_id="ZONE_A",
            zone_name="Production Area 1",
            risk_score=78,
            risk_level="CRITICAL",
            workers_present=12,
            equipment_count=5,
            active_hazards=["gas_leak"],
            active_permits=["permit_001"],
        )
        assert zrs.risk_level == RiskLevel.CRITICAL
        assert "gas_leak" in zrs.active_hazards

    def test_minimal(self):
        zrs = ZoneRiskSummary(
            zone_id="ZONE_B", risk_score=20, risk_level="LOW",
        )
        assert zrs.workers_present == 0
        assert zrs.active_hazards == []
