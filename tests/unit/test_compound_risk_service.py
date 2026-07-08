"""Unit tests for the Compound Risk Aggregation Engine.

Tests cover:
  - Configuration (weights, thresholds)
  - Normalisation of each component
  - Weighted aggregation
  - Risk level classification
  - Confidence calculation
  - Contributing factors
  - Persistence (compute_and_persist)
  - Query helpers
  - Edge cases
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.compound_risk.domain.value_objects import RiskLevel
from app.compound_risk.models.compound_risk_model import CompoundRiskModel
from app.compound_risk.repositories.sqlalchemy_compound_risk_repo import (
    SQLAlchemyCompoundRiskRepository,
)
from app.compound_risk.services.compound_risk_service import (
    CompoundRiskAggregationService,
    CompoundRiskInput,
    CompoundRiskResult,
    CompoundRiskWeights,
    RiskLevelThresholds,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest_asyncio.fixture
async def repo(db_session: AsyncSession) -> SQLAlchemyCompoundRiskRepository:
    return SQLAlchemyCompoundRiskRepository(db_session)


@pytest_asyncio.fixture
async def service(repo) -> CompoundRiskAggregationService:
    return CompoundRiskAggregationService(repo)


@pytest_asyncio.fixture
async def custom_service(repo) -> CompoundRiskAggregationService:
    """Service with custom weights and thresholds."""
    weights = CompoundRiskWeights(
        risk_prediction_weight=0.50,
        isolation_forest_weight=0.10,
        autoencoder_weight=0.10,
        sensor_health_weight=0.10,
        alert_weight=0.10,
        threshold_violation_weight=0.10,
    )
    thresholds = RiskLevelThresholds(
        low_max=20.0, medium_max=40.0, high_max=60.0,
    )
    return CompoundRiskAggregationService(repo, weights=weights, thresholds=thresholds)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestWeightsConfig:
    def test_default_weights_sum_to_one(self):
        w = CompoundRiskWeights()
        assert abs(w.total - 1.0) < 0.001

    def test_custom_weights(self):
        w = CompoundRiskWeights(
            risk_prediction_weight=0.5,
            isolation_forest_weight=0.1,
            autoencoder_weight=0.1,
            sensor_health_weight=0.1,
            alert_weight=0.1,
            threshold_violation_weight=0.1,
        )
        assert abs(w.total - 1.0) < 0.001

    def test_normalise(self):
        w = CompoundRiskWeights(
            risk_prediction_weight=3.0,
            isolation_forest_weight=2.0,
            autoencoder_weight=1.5,
            sensor_health_weight=1.5,
            alert_weight=1.0,
            threshold_violation_weight=1.0,
        )
        n = w.normalised()
        assert abs(n.total - 1.0) < 0.001
        assert n.risk_prediction_weight == pytest.approx(0.3)

    def test_to_dict(self):
        w = CompoundRiskWeights()
        d = w.to_dict()
        assert "risk_prediction_weight" in d
        assert "alert_weight" in d

    def test_auto_normalise_on_init(self, repo):
        weights = CompoundRiskWeights(
            risk_prediction_weight=0.6,
            isolation_forest_weight=0.4,
            autoencoder_weight=0.3,
            sensor_health_weight=0.3,
            alert_weight=0.2,
            threshold_violation_weight=0.2,
        )
        svc = CompoundRiskAggregationService(repo, weights=weights)
        assert abs(svc.weights.total - 1.0) < 0.001


class TestThresholdsConfig:
    def test_defaults(self):
        t = RiskLevelThresholds()
        assert t.low_max == 25.0
        assert t.medium_max == 50.0
        assert t.high_max == 75.0

    def test_custom(self):
        t = RiskLevelThresholds(low_max=20, medium_max=40, high_max=60)
        assert t.low_max == 20.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Normalisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNormalisation:
    def test_zero_inputs(self, service):
        inp = CompoundRiskInput()
        comps = service._normalise_components(inp)
        assert comps["risk_prediction"] == 0.0
        assert comps["isolation_forest"] == 0.0
        assert comps["autoencoder"] == 0.0
        assert comps["sensor_health"] == 0.0  # 100 health → 0 risk
        assert comps["alert"] == 0.0
        assert comps["threshold_violation"] == 0.0

    def test_max_inputs(self, service):
        inp = CompoundRiskInput(
            accident_probability=1.0,
            isolation_forest_score=1.0,
            autoencoder_score=1.0,
            sensor_health_score=0.0,  # Worst health → 100 risk
            active_alert_count=10,
            alert_severity_max=1.0,
            threshold_violation_count=5,
        )
        comps = service._normalise_components(inp)
        assert comps["risk_prediction"] == 100.0
        assert comps["isolation_forest"] == 100.0
        assert comps["autoencoder"] == 100.0
        assert comps["sensor_health"] == 100.0
        assert comps["alert"] == 100.0
        assert comps["threshold_violation"] == 100.0

    def test_sensor_health_inverted(self, service):
        """High health → low risk, low health → high risk."""
        inp_healthy = CompoundRiskInput(sensor_health_score=90.0)
        inp_unhealthy = CompoundRiskInput(sensor_health_score=20.0)
        assert service._normalise_components(inp_healthy)["sensor_health"] == 10.0
        assert service._normalise_components(inp_unhealthy)["sensor_health"] == 80.0

    def test_alert_scaling(self, service):
        """Alert risk scales with count and severity."""
        inp_1 = CompoundRiskInput(active_alert_count=1, alert_severity_max=0.5)
        inp_5 = CompoundRiskInput(active_alert_count=5, alert_severity_max=0.8)
        c1 = service._normalise_components(inp_1)["alert"]
        c5 = service._normalise_components(inp_5)["alert"]
        assert c5 > c1

    def test_threshold_violation_scaling(self, service):
        inp_0 = CompoundRiskInput(threshold_violation_count=0)
        inp_3 = CompoundRiskInput(threshold_violation_count=3)
        inp_5 = CompoundRiskInput(threshold_violation_count=5)
        inp_10 = CompoundRiskInput(threshold_violation_count=10)
        c0 = service._normalise_components(inp_0)["threshold_violation"]
        c3 = service._normalise_components(inp_3)["threshold_violation"]
        c5 = service._normalise_components(inp_5)["threshold_violation"]
        c10 = service._normalise_components(inp_10)["threshold_violation"]
        assert c0 == 0.0
        assert c3 == 60.0
        assert c5 == 100.0
        assert c10 == 100.0  # Capped at 100


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Weighted aggregation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAggregation:
    def test_all_zero_gives_zero(self, service):
        result = service.compute(CompoundRiskInput())
        assert result.compound_risk_score == 0.0

    def test_all_max_gives_hundred(self, service):
        inp = CompoundRiskInput(
            accident_probability=1.0,
            isolation_forest_score=1.0,
            autoencoder_score=1.0,
            sensor_health_score=0.0,
            active_alert_count=10,
            alert_severity_max=1.0,
            threshold_violation_count=5,
        )
        result = service.compute(inp)
        assert result.compound_risk_score == 100.0

    def test_single_component(self, service):
        """Only risk prediction at 0.5 → ~15 (0.5*100*0.30)."""
        inp = CompoundRiskInput(accident_probability=0.5)
        result = service.compute(inp)
        assert result.compound_risk_score == pytest.approx(15.0, abs=0.1)

    def test_custom_weights_emphasise_risk_prediction(self, custom_service):
        """Custom service has 0.50 weight on risk prediction."""
        inp = CompoundRiskInput(accident_probability=0.6)
        result = custom_service.compute(inp)
        # 0.6 * 100 * 0.50 = 30.0
        assert result.compound_risk_score == pytest.approx(30.0, abs=0.1)

    def test_score_clamped_to_zero(self, service):
        """Negative inputs don't produce negative scores."""
        inp = CompoundRiskInput(accident_probability=-0.5)
        result = service.compute(inp)
        assert result.compound_risk_score >= 0.0

    def test_score_clamped_to_hundred(self, service):
        """Extreme inputs don't exceed 100."""
        inp = CompoundRiskInput(
            accident_probability=2.0,  # Out of range
            isolation_forest_score=2.0,
            autoencoder_score=2.0,
            sensor_health_score=-50.0,
            active_alert_count=100,
            alert_severity_max=1.0,
            threshold_violation_count=100,
        )
        result = service.compute(inp)
        assert result.compound_risk_score <= 100.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Risk level classification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestClassification:
    def test_low_risk(self, service):
        inp = CompoundRiskInput(accident_probability=0.1)
        result = service.compute(inp)
        assert result.risk_level == RiskLevel.LOW

    def test_medium_risk(self, service):
        inp = CompoundRiskInput(
            accident_probability=0.5,
            isolation_forest_score=0.5,
            autoencoder_score=0.5,
        )
        result = service.compute(inp)
        assert result.risk_level == RiskLevel.MEDIUM

    def test_high_risk(self, service):
        inp = CompoundRiskInput(
            accident_probability=0.8,
            isolation_forest_score=0.7,
            autoencoder_score=0.6,
            sensor_health_score=40.0,
        )
        result = service.compute(inp)
        assert result.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_critical_risk(self, service):
        inp = CompoundRiskInput(
            accident_probability=1.0,
            isolation_forest_score=1.0,
            autoencoder_score=1.0,
            sensor_health_score=0.0,
            active_alert_count=10,
            alert_severity_max=1.0,
            threshold_violation_count=5,
        )
        result = service.compute(inp)
        assert result.risk_level == RiskLevel.CRITICAL

    def test_custom_thresholds(self, custom_service):
        """Custom thresholds: low_max=20, medium_max=40, high_max=60."""
        # Score ~30 with custom weights: risk_prediction=0.6 → 0.6*100*0.5 = 30
        inp = CompoundRiskInput(accident_probability=0.6)
        result = custom_service.compute(inp)
        assert result.risk_level == RiskLevel.MEDIUM

    def test_boundary_low_medium(self, service):
        """Score exactly at low_max threshold."""
        result = service._classify(25.0)
        assert result == RiskLevel.MEDIUM  # 25 >= low_max(25) → MEDIUM

    def test_boundary_medium_high(self, service):
        result = service._classify(50.0)
        assert result == RiskLevel.HIGH

    def test_boundary_high_critical(self, service):
        result = service._classify(75.0)
        assert result == RiskLevel.CRITICAL

    def test_zero_is_low(self, service):
        result = service._classify(0.0)
        assert result == RiskLevel.LOW


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Confidence calculation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConfidence:
    def test_min_confidence(self, service):
        """Empty inputs → low but non-zero confidence."""
        result = service.compute(CompoundRiskInput())
        assert result.confidence_score >= 0.1

    def test_more_inputs_higher_confidence(self, service):
        """More data sources → higher confidence."""
        sparse = CompoundRiskInput(accident_probability=0.5)
        rich = CompoundRiskInput(
            accident_probability=0.5,
            isolation_forest_score=0.5,
            autoencoder_score=0.5,
            sensor_health_score=50.0,
        )
        c_sparse = service.compute(sparse).confidence_score
        c_rich = service.compute(rich).confidence_score
        assert c_rich > c_sparse

    def test_agreeing_models_higher_confidence(self, service):
        """Models agreeing on similar scores → higher confidence."""
        agreeing = CompoundRiskInput(
            accident_probability=0.7,
            isolation_forest_score=0.7,
            autoencoder_score=0.7,
        )
        disagreeing = CompoundRiskInput(
            accident_probability=0.9,
            isolation_forest_score=0.1,
            autoencoder_score=0.5,
        )
        c_agree = service.compute(agreeing).confidence_score
        c_disagree = service.compute(disagreeing).confidence_score
        assert c_agree > c_disagree

    def test_healthy_sensor_higher_confidence(self, service):
        """Healthy sensors → higher confidence."""
        healthy = CompoundRiskInput(
            accident_probability=0.5,
            sensor_health_score=95.0,
        )
        degraded = CompoundRiskInput(
            accident_probability=0.5,
            sensor_health_score=20.0,
        )
        c_healthy = service.compute(healthy).confidence_score
        c_degraded = service.compute(degraded).confidence_score
        assert c_healthy > c_degraded

    def test_confidence_bounded(self, service):
        """Confidence is always between 0.1 and 1.0."""
        for prob in [0.0, 0.5, 1.0]:
            result = service.compute(
                CompoundRiskInput(accident_probability=prob),
            )
            assert 0.1 <= result.confidence_score <= 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Contributing factors
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestContributingFactors:
    def test_all_factors_present(self, service):
        result = service.compute(CompoundRiskInput(accident_probability=0.5))
        names = {f["factor"] for f in result.contributing_factors}
        assert "Accident Probability" in names
        assert "Isolation Forest Anomaly" in names
        assert "Autoencoder Anomaly" in names
        assert "Sensor Health Degradation" in names
        assert "Active Alerts" in names
        assert "Threshold Violations" in names

    def test_factors_have_weight(self, service):
        result = service.compute(CompoundRiskInput(accident_probability=0.5))
        for f in result.contributing_factors:
            assert "weight" in f
            assert f["weight"] > 0

    def test_factors_sorted_by_contribution(self, service):
        inp = CompoundRiskInput(
            accident_probability=0.9,
            isolation_forest_score=0.1,
        )
        result = service.compute(inp)
        contributions = [
            float(f["current_value"]) * f["weight"]
            for f in result.contributing_factors
        ]
        assert contributions == sorted(contributions, reverse=True)

    def test_component_scores_dict(self, service):
        result = service.compute(CompoundRiskInput(accident_probability=0.5))
        assert "risk_prediction" in result.component_scores
        assert result.component_scores["risk_prediction"] == 50.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Persistence (compute_and_persist)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPersistence:
    async def test_compute_and_persist(self, service, repo):
        inp = CompoundRiskInput(
            accident_probability=0.65,
            isolation_forest_score=0.7,
            autoencoder_score=0.5,
            sensor_health_score=60.0,
            zone_id="ZONE_A",
            equipment_id="EQ001",
        )
        model = await service.compute_and_persist(inp)
        assert model.id is not None
        assert model.zone_id == "ZONE_A"
        assert model.equipment_id == "EQ001"
        assert model.risk_level in [r.value for r in RiskLevel]

        # Verify persisted
        stored = await repo.get_by_id(model.id)
        assert stored is not None
        assert stored.compound_risk_score == model.compound_risk_score

    async def test_persist_contributing_factors(self, service, repo):
        inp = CompoundRiskInput(accident_probability=0.5, zone_id="ZONE_B")
        model = await service.compute_and_persist(inp)
        assert model.contributing_factors is not None
        factors = json.loads(model.contributing_factors)
        assert len(factors) == 6

    async def test_persist_with_recommendation(self, service, repo):
        inp = CompoundRiskInput(accident_probability=0.8)
        model = await service.compute_and_persist(
            inp, recommendation="Stop hot work activities",
        )
        assert model.recommendation == "Stop hot work activities"

    async def test_anomaly_score_is_max(self, service, repo):
        """Persisted anomaly_score = max(IF, AE)."""
        inp = CompoundRiskInput(
            isolation_forest_score=0.3,
            autoencoder_score=0.7,
        )
        model = await service.compute_and_persist(inp)
        assert model.anomaly_score == 0.7

    async def test_compound_score_stored_as_fraction(self, service, repo):
        """compound_risk_score in DB is 0–1 (score/100)."""
        inp = CompoundRiskInput(accident_probability=0.5)
        result = service.compute(inp)
        model = await service.compute_and_persist(inp)
        assert model.compound_risk_score == pytest.approx(
            result.compound_risk_score / 100.0, abs=0.01,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Query helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestQueryHelpers:
    async def test_get_latest(self, service):
        inp1 = CompoundRiskInput(accident_probability=0.3, zone_id="ZONE_A")
        inp2 = CompoundRiskInput(accident_probability=0.8, zone_id="ZONE_A")
        await service.compute_and_persist(inp1)
        m2 = await service.compute_and_persist(inp2)
        latest = await service.get_latest(zone_id="ZONE_A")
        assert latest is not None
        assert latest.id == m2.id

    async def test_get_history(self, service):
        for i in range(3):
            inp = CompoundRiskInput(
                accident_probability=0.1 * (i + 1),
                zone_id="ZONE_B",
            )
            await service.compute_and_persist(inp)
        history = await service.get_history(zone_id="ZONE_B")
        assert len(history) == 3

    async def test_count(self, service):
        for _ in range(4):
            await service.compute_and_persist(
                CompoundRiskInput(accident_probability=0.5),
            )
        assert await service.count() == 4


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Edge cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEdgeCases:
    def test_result_is_dataclass(self, service):
        result = service.compute(CompoundRiskInput())
        assert isinstance(result, CompoundRiskResult)

    def test_risk_level_is_enum(self, service):
        result = service.compute(CompoundRiskInput())
        assert isinstance(result.risk_level, RiskLevel)

    def test_half_inputs_medium_ish(self, service):
        inp = CompoundRiskInput(
            accident_probability=0.5,
            isolation_forest_score=0.5,
            autoencoder_score=0.5,
            sensor_health_score=50.0,
        )
        result = service.compute(inp)
        assert 25 <= result.compound_risk_score <= 60

    def test_alerts_only(self, service):
        inp = CompoundRiskInput(
            active_alert_count=5, alert_severity_max=0.9,
        )
        result = service.compute(inp)
        assert result.compound_risk_score > 0
        assert result.risk_level in list(RiskLevel)

    def test_properties_accessible(self, service):
        assert isinstance(service.weights, CompoundRiskWeights)
        assert isinstance(service.thresholds, RiskLevelThresholds)
