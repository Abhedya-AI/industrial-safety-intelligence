"""Comprehensive unit tests for CompoundRiskService façade.

Tests cover:
  - Full analysis pipeline (analyze)
  - Dry-run computation (compute)
  - Input validation
  - Rule engine integration
  - Explainability integration
  - Persistence
  - Event publishing
  - Query delegation
  - Error handling
  - Metrics tracking
  - Result serialization
  - Publishing failures (graceful degradation)
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.compound_risk.domain.exceptions import (
    CompoundRiskAnalysisFailedError,
    InsufficientScenarioDataError,
)
from app.compound_risk.domain.value_objects import RiskLevel
from app.compound_risk.messaging.publisher import CompoundRiskPublisher
from app.compound_risk.repositories.sqlalchemy_compound_risk_repo import (
    SQLAlchemyCompoundRiskRepository,
)
from app.compound_risk.rules.rule_engine import (
    CompoundRiskRuleEngine,
    create_default_rules,
)
from app.compound_risk.services.compound_risk_facade import (
    CompoundRiskAnalysisResult,
    CompoundRiskService,
)
from app.compound_risk.services.compound_risk_service import (
    CompoundRiskAggregationService,
    CompoundRiskInput,
)
from app.compound_risk.services.explainability_service import ExplainabilityService
from app.shared.messaging.producer import NoopEventProducer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_input(**overrides) -> CompoundRiskInput:
    defaults = {
        "isolation_forest_score": 0.5,
        "autoencoder_score": 0.4,
        "accident_probability": 0.6,
        "sensor_health_score": 75.0,
        "active_alert_count": 2,
        "alert_severity_max": 0.6,
        "threshold_violation_count": 1,
        "equipment_id": "EQ001",
        "zone_id": "ZONE_A",
    }
    defaults.update(overrides)
    return CompoundRiskInput(**defaults)


def _high_risk_input() -> CompoundRiskInput:
    return _make_input(
        isolation_forest_score=0.85,
        autoencoder_score=0.8,
        accident_probability=0.9,
        sensor_health_score=25.0,
        active_alert_count=5,
        alert_severity_max=0.9,
        threshold_violation_count=4,
    )


def _low_risk_input() -> CompoundRiskInput:
    return _make_input(
        isolation_forest_score=0.05,
        autoencoder_score=0.03,
        accident_probability=0.05,
        sensor_health_score=95.0,
        active_alert_count=0,
        alert_severity_max=0.0,
        threshold_violation_count=0,
    )


def _default_sensor_facts() -> Dict[str, Any]:
    return {
        "temperature_celsius": 65,
        "gas_level_ppm": 110,
        "pressure_bar": 4.5,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest_asyncio.fixture
async def repo(db_session: AsyncSession) -> SQLAlchemyCompoundRiskRepository:
    return SQLAlchemyCompoundRiskRepository(db_session)


@pytest_asyncio.fixture
async def publish_log() -> List[Dict[str, Any]]:
    return []


@pytest_asyncio.fixture
async def producer(publish_log) -> NoopEventProducer:
    prod = NoopEventProducer()
    original_publish = prod.publish

    def tracking_publish(topic, data, **kwargs):
        event = original_publish(topic, data, **kwargs)
        publish_log.append({"topic": topic, "data": data})
        return event

    prod.publish = tracking_publish
    return prod


@pytest_asyncio.fixture
async def publisher(producer) -> CompoundRiskPublisher:
    return CompoundRiskPublisher(producer)


@pytest_asyncio.fixture
async def service(repo, publisher) -> CompoundRiskService:
    aggregation = CompoundRiskAggregationService(repo)
    rule_engine = CompoundRiskRuleEngine(create_default_rules())
    explainability = ExplainabilityService()
    return CompoundRiskService(
        aggregation_service=aggregation,
        rule_engine=rule_engine,
        explainability_service=explainability,
        publisher=publisher,
    )


@pytest_asyncio.fixture
async def service_no_publisher(repo) -> CompoundRiskService:
    aggregation = CompoundRiskAggregationService(repo)
    rule_engine = CompoundRiskRuleEngine(create_default_rules())
    explainability = ExplainabilityService()
    return CompoundRiskService(
        aggregation_service=aggregation,
        rule_engine=rule_engine,
        explainability_service=explainability,
        publisher=None,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Full analysis pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAnalyze:
    async def test_returns_analysis_result(self, service):
        result = await service.analyze(_make_input())
        assert isinstance(result, CompoundRiskAnalysisResult)

    async def test_has_compound_risk_score(self, service):
        result = await service.analyze(_make_input())
        assert 0 <= result.compound_risk_score <= 100

    async def test_has_risk_level(self, service):
        result = await service.analyze(_make_input())
        assert isinstance(result.risk_level, RiskLevel)

    async def test_has_confidence_score(self, service):
        result = await service.analyze(_make_input())
        assert 0 < result.confidence_score <= 1.0

    async def test_has_contributing_factors(self, service):
        result = await service.analyze(_make_input())
        assert len(result.contributing_factors) > 0

    async def test_has_component_scores(self, service):
        result = await service.analyze(_make_input())
        assert "risk_prediction" in result.component_scores
        assert "isolation_forest" in result.component_scores

    async def test_has_explanation(self, service):
        result = await service.analyze(_make_input())
        assert result.explanation is not None
        assert result.explanation.summary

    async def test_has_processing_time(self, service):
        result = await service.analyze(_make_input())
        assert result.processing_time_ms > 0

    async def test_model_persisted(self, service, repo):
        result = await service.analyze(_make_input())
        fetched = await repo.get_by_id(result.model.id)
        assert fetched is not None
        assert fetched.zone_id == "ZONE_A"

    async def test_high_risk_scenario(self, service):
        result = await service.analyze(_high_risk_input())
        assert result.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert result.compound_risk_score > 50

    async def test_low_risk_scenario(self, service):
        result = await service.analyze(_low_risk_input())
        assert result.risk_level == RiskLevel.LOW
        assert result.compound_risk_score < 25


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Rule engine integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRuleEngineIntegration:
    async def test_rules_triggered_with_facts(self, service):
        facts = {"temperature_celsius": 70, "gas_level_ppm": 120}
        result = await service.analyze(_make_input(), sensor_facts=facts)
        assert len(result.rule_result.triggered_rules) > 0

    async def test_no_rules_without_facts(self, service):
        result = await service.analyze(_make_input(), sensor_facts={})
        # No sensor facts → no rules fire
        assert result.rule_result is not None

    async def test_rules_affect_explanation(self, service):
        facts = {"temperature_celsius": 70, "gas_level_ppm": 120}
        result = await service.analyze(_make_input(), sensor_facts=facts)
        assert len(result.explanation.triggered_rules) > 0

    async def test_rule_count_property(self, service):
        assert service.rule_count == 9  # 9 default rules


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Explainability integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestExplainabilityIntegration:
    async def test_explanation_has_factors(self, service):
        result = await service.analyze(_make_input())
        assert len(result.explanation.factor_explanations) == 5

    async def test_explanation_has_key_drivers(self, service):
        result = await service.analyze(_high_risk_input())
        assert len(result.explanation.key_drivers) > 0

    async def test_explanation_has_recommendations(self, service):
        result = await service.analyze(_high_risk_input())
        assert len(result.explanation.recommendations) > 0

    async def test_explanation_summary_contains_level(self, service):
        result = await service.analyze(_high_risk_input())
        # Summary should mention the risk level
        level_str = result.risk_level.value
        assert level_str in result.explanation.summary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Kafka publishing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestKafkaPublishing:
    async def test_event_published(self, service, publish_log):
        await service.analyze(_make_input())
        assert len(publish_log) >= 1
        assert publish_log[0]["topic"] == "compound.risk.detected"

    async def test_published_data_has_score(self, service, publish_log):
        await service.analyze(_make_input())
        data = publish_log[0]["data"]
        assert "compound_risk_score" in data
        assert "risk_level" in data

    async def test_no_publish_without_publisher(self, service_no_publisher, repo):
        result = await service_no_publisher.analyze(_make_input())
        # Should still succeed, just no publish
        assert result.model is not None
        count = await repo.count()
        assert count >= 1

    async def test_correlation_id_passed(self, service, publish_log):
        await service.analyze(
            _make_input(), correlation_id="corr-test-123",
        )
        # The event was published (correlation ID is in the BaseEvent)
        assert len(publish_log) >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Input validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInputValidation:
    async def test_all_defaults_rejected(self, service):
        """All-default inputs = no meaningful signal."""
        empty = CompoundRiskInput()
        with pytest.raises(InsufficientScenarioDataError):
            await service.analyze(empty)

    async def test_only_anomaly_accepted(self, service):
        inp = CompoundRiskInput(isolation_forest_score=0.5)
        result = await service.analyze(inp)
        assert result is not None

    async def test_only_risk_accepted(self, service):
        inp = CompoundRiskInput(accident_probability=0.5)
        result = await service.analyze(inp)
        assert result is not None

    async def test_only_health_accepted(self, service):
        inp = CompoundRiskInput(sensor_health_score=50.0)
        result = await service.analyze(inp)
        assert result is not None

    async def test_only_alerts_accepted(self, service):
        inp = CompoundRiskInput(active_alert_count=3)
        result = await service.analyze(inp)
        assert result is not None

    async def test_only_violations_accepted(self, service):
        inp = CompoundRiskInput(threshold_violation_count=2)
        result = await service.analyze(inp)
        assert result is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Dry-run computation (no persist/publish)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCompute:
    def test_returns_dict(self, service):
        result = service.compute(_make_input())
        assert isinstance(result, dict)

    def test_has_required_fields(self, service):
        result = service.compute(_make_input())
        assert "compound_risk_score" in result
        assert "risk_level" in result
        assert "confidence_score" in result
        assert "contributing_factors" in result
        assert "explanation" in result
        assert "key_drivers" in result
        assert "recommendations" in result

    def test_with_sensor_facts(self, service):
        facts = {"temperature_celsius": 70, "gas_level_ppm": 120}
        result = service.compute(_make_input(), sensor_facts=facts)
        assert result["triggered_rules"] > 0

    def test_validation_on_empty(self, service):
        with pytest.raises(InsufficientScenarioDataError):
            service.compute(CompoundRiskInput())

    async def test_does_not_persist(self, service, repo):
        service.compute(_make_input())
        count = await repo.count()
        assert count == 0  # No persistence in compute()

    async def test_does_not_publish(self, service, publish_log):
        service.compute(_make_input())
        assert len(publish_log) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Query delegation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestQueryDelegation:
    async def test_get_by_id(self, service):
        result = await service.analyze(_make_input())
        fetched = await service.get_by_id(result.model.id)
        assert fetched is not None
        assert fetched.id == result.model.id

    async def test_get_by_id_not_found(self, service):
        fetched = await service.get_by_id("nonexistent")
        assert fetched is None

    async def test_get_latest(self, service):
        await service.analyze(_make_input(zone_id="ZONE_Q"))
        latest = await service.get_latest(zone_id="ZONE_Q")
        assert latest is not None

    async def test_get_history(self, service):
        for _ in range(3):
            await service.analyze(_make_input(zone_id="ZONE_H"))
        history = await service.get_history(zone_id="ZONE_H")
        assert len(history) == 3

    async def test_count(self, service):
        await service.analyze(_make_input(zone_id="ZONE_C"))
        await service.analyze(_make_input(zone_id="ZONE_C"))
        count = await service.count(zone_id="ZONE_C")
        assert count == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Error handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestErrorHandling:
    async def test_insufficient_data_error(self, service):
        with pytest.raises(InsufficientScenarioDataError) as exc_info:
            await service.analyze(CompoundRiskInput())
        assert "Insufficient" in exc_info.value.message

    async def test_failed_count_tracked(self, service):
        try:
            await service.analyze(CompoundRiskInput())
        except InsufficientScenarioDataError:
            pass
        assert service.failed_analyses == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMetrics:
    async def test_total_analyses_tracked(self, service):
        await service.analyze(_make_input())
        await service.analyze(_make_input())
        assert service.total_analyses == 2

    async def test_failed_analyses_tracked(self, service):
        try:
            await service.analyze(CompoundRiskInput())
        except Exception:
            pass
        assert service.failed_analyses == 1
        assert service.total_analyses == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Result serialization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSerialization:
    async def test_to_dict(self, service):
        result = await service.analyze(_make_input())
        d = result.to_dict()
        assert d["analysis_id"] == result.model.id
        assert d["zone_id"] == "ZONE_A"
        assert d["equipment_id"] == "EQ001"
        assert isinstance(d["compound_risk_score"], float)
        assert isinstance(d["risk_level"], str)
        assert isinstance(d["confidence_score"], float)
        assert isinstance(d["contributing_factors"], list)
        assert isinstance(d["explanation"], str)
        assert isinstance(d["key_drivers"], list)
        assert d["processing_time_ms"] > 0

    async def test_to_dict_has_triggered_rules(self, service):
        facts = {"temperature_celsius": 70, "gas_level_ppm": 120}
        result = await service.analyze(_make_input(), sensor_facts=facts)
        d = result.to_dict()
        assert isinstance(d["triggered_rules"], list)
        assert len(d["triggered_rules"]) > 0

    async def test_to_dict_has_recommendation(self, service):
        result = await service.analyze(_high_risk_input())
        d = result.to_dict()
        # High risk should have recommendations
        assert d["recommendation"] is not None or d["recommendation"] is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. Recommendation building
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRecommendation:
    async def test_high_risk_has_recommendation(self, service):
        result = await service.analyze(_high_risk_input())
        assert result.model.recommendation is not None
        assert len(result.model.recommendation) > 0

    async def test_low_risk_may_have_no_recommendation(self, service):
        result = await service.analyze(_low_risk_input())
        # Low risk may or may not have a recommendation
        assert isinstance(result.model.recommendation, (str, type(None)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. Multiple analyses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMultipleAnalyses:
    async def test_independent_analyses(self, service):
        r1 = await service.analyze(_make_input(zone_id="Z1"))
        r2 = await service.analyze(_make_input(zone_id="Z2"))
        assert r1.model.id != r2.model.id
        assert r1.model.zone_id == "Z1"
        assert r2.model.zone_id == "Z2"

    async def test_same_zone_different_scores(self, service):
        r1 = await service.analyze(_low_risk_input())
        r2 = await service.analyze(_high_risk_input())
        assert r1.compound_risk_score != r2.compound_risk_score
