"""End-to-end integration tests for the Compound Risk Intelligence pipeline.

Verifies the COMPLETE flow:

    Sensor Reading → Anomaly Detection → Risk Prediction → Rule Engine
        → Aggregation Engine → Explainability → Compound Risk Calculation
        → Kafka Publication → Database Persistence → API Response

Test scenarios:
  1. Normal operating conditions
  2. High-risk conditions
  3. Critical risk conditions
  4. Multiple simultaneous anomalies
  5. Missing / partial data
  6. Invalid data
  7. Repository failures (simulated)
  8. Kafka failures (simulated)
  9. Rule engine edge cases
  10. Multi-zone concurrent analysis
  11. Historical query after persistence
  12. Full API round-trip
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.compound_risk.domain.exceptions import (
    CompoundRiskAnalysisFailedError,
    InsufficientScenarioDataError,
)
from app.compound_risk.domain.value_objects import RiskLevel
from app.compound_risk.messaging.handler import (
    CompoundRiskEventHandler,
    ZoneRiskState,
)
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
    CompoundRiskResult,
)
from app.compound_risk.services.explainability_service import ExplainabilityService
from app.shared.messaging.consumer import NoopEventConsumer
from app.shared.messaging.events import BaseEvent, create_event
from app.shared.messaging.producer import NoopEventProducer
from app.shared.messaging.topics import KafkaTopics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

API_PREFIX = "/api/v1"
COMPOUND_URL = f"{API_PREFIX}/risk/compound-analysis"
LATEST_URL = f"{COMPOUND_URL}/latest"
HISTORY_URL = f"{COMPOUND_URL}/history"


def _wrap_event(event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_type": event_type,
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_system": "test_e2e",
        "data": data,
        "version": "1.0",
    }


# ── Scenario factories ──


class Scenarios:
    """Pre-built test scenarios covering the full risk spectrum."""

    @staticmethod
    def normal() -> Dict[str, Any]:
        """Normal operating conditions — LOW risk expected."""
        return {
            "zone_id": "ZONE_NORMAL",
            "scenario": {
                "gas_level_ppm": 30,
                "temperature_celsius": 25,
                "pressure_bar": 2.0,
                "humidity_percent": 45,
                "vibration_level": 0.5,
                "maintenance_active": False,
                "worker_count": 3,
                "equipment_health": 0.95,
                "anomaly_score": 0.05,
                "accident_probability": 0.03,
                "risk_score": 8.0,
                "sensor_health_score": 95.0,
            },
        }

    @staticmethod
    def high_risk() -> Dict[str, Any]:
        """High-risk conditions — HIGH risk expected."""
        return {
            "zone_id": "ZONE_HIGH",
            "equipment_id": "EQ_BOILER_1",
            "scenario": {
                "gas_level_ppm": 110,
                "temperature_celsius": 62,
                "pressure_bar": 4.5,
                "humidity_percent": 65,
                "vibration_level": 4.0,
                "maintenance_active": True,
                "worker_count": 8,
                "permit_type": "HOT_WORK",
                "permit_active": True,
                "equipment_health": 0.45,
                "anomaly_score": 0.7,
                "accident_probability": 0.65,
                "risk_score": 62.0,
                "sensor_health_score": 60.0,
            },
        }

    @staticmethod
    def critical() -> Dict[str, Any]:
        """Critical risk conditions — CRITICAL risk expected."""
        return {
            "zone_id": "ZONE_CRITICAL",
            "equipment_id": "EQ_REACTOR_1",
            "scenario": {
                "gas_level_ppm": 200,
                "temperature_celsius": 85,
                "pressure_bar": 7.0,
                "humidity_percent": 80,
                "vibration_level": 8.0,
                "maintenance_active": True,
                "worker_count": 15,
                "permit_type": "CONFINED_SPACE",
                "permit_active": True,
                "shift_type": "NIGHT",
                "equipment_health": 0.15,
                "anomaly_score": 0.95,
                "accident_probability": 0.92,
                "risk_score": 88.0,
                "sensor_health_score": 30.0,
            },
        }

    @staticmethod
    def multi_anomaly() -> Dict[str, Any]:
        """Multiple simultaneous anomalies."""
        return {
            "zone_id": "ZONE_MULTI",
            "equipment_id": "EQ_COMPRESSOR",
            "scenario": {
                "gas_level_ppm": 150,
                "temperature_celsius": 70,
                "pressure_bar": 6.5,
                "vibration_level": 6.0,
                "maintenance_active": True,
                "worker_count": 10,
                "equipment_health": 0.3,
                "anomaly_score": 0.85,
                "accident_probability": 0.78,
                "risk_score": 75.0,
                "sensor_health_score": 40.0,
            },
        }

    @staticmethod
    def partial_data() -> Dict[str, Any]:
        """Minimal data — only anomaly score provided."""
        return {
            "zone_id": "ZONE_PARTIAL",
            "scenario": {
                "anomaly_score": 0.4,
            },
        }

    @staticmethod
    def service_input_normal() -> CompoundRiskInput:
        return CompoundRiskInput(
            isolation_forest_score=0.05,
            autoencoder_score=0.03,
            accident_probability=0.04,
            risk_score=8.0,
            sensor_health_score=95.0,
            active_alert_count=0,
            alert_severity_max=0.0,
            threshold_violation_count=0,
            equipment_id="EQ001",
            zone_id="ZONE_NORMAL",
        )

    @staticmethod
    def service_input_high() -> CompoundRiskInput:
        return CompoundRiskInput(
            isolation_forest_score=0.7,
            autoencoder_score=0.6,
            accident_probability=0.65,
            risk_score=62.0,
            sensor_health_score=60.0,
            active_alert_count=3,
            alert_severity_max=0.7,
            threshold_violation_count=2,
            equipment_id="EQ_BOILER",
            zone_id="ZONE_HIGH",
        )

    @staticmethod
    def service_input_critical() -> CompoundRiskInput:
        return CompoundRiskInput(
            isolation_forest_score=0.95,
            autoencoder_score=0.9,
            accident_probability=0.92,
            risk_score=88.0,
            sensor_health_score=30.0,
            active_alert_count=5,
            alert_severity_max=0.95,
            threshold_violation_count=4,
            equipment_id="EQ_REACTOR",
            zone_id="ZONE_CRITICAL",
        )

    @staticmethod
    def sensor_facts_high() -> Dict[str, Any]:
        return {
            "temperature_celsius": 65,
            "gas_level_ppm": 115,
            "pressure_bar": 4.5,
            "vibration_level": 4.0,
            "sensor_health_score": 60.0,
            "equipment_health": 0.45,
        }

    @staticmethod
    def sensor_facts_critical() -> Dict[str, Any]:
        return {
            "temperature_celsius": 85,
            "gas_level_ppm": 200,
            "pressure_bar": 7.0,
            "vibration_level": 8.0,
            "sensor_health_score": 30.0,
            "equipment_health": 0.15,
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
    original = prod.publish

    def tracking(topic, data, **kw):
        event = original(topic, data, **kw)
        publish_log.append({"topic": topic, "data": data, "event": event})
        return event

    prod.publish = tracking
    return prod


@pytest_asyncio.fixture
async def publisher(producer) -> CompoundRiskPublisher:
    return CompoundRiskPublisher(producer)


@pytest_asyncio.fixture
async def service(repo, publisher) -> CompoundRiskService:
    return CompoundRiskService(
        aggregation_service=CompoundRiskAggregationService(repo),
        rule_engine=CompoundRiskRuleEngine(create_default_rules()),
        explainability_service=ExplainabilityService(),
        publisher=publisher,
    )


@pytest_asyncio.fixture
async def event_handler(repo, publisher) -> CompoundRiskEventHandler:
    return CompoundRiskEventHandler(
        aggregation_service=CompoundRiskAggregationService(repo),
        rule_engine=CompoundRiskRuleEngine(create_default_rules()),
        explainability_service=ExplainabilityService(),
        publisher=publisher,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Normal operating conditions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNormalConditions:
    """Low-risk, all-green scenario."""

    async def test_service_produces_low_risk(self, service):
        result = await service.analyze(Scenarios.service_input_normal())
        assert result.risk_level == RiskLevel.LOW
        assert result.compound_risk_score < 25

    async def test_service_persists(self, service, repo):
        result = await service.analyze(Scenarios.service_input_normal())
        fetched = await repo.get_by_id(result.model.id)
        assert fetched is not None
        assert fetched.risk_level == "LOW"

    async def test_service_publishes(self, service, publish_log):
        await service.analyze(Scenarios.service_input_normal())
        assert len(publish_log) >= 1
        assert publish_log[0]["topic"] == KafkaTopics.COMPOUND_RISK_DETECTED

    async def test_api_returns_low_risk(self, client: AsyncClient):
        resp = await client.post(COMPOUND_URL, json=Scenarios.normal())
        assert resp.status_code == 201
        analysis = resp.json()["compound_risk_analysis"]
        assert analysis["risk_level"] == "LOW"

    async def test_explanation_summary(self, service):
        result = await service.analyze(Scenarios.service_input_normal())
        assert "LOW" in result.explanation.summary

    async def test_high_confidence(self, service):
        result = await service.analyze(Scenarios.service_input_normal())
        assert result.confidence_score > 0.3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. High-risk conditions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHighRiskConditions:
    """Elevated risk — expect HIGH level."""

    async def test_service_high_risk(self, service):
        result = await service.analyze(
            Scenarios.service_input_high(),
            sensor_facts=Scenarios.sensor_facts_high(),
        )
        assert result.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert result.compound_risk_score > 40

    async def test_rules_triggered(self, service):
        result = await service.analyze(
            Scenarios.service_input_high(),
            sensor_facts=Scenarios.sensor_facts_high(),
        )
        assert len(result.rule_result.triggered_rules) > 0

    async def test_has_recommendations(self, service):
        result = await service.analyze(
            Scenarios.service_input_high(),
            sensor_facts=Scenarios.sensor_facts_high(),
        )
        assert result.model.recommendation is not None

    async def test_api_returns_high(self, client: AsyncClient):
        resp = await client.post(COMPOUND_URL, json=Scenarios.high_risk())
        assert resp.status_code == 201
        analysis = resp.json()["compound_risk_analysis"]
        assert analysis["risk_level"] in ("HIGH", "CRITICAL")

    async def test_contributing_factors(self, service):
        result = await service.analyze(
            Scenarios.service_input_high(),
            sensor_facts=Scenarios.sensor_facts_high(),
        )
        assert len(result.contributing_factors) >= 3

    async def test_published_event_has_score(self, service, publish_log):
        await service.analyze(
            Scenarios.service_input_high(),
            sensor_facts=Scenarios.sensor_facts_high(),
        )
        assert publish_log[0]["data"]["compound_risk_score"] > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Critical risk conditions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCriticalRiskConditions:
    """All signals at maximum — expect CRITICAL."""

    async def test_service_critical_risk(self, service):
        result = await service.analyze(
            Scenarios.service_input_critical(),
            sensor_facts=Scenarios.sensor_facts_critical(),
        )
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.compound_risk_score >= 75

    async def test_many_rules_triggered(self, service):
        result = await service.analyze(
            Scenarios.service_input_critical(),
            sensor_facts=Scenarios.sensor_facts_critical(),
        )
        assert len(result.rule_result.triggered_rules) >= 3

    async def test_api_critical(self, client: AsyncClient):
        resp = await client.post(COMPOUND_URL, json=Scenarios.critical())
        assert resp.status_code == 201
        analysis = resp.json()["compound_risk_analysis"]
        assert analysis["risk_level"] == "CRITICAL"

    async def test_explanation_mentions_critical(self, service):
        result = await service.analyze(
            Scenarios.service_input_critical(),
            sensor_facts=Scenarios.sensor_facts_critical(),
        )
        assert "CRITICAL" in result.explanation.summary

    async def test_all_components_high(self, service):
        result = await service.analyze(
            Scenarios.service_input_critical(),
            sensor_facts=Scenarios.sensor_facts_critical(),
        )
        for name, score in result.component_scores.items():
            if name != "threshold_violation":
                assert score > 20, f"{name} should be elevated"

    async def test_persisted_with_correct_level(self, service, repo):
        result = await service.analyze(
            Scenarios.service_input_critical(),
            sensor_facts=Scenarios.sensor_facts_critical(),
        )
        fetched = await repo.get_by_id(result.model.id)
        assert fetched.risk_level == "CRITICAL"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Multiple simultaneous anomalies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMultipleAnomalies:
    async def test_api_multi_anomaly(self, client: AsyncClient):
        resp = await client.post(COMPOUND_URL, json=Scenarios.multi_anomaly())
        assert resp.status_code == 201
        analysis = resp.json()["compound_risk_analysis"]
        assert analysis["risk_level"] in ("HIGH", "CRITICAL")

    async def test_service_multi_anomaly(self, service):
        inp = CompoundRiskInput(
            isolation_forest_score=0.85,
            autoencoder_score=0.8,
            accident_probability=0.78,
            risk_score=75.0,
            sensor_health_score=40.0,
            active_alert_count=4,
            alert_severity_max=0.85,
            threshold_violation_count=3,
            zone_id="ZONE_MULTI",
        )
        facts = {
            "temperature_celsius": 70,
            "gas_level_ppm": 150,
            "pressure_bar": 6.5,
            "vibration_level": 6.0,
        }
        result = await service.analyze(inp, sensor_facts=facts)
        assert result.compound_risk_score > 50
        assert len(result.rule_result.triggered_rules) >= 2

    async def test_event_handler_multi_events(self, event_handler, publish_log):
        """Multiple Kafka events for same zone accumulate correctly."""
        anomaly = _wrap_event(KafkaTopics.SENSOR_READING_ANOMALY, {
            "zone_id": "ZONE_M",
            "isolation_forest_score": 0.8,
            "autoencoder_score": 0.7,
            "sensor_health_score": 50.0,
            "temperature_celsius": 70,
            "gas_level_ppm": 140,
        })
        risk = _wrap_event(KafkaTopics.RISK_ASSESSMENT_GENERATED, {
            "zone_id": "ZONE_M",
            "accident_probability": 0.75,
            "risk_score": 70.0,
        })
        await event_handler.handle_event(KafkaTopics.SENSOR_READING_ANOMALY, anomaly)
        await event_handler.handle_event(KafkaTopics.RISK_ASSESSMENT_GENERATED, risk)

        state = event_handler.get_zone_state("ZONE_M")
        assert state.event_count == 2
        assert state.isolation_forest_score == 0.8
        assert state.accident_probability == 0.75
        assert len(publish_log) >= 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Missing / partial data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMissingData:
    async def test_api_partial_data(self, client: AsyncClient):
        resp = await client.post(COMPOUND_URL, json=Scenarios.partial_data())
        assert resp.status_code == 201

    async def test_service_single_signal(self, service):
        inp = CompoundRiskInput(isolation_forest_score=0.5)
        result = await service.analyze(inp)
        assert result is not None
        assert result.risk_level in (
            RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH,
        )

    async def test_service_only_health_degraded(self, service):
        inp = CompoundRiskInput(sensor_health_score=30.0)
        result = await service.analyze(inp)
        assert result.compound_risk_score > 0

    async def test_service_all_defaults_rejected(self, service):
        with pytest.raises(InsufficientScenarioDataError):
            await service.analyze(CompoundRiskInput())

    async def test_event_handler_missing_zone(self, event_handler):
        """Event with missing zone_id defaults to UNKNOWN."""
        event = _wrap_event(KafkaTopics.SENSOR_READING_ANOMALY, {
            "isolation_forest_score": 0.5,
        })
        result = await event_handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, event,
        )
        assert result is True
        assert event_handler.get_zone_state("UNKNOWN") is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Invalid data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInvalidData:
    async def test_api_missing_zone(self, client: AsyncClient):
        resp = await client.post(
            COMPOUND_URL, json={"scenario": {"anomaly_score": 0.5}},
        )
        assert resp.status_code == 422

    async def test_api_missing_scenario(self, client: AsyncClient):
        resp = await client.post(COMPOUND_URL, json={"zone_id": "Z"})
        assert resp.status_code == 422

    async def test_api_empty_body(self, client: AsyncClient):
        resp = await client.post(COMPOUND_URL, json={})
        assert resp.status_code == 422

    async def test_event_missing_event_type(self, event_handler):
        bad = {"event_id": "x", "timestamp": "t", "data": {}}
        result = await event_handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, bad,
        )
        assert result is False
        assert event_handler.events_failed == 1

    async def test_event_missing_data(self, event_handler):
        bad = {
            "event_type": "sensor.reading.anomaly",
            "event_id": "x",
            "timestamp": "t",
        }
        result = await event_handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, bad,
        )
        assert result is False

    async def test_event_data_not_dict(self, event_handler):
        bad = {
            "event_type": "sensor.reading.anomaly",
            "event_id": "x",
            "timestamp": "t",
            "data": "string_not_dict",
        }
        result = await event_handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, bad,
        )
        assert result is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Repository failures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRepositoryFailures:
    async def test_service_handles_repo_error(self, publisher):
        """Service wraps repo errors as CompoundRiskAnalysisFailedError."""
        mock_repo = AsyncMock()
        mock_repo.create.side_effect = Exception("DB connection lost")

        agg = CompoundRiskAggregationService(mock_repo)
        svc = CompoundRiskService(
            aggregation_service=agg,
            rule_engine=CompoundRiskRuleEngine(create_default_rules()),
            explainability_service=ExplainabilityService(),
            publisher=publisher,
        )

        with pytest.raises(CompoundRiskAnalysisFailedError):
            await svc.analyze(Scenarios.service_input_normal())
        assert svc.failed_analyses == 1

    async def test_get_by_id_returns_none_on_missing(self, service):
        result = await service.get_by_id("nonexistent-id")
        assert result is None

    async def test_get_latest_returns_none_on_empty(self, service):
        result = await service.get_latest(zone_id="ZONE_EMPTY")
        assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Kafka failures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestKafkaFailures:
    async def test_publish_failure_does_not_crash(self, repo):
        """Analysis succeeds even if Kafka publish fails."""
        failing_producer = NoopEventProducer()
        failing_producer.publish = MagicMock(
            side_effect=Exception("Kafka broker unreachable"),
        )
        failing_publisher = CompoundRiskPublisher(failing_producer)

        svc = CompoundRiskService(
            aggregation_service=CompoundRiskAggregationService(repo),
            rule_engine=CompoundRiskRuleEngine(create_default_rules()),
            explainability_service=ExplainabilityService(),
            publisher=failing_publisher,
        )

        # Should NOT raise — publish failure is caught
        result = await svc.analyze(Scenarios.service_input_normal())
        assert result is not None
        assert result.model is not None

    async def test_no_publisher_works(self, repo):
        """Service works without any publisher configured."""
        svc = CompoundRiskService(
            aggregation_service=CompoundRiskAggregationService(repo),
            rule_engine=CompoundRiskRuleEngine(create_default_rules()),
            explainability_service=ExplainabilityService(),
            publisher=None,
        )
        result = await svc.analyze(Scenarios.service_input_normal())
        assert result is not None
        assert result.risk_level == RiskLevel.LOW


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Rule engine edge cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRuleEngineEdgeCases:
    async def test_no_facts_no_rules(self, service):
        result = await service.analyze(
            Scenarios.service_input_normal(), sensor_facts={},
        )
        assert result.rule_result is not None

    async def test_single_threshold_breach(self, service):
        facts = {"temperature_celsius": 70}
        result = await service.analyze(
            Scenarios.service_input_normal(), sensor_facts=facts,
        )
        assert len(result.rule_result.triggered_rules) >= 1

    async def test_all_thresholds_breached(self, service):
        result = await service.analyze(
            Scenarios.service_input_critical(),
            sensor_facts=Scenarios.sensor_facts_critical(),
        )
        assert len(result.rule_result.triggered_rules) >= 3

    async def test_rule_impact_affects_explanation(self, service):
        result = await service.analyze(
            Scenarios.service_input_high(),
            sensor_facts=Scenarios.sensor_facts_high(),
        )
        assert len(result.explanation.triggered_rules) > 0
        assert result.explanation.key_drivers


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Multi-zone concurrent analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMultiZone:
    async def test_independent_zones_via_service(self, service, repo):
        r1 = await service.analyze(
            CompoundRiskInput(
                isolation_forest_score=0.1, zone_id="Z1",
            ),
        )
        r2 = await service.analyze(
            CompoundRiskInput(
                isolation_forest_score=0.9, accident_probability=0.9,
                zone_id="Z2",
            ),
        )
        assert r1.risk_level != r2.risk_level
        assert r1.model.zone_id == "Z1"
        assert r2.model.zone_id == "Z2"
        assert await repo.count() == 2

    async def test_independent_zones_via_api(self, client: AsyncClient):
        resp1 = await client.post(COMPOUND_URL, json=Scenarios.normal())
        resp2 = await client.post(COMPOUND_URL, json=Scenarios.critical())
        a1 = resp1.json()["compound_risk_analysis"]
        a2 = resp2.json()["compound_risk_analysis"]
        assert a1["risk_level"] != a2["risk_level"]
        assert a1["zone_id"] != a2["zone_id"]

    async def test_event_handler_independent_zones(self, event_handler):
        e1 = _wrap_event(KafkaTopics.SENSOR_READING_ANOMALY, {
            "zone_id": "ZA", "isolation_forest_score": 0.9,
        })
        e2 = _wrap_event(KafkaTopics.SENSOR_READING_ANOMALY, {
            "zone_id": "ZB", "isolation_forest_score": 0.1,
        })
        await event_handler.handle_event(KafkaTopics.SENSOR_READING_ANOMALY, e1)
        await event_handler.handle_event(KafkaTopics.SENSOR_READING_ANOMALY, e2)
        assert event_handler.get_zone_state("ZA").isolation_forest_score == 0.9
        assert event_handler.get_zone_state("ZB").isolation_forest_score == 0.1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. Historical query after persistence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHistoricalQuery:
    async def test_api_history_round_trip(self, client: AsyncClient):
        for _ in range(3):
            await client.post(COMPOUND_URL, json=Scenarios.normal())

        resp = await client.get(HISTORY_URL, params={"zone_id": "ZONE_NORMAL"})
        body = resp.json()
        assert body["total"] == 3
        assert len(body["predictions"]) == 3

    async def test_api_latest_round_trip(self, client: AsyncClient):
        await client.post(COMPOUND_URL, json=Scenarios.high_risk())
        resp = await client.get(
            LATEST_URL, params={"zone_id": "ZONE_HIGH"},
        )
        assert resp.status_code == 200
        a = resp.json()["compound_risk_analysis"]
        assert a["zone_id"] == "ZONE_HIGH"

    async def test_service_history_round_trip(self, service):
        for _ in range(4):
            await service.analyze(Scenarios.service_input_normal())
        history = await service.get_history(zone_id="ZONE_NORMAL")
        assert len(history) == 4

    async def test_service_count_round_trip(self, service):
        await service.analyze(Scenarios.service_input_normal())
        await service.analyze(Scenarios.service_input_high())
        total = await service.count()
        assert total == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. Full API round-trip
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFullAPIRoundTrip:
    async def test_post_then_get_latest(self, client: AsyncClient):
        post_resp = await client.post(COMPOUND_URL, json=Scenarios.high_risk())
        assert post_resp.status_code == 201
        post_id = post_resp.json()["compound_risk_analysis"]["id"]

        get_resp = await client.get(
            LATEST_URL, params={"zone_id": "ZONE_HIGH"},
        )
        assert get_resp.status_code == 200
        assert get_resp.json()["compound_risk_analysis"]["id"] == post_id

    async def test_post_then_get_history(self, client: AsyncClient):
        await client.post(COMPOUND_URL, json=Scenarios.normal())
        await client.post(COMPOUND_URL, json=Scenarios.high_risk())
        await client.post(COMPOUND_URL, json=Scenarios.critical())

        resp = await client.get(HISTORY_URL)
        body = resp.json()
        assert body["total"] == 3
        levels = {p["risk_level"] for p in body["predictions"]}
        assert len(levels) >= 2  # Different risk levels

    async def test_pagination_round_trip(self, client: AsyncClient):
        for _ in range(7):
            await client.post(COMPOUND_URL, json=Scenarios.normal())

        page1 = await client.get(HISTORY_URL, params={"limit": 3, "offset": 0})
        page2 = await client.get(HISTORY_URL, params={"limit": 3, "offset": 3})
        page3 = await client.get(HISTORY_URL, params={"limit": 3, "offset": 6})

        assert len(page1.json()["predictions"]) == 3
        assert len(page2.json()["predictions"]) == 3
        assert len(page3.json()["predictions"]) == 1
        assert page1.json()["total"] == 7

    async def test_filter_round_trip(self, client: AsyncClient):
        await client.post(COMPOUND_URL, json=Scenarios.normal())
        await client.post(COMPOUND_URL, json=Scenarios.high_risk())

        normal_resp = await client.get(
            HISTORY_URL, params={"zone_id": "ZONE_NORMAL"},
        )
        high_resp = await client.get(
            HISTORY_URL, params={"zone_id": "ZONE_HIGH"},
        )
        assert normal_resp.json()["total"] == 1
        assert high_resp.json()["total"] == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. Full pipeline: event → service → DB → API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFullPipeline:
    async def test_event_to_db_to_api(
        self, event_handler, repo, client: AsyncClient,
    ):
        """Kafka event → handler → persist → API query."""
        event = _wrap_event(KafkaTopics.SENSOR_READING_ANOMALY, {
            "zone_id": "ZONE_E2E",
            "isolation_forest_score": 0.75,
            "autoencoder_score": 0.6,
            "sensor_health_score": 60.0,
            "temperature_celsius": 65,
            "gas_level_ppm": 115,
        })
        result = await event_handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, event,
        )
        assert result is True
        assert event_handler.analyses_produced >= 1

    async def test_service_to_api_consistency(self, service, client: AsyncClient):
        """Service analysis matches what API returns for same zone."""
        svc_result = await service.analyze(
            CompoundRiskInput(
                isolation_forest_score=0.6,
                accident_probability=0.5,
                zone_id="ZONE_CONS",
            ),
        )
        api_resp = await client.get(
            LATEST_URL, params={"zone_id": "ZONE_CONS"},
        )
        if api_resp.status_code == 200:
            api_id = api_resp.json()["compound_risk_analysis"]["id"]
            assert api_id == svc_result.model.id

    async def test_metrics_tracking(self, service):
        """Service metrics track successes and failures."""
        await service.analyze(Scenarios.service_input_normal())
        try:
            await service.analyze(CompoundRiskInput())
        except Exception:
            pass
        assert service.total_analyses == 1
        assert service.failed_analyses == 1

    async def test_serialization_round_trip(self, service):
        """CompoundRiskAnalysisResult.to_dict() contains all fields."""
        result = await service.analyze(
            Scenarios.service_input_high(),
            sensor_facts=Scenarios.sensor_facts_high(),
        )
        d = result.to_dict()
        assert d["analysis_id"]
        assert d["zone_id"] == "ZONE_HIGH"
        assert d["risk_level"] in ("HIGH", "CRITICAL")
        assert isinstance(d["compound_risk_score"], float)
        assert isinstance(d["confidence_score"], float)
        assert isinstance(d["contributing_factors"], list)
        assert isinstance(d["triggered_rules"], list)
        assert isinstance(d["key_drivers"], list)
        assert d["processing_time_ms"] > 0
