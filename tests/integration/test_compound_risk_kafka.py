"""Integration tests for Compound Risk Kafka messaging.

Tests cover:
  - Event consumption (all 3 subscribed topics)
  - Event publication (compound.risk.detected)
  - Invalid/malformed payloads
  - Missing required fields
  - Full pipeline: event → rule engine → aggregation → persist → publish
  - ZoneRiskState accumulation
  - Publisher event data structure
  - Consumer setup registration
  - Error handling (domain errors, unexpected errors)
  - Metrics tracking
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.compound_risk.messaging.consumer import (
    COMPOUND_RISK_SUBSCRIBED_TOPICS,
    CompoundRiskConsumerSetup,
)
from app.compound_risk.messaging.handler import (
    CompoundRiskEventHandler,
    ZoneRiskState,
)
from app.compound_risk.messaging.publisher import CompoundRiskPublisher
from app.compound_risk.repositories.sqlalchemy_compound_risk_repo import (
    SQLAlchemyCompoundRiskRepository,
)
from app.compound_risk.rules.rule_engine import CompoundRiskRuleEngine, create_default_rules
from app.compound_risk.services.compound_risk_service import (
    CompoundRiskAggregationService,
)
from app.compound_risk.services.explainability_service import ExplainabilityService
from app.shared.messaging.consumer import NoopEventConsumer
from app.shared.messaging.events import BaseEvent, create_event
from app.shared.messaging.producer import NoopEventProducer
from app.shared.messaging.topics import KafkaTopics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _wrap_event(event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a full PS-1 v2.0 compliant event dict."""
    return {
        "event_type": event_type,
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_system": "test",
        "data": data,
        "version": "1.0",
    }


def _anomaly_event(
    zone_id: str = "ZONE_A",
    if_score: float = 0.7,
    ae_score: float = 0.5,
    **extra,
) -> Dict[str, Any]:
    data = {
        "zone_id": zone_id,
        "sensor_id": "S001",
        "isolation_forest_score": if_score,
        "autoencoder_score": ae_score,
        "sensor_health_score": 80.0,
        **extra,
    }
    return _wrap_event(KafkaTopics.SENSOR_READING_ANOMALY, data)


def _risk_assessment_event(
    zone_id: str = "ZONE_A",
    accident_prob: float = 0.65,
    risk_score: float = 55.0,
    **extra,
) -> Dict[str, Any]:
    data = {
        "zone_id": zone_id,
        "equipment_id": "EQ001",
        "accident_probability": accident_prob,
        "risk_score": risk_score,
        **extra,
    }
    return _wrap_event(KafkaTopics.RISK_ASSESSMENT_GENERATED, data)


def _risk_score_updated_event(
    zone_id: str = "ZONE_A",
    risk_score: float = 70.0,
    **extra,
) -> Dict[str, Any]:
    data = {
        "zone_id": zone_id,
        "risk_score": risk_score,
        **extra,
    }
    return _wrap_event(KafkaTopics.RISK_SCORE_UPDATED, data)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest_asyncio.fixture
async def repo(db_session: AsyncSession) -> SQLAlchemyCompoundRiskRepository:
    return SQLAlchemyCompoundRiskRepository(db_session)


@pytest_asyncio.fixture
async def publish_log() -> List[Dict[str, Any]]:
    """Track all published events."""
    return []


@pytest_asyncio.fixture
async def producer(publish_log) -> NoopEventProducer:
    """Noop producer that logs publishes."""
    prod = NoopEventProducer()
    original_publish = prod.publish

    def tracking_publish(topic, data, **kwargs):
        event = original_publish(topic, data, **kwargs)
        publish_log.append({"topic": topic, "data": data, "event": event})
        return event

    prod.publish = tracking_publish
    return prod


@pytest_asyncio.fixture
async def publisher(producer) -> CompoundRiskPublisher:
    return CompoundRiskPublisher(producer)


@pytest_asyncio.fixture
async def handler(repo, publisher) -> CompoundRiskEventHandler:
    agg_service = CompoundRiskAggregationService(repo)
    rule_engine = CompoundRiskRuleEngine(create_default_rules())
    explain_service = ExplainabilityService()
    return CompoundRiskEventHandler(
        aggregation_service=agg_service,
        rule_engine=rule_engine,
        explainability_service=explain_service,
        publisher=publisher,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Event consumption — anomaly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAnomalyConsumption:
    async def test_anomaly_event_processed(self, handler):
        event = _anomaly_event()
        result = await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, event,
        )
        assert result is True
        assert handler.events_processed == 1

    async def test_anomaly_updates_zone_state(self, handler):
        event = _anomaly_event(zone_id="ZONE_B", if_score=0.8, ae_score=0.6)
        await handler.handle_event(KafkaTopics.SENSOR_READING_ANOMALY, event)

        state = handler.get_zone_state("ZONE_B")
        assert state is not None
        assert state.isolation_forest_score == 0.8
        assert state.autoencoder_score == 0.6

    async def test_anomaly_with_sensor_facts(self, handler):
        event = _anomaly_event(
            temperature_celsius=75.0, gas_level_ppm=120.0,
        )
        await handler.handle_event(KafkaTopics.SENSOR_READING_ANOMALY, event)

        state = handler.get_zone_state("ZONE_A")
        assert state.sensor_facts["temperature_celsius"] == 75.0
        assert state.sensor_facts["gas_level_ppm"] == 120.0

    async def test_anomaly_max_score_kept(self, handler):
        """Highest anomaly score is retained across events."""
        event1 = _anomaly_event(if_score=0.3)
        event2 = _anomaly_event(if_score=0.9)
        event3 = _anomaly_event(if_score=0.5)

        for e in [event1, event2, event3]:
            await handler.handle_event(KafkaTopics.SENSOR_READING_ANOMALY, e)

        state = handler.get_zone_state("ZONE_A")
        assert state.isolation_forest_score == 0.9


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Event consumption — risk assessment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRiskAssessmentConsumption:
    async def test_risk_assessment_processed(self, handler):
        event = _risk_assessment_event()
        result = await handler.handle_event(
            KafkaTopics.RISK_ASSESSMENT_GENERATED, event,
        )
        assert result is True

    async def test_risk_assessment_updates_state(self, handler):
        event = _risk_assessment_event(
            zone_id="ZONE_C", accident_prob=0.75, risk_score=60.0,
        )
        await handler.handle_event(
            KafkaTopics.RISK_ASSESSMENT_GENERATED, event,
        )
        state = handler.get_zone_state("ZONE_C")
        assert state.accident_probability == 0.75
        assert state.risk_score == 60.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Event consumption — risk score updated
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRiskScoreUpdatedConsumption:
    async def test_risk_score_updated_processed(self, handler):
        event = _risk_score_updated_event()
        result = await handler.handle_event(
            KafkaTopics.RISK_SCORE_UPDATED, event,
        )
        assert result is True

    async def test_risk_score_updates_state(self, handler):
        event = _risk_score_updated_event(
            zone_id="ZONE_D", risk_score=82.0,
            accident_probability=0.85,
        )
        await handler.handle_event(KafkaTopics.RISK_SCORE_UPDATED, event)
        state = handler.get_zone_state("ZONE_D")
        assert state.risk_score == 82.0
        assert state.accident_probability == 0.85


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Event publication
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEventPublication:
    async def test_publishes_compound_risk_detected(self, handler, publish_log):
        event = _anomaly_event(if_score=0.8)
        await handler.handle_event(KafkaTopics.SENSOR_READING_ANOMALY, event)
        assert len(publish_log) >= 1
        assert publish_log[0]["topic"] == KafkaTopics.COMPOUND_RISK_DETECTED

    async def test_published_data_has_required_fields(self, handler, publish_log):
        event = _anomaly_event()
        await handler.handle_event(KafkaTopics.SENSOR_READING_ANOMALY, event)
        data = publish_log[0]["data"]
        assert "analysis_id" in data
        assert "compound_risk_score" in data
        assert "risk_level" in data
        assert "confidence_score" in data
        assert "contributing_factors" in data

    async def test_published_event_is_base_event(self, handler, publish_log):
        event = _anomaly_event()
        await handler.handle_event(KafkaTopics.SENSOR_READING_ANOMALY, event)
        assert publish_log[0]["event"] is not None

    async def test_publication_per_event(self, handler, publish_log):
        """Each processed event triggers a publication."""
        await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, _anomaly_event(),
        )
        await handler.handle_event(
            KafkaTopics.RISK_ASSESSMENT_GENERATED, _risk_assessment_event(),
        )
        assert len(publish_log) >= 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Invalid payloads
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInvalidPayloads:
    async def test_missing_event_type(self, handler):
        bad_event = {
            "event_id": "123",
            "timestamp": "2026-07-08T00:00:00Z",
            "data": {},
        }
        result = await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, bad_event,
        )
        assert result is False
        assert handler.events_failed == 1

    async def test_missing_data_field(self, handler):
        bad_event = {
            "event_type": "sensor.reading.anomaly",
            "event_id": "123",
            "timestamp": "2026-07-08T00:00:00Z",
        }
        result = await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, bad_event,
        )
        assert result is False

    async def test_data_not_dict(self, handler):
        bad_event = {
            "event_type": "sensor.reading.anomaly",
            "event_id": "123",
            "timestamp": "2026-07-08T00:00:00Z",
            "data": "not a dict",
        }
        result = await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, bad_event,
        )
        assert result is False

    async def test_empty_event(self, handler):
        result = await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, {},
        )
        assert result is False

    async def test_missing_timestamp(self, handler):
        bad_event = {
            "event_type": "sensor.reading.anomaly",
            "event_id": "123",
            "data": {"zone_id": "ZONE_A"},
        }
        result = await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, bad_event,
        )
        assert result is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Unhandled topic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestUnhandledTopic:
    async def test_unknown_topic(self, handler):
        event = _wrap_event("unknown.topic", {"x": 1})
        result = await handler.handle_event("unknown.topic", event)
        assert result is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Full pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFullPipeline:
    async def test_anomaly_triggers_full_pipeline(self, handler, repo, publish_log):
        """Anomaly event → persist + publish."""
        event = _anomaly_event(
            if_score=0.85, ae_score=0.7,
            temperature_celsius=75, gas_level_ppm=120,
        )
        result = await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, event,
        )
        assert result is True
        assert handler.analyses_produced >= 1

        # Verify persisted
        count = await repo.count(zone_id="ZONE_A")
        assert count >= 1

        # Verify published
        assert len(publish_log) >= 1

    async def test_multi_event_accumulation(self, handler, repo, publish_log):
        """Multiple events accumulate state before analysis."""
        # First: anomaly scores
        await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY,
            _anomaly_event(if_score=0.6, ae_score=0.4),
        )
        # Second: risk assessment
        await handler.handle_event(
            KafkaTopics.RISK_ASSESSMENT_GENERATED,
            _risk_assessment_event(accident_prob=0.7),
        )

        state = handler.get_zone_state("ZONE_A")
        assert state.event_count == 2
        assert state.isolation_forest_score == 0.6
        assert state.accident_probability == 0.7

    async def test_different_zones_independent(self, handler):
        await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY,
            _anomaly_event(zone_id="ZONE_A", if_score=0.9),
        )
        await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY,
            _anomaly_event(zone_id="ZONE_B", if_score=0.3),
        )
        assert handler.get_zone_state("ZONE_A").isolation_forest_score == 0.9
        assert handler.get_zone_state("ZONE_B").isolation_forest_score == 0.3

    async def test_rule_engine_triggered_on_facts(self, handler, publish_log):
        """Sensor facts pass to rule engine during analysis."""
        event = _anomaly_event(
            temperature_celsius=75, gas_level_ppm=120,
        )
        await handler.handle_event(KafkaTopics.SENSOR_READING_ANOMALY, event)
        # Rules should fire for high temp + gas
        assert handler.analyses_produced >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Publisher unit tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPublisher:
    async def test_build_event_data(self, handler, repo, publisher):
        """Verify published data structure."""
        from app.compound_risk.services.compound_risk_service import (
            CompoundRiskInput, CompoundRiskResult,
        )
        from app.compound_risk.domain.value_objects import RiskLevel
        from app.compound_risk.models.compound_risk_model import CompoundRiskModel
        from datetime import datetime, timezone

        model = CompoundRiskModel(
            id="test-id",
            equipment_id="EQ001",
            zone_id="ZONE_A",
            anomaly_score=0.7,
            accident_probability=0.65,
            risk_score=55.0,
            sensor_health_score=80.0,
            compound_risk_score=0.62,
            risk_level="HIGH",
            confidence_score=0.85,
            created_at=datetime(2026, 7, 8, tzinfo=timezone.utc),
        )
        result = CompoundRiskResult(
            compound_risk_score=62.0,
            risk_level=RiskLevel.HIGH,
            confidence_score=0.85,
            contributing_factors=[],
            component_scores={"risk_prediction": 65.0},
        )
        data = publisher._build_event_data(model, result)
        assert data["analysis_id"] == "test-id"
        assert data["zone_id"] == "ZONE_A"
        assert data["risk_level"] == "HIGH"
        assert data["compound_risk_score"] == 62.0
        assert data["created_at"] is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Consumer setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConsumerSetup:
    def test_subscribed_topics(self):
        assert KafkaTopics.SENSOR_READING_ANOMALY in COMPOUND_RISK_SUBSCRIBED_TOPICS
        assert KafkaTopics.RISK_ASSESSMENT_GENERATED in COMPOUND_RISK_SUBSCRIBED_TOPICS
        assert KafkaTopics.RISK_SCORE_UPDATED in COMPOUND_RISK_SUBSCRIBED_TOPICS
        assert len(COMPOUND_RISK_SUBSCRIBED_TOPICS) == 3

    async def test_register(self, handler):
        consumer = NoopEventConsumer()
        setup = CompoundRiskConsumerSetup(consumer, handler)
        assert setup.is_registered is False
        setup.register()
        assert setup.is_registered is True

    async def test_double_register_safe(self, handler):
        consumer = NoopEventConsumer()
        setup = CompoundRiskConsumerSetup(consumer, handler)
        setup.register()
        setup.register()  # Should not raise
        assert setup.is_registered is True

    async def test_subscribed_topics_property(self, handler):
        consumer = NoopEventConsumer()
        setup = CompoundRiskConsumerSetup(consumer, handler)
        assert len(setup.subscribed_topics) == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. ZoneRiskState
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestZoneRiskState:
    def test_defaults(self):
        state = ZoneRiskState(zone_id="ZONE_A")
        assert state.isolation_forest_score == 0.0
        assert state.sensor_health_score == 100.0
        assert state.event_count == 0

    def test_to_compound_input(self):
        state = ZoneRiskState(
            zone_id="ZONE_A",
            equipment_id="EQ001",
            isolation_forest_score=0.7,
            accident_probability=0.5,
            sensor_health_score=80.0,
        )
        inp = state.to_compound_input()
        assert inp.zone_id == "ZONE_A"
        assert inp.equipment_id == "EQ001"
        assert inp.isolation_forest_score == 0.7

    async def test_reset_zone_state(self, handler):
        await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY,
            _anomaly_event(zone_id="ZONE_X"),
        )
        assert handler.get_zone_state("ZONE_X") is not None
        handler.reset_zone_state("ZONE_X")
        assert handler.get_zone_state("ZONE_X") is None

    async def test_reset_all(self, handler):
        await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, _anomaly_event(zone_id="Z1"),
        )
        await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, _anomaly_event(zone_id="Z2"),
        )
        handler.reset_all()
        assert handler.get_zone_state("Z1") is None
        assert handler.events_processed == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. Metrics tracking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMetrics:
    async def test_processed_count(self, handler):
        await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, _anomaly_event(),
        )
        assert handler.events_processed == 1

    async def test_failed_count(self, handler):
        await handler.handle_event(KafkaTopics.SENSOR_READING_ANOMALY, {})
        assert handler.events_failed == 1

    async def test_analyses_produced_count(self, handler):
        await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, _anomaly_event(),
        )
        assert handler.analyses_produced >= 1

    async def test_mixed_success_failure(self, handler):
        await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, _anomaly_event(),
        )
        await handler.handle_event(KafkaTopics.SENSOR_READING_ANOMALY, {})
        assert handler.events_processed == 1
        assert handler.events_failed == 1
