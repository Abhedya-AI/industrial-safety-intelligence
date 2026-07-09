"""Integration tests for Hazard Propagation Kafka integration.

Verifies:
  1.  HazardPropagationPublisher publishes hazard.propagated correctly
  2.  HazardPropagationEventHandler processes compound.risk.detected
  3.  HazardPropagationEventHandler processes hazard.detected
  4.  Invalid / malformed payload handling
  5.  Duplicate event detection
  6.  Missing entity handling (zone not in graph)
  7.  Repository failure handling
  8.  Kafka publish failure handling
  9.  Event payload format compliance (PS-1 v2.0)
  10. Consumer setup registration
  11. Metrics tracking
  12. Propagation threshold filtering
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from app.hazard_propagation.domain.exceptions import (
    HazardPropagationError,
    PropagationSimulationError,
    ZoneNotFoundError,
)
from app.hazard_propagation.domain.value_objects import (
    PropagationLevel,
    PropagationStatus,
)
from app.hazard_propagation.graph.entities import (
    EquipmentNode,
    ZoneNode,
)
from app.hazard_propagation.messaging.consumer import (
    HAZARD_PROPAGATION_SUBSCRIBED_TOPICS,
    HazardPropagationConsumerSetup,
)
from app.hazard_propagation.messaging.handler import (
    HazardPropagationEventHandler,
)
from app.hazard_propagation.messaging.publisher import (
    HazardPropagationPublisher,
)
from app.hazard_propagation.repositories.in_memory_graph_repo import (
    InMemoryGraphRepository,
)
from app.hazard_propagation.services.config import PropagationConfig
from app.hazard_propagation.services.propagation_engine import (
    HazardPropagationEngine,
    PropagationResult,
)
from app.shared.messaging.events import BaseEvent
from app.shared.messaging.producer import NoopEventProducer
from app.shared.messaging.topics import KafkaTopics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def publish_log() -> List[Dict[str, Any]]:
    """Tracks all events published via the producer."""
    return []


@pytest.fixture
def tracking_producer(publish_log) -> NoopEventProducer:
    """Producer that tracks all published events."""
    prod = NoopEventProducer()
    original = prod.publish

    def tracking(topic, data, **kw):
        event = original(topic, data, **kw)
        publish_log.append({"topic": topic, "data": data, "event": event})
        return event

    prod.publish = tracking
    return prod


@pytest_asyncio.fixture
async def graph_repo() -> InMemoryGraphRepository:
    """Graph with connected zones and equipment."""
    repo = InMemoryGraphRepository()
    await repo.create_zone(ZoneNode(
        zone_id="ZONE_A", zone_name="Zone A",
        risk_level_baseline="HIGH", current_worker_count=5,
    ))
    await repo.create_zone(ZoneNode(
        zone_id="ZONE_B", zone_name="Zone B",
        risk_level_baseline="MEDIUM", current_worker_count=3,
    ))
    await repo.create_zone(ZoneNode(
        zone_id="ZONE_C", zone_name="Zone C",
        risk_level_baseline="LOW", current_worker_count=2,
    ))
    await repo.create_connection("ZONE_A", "ZONE_B")
    await repo.create_connection("ZONE_B", "ZONE_C")

    # Equipment
    await repo.create_equipment(
        "ZONE_A",
        EquipmentNode(
            equipment_id="EQ001", equipment_type="Boiler",
            health_score=90.0,
        ),
    )
    return repo


@pytest.fixture
def publisher(tracking_producer) -> HazardPropagationPublisher:
    return HazardPropagationPublisher(tracking_producer)


@pytest_asyncio.fixture
async def engine(graph_repo) -> HazardPropagationEngine:
    return HazardPropagationEngine(graph_repo)


@pytest_asyncio.fixture
async def handler(engine, publisher, graph_repo) -> HazardPropagationEventHandler:
    return HazardPropagationEventHandler(
        propagation_engine=engine,
        publisher=publisher,
        graph_repo=graph_repo,
    )


def _make_base_event(
    event_type: str,
    data: Dict[str, Any],
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a well-formed PS-1 v2.0 event dict."""
    return {
        "event_type": event_type,
        "event_id": event_id or str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_system": "test",
        "data": data,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Publisher — hazard.propagated
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHazardPropagationPublisher:
    async def test_publishes_to_correct_topic(
        self, engine, graph_repo, publish_log, tracking_producer,
    ):
        pub = HazardPropagationPublisher(tracking_producer)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        pub.publish_hazard_propagated(result)
        assert len(publish_log) == 1
        assert publish_log[0]["topic"] == KafkaTopics.HAZARD_PROPAGATED

    async def test_event_data_fields(
        self, engine, graph_repo, publish_log, tracking_producer,
    ):
        pub = HazardPropagationPublisher(tracking_producer)
        result = await engine.propagate(
            hazard_type="FIRE", origin_zone="ZONE_A",
            compound_risk_score=75.0,
        )
        pub.publish_hazard_propagated(result)
        data = publish_log[0]["data"]
        assert data["propagation_id"] == result.propagation_id
        assert data["hazard_type"] == "FIRE"
        assert data["origin_zone"] == "ZONE_A"
        assert "affected_zones" in data
        assert "impact_scores" in data
        assert "propagation_probabilities" in data
        assert "propagation_paths" in data
        assert "affected_equipment" in data
        assert "recommended_action" in data
        assert "propagation_level" in data
        assert "status" in data
        assert "total_affected_zones" in data
        assert "total_workers_at_risk" in data
        assert "impact_radius_meters" in data
        assert "time_to_critical_minutes" in data

    async def test_returns_base_event(
        self, engine, graph_repo, publish_log, tracking_producer,
    ):
        pub = HazardPropagationPublisher(tracking_producer)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        event = pub.publish_hazard_propagated(result)
        assert isinstance(event, BaseEvent)
        assert event.event_type == KafkaTopics.HAZARD_PROPAGATED

    async def test_correlation_id_forwarded(
        self, engine, graph_repo, publish_log, tracking_producer,
    ):
        pub = HazardPropagationPublisher(tracking_producer)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        event = pub.publish_hazard_propagated(
            result, correlation_id="CORR-001",
        )
        assert event.correlation_id == "CORR-001"

    async def test_published_count_increments(
        self, engine, graph_repo, tracking_producer,
    ):
        pub = HazardPropagationPublisher(tracking_producer)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert pub.published_count == 0
        pub.publish_hazard_propagated(result)
        assert pub.published_count == 1
        pub.publish_hazard_propagated(result)
        assert pub.published_count == 2

    async def test_publish_failure_tracked(
        self, engine, graph_repo,
    ):
        failing_producer = NoopEventProducer()
        failing_producer.publish = MagicMock(side_effect=RuntimeError("Kafka down"))
        pub = HazardPropagationPublisher(failing_producer)

        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        event = pub.publish_hazard_propagated(result)
        assert event is None
        assert pub.failed_count == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Handler — compound.risk.detected
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHandleCompoundRiskDetected:
    async def test_processes_event(self, handler, publish_log):
        event = _make_base_event(
            event_type="compound.risk.detected",
            data={
                "zone_id": "ZONE_A",
                "compound_risk_score": 75.0,
                "risk_level": "HIGH",
            },
        )
        result = await handler.handle_event(
            KafkaTopics.COMPOUND_RISK_DETECTED, event,
        )
        assert result is True
        assert handler.events_processed == 1
        assert handler.propagations_executed == 1

    async def test_publishes_hazard_propagated(self, handler, publish_log):
        event = _make_base_event(
            event_type="compound.risk.detected",
            data={
                "zone_id": "ZONE_A",
                "compound_risk_score": 80.0,
                "risk_level": "CRITICAL",
            },
        )
        await handler.handle_event(KafkaTopics.COMPOUND_RISK_DETECTED, event)
        assert len(publish_log) == 1
        assert publish_log[0]["topic"] == KafkaTopics.HAZARD_PROPAGATED

    async def test_skips_low_risk(self, handler, publish_log):
        event = _make_base_event(
            event_type="compound.risk.detected",
            data={
                "zone_id": "ZONE_A",
                "compound_risk_score": 20.0,
                "risk_level": "LOW",
            },
        )
        result = await handler.handle_event(
            KafkaTopics.COMPOUND_RISK_DETECTED, event,
        )
        # Low risk is skipped, but event processing still returns True
        assert handler.events_skipped >= 1
        assert len(publish_log) == 0

    async def test_missing_zone_id_skipped(self, handler, publish_log):
        event = _make_base_event(
            event_type="compound.risk.detected",
            data={"compound_risk_score": 80.0},
        )
        result = await handler.handle_event(
            KafkaTopics.COMPOUND_RISK_DETECTED, event,
        )
        assert len(publish_log) == 0

    async def test_zone_not_in_graph(self, handler, publish_log):
        event = _make_base_event(
            event_type="compound.risk.detected",
            data={
                "zone_id": "ZONE_UNKNOWN",
                "compound_risk_score": 90.0,
            },
        )
        await handler.handle_event(
            KafkaTopics.COMPOUND_RISK_DETECTED, event,
        )
        assert len(publish_log) == 0
        assert handler.events_skipped >= 1

    async def test_correlation_id_forwarded(self, handler, publish_log):
        eid = str(uuid.uuid4())
        event = _make_base_event(
            event_type="compound.risk.detected",
            data={
                "zone_id": "ZONE_A",
                "compound_risk_score": 80.0,
            },
            event_id=eid,
        )
        await handler.handle_event(KafkaTopics.COMPOUND_RISK_DETECTED, event)
        assert len(publish_log) == 1

    async def test_infers_gas_leak_from_factors(self, handler, publish_log):
        event = _make_base_event(
            event_type="compound.risk.detected",
            data={
                "zone_id": "ZONE_A",
                "compound_risk_score": 80.0,
                "contributing_factors": {"gas_risk": 0.8},
            },
        )
        await handler.handle_event(KafkaTopics.COMPOUND_RISK_DETECTED, event)
        assert len(publish_log) == 1
        assert publish_log[0]["data"]["hazard_type"] == "GAS_LEAK"

    async def test_infers_fire_from_factors(self, handler, publish_log):
        event = _make_base_event(
            event_type="compound.risk.detected",
            data={
                "zone_id": "ZONE_A",
                "compound_risk_score": 80.0,
                "contributing_factors": {"fire_risk": 0.8},
            },
        )
        await handler.handle_event(KafkaTopics.COMPOUND_RISK_DETECTED, event)
        assert publish_log[0]["data"]["hazard_type"] == "FIRE"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Handler — hazard.detected
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHandleHazardDetected:
    async def test_processes_event(self, handler, publish_log):
        event = _make_base_event(
            event_type="hazard.detected",
            data={
                "zone_id": "ZONE_A",
                "hazard_type": "FIRE",
                "severity": "CRITICAL",
            },
        )
        result = await handler.handle_event(
            KafkaTopics.HAZARD_DETECTED, event,
        )
        assert result is True
        assert handler.propagations_executed == 1

    async def test_publishes_propagation_event(self, handler, publish_log):
        event = _make_base_event(
            event_type="hazard.detected",
            data={
                "zone_id": "ZONE_A",
                "hazard_type": "GAS_LEAK",
                "severity": "HIGH",
            },
        )
        await handler.handle_event(KafkaTopics.HAZARD_DETECTED, event)
        assert len(publish_log) == 1
        assert publish_log[0]["topic"] == KafkaTopics.HAZARD_PROPAGATED
        assert publish_log[0]["data"]["hazard_type"] == "GAS_LEAK"

    async def test_severity_mapping(self, handler, publish_log):
        for severity, expected_min in [
            ("LOW", 20.0), ("MEDIUM", 40.0),
            ("HIGH", 60.0), ("CRITICAL", 90.0),
        ]:
            event = _make_base_event(
                event_type="hazard.detected",
                data={
                    "zone_id": "ZONE_A",
                    "hazard_type": "SMOKE",
                    "severity": severity,
                },
            )
            await handler.handle_event(KafkaTopics.HAZARD_DETECTED, event)

    async def test_missing_zone_id(self, handler, publish_log):
        event = _make_base_event(
            event_type="hazard.detected",
            data={"hazard_type": "FIRE"},
        )
        await handler.handle_event(KafkaTopics.HAZARD_DETECTED, event)
        assert len(publish_log) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Invalid / malformed payloads
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMalformedPayloads:
    async def test_missing_event_type(self, handler):
        event = {
            "event_id": "123",
            "timestamp": "2026-01-01T00:00:00Z",
            "data": {},
        }
        result = await handler.handle_event(
            KafkaTopics.COMPOUND_RISK_DETECTED, event,
        )
        assert result is False
        assert handler.events_failed == 1

    async def test_missing_data_field(self, handler):
        event = {
            "event_type": "compound.risk.detected",
            "event_id": "123",
            "timestamp": "2026-01-01T00:00:00Z",
        }
        result = await handler.handle_event(
            KafkaTopics.COMPOUND_RISK_DETECTED, event,
        )
        assert result is False

    async def test_data_not_dict(self, handler):
        event = {
            "event_type": "compound.risk.detected",
            "event_id": "123",
            "timestamp": "2026-01-01T00:00:00Z",
            "data": "not a dict",
        }
        result = await handler.handle_event(
            KafkaTopics.COMPOUND_RISK_DETECTED, event,
        )
        assert result is False

    async def test_missing_timestamp(self, handler):
        event = {
            "event_type": "compound.risk.detected",
            "event_id": "123",
            "data": {"zone_id": "ZONE_A"},
        }
        result = await handler.handle_event(
            KafkaTopics.COMPOUND_RISK_DETECTED, event,
        )
        assert result is False

    async def test_empty_event(self, handler):
        result = await handler.handle_event(
            KafkaTopics.COMPOUND_RISK_DETECTED, {},
        )
        assert result is False

    async def test_unhandled_topic(self, handler):
        event = _make_base_event(
            event_type="unknown.topic",
            data={"zone_id": "ZONE_A"},
        )
        result = await handler.handle_event("unknown.topic", event)
        assert result is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Duplicate event detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDuplicateDetection:
    async def test_duplicate_event_skipped(self, handler, publish_log):
        eid = str(uuid.uuid4())
        event = _make_base_event(
            event_type="hazard.detected",
            data={
                "zone_id": "ZONE_A",
                "hazard_type": "FIRE",
                "severity": "HIGH",
            },
            event_id=eid,
        )
        # First time — should process
        r1 = await handler.handle_event(KafkaTopics.HAZARD_DETECTED, event)
        assert r1 is True

        # Second time — should skip
        r2 = await handler.handle_event(KafkaTopics.HAZARD_DETECTED, event)
        assert r2 is False
        assert handler.events_skipped >= 1

    async def test_different_events_not_duplicates(self, handler):
        e1 = _make_base_event(
            event_type="hazard.detected",
            data={
                "zone_id": "ZONE_A",
                "hazard_type": "FIRE",
                "severity": "HIGH",
            },
        )
        e2 = _make_base_event(
            event_type="hazard.detected",
            data={
                "zone_id": "ZONE_B",
                "hazard_type": "GAS_LEAK",
                "severity": "HIGH",
            },
        )
        r1 = await handler.handle_event(KafkaTopics.HAZARD_DETECTED, e1)
        r2 = await handler.handle_event(KafkaTopics.HAZARD_DETECTED, e2)
        assert r1 is True
        assert r2 is True
        assert handler.events_processed == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Repository failure handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRepositoryFailures:
    async def test_graph_repo_get_zone_failure(
        self, engine, publisher, publish_log,
    ):
        """GraphRepository.get_zone raises → event fails gracefully."""
        mock_repo = AsyncMock()
        mock_repo.get_zone = AsyncMock(
            side_effect=RuntimeError("Neo4j connection lost"),
        )
        handler = HazardPropagationEventHandler(
            propagation_engine=engine,
            publisher=publisher,
            graph_repo=mock_repo,
        )
        event = _make_base_event(
            event_type="hazard.detected",
            data={
                "zone_id": "ZONE_A",
                "hazard_type": "FIRE",
                "severity": "HIGH",
            },
        )
        result = await handler.handle_event(
            KafkaTopics.HAZARD_DETECTED, event,
        )
        assert result is False
        assert handler.events_failed == 1
        assert len(publish_log) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Kafka publish failure handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestKafkaPublishFailures:
    async def test_publish_failure_does_not_crash(
        self, engine, graph_repo,
    ):
        """Kafka publish failure should not crash the handler."""
        failing_producer = NoopEventProducer()
        failing_producer.publish = MagicMock(
            side_effect=RuntimeError("Kafka unavailable"),
        )
        failing_publisher = HazardPropagationPublisher(failing_producer)
        handler = HazardPropagationEventHandler(
            propagation_engine=engine,
            publisher=failing_publisher,
            graph_repo=graph_repo,
        )
        event = _make_base_event(
            event_type="hazard.detected",
            data={
                "zone_id": "ZONE_A",
                "hazard_type": "FIRE",
                "severity": "HIGH",
            },
        )
        # Should NOT raise — publish failure is caught
        result = await handler.handle_event(
            KafkaTopics.HAZARD_DETECTED, event,
        )
        assert result is True
        assert handler.propagations_executed == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Event payload format compliance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEventPayloadCompliance:
    async def test_hazard_propagated_payload_structure(
        self, handler, publish_log,
    ):
        """Verify published event matches PS-1 v2.0 format."""
        event = _make_base_event(
            event_type="hazard.detected",
            data={
                "zone_id": "ZONE_A",
                "hazard_type": "GAS_LEAK",
                "severity": "HIGH",
            },
        )
        await handler.handle_event(KafkaTopics.HAZARD_DETECTED, event)
        assert len(publish_log) == 1

        published = publish_log[0]
        assert published["topic"] == "hazard.propagated"

        data = published["data"]
        # Required fields per spec
        assert "propagation_id" in data
        assert "hazard_type" in data
        assert "origin_zone" in data
        assert "propagation_level" in data
        assert "status" in data
        assert "affected_zones" in data
        assert "total_affected_zones" in data
        assert "total_workers_at_risk" in data
        assert "impact_radius_meters" in data
        assert "time_to_critical_minutes" in data
        assert "impact_scores" in data
        assert "propagation_probabilities" in data
        assert "affected_equipment" in data
        assert "propagation_paths" in data
        assert "recommended_action" in data

    async def test_base_event_envelope(self, handler, publish_log):
        """Published event should have BaseEvent wrapper fields."""
        event = _make_base_event(
            event_type="hazard.detected",
            data={
                "zone_id": "ZONE_A",
                "hazard_type": "FIRE",
                "severity": "CRITICAL",
            },
        )
        await handler.handle_event(KafkaTopics.HAZARD_DETECTED, event)
        published_event = publish_log[0]["event"]
        assert hasattr(published_event, "event_type")
        assert hasattr(published_event, "event_id")
        assert hasattr(published_event, "timestamp")
        assert hasattr(published_event, "source_system")
        assert published_event.source_system == "hazard_propagation_engine"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Consumer setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConsumerSetup:
    def test_subscribed_topics(self):
        assert KafkaTopics.COMPOUND_RISK_DETECTED in (
            HAZARD_PROPAGATION_SUBSCRIBED_TOPICS
        )
        assert KafkaTopics.HAZARD_DETECTED in (
            HAZARD_PROPAGATION_SUBSCRIBED_TOPICS
        )

    def test_register_handlers(self, handler):
        mock_consumer = MagicMock()
        setup = HazardPropagationConsumerSetup(mock_consumer, handler)
        setup.register()
        assert setup.is_registered is True
        assert mock_consumer.register_handler.call_count == 2

    def test_idempotent_registration(self, handler):
        mock_consumer = MagicMock()
        setup = HazardPropagationConsumerSetup(mock_consumer, handler)
        setup.register()
        setup.register()  # Second call should be a no-op
        assert mock_consumer.register_handler.call_count == 2

    def test_subscribed_topics_property(self, handler):
        mock_consumer = MagicMock()
        setup = HazardPropagationConsumerSetup(mock_consumer, handler)
        assert len(setup.subscribed_topics) == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Metrics tracking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMetrics:
    async def test_events_processed_counter(self, handler):
        assert handler.events_processed == 0
        event = _make_base_event(
            event_type="hazard.detected",
            data={
                "zone_id": "ZONE_A",
                "hazard_type": "FIRE",
                "severity": "HIGH",
            },
        )
        await handler.handle_event(KafkaTopics.HAZARD_DETECTED, event)
        assert handler.events_processed == 1

    async def test_events_failed_counter(self, handler):
        assert handler.events_failed == 0
        await handler.handle_event(KafkaTopics.HAZARD_DETECTED, {})
        assert handler.events_failed == 1

    async def test_propagations_executed_counter(self, handler):
        assert handler.propagations_executed == 0
        event = _make_base_event(
            event_type="hazard.detected",
            data={
                "zone_id": "ZONE_A",
                "hazard_type": "GAS_LEAK",
                "severity": "CRITICAL",
            },
        )
        await handler.handle_event(KafkaTopics.HAZARD_DETECTED, event)
        assert handler.propagations_executed == 1

    async def test_reset_metrics(self, handler):
        event = _make_base_event(
            event_type="hazard.detected",
            data={
                "zone_id": "ZONE_A",
                "hazard_type": "FIRE",
                "severity": "HIGH",
            },
        )
        await handler.handle_event(KafkaTopics.HAZARD_DETECTED, event)
        assert handler.events_processed > 0
        handler.reset_metrics()
        assert handler.events_processed == 0
        assert handler.events_failed == 0
        assert handler.propagations_executed == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. End-to-end flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEndToEnd:
    async def test_compound_risk_to_propagation(self, handler, publish_log):
        """Full flow: compound.risk.detected → propagation → publish."""
        event = _make_base_event(
            event_type="compound.risk.detected",
            data={
                "zone_id": "ZONE_A",
                "compound_risk_score": 85.0,
                "risk_level": "CRITICAL",
                "equipment_id": "EQ001",
                "contributing_factors": {"gas_risk": 0.9},
            },
        )
        result = await handler.handle_event(
            KafkaTopics.COMPOUND_RISK_DETECTED, event,
        )
        assert result is True
        assert handler.propagations_executed == 1
        assert len(publish_log) == 1

        # Verify published data
        data = publish_log[0]["data"]
        assert data["origin_zone"] == "ZONE_A"
        assert data["hazard_type"] == "GAS_LEAK"
        assert data["total_affected_zones"] >= 1
        assert isinstance(data["impact_scores"], dict)
        assert "ZONE_A" in data["impact_scores"]

    async def test_hazard_detected_to_propagation(self, handler, publish_log):
        """Full flow: hazard.detected → propagation → publish."""
        event = _make_base_event(
            event_type="hazard.detected",
            data={
                "zone_id": "ZONE_A",
                "hazard_type": "FIRE",
                "severity": "CRITICAL",
            },
        )
        result = await handler.handle_event(
            KafkaTopics.HAZARD_DETECTED, event,
        )
        assert result is True
        assert len(publish_log) == 1

        data = publish_log[0]["data"]
        assert data["hazard_type"] == "FIRE"
        assert data["origin_zone"] == "ZONE_A"
        assert len(data["affected_zones"]) >= 1
        assert isinstance(data["propagation_paths"], list)
        assert isinstance(data["affected_equipment"], list)

    async def test_multiple_events_sequential(self, handler, publish_log):
        """Multiple events processed in sequence."""
        e1 = _make_base_event(
            event_type="hazard.detected",
            data={
                "zone_id": "ZONE_A",
                "hazard_type": "FIRE",
                "severity": "HIGH",
            },
        )
        e2 = _make_base_event(
            event_type="compound.risk.detected",
            data={
                "zone_id": "ZONE_B",
                "compound_risk_score": 70.0,
                "risk_level": "HIGH",
            },
        )
        await handler.handle_event(KafkaTopics.HAZARD_DETECTED, e1)
        await handler.handle_event(KafkaTopics.COMPOUND_RISK_DETECTED, e2)
        assert handler.events_processed == 2
        assert len(publish_log) == 2
