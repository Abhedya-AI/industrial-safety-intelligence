"""Mode 1: Manual Handler Invocation — Fast Integration Verification.

Verification steps:
  1. Check Kafka broker connectivity
  2. Publish sensor.reading.anomaly event
  3. Verify Compound Risk handler consumes and processes it
  4. Verify compound.risk.detected was published to Kafka
  5. Verify Hazard Propagation handler consumes and processes it
  6. Verify hazard.propagated was published to Kafka
  7. Verify Kafka offsets advanced correctly

Handlers are invoked directly (not via the consumer loop). All Kafka
I/O is REAL — no mocks, no NoopProducer.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from app.compound_risk.messaging.handler import CompoundRiskEventHandler
from app.compound_risk.messaging.publisher import CompoundRiskPublisher
from app.compound_risk.repositories.session_scoped_compound_risk_repo import (
    SessionScopedCompoundRiskRepository,
)
from app.compound_risk.rules.rule_engine import (
    CompoundRiskRuleEngine,
    create_default_rules,
)
from app.compound_risk.services.compound_risk_service import (
    CompoundRiskAggregationService,
)
from app.compound_risk.services.explainability_service import ExplainabilityService
from app.hazard_propagation.messaging.handler import HazardPropagationEventHandler
from app.hazard_propagation.messaging.publisher import HazardPropagationPublisher
from app.hazard_propagation.services.propagation_engine import HazardPropagationEngine
from app.shared.messaging.topics import KafkaTopics
from tests.e2e_kafka.conftest import (
    E2E_TOPICS,
    e2e_session_factory,
    get_topic_offsets,
)
from tests.e2e_kafka.report_generator import (
    ModeResult,
    OffsetSnapshot,
    StepResult,
)

logger = logging.getLogger(__name__)

# Module-level result collector for report generation
_mode1_result = ModeResult(mode_name="Mode 1: Manual Handler Invocation")


@pytest.mark.usefixtures("kafka_available", "ensure_topics")
class TestMode1ManualHandler:
    """Mode 1: Direct handler invocation with real Kafka I/O."""

    def test_step1_broker_connectivity(self, kafka_admin, kafka_bootstrap):
        """Step 1: Verify Kafka broker is reachable and topics exist."""
        t0 = time.monotonic()

        topics = kafka_admin.list_topics()
        assert isinstance(topics, list), "list_topics() must return a list"
        assert len(topics) > 0, "Broker must have at least one topic"

        # Verify our E2E topics exist
        for topic in E2E_TOPICS:
            assert topic in topics, f"Topic '{topic}' not found on broker"

        latency = (time.monotonic() - t0) * 1000
        _mode1_result.steps.append(StepResult(
            step_number=1,
            step_name="Broker Connectivity",
            passed=True,
            latency_ms=latency,
            details=f"Found {len(topics)} topics on broker",
        ))

    def test_step2_publish_anomaly_event(
        self, kafka_producer, kafka_bootstrap, anomaly_event_payload,
        verification_consumer_factory,
    ):
        """Step 2: Publish sensor.reading.anomaly and verify it lands on Kafka."""
        t0 = time.monotonic()

        event = kafka_producer.publish(
            topic=KafkaTopics.SENSOR_READING_ANOMALY,
            data=anomaly_event_payload["data"],
            source_system="sensor_intelligence",
            key="ZONE_A",
            correlation_id=anomaly_event_payload.get("correlation_id"),
        )
        kafka_producer.flush(timeout=10.0)

        assert event is not None, "publish() must return a BaseEvent"
        assert event.event_id, "Event must have an event_id"

        # Verify via a verification consumer
        consumer = verification_consumer_factory([KafkaTopics.SENSOR_READING_ANOMALY])
        messages = []
        for msg in consumer:
            messages.append(msg)
            if msg.value.get("event_id") == event.event_id:
                break

        assert any(
            m.value.get("event_id") == event.event_id for m in messages
        ), "Published event not found on Kafka topic"

        latency = (time.monotonic() - t0) * 1000
        _mode1_result.steps.append(StepResult(
            step_number=2,
            step_name="Publish sensor.reading.anomaly",
            passed=True,
            latency_ms=latency,
            event_id=event.event_id,
            details=f"Published to {KafkaTopics.SENSOR_READING_ANOMALY}",
        ))

        # Store event_id for downstream steps
        self.__class__._anomaly_event_id = event.event_id
        self.__class__._anomaly_correlation_id = event.correlation_id

    def test_step3_compound_risk_handler_processes(
        self, kafka_producer, populated_graph_repo,
        anomaly_event_payload,
    ):
        """Step 3: CompoundRiskEventHandler processes the anomaly event."""
        t0 = time.monotonic()

        # Build handler with real publisher and session-scoped repo
        repo = SessionScopedCompoundRiskRepository(e2e_session_factory)
        aggregation = CompoundRiskAggregationService(repository=repo)
        rule_engine = CompoundRiskRuleEngine(create_default_rules())
        explainability = ExplainabilityService()
        publisher = CompoundRiskPublisher(kafka_producer)

        handler = CompoundRiskEventHandler(
            aggregation_service=aggregation,
            rule_engine=rule_engine,
            explainability_service=explainability,
            publisher=publisher,
        )

        # Invoke handler directly with the anomaly event
        success = asyncio.get_event_loop().run_until_complete(
            handler.handle_event(
                KafkaTopics.SENSOR_READING_ANOMALY,
                anomaly_event_payload,
            )
        )
        kafka_producer.flush(timeout=10.0)

        assert success is True, "Handler must return True on success"
        assert handler.events_processed == 1, "events_processed must be 1"
        assert handler.analyses_produced == 1, "analyses_produced must be 1"
        assert handler.events_failed == 0, "events_failed must be 0"

        latency = (time.monotonic() - t0) * 1000
        _mode1_result.steps.append(StepResult(
            step_number=3,
            step_name="Compound Risk handler processes event",
            passed=True,
            latency_ms=latency,
            details=(
                f"processed={handler.events_processed}, "
                f"analyses={handler.analyses_produced}"
            ),
        ))

    def test_step4_verify_compound_risk_published(
        self, kafka_bootstrap, verification_consumer_factory,
    ):
        """Step 4: Verify compound.risk.detected was published to Kafka."""
        t0 = time.monotonic()

        consumer = verification_consumer_factory(
            [KafkaTopics.COMPOUND_RISK_DETECTED],
        )

        found_event = None
        for msg in consumer:
            data = msg.value
            # Match by zone_id since correlation varies
            if (
                data.get("event_type") == KafkaTopics.COMPOUND_RISK_DETECTED
                and isinstance(data.get("data"), dict)
                and data["data"].get("zone_id") == "ZONE_A"
            ):
                found_event = data
                break

        assert found_event is not None, (
            "compound.risk.detected event not found on Kafka"
        )

        # Validate PS-1 v2.0 schema compliance
        assert "event_type" in found_event
        assert "event_id" in found_event
        assert "timestamp" in found_event
        assert "data" in found_event
        assert isinstance(found_event["data"], dict)

        event_data = found_event["data"]
        assert "zone_id" in event_data
        assert "compound_risk_score" in event_data
        assert "risk_level" in event_data
        assert event_data["zone_id"] == "ZONE_A"
        assert event_data["compound_risk_score"] >= 0

        latency = (time.monotonic() - t0) * 1000
        _mode1_result.steps.append(StepResult(
            step_number=4,
            step_name="Verify compound.risk.detected published",
            passed=True,
            latency_ms=latency,
            event_id=found_event.get("event_id"),
            details=(
                f"zone={event_data['zone_id']}, "
                f"score={event_data['compound_risk_score']}, "
                f"level={event_data['risk_level']}"
            ),
            payload_summary=event_data,
        ))

        # Store for downstream
        self.__class__._compound_risk_event = found_event

    def test_step5_hazard_propagation_handler_processes(
        self, kafka_producer, populated_graph_repo,
    ):
        """Step 5: HazardPropagationEventHandler processes compound risk event."""
        t0 = time.monotonic()

        compound_event = getattr(self.__class__, "_compound_risk_event", None)
        assert compound_event is not None, (
            "Step 4 must run first to capture compound risk event"
        )

        # Build handler with real publisher and graph repo
        engine = HazardPropagationEngine(graph_repo=populated_graph_repo)
        hp_publisher = HazardPropagationPublisher(kafka_producer)

        handler = HazardPropagationEventHandler(
            propagation_engine=engine,
            publisher=hp_publisher,
            graph_repo=populated_graph_repo,
        )

        success = asyncio.get_event_loop().run_until_complete(
            handler.handle_event(
                KafkaTopics.COMPOUND_RISK_DETECTED,
                compound_event,
            )
        )
        kafka_producer.flush(timeout=10.0)

        assert success is True, "Handler must return True on success"
        assert handler.events_processed == 1, "events_processed must be 1"
        assert handler.propagations_executed == 1, (
            "propagations_executed must be 1"
        )

        latency = (time.monotonic() - t0) * 1000
        _mode1_result.steps.append(StepResult(
            step_number=5,
            step_name="Hazard Propagation handler processes event",
            passed=True,
            latency_ms=latency,
            details=(
                f"processed={handler.events_processed}, "
                f"propagations={handler.propagations_executed}"
            ),
        ))

    def test_step6_verify_hazard_propagated_published(
        self, kafka_bootstrap, verification_consumer_factory,
    ):
        """Step 6: Verify hazard.propagated was published to Kafka."""
        t0 = time.monotonic()

        consumer = verification_consumer_factory(
            [KafkaTopics.HAZARD_PROPAGATED],
        )

        found_event = None
        for msg in consumer:
            data = msg.value
            if (
                data.get("event_type") == KafkaTopics.HAZARD_PROPAGATED
                and isinstance(data.get("data"), dict)
                and data["data"].get("origin_zone") == "ZONE_A"
            ):
                found_event = data
                break

        assert found_event is not None, (
            "hazard.propagated event not found on Kafka"
        )

        # Validate propagation data
        event_data = found_event["data"]
        assert "propagation_id" in event_data
        assert "hazard_type" in event_data
        assert "origin_zone" in event_data
        assert "affected_zones" in event_data
        assert "propagation_level" in event_data
        assert event_data["origin_zone"] == "ZONE_A"
        assert isinstance(event_data["affected_zones"], list)

        latency = (time.monotonic() - t0) * 1000
        _mode1_result.steps.append(StepResult(
            step_number=6,
            step_name="Verify hazard.propagated published",
            passed=True,
            latency_ms=latency,
            event_id=found_event.get("event_id"),
            details=(
                f"origin={event_data['origin_zone']}, "
                f"type={event_data['hazard_type']}, "
                f"level={event_data['propagation_level']}, "
                f"affected={len(event_data['affected_zones'])} zones"
            ),
            payload_summary=event_data,
        ))

    def test_step7_verify_offsets_advanced(self, kafka_bootstrap):
        """Step 7: Verify Kafka offsets have advanced."""
        t0 = time.monotonic()

        offsets = get_topic_offsets(
            kafka_bootstrap,
            [
                KafkaTopics.SENSOR_READING_ANOMALY,
                KafkaTopics.COMPOUND_RISK_DETECTED,
                KafkaTopics.HAZARD_PROPAGATED,
            ],
        )

        all_advanced = True
        for topic, partitions in offsets.items():
            for partition, offset in partitions.items():
                snapshot = OffsetSnapshot(
                    topic=topic,
                    partition=partition,
                    start_offset=0,
                    end_offset=offset,
                )
                _mode1_result.offsets.append(snapshot)
                if offset <= 0:
                    all_advanced = False

        assert all_advanced, f"Some topic offsets did not advance: {offsets}"

        latency = (time.monotonic() - t0) * 1000
        _mode1_result.steps.append(StepResult(
            step_number=7,
            step_name="Verify Kafka offsets advanced",
            passed=True,
            latency_ms=latency,
            details=f"Checked {len(offsets)} topics",
        ))

        # Finalize mode result
        _mode1_result.total_latency_ms = sum(
            s.latency_ms for s in _mode1_result.steps
        )
        _mode1_result.e2e_latency_ms = _mode1_result.total_latency_ms
        _mode1_result.completed_at = datetime.now(timezone.utc).isoformat()


def get_mode1_result() -> ModeResult:
    """Return the Mode 1 result for report generation."""
    if not _mode1_result.started_at:
        _mode1_result.started_at = datetime.now(timezone.utc).isoformat()
    return _mode1_result
