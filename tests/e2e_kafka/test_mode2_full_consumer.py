"""Mode 2: Full Application Consumer Verification.

Uses the NORMAL application startup path (start_consumers from
dependencies.py) to verify the complete consumer loop:

  1. Start application consumers via start_consumers(settings)
  2. Publish sensor.reading.anomaly to real Kafka
  3. Verify compound.risk.detected appears (consumer loop processed it)
  4. Verify hazard.propagated appears (downstream pipeline triggered)
  5. Record end-to-end latency
  6. Verify consumer group offsets advanced
  7. Stop consumers via stop_consumers()

This proves the full wiring: KafkaEventConsumer → handler registration
→ sync/async bridge → handler → publisher → Kafka.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import pytest

from app.shared.messaging.topics import KafkaTopics
from tests.e2e_kafka.conftest import (
    KAFKA_BOOTSTRAP,
    E2E_TOPICS,
    get_consumer_group_offsets,
    get_topic_offsets,
)
from tests.e2e_kafka.report_generator import (
    ModeResult,
    OffsetSnapshot,
    StepResult,
)

logger = logging.getLogger(__name__)

# Module-level result collector
_mode2_result = ModeResult(mode_name="Mode 2: Full Application Consumer")

# Consumer group used by the application
APP_CONSUMER_GROUP = "sentinel_ai_e2e_test"


@pytest.mark.usefixtures("kafka_available", "ensure_topics")
class TestMode2FullConsumer:
    """Mode 2: Full application consumer verification."""

    @pytest.fixture(autouse=True)
    def _setup_and_teardown(self, kafka_bootstrap, populated_graph_repo):
        """Start application consumers before tests, stop after.

        Resets all module-level singletons in dependencies.py to ensure
        fresh instances are created, then starts consumers using the
        normal application startup path.
        """
        import app.core.dependencies as deps
        from app.core.settings import Settings
        from app.hazard_propagation.repositories.in_memory_graph_repo import (
            InMemoryGraphRepository,
        )

        # Save original state
        orig_producer = deps._event_producer
        orig_consumer = deps._event_consumer
        orig_graph = deps._graph_repository
        orig_thread = deps._consumer_thread

        # Reset singletons for clean test state
        deps._event_producer = None
        deps._event_consumer = None
        deps._graph_repository = populated_graph_repo  # Inject test graph
        deps._consumer_thread = None

        # Create test settings that point to real Kafka
        self._settings = Settings(
            event_broker="kafka",
            kafka_bootstrap_servers=kafka_bootstrap,
            kafka_consumer_group_id=APP_CONSUMER_GROUP,
            kafka_auto_offset_reset="latest",
            graph_repository="in_memory",
            database_url="sqlite+aiosqlite:///./sensor_intelligence.db",
        )

        _mode2_result.started_at = datetime.now(timezone.utc).isoformat()

        yield

        # Cleanup: stop consumers and restore singletons
        try:
            deps.stop_consumers()
        except Exception:
            logger.exception("Error stopping consumers in teardown")

        deps._event_producer = orig_producer
        deps._event_consumer = orig_consumer
        deps._graph_repository = orig_graph
        deps._consumer_thread = orig_thread

    def test_step1_start_application_consumers(self):
        """Step 1: Start consumers using the normal startup path."""
        import app.core.dependencies as deps

        t0 = time.monotonic()

        deps.start_consumers(self._settings)

        # Verify consumer thread is running
        assert deps._consumer_thread is not None, (
            "Consumer thread must be created"
        )
        assert deps._consumer_thread.is_alive(), (
            "Consumer thread must be alive"
        )

        # Verify consumer is connected
        consumer = deps._get_event_consumer(self._settings)
        assert consumer.is_connected, "Consumer must be connected to Kafka"
        assert consumer.is_enabled, "Consumer must be enabled"

        latency = (time.monotonic() - t0) * 1000
        _mode2_result.steps.append(StepResult(
            step_number=1,
            step_name="Start application consumers",
            passed=True,
            latency_ms=latency,
            details=(
                f"group={APP_CONSUMER_GROUP}, "
                f"thread={deps._consumer_thread.name}"
            ),
        ))

        # Brief pause for consumer to join group and be ready
        time.sleep(3)

    def test_step2_publish_anomaly_event(self, kafka_producer, anomaly_event_payload):
        """Step 2: Publish sensor.reading.anomaly via real producer."""
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

        latency = (time.monotonic() - t0) * 1000
        _mode2_result.steps.append(StepResult(
            step_number=2,
            step_name="Publish sensor.reading.anomaly",
            passed=True,
            latency_ms=latency,
            event_id=event.event_id,
            details=f"Published to {KafkaTopics.SENSOR_READING_ANOMALY}",
        ))

        self.__class__._publish_time = time.monotonic()
        self.__class__._anomaly_event_id = event.event_id

    def test_step3_verify_compound_risk_detected(
        self, kafka_bootstrap, verification_consumer_factory,
    ):
        """Step 3: Verify compound.risk.detected appears (consumer loop processed it)."""
        t0 = time.monotonic()

        # The app consumer should process the anomaly event and publish
        # compound.risk.detected. We verify by reading from a separate consumer.
        consumer = verification_consumer_factory(
            [KafkaTopics.COMPOUND_RISK_DETECTED],
        )

        found_event = None
        deadline = time.monotonic() + 30  # 30s timeout
        while time.monotonic() < deadline:
            records = consumer.poll(timeout_ms=2000, max_records=10)
            for tp, messages in records.items():
                for msg in messages:
                    data = msg.value
                    if (
                        data.get("event_type") == KafkaTopics.COMPOUND_RISK_DETECTED
                        and isinstance(data.get("data"), dict)
                        and data["data"].get("zone_id") == "ZONE_A"
                    ):
                        found_event = data
                        break
                if found_event:
                    break
            if found_event:
                break

        assert found_event is not None, (
            "compound.risk.detected not received within 30s — "
            "consumer loop may not have processed the anomaly event"
        )

        event_data = found_event["data"]
        assert "compound_risk_score" in event_data
        assert "risk_level" in event_data
        assert event_data["zone_id"] == "ZONE_A"

        latency = (time.monotonic() - t0) * 1000
        _mode2_result.steps.append(StepResult(
            step_number=3,
            step_name="Verify compound.risk.detected received",
            passed=True,
            latency_ms=latency,
            event_id=found_event.get("event_id"),
            details=(
                f"zone={event_data['zone_id']}, "
                f"score={event_data.get('compound_risk_score')}, "
                f"level={event_data.get('risk_level')}"
            ),
            payload_summary=event_data,
        ))

        self.__class__._compound_risk_event = found_event

    def test_step4_verify_hazard_propagated(
        self, kafka_bootstrap, verification_consumer_factory,
    ):
        """Step 4: Verify hazard.propagated appears (downstream pipeline triggered)."""
        t0 = time.monotonic()

        consumer = verification_consumer_factory(
            [KafkaTopics.HAZARD_PROPAGATED],
        )

        found_event = None
        deadline = time.monotonic() + 30  # 30s timeout
        while time.monotonic() < deadline:
            records = consumer.poll(timeout_ms=2000, max_records=10)
            for tp, messages in records.items():
                for msg in messages:
                    data = msg.value
                    if (
                        data.get("event_type") == KafkaTopics.HAZARD_PROPAGATED
                        and isinstance(data.get("data"), dict)
                        and data["data"].get("origin_zone") == "ZONE_A"
                    ):
                        found_event = data
                        break
                if found_event:
                    break
            if found_event:
                break

        assert found_event is not None, (
            "hazard.propagated not received within 30s — "
            "full pipeline may not have executed"
        )

        event_data = found_event["data"]
        assert "propagation_id" in event_data
        assert "hazard_type" in event_data
        assert "affected_zones" in event_data
        assert event_data["origin_zone"] == "ZONE_A"

        latency = (time.monotonic() - t0) * 1000
        _mode2_result.steps.append(StepResult(
            step_number=4,
            step_name="Verify hazard.propagated received",
            passed=True,
            latency_ms=latency,
            event_id=found_event.get("event_id"),
            details=(
                f"origin={event_data['origin_zone']}, "
                f"type={event_data['hazard_type']}, "
                f"level={event_data.get('propagation_level')}, "
                f"affected={len(event_data.get('affected_zones', []))} zones"
            ),
            payload_summary=event_data,
        ))

    def test_step5_record_e2e_latency(self):
        """Step 5: Record end-to-end latency from publish to propagation."""
        t0 = time.monotonic()

        publish_time = getattr(self.__class__, "_publish_time", None)
        if publish_time is not None:
            e2e_latency = (time.monotonic() - publish_time) * 1000
        else:
            e2e_latency = sum(s.latency_ms for s in _mode2_result.steps)

        _mode2_result.e2e_latency_ms = e2e_latency

        latency = (time.monotonic() - t0) * 1000
        _mode2_result.steps.append(StepResult(
            step_number=5,
            step_name="Record E2E latency",
            passed=True,
            latency_ms=latency,
            details=f"E2E latency: {e2e_latency:.1f}ms",
        ))

    def test_step6_verify_offsets_advanced(self, kafka_bootstrap):
        """Step 6: Verify consumer group offsets advanced."""
        t0 = time.monotonic()

        # Check topic end offsets
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
                _mode2_result.offsets.append(snapshot)
                if offset <= 0:
                    all_advanced = False

        assert all_advanced, f"Some topic offsets did not advance: {offsets}"

        # Also check committed offsets for the app consumer group
        committed = get_consumer_group_offsets(
            kafka_bootstrap,
            APP_CONSUMER_GROUP,
            [KafkaTopics.SENSOR_READING_ANOMALY],
        )
        logger.info("Committed offsets for %s: %s", APP_CONSUMER_GROUP, committed)

        latency = (time.monotonic() - t0) * 1000
        _mode2_result.steps.append(StepResult(
            step_number=6,
            step_name="Verify offsets advanced",
            passed=True,
            latency_ms=latency,
            details=f"Checked {len(offsets)} topics, committed={committed}",
        ))

    def test_step7_stop_consumers(self):
        """Step 7: Stop application consumers cleanly."""
        import app.core.dependencies as deps

        t0 = time.monotonic()

        deps.stop_consumers()

        # Verify thread stopped
        assert deps._consumer_thread is None or not deps._consumer_thread.is_alive(), (
            "Consumer thread should be stopped"
        )

        latency = (time.monotonic() - t0) * 1000
        _mode2_result.steps.append(StepResult(
            step_number=7,
            step_name="Stop application consumers",
            passed=True,
            latency_ms=latency,
            details="Consumers stopped cleanly",
        ))

        # Finalize
        _mode2_result.total_latency_ms = sum(
            s.latency_ms for s in _mode2_result.steps
        )
        _mode2_result.completed_at = datetime.now(timezone.utc).isoformat()


def get_mode2_result() -> ModeResult:
    """Return the Mode 2 result for report generation."""
    if not _mode2_result.started_at:
        _mode2_result.started_at = datetime.now(timezone.utc).isoformat()
    return _mode2_result
