"""Unit tests for shared Kafka messaging infrastructure.

Tests cover:
  - Topic constants (PS-1 v2.0 compliance)
  - BaseEvent schema
  - Serialization / deserialization utilities
  - KafkaEventProducer (noop mode)
  - KafkaEventConsumer (noop mode, handler registration)
  - Edge cases (malformed data, missing fields)

Note: These tests do NOT require a running Kafka broker. Producer and
consumer are tested in noop mode. Serialization tests are pure unit tests.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum

import pytest
from pydantic import BaseModel

from app.shared.messaging.events import BaseEvent, create_event
from app.shared.messaging.serialization import (
    DeserializationError,
    EventJSONEncoder,
    SerializationError,
    deserialize_dict,
    deserialize_event,
    deserialize_to_model,
    kafka_key_deserializer,
    kafka_key_serializer,
    kafka_value_deserializer,
    kafka_value_serializer,
    serialize_dict,
    serialize_event,
)
from app.shared.messaging.topics import KafkaTopics
from app.shared.messaging.producer import KafkaEventProducer, NoopEventProducer
from app.shared.messaging.consumer import KafkaEventConsumer, NoopEventConsumer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Topic constants — PS-1 v2.0 compliance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTopics:
    def test_sensor_topics_exist(self):
        assert KafkaTopics.SENSOR_READING_CREATED == "sensor.reading.created"
        assert KafkaTopics.SENSOR_READING_ANOMALY == "sensor.reading.anomaly"
        assert KafkaTopics.SENSOR_STATUS_CHANGED == "sensor.status.changed"

    def test_risk_topics_exist(self):
        assert KafkaTopics.RISK_ASSESSMENT_GENERATED == "risk.assessment.generated"
        assert KafkaTopics.RISK_SCORE_UPDATED == "risk.score.updated"
        assert KafkaTopics.RISK_THRESHOLD_EXCEEDED == "risk.threshold.exceeded"
        assert KafkaTopics.COMPOUND_RISK_DETECTED == "compound.risk.detected"

    def test_vision_topics_exist(self):
        assert KafkaTopics.VISION_EVENT_DETECTED == "vision.event.detected"
        assert KafkaTopics.VISION_WORKER_LOCATED == "vision.worker.located"

    def test_incident_topics_exist(self):
        assert KafkaTopics.INCIDENT_CREATED == "incident.created"
        assert KafkaTopics.INCIDENT_UPDATED == "incident.updated"
        assert KafkaTopics.INCIDENT_RESOLVED == "incident.resolved"

    def test_emergency_topics_exist(self):
        assert KafkaTopics.EMERGENCY_TRIGGERED == "emergency.triggered"
        assert KafkaTopics.EVACUATION_INITIATED == "evacuation.initiated"
        assert KafkaTopics.ALL_CLEAR_SIGNAL == "all.clear.signal"

    def test_alert_topics_exist(self):
        assert KafkaTopics.ALERT_CREATED == "alert.created"
        assert KafkaTopics.ALERT_ACKNOWLEDGED == "alert.acknowledged"

    def test_permit_topics_exist(self):
        assert KafkaTopics.PERMIT_CREATED == "permit.created"
        assert KafkaTopics.PERMIT_APPROVED == "permit.approved"
        assert KafkaTopics.PERMIT_ACTIVATED == "permit.activated"
        assert KafkaTopics.PERMIT_EXPIRED == "permit.expired"
        assert KafkaTopics.PERMIT_REVOKED == "permit.revoked"

    def test_maintenance_topics_exist(self):
        assert KafkaTopics.MAINTENANCE_CREATED == "maintenance.created"
        assert KafkaTopics.MAINTENANCE_STARTED == "maintenance.started"
        assert KafkaTopics.MAINTENANCE_COMPLETED == "maintenance.completed"

    def test_agent_topics_exist(self):
        assert KafkaTopics.AGENT_DECISION_GENERATED == "agent.decision.generated"
        assert KafkaTopics.ROOT_CAUSE_ANALYSIS_COMPLETED == "root.cause.analysis.completed"

    def test_all_topics_snake_case_with_dots(self):
        """PS-1 v2.0 §5.2: snake_case with dots."""
        for topic in KafkaTopics.all_topics():
            assert "." in topic
            assert topic == topic.lower()
            assert " " not in topic

    def test_all_topics_helper(self):
        topics = KafkaTopics.all_topics()
        assert len(topics) >= 30

    def test_sensor_topics_helper(self):
        assert len(KafkaTopics.sensor_topics()) == 3

    def test_risk_topics_helper(self):
        assert len(KafkaTopics.risk_topics()) == 4

    def test_alert_topics_helper(self):
        assert len(KafkaTopics.alert_topics()) == 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. BaseEvent schema
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBaseEvent:
    def test_create_with_defaults(self):
        event = BaseEvent(event_type="sensor.reading.created")
        assert event.event_type == "sensor.reading.created"
        assert event.event_id is not None
        uuid.UUID(event.event_id)  # Valid UUID
        assert event.timestamp is not None
        assert event.source_system == "sentinel_ai"
        assert event.data == {}
        assert event.version == "1.0"

    def test_create_with_all_fields(self):
        event = BaseEvent(
            event_type="risk.score.updated",
            event_id="custom-id",
            timestamp="2026-07-08T00:00:00Z",
            source_system="risk_prediction",
            data={"score": 85.0},
            correlation_id="corr-123",
            version="2.0",
        )
        assert event.event_type == "risk.score.updated"
        assert event.event_id == "custom-id"
        assert event.source_system == "risk_prediction"
        assert event.data["score"] == 85.0
        assert event.correlation_id == "corr-123"

    def test_create_event_factory(self):
        event = create_event(
            event_type="alert.created",
            data={"alert_id": "A001", "severity": "HIGH"},
            source_system="compound_risk",
            correlation_id="corr-456",
        )
        assert event.event_type == "alert.created"
        assert event.data["alert_id"] == "A001"
        assert event.source_system == "compound_risk"
        assert event.correlation_id == "corr-456"
        uuid.UUID(event.event_id)

    def test_event_model_dump(self):
        event = create_event(
            event_type="sensor.reading.created",
            data={"sensor_id": "S001"},
        )
        d = event.model_dump()
        assert d["event_type"] == "sensor.reading.created"
        assert d["data"]["sensor_id"] == "S001"
        assert "event_id" in d
        assert "timestamp" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. JSON encoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestJSONEncoder:
    def test_datetime(self):
        dt = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)
        result = json.dumps({"ts": dt}, cls=EventJSONEncoder)
        assert "2026-07-08" in result

    def test_uuid(self):
        u = uuid.uuid4()
        result = json.dumps({"id": u}, cls=EventJSONEncoder)
        assert str(u) in result

    def test_enum(self):
        class Color(Enum):
            RED = "red"
        result = json.dumps({"c": Color.RED}, cls=EventJSONEncoder)
        assert '"red"' in result

    def test_pydantic_model(self):
        class Sub(BaseModel):
            x: int = 1
        result = json.dumps({"m": Sub()}, cls=EventJSONEncoder)
        assert '"x": 1' in result

    def test_set(self):
        result = json.dumps({"s": {1, 2, 3}}, cls=EventJSONEncoder)
        parsed = json.loads(result)
        assert sorted(parsed["s"]) == [1, 2, 3]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Serialization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSerialization:
    def test_serialize_event(self):
        event = create_event("test.event", {"key": "value"})
        raw = serialize_event(event)
        assert isinstance(raw, bytes)
        parsed = json.loads(raw.decode("utf-8"))
        assert parsed["event_type"] == "test.event"
        assert parsed["data"]["key"] == "value"

    def test_serialize_dict(self):
        raw = serialize_dict({"sensor_id": "S001", "value": 42.5})
        parsed = json.loads(raw.decode("utf-8"))
        assert parsed["sensor_id"] == "S001"

    def test_serialize_dict_with_datetime(self):
        dt = datetime(2026, 7, 8, tzinfo=timezone.utc)
        raw = serialize_dict({"ts": dt})
        parsed = json.loads(raw.decode("utf-8"))
        assert "2026-07-08" in parsed["ts"]

    def test_kafka_value_serializer_event(self):
        event = create_event("test", {"a": 1})
        raw = kafka_value_serializer(event)
        assert isinstance(raw, bytes)

    def test_kafka_value_serializer_dict(self):
        raw = kafka_value_serializer({"a": 1})
        assert isinstance(raw, bytes)

    def test_kafka_value_serializer_unsupported(self):
        with pytest.raises(SerializationError):
            kafka_value_serializer(12345)

    def test_kafka_key_serializer(self):
        assert kafka_key_serializer("S001") == b"S001"
        assert kafka_key_serializer(None) is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Deserialization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeserialization:
    def test_deserialize_event(self):
        event = create_event("test.event", {"x": 1})
        raw = serialize_event(event)
        parsed = deserialize_event(raw)
        assert parsed.event_type == "test.event"
        assert parsed.data["x"] == 1

    def test_deserialize_dict(self):
        raw = b'{"sensor_id": "S001", "value": 42.5}'
        result = deserialize_dict(raw)
        assert result["sensor_id"] == "S001"

    def test_deserialize_to_model(self):
        class SensorReading(BaseModel):
            sensor_id: str
            value: float
        raw = b'{"sensor_id": "S001", "value": 42.5}'
        result = deserialize_to_model(raw, SensorReading)
        assert result.sensor_id == "S001"
        assert result.value == 42.5

    def test_deserialize_invalid_json(self):
        with pytest.raises(DeserializationError):
            deserialize_event(b"not json at all")

    def test_deserialize_dict_invalid(self):
        with pytest.raises(DeserializationError):
            deserialize_dict(b"{invalid json}")

    def test_kafka_value_deserializer(self):
        raw = b'{"event_type": "test", "data": {}}'
        result = kafka_value_deserializer(raw)
        assert result["event_type"] == "test"

    def test_kafka_key_deserializer(self):
        assert kafka_key_deserializer(b"S001") == "S001"
        assert kafka_key_deserializer(None) is None

    def test_roundtrip_event(self):
        """Serialize → Deserialize roundtrip preserves data."""
        original = create_event(
            "sensor.reading.created",
            {"sensor_id": "S001", "value": 72.5},
            source_system="sensor_intelligence",
            correlation_id="corr-789",
        )
        raw = serialize_event(original)
        restored = deserialize_event(raw)
        assert restored.event_type == original.event_type
        assert restored.event_id == original.event_id
        assert restored.data["sensor_id"] == "S001"
        assert restored.correlation_id == "corr-789"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Producer (noop mode)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNoopProducer:
    def test_noop_producer_not_connected(self):
        producer = NoopEventProducer()
        assert producer.is_connected is False
        assert producer.is_enabled is False

    def test_noop_publish_returns_event(self):
        producer = NoopEventProducer()
        event = producer.publish(
            topic=KafkaTopics.SENSOR_READING_CREATED,
            data={"sensor_id": "S001", "value": 42.5},
            source_system="sensor_intelligence",
        )
        assert event is not None
        assert event.event_type == KafkaTopics.SENSOR_READING_CREATED
        assert event.data["sensor_id"] == "S001"

    def test_noop_publish_with_key_and_correlation(self):
        producer = NoopEventProducer()
        event = producer.publish(
            topic=KafkaTopics.ALERT_CREATED,
            data={"severity": "HIGH"},
            key="ZONE_A",
            correlation_id="corr-123",
        )
        assert event is not None
        assert event.correlation_id == "corr-123"

    def test_disabled_producer(self):
        producer = KafkaEventProducer(enabled=False)
        assert producer.is_enabled is False
        event = producer.publish("test.topic", {"x": 1})
        assert event is not None

    def test_flush_noop(self):
        producer = NoopEventProducer()
        producer.flush()  # Should not raise

    def test_close_noop(self):
        producer = NoopEventProducer()
        producer.close()  # Should not raise


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Consumer (noop mode)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNoopConsumer:
    def test_noop_consumer_not_connected(self):
        consumer = NoopEventConsumer()
        assert consumer.is_connected is False
        assert consumer.is_enabled is False

    def test_register_handler(self):
        consumer = NoopEventConsumer()
        handler_called = []
        consumer.register_handler(
            KafkaTopics.SENSOR_READING_CREATED,
            lambda topic, data: handler_called.append((topic, data)),
        )
        # Dispatch manually
        consumer._dispatch(KafkaTopics.SENSOR_READING_CREATED, {"x": 1})
        assert len(handler_called) == 1
        assert handler_called[0][0] == KafkaTopics.SENSOR_READING_CREATED

    def test_register_default_handler(self):
        consumer = NoopEventConsumer()
        received = []
        consumer.register_default_handler(
            lambda topic, data: received.append(topic),
        )
        consumer._dispatch("any.topic", {})
        assert "any.topic" in received

    def test_multiple_handlers_per_topic(self):
        consumer = NoopEventConsumer()
        calls = []
        consumer.register_handler("test.topic", lambda t, d: calls.append("h1"))
        consumer.register_handler("test.topic", lambda t, d: calls.append("h2"))
        consumer._dispatch("test.topic", {})
        assert calls == ["h1", "h2"]

    def test_handler_error_does_not_crash(self):
        consumer = NoopEventConsumer()
        consumer.register_handler(
            "test.topic",
            lambda t, d: (_ for _ in ()).throw(ValueError("boom")),
        )
        # Should not raise — error is logged
        consumer._dispatch("test.topic", {})

    def test_consume_one_noop(self):
        consumer = NoopEventConsumer()
        assert consumer.consume_one() is None

    def test_start_noop(self):
        consumer = NoopEventConsumer()
        consumer.start()  # Should return immediately

    def test_close_noop(self):
        consumer = NoopEventConsumer()
        consumer.close()  # Should not raise

    def test_disabled_consumer(self):
        consumer = KafkaEventConsumer(enabled=False)
        assert consumer.is_enabled is False
        assert consumer.consume_one() is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. PS-1 v2.0 Kafka message format compliance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPS1Compliance:
    def test_event_has_required_fields(self):
        """§5.3: event_type, event_id, timestamp, source_system, data."""
        event = create_event("sensor.reading.created", {"sensor_id": "S001"})
        d = event.model_dump()
        assert "event_type" in d
        assert "event_id" in d
        assert "timestamp" in d
        assert "source_system" in d
        assert "data" in d

    def test_timestamp_is_iso8601(self):
        event = create_event("test", {})
        # Should parse as ISO 8601
        datetime.fromisoformat(event.timestamp)

    def test_event_id_is_uuid(self):
        event = create_event("test", {})
        uuid.UUID(event.event_id)

    def test_serialized_format(self):
        """The serialized message must have all §5.3 fields."""
        event = create_event(
            "risk.score.updated",
            {"score": 85.0, "zone_id": "ZONE_A"},
            source_system="risk_prediction",
        )
        raw = serialize_event(event)
        parsed = json.loads(raw)
        assert parsed["event_type"] == "risk.score.updated"
        assert parsed["source_system"] == "risk_prediction"
        assert parsed["data"]["score"] == 85.0
        uuid.UUID(parsed["event_id"])
        datetime.fromisoformat(parsed["timestamp"])
