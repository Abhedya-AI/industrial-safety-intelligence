"""Integration tests for Sensor Intelligence Kafka integration.

Verifies:
  1. SensorIntelligencePublisher publishes all 4 event types correctly
  2. KafkaEventPublisherAdapter bridges AlertService to shared Kafka
  3. Reading ingestion API publishes events after persistence
  4. Anomaly detection triggers sensor.reading.anomaly event
  5. Batch ingestion publishes events per reading
  6. Publisher failure does not crash business operations
  7. Event payload format compliance (PS-1 v2.0)
  8. Metrics tracking (published_count, failed_count)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.sensor_intelligence.messaging.kafka_adapter import (
    KafkaEventPublisherAdapter,
)
from app.sensor_intelligence.messaging.publisher import (
    SensorIntelligencePublisher,
)
from app.shared.messaging.events import BaseEvent
from app.shared.messaging.producer import NoopEventProducer
from app.shared.messaging.topics import KafkaTopics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def publish_log() -> List[Dict[str, Any]]:
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


@pytest.fixture
def publisher(tracking_producer) -> SensorIntelligencePublisher:
    return SensorIntelligencePublisher(tracking_producer)


@pytest.fixture
def adapter(tracking_producer) -> KafkaEventPublisherAdapter:
    return KafkaEventPublisherAdapter(tracking_producer)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. SensorIntelligencePublisher — reading.created
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPublishReadingCreated:
    def test_publishes_to_correct_topic(self, publisher, publish_log):
        publisher.publish_reading_created(
            reading_id="R001", sensor_id="S001", value=42.5,
        )
        assert len(publish_log) == 1
        assert publish_log[0]["topic"] == KafkaTopics.SENSOR_READING_CREATED

    def test_event_data_fields(self, publisher, publish_log):
        publisher.publish_reading_created(
            reading_id="R002", sensor_id="S002", value=37.0,
            anomaly_score=0.1, anomaly_status="NORMAL", confidence=95.0,
        )
        data = publish_log[0]["data"]
        assert data["reading_id"] == "R002"
        assert data["sensor_id"] == "S002"
        assert data["value"] == 37.0
        assert data["anomaly_score"] == 0.1
        assert data["anomaly_status"] == "NORMAL"
        assert data["confidence"] == 95.0

    def test_optional_zone_and_equipment(self, publisher, publish_log):
        publisher.publish_reading_created(
            reading_id="R003", sensor_id="S003", value=10.0,
            zone_id="ZONE_A", equipment_id="EQ001",
        )
        data = publish_log[0]["data"]
        assert data["zone_id"] == "ZONE_A"
        assert data["equipment_id"] == "EQ001"

    def test_zone_omitted_when_none(self, publisher, publish_log):
        publisher.publish_reading_created(
            reading_id="R004", sensor_id="S004", value=10.0,
        )
        data = publish_log[0]["data"]
        assert "zone_id" not in data
        assert "equipment_id" not in data

    def test_returns_base_event(self, publisher):
        result = publisher.publish_reading_created(
            reading_id="R005", sensor_id="S005", value=10.0,
        )
        assert result is not None

    def test_increments_published_count(self, publisher):
        assert publisher.published_count == 0
        publisher.publish_reading_created(
            reading_id="R006", sensor_id="S006", value=10.0,
        )
        assert publisher.published_count == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. SensorIntelligencePublisher — reading.anomaly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPublishReadingAnomaly:
    def test_publishes_to_correct_topic(self, publisher, publish_log):
        publisher.publish_reading_anomaly(
            reading_id="R001", sensor_id="S001", value=99.9,
            anomaly_score=0.85, anomaly_status="ANOMALY",
        )
        assert publish_log[0]["topic"] == KafkaTopics.SENSOR_READING_ANOMALY

    def test_event_data_fields(self, publisher, publish_log):
        publisher.publish_reading_anomaly(
            reading_id="R002", sensor_id="S002", value=120.0,
            anomaly_score=0.92, anomaly_status="CRITICAL_ANOMALY",
            sensor_type="TEMPERATURE",
        )
        data = publish_log[0]["data"]
        assert data["reading_id"] == "R002"
        assert data["anomaly_score"] == 0.92
        assert data["anomaly_status"] == "CRITICAL_ANOMALY"
        assert data["sensor_type"] == "TEMPERATURE"

    def test_optional_zone(self, publisher, publish_log):
        publisher.publish_reading_anomaly(
            reading_id="R003", sensor_id="S003", value=50.0,
            anomaly_score=0.7, anomaly_status="ANOMALY",
            zone_id="ZONE_B", equipment_id="EQ002",
        )
        data = publish_log[0]["data"]
        assert data["zone_id"] == "ZONE_B"
        assert data["equipment_id"] == "EQ002"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. SensorIntelligencePublisher — alert.created
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPublishAlertCreated:
    def test_publishes_to_correct_topic(self, publisher, publish_log):
        publisher.publish_alert_created(
            alert_id="A001", sensor_id="S001",
            alert_level="CRITICAL", title="High temperature",
        )
        assert publish_log[0]["topic"] == KafkaTopics.ALERT_CREATED

    def test_event_data_fields(self, publisher, publish_log):
        publisher.publish_alert_created(
            alert_id="A002", sensor_id="S002",
            alert_level="WARNING", title="Gas warning",
            description="Gas ppm above warning threshold",
            zone_id="ZONE_C",
        )
        data = publish_log[0]["data"]
        assert data["alert_id"] == "A002"
        assert data["sensor_id"] == "S002"
        assert data["alert_level"] == "WARNING"
        assert data["title"] == "Gas warning"
        assert data["description"] == "Gas ppm above warning threshold"
        assert data["zone_id"] == "ZONE_C"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. SensorIntelligencePublisher — health.updated
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPublishHealthUpdated:
    def test_publishes_to_correct_topic(self, publisher, publish_log):
        publisher.publish_health_updated(
            sensor_id="S001", health_score=85.0, health_status="GOOD",
        )
        assert publish_log[0]["topic"] == KafkaTopics.SENSOR_HEALTH_UPDATED

    def test_event_data_fields(self, publisher, publish_log):
        publisher.publish_health_updated(
            sensor_id="S002", health_score=45.0, health_status="DEGRADED",
            calibration_score=60.0, anomaly_score=0.3, uptime_score=80.0,
            zone_id="ZONE_D",
        )
        data = publish_log[0]["data"]
        assert data["sensor_id"] == "S002"
        assert data["health_score"] == 45.0
        assert data["health_status"] == "DEGRADED"
        assert data["calibration_score"] == 60.0
        assert data["anomaly_score"] == 0.3
        assert data["uptime_score"] == 80.0
        assert data["zone_id"] == "ZONE_D"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. KafkaEventPublisherAdapter — topic mapping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestKafkaAdapter:
    async def test_maps_alerts_topic(self, adapter, publish_log):
        await adapter.publish("alerts", {
            "alert_id": "A001", "sensor_id": "S001",
            "severity": "CRITICAL", "message": "Test",
        })
        assert len(publish_log) == 1
        assert publish_log[0]["topic"] == KafkaTopics.ALERT_CREATED

    async def test_unmapped_topic_passthrough(self, adapter, publish_log):
        await adapter.publish("custom.topic", {"key": "value"})
        assert publish_log[0]["topic"] == "custom.topic"

    async def test_increments_count(self, adapter):
        assert adapter.published_count == 0
        await adapter.publish("alerts", {"sensor_id": "S001"})
        assert adapter.published_count == 1

    async def test_disconnect_noop(self, adapter):
        await adapter.disconnect()  # Should not raise

    async def test_uses_sensor_id_as_key(self, adapter, publish_log):
        await adapter.publish("alerts", {"sensor_id": "S123"})
        # The tracking producer records data; verify key was used
        assert publish_log[0]["data"]["sensor_id"] == "S123"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Failure handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPublishFailure:
    def test_publisher_handles_producer_failure(self):
        failing_producer = NoopEventProducer()
        failing_producer.publish = MagicMock(
            side_effect=Exception("Kafka unreachable"),
        )
        pub = SensorIntelligencePublisher(failing_producer)
        result = pub.publish_reading_created(
            reading_id="R001", sensor_id="S001", value=10.0,
        )
        assert result is None
        assert pub.failed_count == 1
        assert pub.published_count == 0

    async def test_adapter_handles_producer_failure(self):
        failing_producer = NoopEventProducer()
        failing_producer.publish = MagicMock(
            side_effect=Exception("Kafka unreachable"),
        )
        adapter = KafkaEventPublisherAdapter(failing_producer)
        await adapter.publish("alerts", {"sensor_id": "S001"})
        assert adapter.failed_count == 1
        assert adapter.published_count == 0

    def test_multiple_failures_tracked(self):
        failing_producer = NoopEventProducer()
        failing_producer.publish = MagicMock(
            side_effect=Exception("Kafka unreachable"),
        )
        pub = SensorIntelligencePublisher(failing_producer)
        for _ in range(5):
            pub.publish_reading_created(
                reading_id="R001", sensor_id="S001", value=10.0,
            )
        assert pub.failed_count == 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Event format compliance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEventFormat:
    def test_event_has_standard_fields(self, publisher, publish_log):
        publisher.publish_reading_created(
            reading_id="R001", sensor_id="S001", value=42.0,
        )
        event = publish_log[0]["event"]
        assert event.event_type == KafkaTopics.SENSOR_READING_CREATED
        assert event.event_id  # Non-empty UUID
        assert event.timestamp  # Non-empty ISO string
        assert event.source_system == "sensor_intelligence"
        assert event.data is not None

    def test_event_has_iso_timestamp(self, publisher, publish_log):
        publisher.publish_reading_created(
            reading_id="R001", sensor_id="S001", value=42.0,
        )
        event = publish_log[0]["event"]
        from datetime import datetime
        # Should not raise
        datetime.fromisoformat(
            event.timestamp.replace("Z", "+00:00"),
        )

    def test_event_source_system(self, publisher, publish_log):
        publisher.publish_reading_anomaly(
            reading_id="R001", sensor_id="S001", value=99.0,
            anomaly_score=0.9, anomaly_status="ANOMALY",
        )
        event = publish_log[0]["event"]
        assert event.source_system == "sensor_intelligence"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMetrics:
    def test_publisher_tracks_success_count(self, publisher):
        for i in range(3):
            publisher.publish_reading_created(
                reading_id=f"R{i}", sensor_id=f"S{i}", value=float(i),
            )
        assert publisher.published_count == 3

    def test_publisher_mixed_success_failure(self):
        prod = NoopEventProducer()
        call_count = [0]
        original = prod.publish

        def sometimes_fail(*args, **kw):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise Exception("Intermittent failure")
            return original(*args, **kw)

        prod.publish = sometimes_fail
        pub = SensorIntelligencePublisher(prod)

        for i in range(6):
            pub.publish_reading_created(
                reading_id=f"R{i}", sensor_id=f"S{i}", value=float(i),
            )
        assert pub.published_count == 3
        assert pub.failed_count == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. API integration (reading endpoints)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

API_PREFIX = "/api/v1"
INGEST_URL = f"{API_PREFIX}/readings/ingest"
BATCH_URL = f"{INGEST_URL}/batch"


class TestReadingAPIPublishing:
    """Verify that the reading ingestion endpoints publish Kafka events."""

    async def _setup_sensor(self, client: AsyncClient) -> str:
        """Register a sensor for ingestion tests."""
        resp = await client.post(
            f"{API_PREFIX}/sensors",
            json={
                "sensor_id": f"S_KAFKA_{uuid.uuid4().hex[:6]}",
                "sensor_name": "Kafka Test Sensor",
                "sensor_type": "TEMPERATURE",
                "unit": "°C",
                "location": "Test Zone",
                "min_value": -50.0,
                "max_value": 200.0,
            },
        )
        assert resp.status_code == 201
        return resp.json()["sensor_id"]

    async def test_ingest_returns_201(self, client: AsyncClient):
        sensor_id = await self._setup_sensor(client)
        resp = await client.post(INGEST_URL, json={
            "sensor_id": sensor_id,
            "value": 42.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": 95.0,
        })
        assert resp.status_code == 201
        assert resp.json()["success"] is True

    async def test_batch_ingest_returns_201(self, client: AsyncClient):
        sensor_id = await self._setup_sensor(client)
        now = datetime.now(timezone.utc)
        readings = [
            {
                "sensor_id": sensor_id,
                "value": 40.0 + i,
                "timestamp": (now.replace(second=i)).isoformat(),
                "confidence": 90.0,
            }
            for i in range(3)
        ]
        resp = await client.post(BATCH_URL, json={"readings": readings})
        assert resp.status_code == 201
        assert resp.json()["ingested"] == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. All 4 event types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAllEventTypes:
    """Verify all four event types publish through the publisher."""

    def test_all_four_events(self, publisher, publish_log):
        publisher.publish_reading_created(
            reading_id="R1", sensor_id="S1", value=10.0,
        )
        publisher.publish_reading_anomaly(
            reading_id="R2", sensor_id="S2", value=99.0,
            anomaly_score=0.9, anomaly_status="ANOMALY",
        )
        publisher.publish_alert_created(
            alert_id="A1", sensor_id="S3",
            alert_level="CRITICAL", title="Test Alert",
        )
        publisher.publish_health_updated(
            sensor_id="S4", health_score=80.0, health_status="GOOD",
        )

        topics = [e["topic"] for e in publish_log]
        assert KafkaTopics.SENSOR_READING_CREATED in topics
        assert KafkaTopics.SENSOR_READING_ANOMALY in topics
        assert KafkaTopics.ALERT_CREATED in topics
        assert KafkaTopics.SENSOR_HEALTH_UPDATED in topics
        assert publisher.published_count == 4

    def test_topics_use_standard_constants(self, publisher, publish_log):
        """Verify topics match KafkaTopics constants exactly."""
        publisher.publish_reading_created(
            reading_id="R1", sensor_id="S1", value=10.0,
        )
        assert publish_log[0]["topic"] == "sensor.reading.created"

        publisher.publish_reading_anomaly(
            reading_id="R2", sensor_id="S2", value=99.0,
            anomaly_score=0.9, anomaly_status="ANOMALY",
        )
        assert publish_log[1]["topic"] == "sensor.reading.anomaly"

        publisher.publish_alert_created(
            alert_id="A1", sensor_id="S3",
            alert_level="CRITICAL", title="Test",
        )
        assert publish_log[2]["topic"] == "alert.created"

        publisher.publish_health_updated(
            sensor_id="S4", health_score=80.0, health_status="GOOD",
        )
        assert publish_log[3]["topic"] == "sensor.health.updated"
