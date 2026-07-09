"""Sensor Intelligence event publisher.

Publishes domain events after successful business operations via the
shared Kafka producer. No new producer/schema/topic infrastructure is
created — everything is reused from ``app.shared.messaging``.

Events published:
  - sensor.reading.created    — after a reading is successfully stored
  - sensor.reading.anomaly    — after anomaly detection flags a reading
  - alert.created             — after an alert is generated
  - sensor.health.updated     — after sensor health is recalculated
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.shared.messaging.events import BaseEvent
from app.shared.messaging.producer import KafkaEventProducer
from app.shared.messaging.topics import KafkaTopics

logger = logging.getLogger(__name__)

SOURCE_SYSTEM = "sensor_intelligence"


class SensorIntelligencePublisher:
    """Publishes Sensor Intelligence domain events to Kafka.

    Reuses the shared ``KafkaEventProducer`` — does NOT create its own.
    All publishing failures are caught and logged — they must never
    crash the business operation that triggered them.

    Usage:
        publisher = SensorIntelligencePublisher(producer)
        publisher.publish_reading_created(reading_model)
    """

    def __init__(self, producer: KafkaEventProducer) -> None:
        self._producer = producer
        self._published_count: int = 0
        self._failed_count: int = 0

    # ── Properties ──

    @property
    def published_count(self) -> int:
        return self._published_count

    @property
    def failed_count(self) -> int:
        return self._failed_count

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # sensor.reading.created
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def publish_reading_created(
        self,
        reading_id: str,
        sensor_id: str,
        value: float,
        timestamp: Optional[datetime] = None,
        anomaly_score: float = 0.0,
        anomaly_status: str = "NORMAL",
        confidence: float = 100.0,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
    ) -> Optional[BaseEvent]:
        """Publish a ``sensor.reading.created`` event.

        Called after a reading is successfully persisted.
        """
        data: Dict[str, Any] = {
            "reading_id": reading_id,
            "sensor_id": sensor_id,
            "value": value,
            "timestamp": (
                timestamp.isoformat() if timestamp
                else datetime.now(timezone.utc).isoformat()
            ),
            "anomaly_score": anomaly_score,
            "anomaly_status": anomaly_status,
            "confidence": confidence,
        }
        if zone_id:
            data["zone_id"] = zone_id
        if equipment_id:
            data["equipment_id"] = equipment_id

        return self._publish(
            KafkaTopics.SENSOR_READING_CREATED, data, key=sensor_id,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # sensor.reading.anomaly
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def publish_reading_anomaly(
        self,
        reading_id: str,
        sensor_id: str,
        value: float,
        anomaly_score: float,
        anomaly_status: str,
        sensor_type: Optional[str] = None,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
    ) -> Optional[BaseEvent]:
        """Publish a ``sensor.reading.anomaly`` event.

        Called after anomaly detection identifies an anomaly (score above
        threshold or status != NORMAL).
        """
        data: Dict[str, Any] = {
            "reading_id": reading_id,
            "sensor_id": sensor_id,
            "value": value,
            "anomaly_score": anomaly_score,
            "anomaly_status": anomaly_status,
        }
        if sensor_type:
            data["sensor_type"] = sensor_type
        if zone_id:
            data["zone_id"] = zone_id
        if equipment_id:
            data["equipment_id"] = equipment_id

        return self._publish(
            KafkaTopics.SENSOR_READING_ANOMALY, data, key=sensor_id,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # alert.created
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def publish_alert_created(
        self,
        alert_id: str,
        sensor_id: str,
        alert_level: str,
        title: str,
        description: str = "",
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
    ) -> Optional[BaseEvent]:
        """Publish an ``alert.created`` event.

        Called after an alert is successfully persisted.
        """
        data: Dict[str, Any] = {
            "alert_id": alert_id,
            "sensor_id": sensor_id,
            "alert_level": alert_level,
            "title": title,
            "description": description,
        }
        if zone_id:
            data["zone_id"] = zone_id
        if equipment_id:
            data["equipment_id"] = equipment_id

        return self._publish(
            KafkaTopics.ALERT_CREATED, data, key=sensor_id,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # sensor.health.updated
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def publish_health_updated(
        self,
        sensor_id: str,
        health_score: float,
        health_status: str,
        calibration_score: float = 0.0,
        anomaly_score: float = 0.0,
        uptime_score: float = 0.0,
        zone_id: Optional[str] = None,
    ) -> Optional[BaseEvent]:
        """Publish a ``sensor.health.updated`` event.

        Called after sensor health is recalculated and persisted.
        """
        data: Dict[str, Any] = {
            "sensor_id": sensor_id,
            "health_score": health_score,
            "health_status": health_status,
            "calibration_score": calibration_score,
            "anomaly_score": anomaly_score,
            "uptime_score": uptime_score,
        }
        if zone_id:
            data["zone_id"] = zone_id

        return self._publish(
            KafkaTopics.SENSOR_HEALTH_UPDATED, data, key=sensor_id,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _publish(
        self,
        topic: str,
        data: Dict[str, Any],
        key: Optional[str] = None,
    ) -> Optional[BaseEvent]:
        """Publish an event with error handling.

        Publishing failures are caught and logged — they must NEVER
        crash the caller.
        """
        try:
            event = self._producer.publish(
                topic=topic,
                data=data,
                source_system=SOURCE_SYSTEM,
                key=key,
            )
            self._published_count += 1
            logger.info("Published %s: %s", topic, data.get("sensor_id", ""))
            return event
        except Exception:
            self._failed_count += 1
            logger.exception("Failed to publish %s", topic)
            return None
