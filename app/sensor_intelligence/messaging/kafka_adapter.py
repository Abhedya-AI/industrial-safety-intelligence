"""Kafka-backed EventPublisher adapter for Sensor Intelligence.

Bridges the existing ``EventPublisher`` interface (used by AlertService)
to the shared Kafka producer infrastructure. No business logic is
modified — the AlertService continues to call ``publisher.publish()``
exactly as before, but events now flow through the shared Kafka producer.

This adapter:
  - Maps the legacy ``"alerts"`` topic to ``KafkaTopics.ALERT_CREATED``
  - Wraps events in the PS-1 v2.0 standard envelope
  - Catches and logs publish failures (never crashes the caller)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.sensor_intelligence.repositories.noop_publisher import EventPublisher
from app.shared.messaging.producer import KafkaEventProducer
from app.shared.messaging.topics import KafkaTopics

logger = logging.getLogger(__name__)

SOURCE_SYSTEM = "sensor_intelligence"

# Map legacy topic names used by AlertService to standard Kafka topics
_TOPIC_MAP: Dict[str, str] = {
    "alerts": KafkaTopics.ALERT_CREATED,
}


class KafkaEventPublisherAdapter(EventPublisher):
    """Adapts the SI EventPublisher interface to the shared Kafka producer.

    Drop-in replacement for ``NoOpPublisher`` — inject this into
    ``AlertService`` to route alert events through Kafka.

    Usage:
        adapter = KafkaEventPublisherAdapter(shared_producer)
        alert_service = AlertService(repo, adapter, config)
    """

    def __init__(self, producer: KafkaEventProducer) -> None:
        self._producer = producer
        self._published_count: int = 0
        self._failed_count: int = 0

    @property
    def published_count(self) -> int:
        return self._published_count

    @property
    def failed_count(self) -> int:
        return self._failed_count

    async def publish(self, topic: str, event: Dict[str, Any]) -> None:
        """Publish an event through the shared Kafka producer.

        Maps legacy topic names to standard KafkaTopics constants.
        Publishing failures are caught and logged.
        """
        kafka_topic = _TOPIC_MAP.get(topic, topic)

        try:
            # Extract a partition key from the event if available
            key = event.get("sensor_id") or event.get("zone_id")

            self._producer.publish(
                topic=kafka_topic,
                data=event,
                source_system=SOURCE_SYSTEM,
                key=key,
            )
            self._published_count += 1
            logger.info(
                "Published %s (mapped from '%s'): sensor=%s alert=%s",
                kafka_topic, topic,
                event.get("sensor_id", ""),
                event.get("alert_id", ""),
            )
        except Exception:
            self._failed_count += 1
            logger.exception(
                "Failed to publish %s (mapped from '%s')",
                kafka_topic, topic,
            )

    async def disconnect(self) -> None:
        """No-op — the shared producer manages its own lifecycle."""
        pass
