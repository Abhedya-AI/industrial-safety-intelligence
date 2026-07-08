"""Kafka producer abstraction.

Provides a ``KafkaEventProducer`` that wraps the kafka-python
``KafkaProducer`` with:
  - Automatic JSON serialization via shared utilities
  - BaseEvent envelope creation
  - Graceful degradation (noop mode when Kafka is unavailable)
  - Configurable via application Settings

All modules MUST use this producer — no direct ``KafkaProducer`` usage.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from app.shared.messaging.events import BaseEvent, create_event
from app.shared.messaging.serialization import (
    kafka_key_serializer,
    kafka_value_serializer,
    serialize_event,
)

logger = logging.getLogger(__name__)


class KafkaEventProducer:
    """Shared Kafka event producer.

    Wraps kafka-python's KafkaProducer with standardised serialization
    and BaseEvent envelope support. Falls back to noop mode when Kafka
    is not configured or unavailable.

    Args:
        bootstrap_servers: Kafka bootstrap server address(es).
        enabled: If False, operates in noop mode (logs events, doesn't send).
        on_delivery: Optional callback for delivery reports.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        enabled: bool = True,
        on_delivery: Optional[Callable] = None,
    ) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._enabled = enabled
        self._on_delivery = on_delivery
        self._producer = None
        self._connected = False

        if enabled:
            self._connect()

    def _connect(self) -> None:
        """Attempt to connect to Kafka. Degrades to noop on failure."""
        try:
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=self._bootstrap_servers,
                value_serializer=kafka_value_serializer,
                key_serializer=kafka_key_serializer,
                acks="all",
                retries=3,
                max_in_flight_requests_per_connection=1,
            )
            self._connected = True
            logger.info(
                "Kafka producer connected to %s", self._bootstrap_servers,
            )
        except Exception:
            self._connected = False
            logger.warning(
                "Kafka connection failed (%s). Running in noop mode.",
                self._bootstrap_servers,
                exc_info=True,
            )

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def publish(
        self,
        topic: str,
        data: Dict[str, Any],
        source_system: str = "sentinel_ai",
        key: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> Optional[BaseEvent]:
        """Publish a standardised event to a Kafka topic.

        Creates a BaseEvent envelope, serializes, and sends.

        Args:
            topic: Target Kafka topic (use KafkaTopics constants).
            data: Event-specific payload.
            source_system: Producing service name.
            key: Optional message key for partitioning.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            The published BaseEvent, or None if in noop mode.
        """
        event = create_event(
            event_type=topic,
            data=data,
            source_system=source_system,
            correlation_id=correlation_id,
        )

        if not self._enabled or not self._connected:
            logger.debug(
                "[noop] Would publish to '%s': event_id=%s",
                topic, event.event_id,
            )
            return event

        try:
            future = self._producer.send(
                topic, value=event, key=key,
            )
            if self._on_delivery:
                future.add_callback(self._on_delivery)
            logger.info(
                "Published event to '%s': event_id=%s",
                topic, event.event_id,
            )
            return event
        except Exception:
            logger.exception(
                "Failed to publish event to '%s': event_id=%s",
                topic, event.event_id,
            )
            return event  # Return event even on failure for tracing

    def publish_raw(
        self,
        topic: str,
        event: BaseEvent,
        key: Optional[str] = None,
    ) -> None:
        """Publish a pre-built BaseEvent to Kafka.

        Use this when you already have a fully constructed BaseEvent.

        Args:
            topic: Target Kafka topic.
            event: Pre-built BaseEvent instance.
            key: Optional message key.
        """
        if not self._enabled or not self._connected:
            logger.debug(
                "[noop] Would publish raw to '%s': event_id=%s",
                topic, event.event_id,
            )
            return

        try:
            self._producer.send(topic, value=event, key=key)
            logger.info(
                "Published raw event to '%s': event_id=%s",
                topic, event.event_id,
            )
        except Exception:
            logger.exception(
                "Failed to publish raw event to '%s': event_id=%s",
                topic, event.event_id,
            )

    def flush(self, timeout: float = 10.0) -> None:
        """Flush pending messages. Blocks until all messages are sent."""
        if self._producer and self._connected:
            self._producer.flush(timeout=timeout)

    def close(self) -> None:
        """Gracefully close the producer."""
        if self._producer:
            try:
                self._producer.flush(timeout=5.0)
                self._producer.close(timeout=5.0)
                logger.info("Kafka producer closed.")
            except Exception:
                logger.exception("Error closing Kafka producer.")
            finally:
                self._connected = False
                self._producer = None


class NoopEventProducer(KafkaEventProducer):
    """A no-op producer for development/testing.

    Logs all events without connecting to Kafka.
    """

    def __init__(self) -> None:
        super().__init__(enabled=False)
        logger.info("NoopEventProducer initialised (no Kafka connection).")
