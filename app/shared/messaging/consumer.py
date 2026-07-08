"""Kafka consumer abstraction.

Provides a ``KafkaEventConsumer`` that wraps the kafka-python
``KafkaConsumer`` with:
  - Automatic JSON deserialization via shared utilities
  - BaseEvent parsing
  - Handler registration pattern
  - Graceful degradation (noop mode when Kafka is unavailable)

All modules MUST use this consumer — no direct ``KafkaConsumer`` usage.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from app.shared.messaging.events import BaseEvent
from app.shared.messaging.serialization import (
    kafka_key_deserializer,
    kafka_value_deserializer,
)

logger = logging.getLogger(__name__)

# Type for event handlers: (topic, event_data_dict) → None
EventHandler = Callable[[str, Dict[str, Any]], None]


class KafkaEventConsumer:
    """Shared Kafka event consumer with handler registration.

    Wraps kafka-python's KafkaConsumer with standardised deserialization,
    event routing to registered handlers, and noop fallback.

    Args:
        bootstrap_servers: Kafka bootstrap server address(es).
        group_id: Consumer group ID.
        topics: List of topics to subscribe to.
        enabled: If False, operates in noop mode.
        auto_offset_reset: Where to start reading (earliest/latest).
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "sentinel_ai",
        topics: Optional[List[str]] = None,
        enabled: bool = True,
        auto_offset_reset: str = "earliest",
    ) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._group_id = group_id
        self._topics = topics or []
        self._enabled = enabled
        self._auto_offset_reset = auto_offset_reset
        self._consumer = None
        self._connected = False
        self._running = False
        self._handlers: Dict[str, List[EventHandler]] = {}

        if enabled and self._topics:
            self._connect()

    def _connect(self) -> None:
        """Attempt to connect to Kafka. Degrades to noop on failure."""
        try:
            from kafka import KafkaConsumer

            self._consumer = KafkaConsumer(
                *self._topics,
                bootstrap_servers=self._bootstrap_servers,
                group_id=self._group_id,
                auto_offset_reset=self._auto_offset_reset,
                enable_auto_commit=True,
                value_deserializer=kafka_value_deserializer,
                key_deserializer=kafka_key_deserializer,
            )
            self._connected = True
            logger.info(
                "Kafka consumer connected to %s, group=%s, topics=%s",
                self._bootstrap_servers, self._group_id, self._topics,
            )
        except Exception:
            self._connected = False
            logger.warning(
                "Kafka consumer connection failed (%s). Running in noop mode.",
                self._bootstrap_servers,
                exc_info=True,
            )

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def register_handler(self, topic: str, handler: EventHandler) -> None:
        """Register an event handler for a specific topic.

        Multiple handlers can be registered per topic.

        Args:
            topic: Kafka topic to handle (use KafkaTopics constants).
            handler: Callable that receives (topic, event_data_dict).
        """
        if topic not in self._handlers:
            self._handlers[topic] = []
        self._handlers[topic].append(handler)
        logger.debug("Registered handler for topic '%s'", topic)

    def register_default_handler(self, handler: EventHandler) -> None:
        """Register a handler that receives events from ALL topics.

        Args:
            handler: Callable that receives (topic, event_data_dict).
        """
        self._handlers["*"] = self._handlers.get("*", [])
        self._handlers["*"].append(handler)
        logger.debug("Registered default handler for all topics")

    def _dispatch(self, topic: str, data: Dict[str, Any]) -> None:
        """Route an event to registered handlers."""
        handlers = self._handlers.get(topic, []) + self._handlers.get("*", [])
        for handler in handlers:
            try:
                handler(topic, data)
            except Exception:
                logger.exception(
                    "Handler error for topic '%s': event_type=%s",
                    topic, data.get("event_type", "unknown"),
                )

    def consume_one(self, timeout_ms: int = 1000) -> Optional[Dict[str, Any]]:
        """Consume a single message (non-blocking with timeout).

        Returns the event data dict, or None if no message available.
        Does not dispatch to handlers — caller processes directly.
        """
        if not self._enabled or not self._connected:
            return None

        try:
            records = self._consumer.poll(timeout_ms=timeout_ms, max_records=1)
            for topic_partition, messages in records.items():
                for msg in messages:
                    return msg.value
            return None
        except Exception:
            logger.exception("Error consuming message")
            return None

    def start(self, max_messages: Optional[int] = None) -> None:
        """Start the consumer loop.

        Consumes messages and dispatches to registered handlers.
        Blocks until stopped or max_messages reached.

        Args:
            max_messages: If set, stop after processing this many messages.
        """
        if not self._enabled or not self._connected:
            logger.info(
                "[noop] Consumer not started (enabled=%s, connected=%s)",
                self._enabled, self._connected,
            )
            return

        self._running = True
        count = 0
        logger.info("Consumer loop started for topics: %s", self._topics)

        try:
            for message in self._consumer:
                if not self._running:
                    break

                topic = message.topic
                data = message.value

                logger.debug(
                    "Received message from '%s': event_type=%s",
                    topic, data.get("event_type", "unknown"),
                )

                self._dispatch(topic, data)
                count += 1

                if max_messages and count >= max_messages:
                    break
        except Exception:
            logger.exception("Consumer loop error")
        finally:
            self._running = False
            logger.info("Consumer loop stopped after %d messages.", count)

    def stop(self) -> None:
        """Signal the consumer loop to stop."""
        self._running = False
        logger.info("Consumer stop requested.")

    def close(self) -> None:
        """Gracefully close the consumer."""
        self.stop()
        if self._consumer:
            try:
                self._consumer.close()
                logger.info("Kafka consumer closed.")
            except Exception:
                logger.exception("Error closing Kafka consumer.")
            finally:
                self._connected = False
                self._consumer = None


class NoopEventConsumer(KafkaEventConsumer):
    """A no-op consumer for development/testing.

    Does not connect to Kafka.
    """

    def __init__(self) -> None:
        super().__init__(enabled=False)
        logger.info("NoopEventConsumer initialised (no Kafka connection).")
