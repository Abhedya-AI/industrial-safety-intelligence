"""Compound Risk Kafka consumer setup.

Registers the CompoundRiskEventHandler with the shared
KafkaEventConsumer. Does NOT create a new consumer — reuses
the shared infrastructure.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from app.compound_risk.messaging.handler import CompoundRiskEventHandler
from app.shared.messaging.consumer import KafkaEventConsumer
from app.shared.messaging.topics import KafkaTopics

logger = logging.getLogger(__name__)

# Topics this module consumes
COMPOUND_RISK_SUBSCRIBED_TOPICS = [
    KafkaTopics.SENSOR_READING_ANOMALY,
    KafkaTopics.RISK_ASSESSMENT_GENERATED,
    KafkaTopics.RISK_SCORE_UPDATED,
]


class CompoundRiskConsumerSetup:
    """Registers compound risk event handlers with the shared consumer.

    Does NOT instantiate a new KafkaConsumer. Receives the shared
    consumer from DI and registers topic-specific handlers.

    Usage:
        setup = CompoundRiskConsumerSetup(consumer, handler)
        setup.register()
    """

    def __init__(
        self,
        consumer: KafkaEventConsumer,
        handler: CompoundRiskEventHandler,
    ) -> None:
        self._consumer = consumer
        self._handler = handler
        self._registered = False

    @property
    def is_registered(self) -> bool:
        return self._registered

    @property
    def subscribed_topics(self) -> list[str]:
        return list(COMPOUND_RISK_SUBSCRIBED_TOPICS)

    def register(self) -> None:
        """Register compound risk handlers on the shared consumer.

        Safe to call multiple times — only registers once.
        """
        if self._registered:
            logger.debug("Compound risk handlers already registered.")
            return

        for topic in COMPOUND_RISK_SUBSCRIBED_TOPICS:
            self._consumer.register_handler(topic, self._make_sync_handler())
            logger.info(
                "Registered compound risk handler for topic '%s'", topic,
            )

        self._registered = True
        logger.info(
            "Compound risk consumer setup complete. "
            "Subscribed to %d topics.", len(COMPOUND_RISK_SUBSCRIBED_TOPICS),
        )

    def _make_sync_handler(self) -> Callable[[str, Dict[str, Any]], None]:
        """Create a synchronous handler wrapper for the async handler.

        The shared consumer dispatches synchronously, but our handler
        is async. This wrapper bridges the gap using asyncio.
        """
        handler = self._handler

        def sync_handler(topic: str, data: Dict[str, Any]) -> None:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(handler.handle_event(topic, data))
            except RuntimeError:
                # No running event loop — create one (worker thread)
                asyncio.run(handler.handle_event(topic, data))

        return sync_handler
