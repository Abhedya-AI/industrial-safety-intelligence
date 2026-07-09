"""Hazard Propagation Kafka consumer setup.

Registers the HazardPropagationEventHandler with the shared
KafkaEventConsumer. Does NOT create a new consumer — reuses
the shared infrastructure.

Follows the same pattern as CompoundRiskConsumerSetup.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from app.hazard_propagation.messaging.handler import (
    HazardPropagationEventHandler,
)
from app.shared.messaging.consumer import KafkaEventConsumer
from app.shared.messaging.topics import KafkaTopics

logger = logging.getLogger(__name__)

# Topics this module consumes
HAZARD_PROPAGATION_SUBSCRIBED_TOPICS = [
    KafkaTopics.COMPOUND_RISK_DETECTED,
    KafkaTopics.HAZARD_DETECTED,
]


class HazardPropagationConsumerSetup:
    """Registers hazard propagation event handlers with the shared consumer.

    Does NOT instantiate a new KafkaConsumer. Receives the shared
    consumer from DI and registers topic-specific handlers.

    Usage:
        setup = HazardPropagationConsumerSetup(consumer, handler)
        setup.register()
    """

    def __init__(
        self,
        consumer: KafkaEventConsumer,
        handler: HazardPropagationEventHandler,
    ) -> None:
        self._consumer = consumer
        self._handler = handler
        self._registered = False

    @property
    def is_registered(self) -> bool:
        return self._registered

    @property
    def subscribed_topics(self) -> list[str]:
        return list(HAZARD_PROPAGATION_SUBSCRIBED_TOPICS)

    def register(self) -> None:
        """Register hazard propagation handlers on the shared consumer.

        Safe to call multiple times — only registers once.
        """
        if self._registered:
            logger.debug("Hazard propagation handlers already registered.")
            return

        for topic in HAZARD_PROPAGATION_SUBSCRIBED_TOPICS:
            self._consumer.register_handler(
                topic, self._make_sync_handler(),
            )
            logger.info(
                "Registered hazard propagation handler for topic '%s'",
                topic,
            )

        self._registered = True
        logger.info(
            "Hazard propagation consumer setup complete. "
            "Subscribed to %d topics.",
            len(HAZARD_PROPAGATION_SUBSCRIBED_TOPICS),
        )

    def _make_sync_handler(
        self,
    ) -> Callable[[str, Dict[str, Any]], None]:
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
