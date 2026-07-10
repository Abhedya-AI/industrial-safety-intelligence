"""Digital Twin Kafka consumer setup.

Registers the DigitalTwinEventHandler with the shared
KafkaEventConsumer. Does NOT create a new consumer — reuses
the shared infrastructure.

Follows the same pattern as:
  - CompoundRiskConsumerSetup
  - HazardPropagationConsumerSetup
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from app.digital_twin.messaging.handler import DigitalTwinEventHandler
from app.shared.messaging.consumer import KafkaEventConsumer
from app.shared.messaging.topics import KafkaTopics

logger = logging.getLogger(__name__)

# Topics this module consumes
DIGITAL_TWIN_SUBSCRIBED_TOPICS = [
    # Sensor Intelligence
    KafkaTopics.SENSOR_READING_ANOMALY,
    KafkaTopics.SENSOR_STATUS_CHANGED,
    KafkaTopics.SENSOR_HEALTH_UPDATED,
    # Risk Prediction
    KafkaTopics.RISK_ASSESSMENT_GENERATED,
    KafkaTopics.RISK_SCORE_UPDATED,
    KafkaTopics.RISK_THRESHOLD_EXCEEDED,
    # Compound Risk Intelligence
    KafkaTopics.COMPOUND_RISK_DETECTED,
    # Hazard Propagation
    KafkaTopics.HAZARD_DETECTED,
    KafkaTopics.HAZARD_PROPAGATED,
]


class DigitalTwinConsumerSetup:
    """Registers digital twin event handlers with the shared consumer.

    Does NOT instantiate a new KafkaConsumer. Receives the shared
    consumer from DI and registers topic-specific handlers.

    Usage:
        setup = DigitalTwinConsumerSetup(consumer, handler)
        setup.register()
    """

    def __init__(
        self,
        consumer: KafkaEventConsumer,
        handler: DigitalTwinEventHandler,
    ) -> None:
        self._consumer = consumer
        self._handler = handler
        self._registered = False

    @property
    def is_registered(self) -> bool:
        return self._registered

    @property
    def subscribed_topics(self) -> list[str]:
        return list(DIGITAL_TWIN_SUBSCRIBED_TOPICS)

    def register(self) -> None:
        """Register digital twin handlers on the shared consumer.

        Safe to call multiple times — only registers once.
        """
        if self._registered:
            logger.debug("Digital twin handlers already registered.")
            return

        for topic in DIGITAL_TWIN_SUBSCRIBED_TOPICS:
            self._consumer.register_handler(
                topic, self._make_sync_handler(),
            )
            logger.info(
                "Registered digital twin handler for topic '%s'",
                topic,
            )

        self._registered = True
        logger.info(
            "Digital twin consumer setup complete. "
            "Subscribed to %d topics.",
            len(DIGITAL_TWIN_SUBSCRIBED_TOPICS),
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
