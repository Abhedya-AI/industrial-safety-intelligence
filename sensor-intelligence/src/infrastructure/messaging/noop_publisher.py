"""No-op event publisher — default implementation for local development.

Logs events instead of publishing them to a message broker.
Swap this for KafkaPublisher or MQTTPublisher in production via DI config.
"""

import logging
from typing import Any

from src.application.interfaces.event_publisher import EventPublisher

logger = logging.getLogger(__name__)


class NoOpPublisher(EventPublisher):
    """Event publisher that logs events without sending them anywhere.

    Used during local development and hackathon demos where no
    message broker is available.
    """

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Log the event instead of publishing it."""
        logger.info(
            "Event published (NoOp) | topic=%s | payload=%s",
            topic,
            event,
        )

    async def disconnect(self) -> None:
        """No-op — nothing to disconnect."""
        logger.debug("NoOpPublisher disconnect called (no-op)")
