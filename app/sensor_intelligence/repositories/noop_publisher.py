"""Event publisher interface (port)."""

from abc import ABC, abstractmethod
from typing import Any


class EventPublisher(ABC):
    """Abstract interface for publishing domain events."""

    @abstractmethod
    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Publish an event to the given topic."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connections and release resources."""
        ...
class NoOpPublisher(EventPublisher):
    """Event publisher that logs events without sending them anywhere."""

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Log the event instead of publishing it."""
        import logging
        logging.getLogger(__name__).info(
            "Event published (NoOp) | topic=%s | payload=%s",
            topic,
            event,
        )

    async def disconnect(self) -> None:
        """No-op — nothing to disconnect."""
        pass
