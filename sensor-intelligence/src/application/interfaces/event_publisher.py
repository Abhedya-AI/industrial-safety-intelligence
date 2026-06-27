"""Event publisher interface (port).

Defines the contract for publishing domain events to external systems.
Implementations: NoOpPublisher (default), KafkaPublisher, MQTTPublisher.
"""

from abc import ABC, abstractmethod
from typing import Any


class EventPublisher(ABC):
    """Abstract interface for publishing domain events."""

    @abstractmethod
    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Publish an event to the given topic.

        Args:
            topic: Event topic/channel name (e.g. "sensor.anomaly.detected").
            event: Event payload as a dictionary.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connections and release resources."""
        ...
