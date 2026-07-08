"""Base event schema for Kafka messages.

Source of truth: PS1 SentinelAI Common Domain Names v2.0 (§5.3).

All Kafka messages MUST include:
  - event_type: from standardised event types (§3)
  - event_id: unique identifier (UUID)
  - timestamp: ISO 8601 UTC
  - source_system: which service produced the event
  - data: event-specific payload (dict)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    """Standard Kafka event envelope.

    Every message published to Kafka MUST use this schema or a subclass.
    The ``data`` field carries the module-specific payload.

    Attributes:
        event_type: Standardised event type from §3 (e.g. ``sensor.reading.created``).
        event_id: Unique event identifier (UUID4, auto-generated).
        timestamp: ISO 8601 UTC timestamp (auto-set to now if omitted).
        source_system: Name of the producing service/module.
        data: Event-specific payload dictionary.
        correlation_id: Optional correlation ID for tracing event chains.
        version: Schema version for forward compatibility.
    """

    event_type: str
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    source_system: str = "sentinel_ai"
    data: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None
    version: str = "1.0"

    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "sensor.reading.created",
                "event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "timestamp": "2026-07-08T04:30:00+00:00",
                "source_system": "sensor_intelligence",
                "data": {
                    "sensor_id": "S001",
                    "reading_type": "TEMPERATURE",
                    "value": 72.5,
                    "unit": "celsius",
                },
                "correlation_id": None,
                "version": "1.0",
            },
        }


# ── Convenience factory functions ──


def create_event(
    event_type: str,
    data: Dict[str, Any],
    source_system: str = "sentinel_ai",
    correlation_id: Optional[str] = None,
) -> BaseEvent:
    """Create a new event with auto-generated ID and timestamp.

    Args:
        event_type: Standardised topic/event type string.
        data: Module-specific payload.
        source_system: Name of the producing service.
        correlation_id: Optional ID for event chain tracing.

    Returns:
        A fully populated BaseEvent instance.
    """
    return BaseEvent(
        event_type=event_type,
        data=data,
        source_system=source_system,
        correlation_id=correlation_id,
    )
