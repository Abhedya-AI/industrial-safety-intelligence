"""Serialization and deserialization utilities for Kafka messages.

Handles JSON encoding/decoding with proper error handling and type
coercion for datetime objects, UUIDs, enums, and Pydantic models.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from app.shared.messaging.events import BaseEvent

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSON encoder with extended type support
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class EventJSONEncoder(json.JSONEncoder):
    """Extended JSON encoder that handles common Python types.

    Supports: datetime, date, UUID, Enum, Pydantic BaseModel, sets.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        return super().default(obj)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Serialization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def serialize_event(event: BaseEvent) -> bytes:
    """Serialize a BaseEvent to UTF-8 encoded JSON bytes.

    This is the standard Kafka value_serializer.

    Args:
        event: The event to serialize.

    Returns:
        UTF-8 bytes suitable for Kafka producer.

    Raises:
        SerializationError: If serialization fails.
    """
    try:
        payload = event.model_dump()
        return json.dumps(payload, cls=EventJSONEncoder).encode("utf-8")
    except (TypeError, ValueError) as exc:
        logger.error("Serialization failed for event %s: %s", event.event_type, exc)
        raise SerializationError(f"Failed to serialize event: {exc}") from exc


def serialize_dict(data: Dict[str, Any]) -> bytes:
    """Serialize a plain dictionary to UTF-8 encoded JSON bytes.

    Args:
        data: Dictionary to serialize.

    Returns:
        UTF-8 bytes suitable for Kafka producer.
    """
    try:
        return json.dumps(data, cls=EventJSONEncoder).encode("utf-8")
    except (TypeError, ValueError) as exc:
        logger.error("Dict serialization failed: %s", exc)
        raise SerializationError(f"Failed to serialize dict: {exc}") from exc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Deserialization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def deserialize_event(raw: bytes) -> BaseEvent:
    """Deserialize UTF-8 JSON bytes into a BaseEvent.

    This is the standard Kafka value_deserializer.

    Args:
        raw: UTF-8 encoded JSON bytes.

    Returns:
        Parsed BaseEvent instance.

    Raises:
        DeserializationError: If parsing fails.
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
        return BaseEvent(**payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        logger.error("Deserialization failed: %s", exc)
        raise DeserializationError(f"Failed to deserialize event: {exc}") from exc


def deserialize_dict(raw: bytes) -> Dict[str, Any]:
    """Deserialize UTF-8 JSON bytes into a dictionary.

    Args:
        raw: UTF-8 encoded JSON bytes.

    Returns:
        Parsed dictionary.

    Raises:
        DeserializationError: If parsing fails.
    """
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.error("Dict deserialization failed: %s", exc)
        raise DeserializationError(f"Failed to deserialize dict: {exc}") from exc


def deserialize_to_model(raw: bytes, model_class: Type[T]) -> T:
    """Deserialize UTF-8 JSON bytes into a specific Pydantic model.

    Args:
        raw: UTF-8 encoded JSON bytes.
        model_class: Target Pydantic model class.

    Returns:
        Parsed model instance.

    Raises:
        DeserializationError: If parsing or validation fails.
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
        return model_class(**payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
        logger.error("Model deserialization failed for %s: %s", model_class.__name__, exc)
        raise DeserializationError(
            f"Failed to deserialize to {model_class.__name__}: {exc}",
        ) from exc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Kafka-compatible serializer/deserializer callables
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def kafka_value_serializer(value: Any) -> bytes:
    """Kafka-compatible value_serializer callable.

    Handles BaseEvent, dict, and Pydantic BaseModel instances.
    """
    if isinstance(value, BaseEvent):
        return serialize_event(value)
    if isinstance(value, BaseModel):
        return serialize_dict(value.model_dump())
    if isinstance(value, dict):
        return serialize_dict(value)
    raise SerializationError(f"Unsupported type for Kafka serialization: {type(value)}")


def kafka_value_deserializer(raw: bytes) -> Dict[str, Any]:
    """Kafka-compatible value_deserializer callable.

    Returns a plain dictionary. Callers can convert to BaseEvent or
    domain models as needed.
    """
    return deserialize_dict(raw)


def kafka_key_serializer(key: Optional[str]) -> Optional[bytes]:
    """Serialize a Kafka message key (string → bytes)."""
    if key is None:
        return None
    return key.encode("utf-8")


def kafka_key_deserializer(raw: Optional[bytes]) -> Optional[str]:
    """Deserialize a Kafka message key (bytes → string)."""
    if raw is None:
        return None
    return raw.decode("utf-8")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Exceptions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SerializationError(Exception):
    """Raised when event serialization fails."""


class DeserializationError(Exception):
    """Raised when event deserialization fails."""
