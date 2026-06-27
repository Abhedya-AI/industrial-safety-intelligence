"""Reading data transfer objects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class ReadingCreateDTO:
    """Input DTO for ingesting a new sensor reading."""

    sensor_id: str
    value: float
    timestamp: datetime
    confidence: float = 100.0
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ReadingResponseDTO:
    """Output DTO for sensor reading data."""

    id: UUID
    sensor_id: UUID
    value: float
    timestamp: datetime
    confidence: float
    received_at: datetime


@dataclass(frozen=True)
class BatchReadingCreateDTO:
    """Input DTO for batch ingestion of readings."""

    readings: list[ReadingCreateDTO]
