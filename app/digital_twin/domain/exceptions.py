"""Digital Twin domain exceptions."""

from __future__ import annotations


class TwinStateError(Exception):
    """Base exception for Digital Twin state errors."""


class ZoneNotFoundInTwinError(TwinStateError):
    """Raised when a zone ID is not found in the twin state."""

    def __init__(self, zone_id: str) -> None:
        self.zone_id = zone_id
        super().__init__(f"Zone not found in twin state: {zone_id}")


class TwinNotInitializedError(TwinStateError):
    """Raised when the twin state has not been initialized yet."""

    def __init__(self) -> None:
        super().__init__(
            "Digital Twin state not initialized. "
            "Call initialize() or wait for first event."
        )
