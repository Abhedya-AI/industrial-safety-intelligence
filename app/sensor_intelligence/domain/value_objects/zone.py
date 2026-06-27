"""Zone value object for plant location zones."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Zone:
    """Represents a physical zone in the industrial facility."""

    name: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Zone name cannot be empty")

    def __str__(self) -> str:
        return self.name
