"""Domain exceptions for the Hazard Propagation Engine.

Follows the same exception hierarchy pattern used across the platform.
"""

from __future__ import annotations


class HazardPropagationError(Exception):
    """Base exception for all Hazard Propagation Engine errors."""


class InvalidHazardError(HazardPropagationError):
    """Raised when hazard input is invalid or incomplete."""


class ZoneNotFoundError(HazardPropagationError):
    """Raised when a referenced zone does not exist in the graph."""

    def __init__(self, zone_id: str) -> None:
        self.zone_id = zone_id
        super().__init__(f"Zone not found: {zone_id}")


class GraphNotInitializedError(HazardPropagationError):
    """Raised when the graph has not been built before simulation."""


class PropagationSimulationError(HazardPropagationError):
    """Raised when the propagation simulation fails."""


class CyclicPropagationError(HazardPropagationError):
    """Raised when a cyclic dependency is detected in the propagation graph."""
