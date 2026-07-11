"""Abstract repository interface for Digital Twin snapshots.

Follows the same ABC pattern as CompoundRiskRepository.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from app.digital_twin.models.facility_snapshot_model import (
    FacilitySnapshotModel,
)
from app.digital_twin.models.zone_state_model import ZoneStateModel


class TwinSnapshotRepository(ABC):
    """Interface for Digital Twin snapshot persistence.

    Implementations:
      - SQLAlchemySnapshotRepository: async SQLAlchemy backend
    """

    # ── Facility Snapshots ──

    @abstractmethod
    async def save_snapshot(
        self,
        snapshot: FacilitySnapshotModel,
        zone_states: List[ZoneStateModel],
    ) -> FacilitySnapshotModel:
        """Persist a facility snapshot with its zone states."""
        ...

    @abstractmethod
    async def get_snapshot(
        self, snapshot_id: str,
    ) -> Optional[FacilitySnapshotModel]:
        """Get a snapshot by its snapshot_id."""
        ...

    @abstractmethod
    async def get_latest_snapshot(
        self,
    ) -> Optional[FacilitySnapshotModel]:
        """Get the most recent snapshot."""
        ...

    @abstractmethod
    async def list_snapshots(
        self, offset: int = 0, limit: int = 50,
    ) -> List[FacilitySnapshotModel]:
        """List snapshots in reverse chronological order."""
        ...

    @abstractmethod
    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot and its zone states."""
        ...

    @abstractmethod
    async def count_snapshots(self) -> int:
        """Count total snapshots."""
        ...

    # ── Zone States ──

    @abstractmethod
    async def get_zone_states_for_snapshot(
        self, snapshot_id: str,
    ) -> List[ZoneStateModel]:
        """Get all zone states for a given snapshot."""
        ...

    # ── Retention ──

    @abstractmethod
    async def delete_oldest_snapshots(
        self, keep_count: int,
    ) -> int:
        """Delete oldest snapshots, keeping the most recent keep_count.

        Returns the number of deleted snapshots.
        """
        ...
