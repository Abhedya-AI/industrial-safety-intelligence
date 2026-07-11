"""SQLAlchemy implementation of TwinSnapshotRepository.

Follows the SessionScopedCompoundRiskRepository pattern: each
operation creates its own session from the factory and commits
within that scope. This makes it safe for use from both the
FastAPI request lifecycle and the background Kafka consumer thread.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.digital_twin.models.facility_snapshot_model import (
    FacilitySnapshotModel,
)
from app.digital_twin.models.zone_state_model import ZoneStateModel
from app.digital_twin.repositories.twin_snapshot_repository import (
    TwinSnapshotRepository,
)

logger = logging.getLogger(__name__)


class SQLAlchemySnapshotRepository(TwinSnapshotRepository):
    """Concrete snapshot repository using async SQLAlchemy.

    Uses a session factory (not a single session) to create
    per-operation sessions — safe for background threads and
    the FastAPI request lifecycle.

    Args:
        session_factory: An ``async_sessionmaker`` bound to the
            application engine.
    """

    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Facility Snapshots
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def save_snapshot(
        self,
        snapshot: FacilitySnapshotModel,
        zone_states: List[ZoneStateModel],
    ) -> FacilitySnapshotModel:
        """Persist a facility snapshot and all associated zone states."""
        try:
            async with self._session_factory() as session:
                session.add(snapshot)
                for zs in zone_states:
                    session.add(zs)
                await session.flush()
                await session.refresh(snapshot)
                await session.commit()
                logger.debug(
                    "Saved snapshot: id=%s zones=%d",
                    snapshot.snapshot_id, len(zone_states),
                )
                return snapshot
        except SQLAlchemyError:
            logger.exception("DB error in save_snapshot")
            raise

    async def get_snapshot(
        self, snapshot_id: str,
    ) -> Optional[FacilitySnapshotModel]:
        """Get a snapshot by its snapshot_id."""
        try:
            async with self._session_factory() as session:
                stmt = select(FacilitySnapshotModel).where(
                    FacilitySnapshotModel.snapshot_id == snapshot_id,
                )
                result = await session.execute(stmt)
                return result.scalar_one_or_none()
        except SQLAlchemyError:
            logger.exception(
                "DB error in get_snapshot (id=%s)", snapshot_id,
            )
            raise

    async def get_latest_snapshot(
        self,
    ) -> Optional[FacilitySnapshotModel]:
        """Get the most recent snapshot."""
        try:
            async with self._session_factory() as session:
                stmt = (
                    select(FacilitySnapshotModel)
                    .order_by(FacilitySnapshotModel.created_at.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                return result.scalar_one_or_none()
        except SQLAlchemyError:
            logger.exception("DB error in get_latest_snapshot")
            raise

    async def list_snapshots(
        self, offset: int = 0, limit: int = 50,
    ) -> List[FacilitySnapshotModel]:
        """List snapshots in reverse chronological order."""
        try:
            async with self._session_factory() as session:
                stmt = (
                    select(FacilitySnapshotModel)
                    .order_by(FacilitySnapshotModel.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                )
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except SQLAlchemyError:
            logger.exception("DB error in list_snapshots")
            raise

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot and its associated zone states."""
        try:
            async with self._session_factory() as session:
                # Delete zone states first
                zone_stmt = delete(ZoneStateModel).where(
                    ZoneStateModel.snapshot_id == snapshot_id,
                )
                await session.execute(zone_stmt)

                # Delete the facility snapshot
                snap_stmt = delete(FacilitySnapshotModel).where(
                    FacilitySnapshotModel.snapshot_id == snapshot_id,
                )
                result = await session.execute(snap_stmt)
                await session.commit()
                deleted = result.rowcount > 0
                if deleted:
                    logger.debug(
                        "Deleted snapshot: id=%s", snapshot_id,
                    )
                return deleted
        except SQLAlchemyError:
            logger.exception(
                "DB error in delete_snapshot (id=%s)", snapshot_id,
            )
            raise

    async def count_snapshots(self) -> int:
        """Count total snapshots."""
        try:
            async with self._session_factory() as session:
                stmt = select(
                    func.count(FacilitySnapshotModel.id),
                )
                result = await session.execute(stmt)
                return result.scalar_one()
        except SQLAlchemyError:
            logger.exception("DB error in count_snapshots")
            raise

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Zone States
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_zone_states_for_snapshot(
        self, snapshot_id: str,
    ) -> List[ZoneStateModel]:
        """Get all zone states for a given snapshot."""
        try:
            async with self._session_factory() as session:
                stmt = (
                    select(ZoneStateModel)
                    .where(ZoneStateModel.snapshot_id == snapshot_id)
                    .order_by(ZoneStateModel.zone_id)
                )
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except SQLAlchemyError:
            logger.exception(
                "DB error in get_zone_states (snapshot=%s)",
                snapshot_id,
            )
            raise

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Retention
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def delete_oldest_snapshots(
        self, keep_count: int,
    ) -> int:
        """Delete oldest snapshots, keeping the newest keep_count.

        Returns number of deleted snapshots.
        """
        try:
            async with self._session_factory() as session:
                # Get IDs to keep (newest N)
                keep_stmt = (
                    select(FacilitySnapshotModel.snapshot_id)
                    .order_by(FacilitySnapshotModel.created_at.desc())
                    .limit(keep_count)
                )
                keep_result = await session.execute(keep_stmt)
                keep_ids = set(keep_result.scalars().all())

                if not keep_ids:
                    return 0

                # Get IDs to delete
                all_stmt = select(FacilitySnapshotModel.snapshot_id)
                all_result = await session.execute(all_stmt)
                all_ids = set(all_result.scalars().all())

                delete_ids = all_ids - keep_ids
                if not delete_ids:
                    return 0

                # Delete zone states for those snapshots
                zone_del_stmt = delete(ZoneStateModel).where(
                    ZoneStateModel.snapshot_id.in_(delete_ids),
                )
                await session.execute(zone_del_stmt)

                # Delete facility snapshots
                snap_del_stmt = delete(FacilitySnapshotModel).where(
                    FacilitySnapshotModel.snapshot_id.in_(delete_ids),
                )
                result = await session.execute(snap_del_stmt)
                await session.commit()

                deleted = result.rowcount
                if deleted > 0:
                    logger.info(
                        "Retention cleanup: deleted %d snapshots "
                        "(kept %d)",
                        deleted, keep_count,
                    )
                return deleted
        except SQLAlchemyError:
            logger.exception("DB error in delete_oldest_snapshots")
            raise
