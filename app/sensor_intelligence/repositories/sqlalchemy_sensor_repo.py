"""SQLAlchemy concrete implementation of the SensorRepository interface.

Handles all database interaction for the ``sensors`` table.
Business validation is NOT performed here.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy import exists, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.sensor_intelligence.models.sensor_model import SensorModel
from app.sensor_intelligence.repositories.sensor_repository import SensorRepository

logger = logging.getLogger(__name__)


class SQLAlchemySensorRepository(SensorRepository):
    """Concrete repository that persists sensors via SQLAlchemy.

    Each instance is scoped to a single ``AsyncSession``, which is
    managed externally by the FastAPI dependency injection container.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Queries ──

    async def get_sensor_by_id(self, id: UUID) -> Optional[SensorModel]:
        """Retrieve a sensor by its internal UUID primary key."""
        try:
            stmt = select(SensorModel).where(SensorModel.id == str(id))
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError:
            logger.exception("DB error in get_sensor_by_id (id=%s)", id)
            raise

    async def get_sensor_by_code(self, sensor_id: str) -> Optional[SensorModel]:
        """Retrieve a sensor by its external business code (e.g. 'S001')."""
        try:
            stmt = select(SensorModel).where(SensorModel.sensor_id == sensor_id)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError:
            logger.exception("DB error in get_sensor_by_code (sensor_id=%s)", sensor_id)
            raise

    async def list_sensors(
        self,
        sensor_type: Optional[str] = None,
        status: Optional[str] = None,
        location_zone: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[SensorModel]:
        """List sensors with optional filters and pagination."""
        try:
            stmt = select(SensorModel)

            if sensor_type is not None:
                stmt = stmt.where(SensorModel.sensor_type == sensor_type)
            if status is not None:
                stmt = stmt.where(SensorModel.status == status)
            if location_zone is not None:
                stmt = stmt.where(SensorModel.location_zone == location_zone)

            stmt = stmt.order_by(SensorModel.sensor_id).offset(offset).limit(limit)
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError:
            logger.exception("DB error in list_sensors")
            raise

    async def count(
        self,
        sensor_type: Optional[str] = None,
        status: Optional[str] = None,
        location_zone: Optional[str] = None,
    ) -> int:
        """Count sensors matching the given filters."""
        try:
            stmt = select(func.count(SensorModel.id))

            if sensor_type is not None:
                stmt = stmt.where(SensorModel.sensor_type == sensor_type)
            if status is not None:
                stmt = stmt.where(SensorModel.status == status)
            if location_zone is not None:
                stmt = stmt.where(SensorModel.location_zone == location_zone)

            result = await self._session.execute(stmt)
            return result.scalar_one()
        except SQLAlchemyError:
            logger.exception("DB error in count")
            raise

    async def count_by_status(self) -> dict[str, int]:
        """Count sensors grouped by status."""
        try:
            stmt = (
                select(SensorModel.status, func.count(SensorModel.id))
                .group_by(SensorModel.status)
            )
            result = await self._session.execute(stmt)
            return {row[0]: row[1] for row in result.all()}
        except SQLAlchemyError:
            logger.exception("DB error in count_by_status")
            raise

    async def sensor_exists(self, sensor_id: str) -> bool:
        """Check whether a sensor with the given business code exists.

        Uses an EXISTS sub-query for efficiency — no full row fetch.
        """
        try:
            stmt = select(
                exists().where(SensorModel.sensor_id == sensor_id)
            )
            result = await self._session.execute(stmt)
            return result.scalar_one()
        except SQLAlchemyError:
            logger.exception("DB error in sensor_exists (sensor_id=%s)", sensor_id)
            raise

    # ── Mutations ──

    async def create_sensor(self, sensor: SensorModel) -> SensorModel:
        """Persist a new sensor.

        Raises ``IntegrityError`` if a unique constraint is violated
        (e.g. duplicate ``sensor_id``).  The caller (service layer)
        should handle this and translate it to a domain exception.
        """
        try:
            self._session.add(sensor)
            await self._session.flush()
            await self._session.refresh(sensor)
            return sensor
        except IntegrityError:
            logger.warning(
                "Integrity error creating sensor (sensor_id=%s)", sensor.sensor_id
            )
            raise
        except SQLAlchemyError:
            logger.exception("DB error in create_sensor")
            raise

    async def update_sensor(self, sensor: SensorModel) -> SensorModel:
        """Flush pending attribute changes on an already-tracked instance."""
        try:
            await self._session.flush()
            await self._session.refresh(sensor)
            return sensor
        except IntegrityError:
            logger.warning(
                "Integrity error updating sensor (id=%s)", sensor.id
            )
            raise
        except SQLAlchemyError:
            logger.exception("DB error in update_sensor")
            raise

    async def delete_sensor(self, sensor: SensorModel) -> None:
        """Delete a sensor from persistent storage."""
        try:
            await self._session.delete(sensor)
            await self._session.flush()
        except SQLAlchemyError:
            logger.exception("DB error in delete_sensor (id=%s)", sensor.id)
            raise
