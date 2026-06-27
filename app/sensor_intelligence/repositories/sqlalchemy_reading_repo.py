"""SQLAlchemy concrete implementation of the ReadingRepository interface."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.sensor_intelligence.models.reading_model import ReadingModel
from app.sensor_intelligence.repositories.reading_repository import (
    ReadingRepository,
    ReadingStats,
)

logger = logging.getLogger(__name__)


class SQLAlchemyReadingRepository(ReadingRepository):
    """Concrete repository for sensor readings via SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Queries
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_reading_by_id(self, id: str) -> Optional[ReadingModel]:
        try:
            stmt = select(ReadingModel).where(ReadingModel.id == id)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError:
            logger.exception("DB error in get_reading_by_id (id=%s)", id)
            raise

    async def get_latest_reading(self, sensor_pk: str) -> Optional[ReadingModel]:
        try:
            stmt = (
                select(ReadingModel)
                .where(ReadingModel.sensor_id == sensor_pk)
                .order_by(ReadingModel.timestamp.desc())
                .limit(1)
            )
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError:
            logger.exception(
                "DB error in get_latest_reading (sensor_pk=%s)", sensor_pk
            )
            raise

    async def get_sensor_history(
        self,
        sensor_pk: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
    ) -> list[ReadingModel]:
        try:
            stmt = (
                select(ReadingModel)
                .where(
                    ReadingModel.sensor_id == sensor_pk,
                    ReadingModel.timestamp >= start_time,
                    ReadingModel.timestamp <= end_time,
                )
                .order_by(ReadingModel.timestamp.asc())
                .limit(limit)
            )
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError:
            logger.exception(
                "DB error in get_sensor_history (sensor_pk=%s)", sensor_pk
            )
            raise

    async def list_readings(
        self,
        sensor_pk: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[ReadingModel]:
        try:
            stmt = select(ReadingModel)
            if sensor_pk is not None:
                stmt = stmt.where(ReadingModel.sensor_id == sensor_pk)
            stmt = (
                stmt.order_by(ReadingModel.timestamp.desc())
                .offset(offset)
                .limit(limit)
            )
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError:
            logger.exception("DB error in list_readings")
            raise

    async def count_for_sensor(self, sensor_pk: str) -> int:
        try:
            stmt = select(func.count(ReadingModel.id)).where(
                ReadingModel.sensor_id == sensor_pk
            )
            result = await self._session.execute(stmt)
            return result.scalar_one()
        except SQLAlchemyError:
            logger.exception(
                "DB error in count_for_sensor (sensor_pk=%s)", sensor_pk
            )
            raise

    async def reading_exists(self, id: str) -> bool:
        try:
            stmt = select(
                select(ReadingModel.id)
                .where(ReadingModel.id == id)
                .limit(1)
                .exists()
            )
            result = await self._session.execute(stmt)
            return result.scalar_one()
        except SQLAlchemyError:
            logger.exception("DB error in reading_exists (id=%s)", id)
            raise

    async def get_stats(
        self,
        sensor_pk: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Optional[ReadingStats]:
        """Compute AVG, MIN, MAX, COUNT, and STD DEV.

        Uses a two-pass approach for STD DEV because SQLite
        lacks a native stddev aggregate function.
        """
        try:
            stmt = select(
                func.avg(ReadingModel.value),
                func.min(ReadingModel.value),
                func.max(ReadingModel.value),
                func.count(ReadingModel.id),
            ).where(
                ReadingModel.sensor_id == sensor_pk,
                ReadingModel.timestamp >= start_time,
                ReadingModel.timestamp <= end_time,
            )
            result = await self._session.execute(stmt)
            row = result.one()
            avg_val, min_val, max_val, count = row

            if count == 0:
                return None

            # Second pass for stddev (SQLite compat)
            vals_stmt = select(ReadingModel.value).where(
                ReadingModel.sensor_id == sensor_pk,
                ReadingModel.timestamp >= start_time,
                ReadingModel.timestamp <= end_time,
            )
            vals_result = await self._session.execute(vals_stmt)
            values = [r[0] for r in vals_result.all()]
            variance = sum((v - avg_val) ** 2 for v in values) / len(values)
            std_dev = math.sqrt(variance)

            return ReadingStats(
                sensor_id=sensor_pk,
                mean=round(avg_val, 2),
                std_dev=round(std_dev, 2),
                min_value=min_val,
                max_value=max_val,
                count=count,
                window_start=start_time,
                window_end=end_time,
            )
        except SQLAlchemyError:
            logger.exception("DB error in get_stats (sensor_pk=%s)", sensor_pk)
            raise

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Mutations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def create_reading(self, reading: ReadingModel) -> ReadingModel:
        try:
            self._session.add(reading)
            await self._session.flush()
            await self._session.refresh(reading)
            return reading
        except SQLAlchemyError:
            logger.exception("DB error in create_reading")
            raise

    async def create_readings_batch(
        self, readings: list[ReadingModel]
    ) -> list[ReadingModel]:
        try:
            self._session.add_all(readings)
            await self._session.flush()
            for r in readings:
                await self._session.refresh(r)
            return readings
        except SQLAlchemyError:
            logger.exception("DB error in create_readings_batch")
            raise
