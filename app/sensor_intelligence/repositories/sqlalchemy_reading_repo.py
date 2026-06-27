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

    # ── Queries ──

    async def get_by_id(self, id: str) -> Optional[ReadingModel]:
        try:
            stmt = select(ReadingModel).where(ReadingModel.id == id)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError:
            logger.exception("DB error in get_by_id (id=%s)", id)
            raise

    async def get_latest(self, sensor_pk: str) -> Optional[ReadingModel]:
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
            logger.exception("DB error in get_latest (sensor_pk=%s)", sensor_pk)
            raise

    async def get_range(
        self,
        sensor_pk: str,
        from_dt: datetime,
        to_dt: datetime,
        limit: int = 1000,
    ) -> list[ReadingModel]:
        try:
            stmt = (
                select(ReadingModel)
                .where(
                    ReadingModel.sensor_id == sensor_pk,
                    ReadingModel.timestamp >= from_dt,
                    ReadingModel.timestamp <= to_dt,
                )
                .order_by(ReadingModel.timestamp.asc())
                .limit(limit)
            )
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError:
            logger.exception("DB error in get_range (sensor_pk=%s)", sensor_pk)
            raise

    async def count_for_sensor(self, sensor_pk: str) -> int:
        try:
            stmt = select(func.count(ReadingModel.id)).where(
                ReadingModel.sensor_id == sensor_pk
            )
            result = await self._session.execute(stmt)
            return result.scalar_one()
        except SQLAlchemyError:
            logger.exception("DB error in count_for_sensor (sensor_pk=%s)", sensor_pk)
            raise

    async def get_stats(
        self,
        sensor_pk: str,
        from_dt: datetime,
        to_dt: datetime,
    ) -> Optional[ReadingStats]:
        """Compute AVG, MIN, MAX, COUNT, and approximate STD DEV."""
        try:
            stmt = select(
                func.avg(ReadingModel.value),
                func.min(ReadingModel.value),
                func.max(ReadingModel.value),
                func.count(ReadingModel.id),
            ).where(
                ReadingModel.sensor_id == sensor_pk,
                ReadingModel.timestamp >= from_dt,
                ReadingModel.timestamp <= to_dt,
            )
            result = await self._session.execute(stmt)
            row = result.one()
            avg_val, min_val, max_val, count = row

            if count == 0:
                return None

            # Compute std_dev in a second pass (SQLite lacks native stddev)
            vals_stmt = select(ReadingModel.value).where(
                ReadingModel.sensor_id == sensor_pk,
                ReadingModel.timestamp >= from_dt,
                ReadingModel.timestamp <= to_dt,
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
                window_start=from_dt,
                window_end=to_dt,
            )
        except SQLAlchemyError:
            logger.exception("DB error in get_stats (sensor_pk=%s)", sensor_pk)
            raise

    # ── Mutations ──

    async def save(self, reading: ReadingModel) -> ReadingModel:
        try:
            self._session.add(reading)
            await self._session.flush()
            await self._session.refresh(reading)
            return reading
        except SQLAlchemyError:
            logger.exception("DB error in save")
            raise

    async def save_batch(self, readings: list[ReadingModel]) -> list[ReadingModel]:
        try:
            self._session.add_all(readings)
            await self._session.flush()
            for r in readings:
                await self._session.refresh(r)
            return readings
        except SQLAlchemyError:
            logger.exception("DB error in save_batch")
            raise
