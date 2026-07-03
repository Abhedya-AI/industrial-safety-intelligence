"""SQLAlchemy concrete implementation of the RiskPredictionRepository interface."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.risk_prediction.models.risk_prediction_model import RiskPredictionModel
from app.risk_prediction.repositories.risk_prediction_repository import (
    RiskPredictionRepository,
)

logger = logging.getLogger(__name__)


class SQLAlchemyRiskPredictionRepository(RiskPredictionRepository):
    """Concrete repository for risk predictions via SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Queries
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_prediction(self, prediction_id: str) -> Optional[RiskPredictionModel]:
        try:
            stmt = select(RiskPredictionModel).where(
                RiskPredictionModel.id == prediction_id,
            )
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError:
            logger.exception(
                "DB error in get_prediction (id=%s)", prediction_id,
            )
            raise

    async def get_latest_prediction(
        self,
        sensor_id: Optional[str] = None,
        zone_id: Optional[str] = None,
    ) -> Optional[RiskPredictionModel]:
        try:
            stmt = select(RiskPredictionModel)
            if sensor_id is not None:
                stmt = stmt.where(RiskPredictionModel.sensor_id == sensor_id)
            if zone_id is not None:
                stmt = stmt.where(RiskPredictionModel.zone_id == zone_id)
            stmt = stmt.order_by(
                RiskPredictionModel.prediction_timestamp.desc(),
            ).limit(1)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError:
            logger.exception(
                "DB error in get_latest_prediction "
                "(sensor_id=%s, zone_id=%s)", sensor_id, zone_id,
            )
            raise

    async def get_prediction_history(
        self,
        sensor_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[RiskPredictionModel]:
        try:
            stmt = select(RiskPredictionModel)
            stmt = self._apply_filters(
                stmt, sensor_id, zone_id, risk_level, start_time, end_time,
            )
            stmt = (
                stmt.order_by(RiskPredictionModel.prediction_timestamp.desc())
                .offset(offset)
                .limit(limit)
            )
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError:
            logger.exception("DB error in get_prediction_history")
            raise

    async def count_predictions(
        self,
        sensor_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        try:
            stmt = select(func.count(RiskPredictionModel.id))
            stmt = self._apply_filters(
                stmt, sensor_id, zone_id, risk_level, start_time, end_time,
            )
            result = await self._session.execute(stmt)
            return result.scalar_one()
        except SQLAlchemyError:
            logger.exception("DB error in count_predictions")
            raise

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Mutations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def create_prediction(
        self, prediction: RiskPredictionModel,
    ) -> RiskPredictionModel:
        try:
            self._session.add(prediction)
            await self._session.flush()
            await self._session.refresh(prediction)
            return prediction
        except SQLAlchemyError:
            logger.exception("DB error in create_prediction")
            raise

    async def delete_prediction(self, prediction_id: str) -> bool:
        try:
            stmt = delete(RiskPredictionModel).where(
                RiskPredictionModel.id == prediction_id,
            )
            result = await self._session.execute(stmt)
            await self._session.flush()
            return result.rowcount > 0
        except SQLAlchemyError:
            logger.exception(
                "DB error in delete_prediction (id=%s)", prediction_id,
            )
            raise

    async def delete_prediction_history(
        self,
        sensor_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        before: Optional[datetime] = None,
    ) -> int:
        if sensor_id is None and zone_id is None and before is None:
            raise ValueError(
                "At least one filter (sensor_id, zone_id, before) must be "
                "provided to prevent accidental full table deletion."
            )
        try:
            stmt = delete(RiskPredictionModel)
            if sensor_id is not None:
                stmt = stmt.where(RiskPredictionModel.sensor_id == sensor_id)
            if zone_id is not None:
                stmt = stmt.where(RiskPredictionModel.zone_id == zone_id)
            if before is not None:
                stmt = stmt.where(
                    RiskPredictionModel.prediction_timestamp < before,
                )
            result = await self._session.execute(stmt)
            await self._session.flush()
            return result.rowcount
        except SQLAlchemyError:
            logger.exception("DB error in delete_prediction_history")
            raise

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal Helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _apply_filters(stmt, sensor_id, zone_id, risk_level, start_time, end_time):
        """Apply common WHERE clauses to a SELECT or COUNT statement."""
        if sensor_id is not None:
            stmt = stmt.where(RiskPredictionModel.sensor_id == sensor_id)
        if zone_id is not None:
            stmt = stmt.where(RiskPredictionModel.zone_id == zone_id)
        if risk_level is not None:
            stmt = stmt.where(RiskPredictionModel.risk_level == risk_level)
        if start_time is not None:
            stmt = stmt.where(
                RiskPredictionModel.prediction_timestamp >= start_time,
            )
        if end_time is not None:
            stmt = stmt.where(
                RiskPredictionModel.prediction_timestamp <= end_time,
            )
        return stmt
