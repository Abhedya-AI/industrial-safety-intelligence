"""SQLAlchemy concrete implementation of the CompoundRiskRepository interface."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.compound_risk.models.compound_risk_model import CompoundRiskModel
from app.compound_risk.repositories.compound_risk_repository import (
    CompoundRiskRepository,
)

logger = logging.getLogger(__name__)


class SQLAlchemyCompoundRiskRepository(CompoundRiskRepository):
    """Concrete repository for compound risk analyses via SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Queries
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_by_id(self, analysis_id: str) -> Optional[CompoundRiskModel]:
        try:
            stmt = select(CompoundRiskModel).where(
                CompoundRiskModel.id == analysis_id,
            )
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError:
            logger.exception(
                "DB error in get_by_id (id=%s)", analysis_id,
            )
            raise

    async def get_latest(
        self,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
    ) -> Optional[CompoundRiskModel]:
        try:
            stmt = select(CompoundRiskModel)
            if zone_id is not None:
                stmt = stmt.where(CompoundRiskModel.zone_id == zone_id)
            if equipment_id is not None:
                stmt = stmt.where(CompoundRiskModel.equipment_id == equipment_id)
            stmt = stmt.order_by(
                CompoundRiskModel.created_at.desc(),
            ).limit(1)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError:
            logger.exception(
                "DB error in get_latest "
                "(zone_id=%s, equipment_id=%s)", zone_id, equipment_id,
            )
            raise

    async def get_history(
        self,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[CompoundRiskModel]:
        try:
            stmt = select(CompoundRiskModel)
            stmt = self._apply_filters(
                stmt, zone_id, equipment_id, risk_level, start_time, end_time,
            )
            stmt = (
                stmt.order_by(CompoundRiskModel.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError:
            logger.exception("DB error in get_history")
            raise

    async def count(
        self,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        try:
            stmt = select(func.count(CompoundRiskModel.id))
            stmt = self._apply_filters(
                stmt, zone_id, equipment_id, risk_level, start_time, end_time,
            )
            result = await self._session.execute(stmt)
            return result.scalar_one()
        except SQLAlchemyError:
            logger.exception("DB error in count")
            raise

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Mutations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def create(
        self, analysis: CompoundRiskModel,
    ) -> CompoundRiskModel:
        try:
            self._session.add(analysis)
            await self._session.flush()
            await self._session.refresh(analysis)
            return analysis
        except SQLAlchemyError:
            logger.exception("DB error in create")
            raise

    async def delete(self, analysis_id: str) -> bool:
        try:
            stmt = delete(CompoundRiskModel).where(
                CompoundRiskModel.id == analysis_id,
            )
            result = await self._session.execute(stmt)
            await self._session.flush()
            return result.rowcount > 0
        except SQLAlchemyError:
            logger.exception(
                "DB error in delete (id=%s)", analysis_id,
            )
            raise

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal Helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _apply_filters(stmt, zone_id, equipment_id, risk_level, start_time, end_time):
        """Apply common WHERE clauses to a SELECT or COUNT statement."""
        if zone_id is not None:
            stmt = stmt.where(CompoundRiskModel.zone_id == zone_id)
        if equipment_id is not None:
            stmt = stmt.where(CompoundRiskModel.equipment_id == equipment_id)
        if risk_level is not None:
            stmt = stmt.where(CompoundRiskModel.risk_level == risk_level)
        if start_time is not None:
            stmt = stmt.where(CompoundRiskModel.created_at >= start_time)
        if end_time is not None:
            stmt = stmt.where(CompoundRiskModel.created_at <= end_time)
        return stmt
