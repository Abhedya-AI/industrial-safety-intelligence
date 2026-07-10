"""Session-scoped implementation of CompoundRiskRepository.

Wraps ``SQLAlchemyCompoundRiskRepository`` with per-operation session
management. Each repository call:

  1. Creates a new ``AsyncSession`` from the factory
  2. Delegates to ``SQLAlchemyCompoundRiskRepository``
  3. Commits and closes the session

This is required for the **Kafka consumer path**, where the handler
runs inside ``asyncio.run()`` per event (no shared request-scoped
session). The API path uses FastAPI's ``Depends(get_async_session)``
instead.

Thread-safety: safe — each call gets its own session.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.compound_risk.models.compound_risk_model import CompoundRiskModel
from app.compound_risk.repositories.compound_risk_repository import (
    CompoundRiskRepository,
)
from app.compound_risk.repositories.sqlalchemy_compound_risk_repo import (
    SQLAlchemyCompoundRiskRepository,
)

logger = logging.getLogger(__name__)


class SessionScopedCompoundRiskRepository(CompoundRiskRepository):
    """Repository that creates a fresh session for each operation.

    Designed for use outside the FastAPI request lifecycle (e.g. Kafka
    consumer background thread).

    Args:
        session_factory: An ``async_sessionmaker`` that produces
            ``AsyncSession`` instances bound to the application engine.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    # ── Queries ──

    async def get_by_id(self, analysis_id: str) -> Optional[CompoundRiskModel]:
        async with self._session_factory() as session:
            repo = SQLAlchemyCompoundRiskRepository(session)
            return await repo.get_by_id(analysis_id)

    async def get_latest(
        self,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
    ) -> Optional[CompoundRiskModel]:
        async with self._session_factory() as session:
            repo = SQLAlchemyCompoundRiskRepository(session)
            return await repo.get_latest(zone_id=zone_id, equipment_id=equipment_id)

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
        async with self._session_factory() as session:
            repo = SQLAlchemyCompoundRiskRepository(session)
            return await repo.get_history(
                zone_id=zone_id, equipment_id=equipment_id,
                risk_level=risk_level, start_time=start_time,
                end_time=end_time, offset=offset, limit=limit,
            )

    async def count(
        self,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        async with self._session_factory() as session:
            repo = SQLAlchemyCompoundRiskRepository(session)
            return await repo.count(
                zone_id=zone_id, equipment_id=equipment_id,
                risk_level=risk_level, start_time=start_time,
                end_time=end_time,
            )

    # ── Mutations ──

    async def create(self, analysis: CompoundRiskModel) -> CompoundRiskModel:
        async with self._session_factory() as session:
            repo = SQLAlchemyCompoundRiskRepository(session)
            result = await repo.create(analysis)
            await session.commit()
            logger.debug(
                "Persisted compound risk analysis: id=%s zone=%s",
                result.id, result.zone_id,
            )
            return result

    async def delete(self, analysis_id: str) -> bool:
        async with self._session_factory() as session:
            repo = SQLAlchemyCompoundRiskRepository(session)
            deleted = await repo.delete(analysis_id)
            await session.commit()
            return deleted
