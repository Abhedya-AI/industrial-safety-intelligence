"""SQLAlchemy concrete implementation of the AlertRepository interface."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.sensor_intelligence.domain.entities.alert import Alert
from app.sensor_intelligence.domain.value_objects.alert_level import AlertLevel
from app.sensor_intelligence.models.alert_model import AlertModel
from app.sensor_intelligence.repositories.alert_repository import (
    AlertRepository,
    AlertSummary,
)

logger = logging.getLogger(__name__)


class SQLAlchemyAlertRepository(AlertRepository):
    """Concrete repository for alerts via SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Mapping helpers ──

    @staticmethod
    def _to_domain(model: AlertModel) -> Alert:
        """Map ORM model → domain entity."""
        return Alert(
            id=UUID(model.id),
            sensor_id=UUID(model.sensor_id),
            level=AlertLevel(model.level),
            title=model.title,
            message=model.message,
            anomaly_id=UUID(model.anomaly_id) if model.anomaly_id else None,
            is_acknowledged=model.is_acknowledged,
            acknowledged_by=model.acknowledged_by,
            acknowledged_at=model.acknowledged_at,
            created_at=model.created_at,
        )

    @staticmethod
    def _to_model(entity: Alert) -> AlertModel:
        """Map domain entity → ORM model."""
        return AlertModel(
            id=str(entity.id),
            sensor_id=str(entity.sensor_id),
            level=entity.level.value,
            title=entity.title,
            message=entity.message,
            anomaly_id=str(entity.anomaly_id) if entity.anomaly_id else None,
            is_acknowledged=entity.is_acknowledged,
            acknowledged_by=entity.acknowledged_by,
            acknowledged_at=entity.acknowledged_at,
        )

    # ── CRUD ──

    async def save(self, alert: Alert) -> Alert:
        try:
            model = self._to_model(alert)
            self._session.add(model)
            await self._session.flush()
            await self._session.refresh(model)
            logger.info("Saved alert %s (level=%s)", model.id, model.level)
            return self._to_domain(model)
        except SQLAlchemyError:
            logger.exception("DB error saving alert")
            raise

    async def get_by_id(self, id: UUID) -> Alert | None:
        try:
            stmt = select(AlertModel).where(AlertModel.id == str(id))
            result = await self._session.execute(stmt)
            model = result.scalar_one_or_none()
            return self._to_domain(model) if model else None
        except SQLAlchemyError:
            logger.exception("DB error in get_by_id (id=%s)", id)
            raise

    async def list_all(
        self,
        level: AlertLevel | None = None,
        is_acknowledged: bool | None = None,
        sensor_id: UUID | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[Alert]:
        try:
            stmt = select(AlertModel)
            if level is not None:
                stmt = stmt.where(AlertModel.level == level.value)
            if is_acknowledged is not None:
                stmt = stmt.where(AlertModel.is_acknowledged == is_acknowledged)
            if sensor_id is not None:
                stmt = stmt.where(AlertModel.sensor_id == str(sensor_id))
            stmt = stmt.order_by(AlertModel.created_at.desc()).offset(offset).limit(limit)
            result = await self._session.execute(stmt)
            return [self._to_domain(m) for m in result.scalars().all()]
        except SQLAlchemyError:
            logger.exception("DB error in list_all")
            raise

    async def count(
        self,
        level: AlertLevel | None = None,
        is_acknowledged: bool | None = None,
        sensor_id: UUID | None = None,
    ) -> int:
        try:
            stmt = select(func.count(AlertModel.id))
            if level is not None:
                stmt = stmt.where(AlertModel.level == level.value)
            if is_acknowledged is not None:
                stmt = stmt.where(AlertModel.is_acknowledged == is_acknowledged)
            if sensor_id is not None:
                stmt = stmt.where(AlertModel.sensor_id == str(sensor_id))
            result = await self._session.execute(stmt)
            return result.scalar_one()
        except SQLAlchemyError:
            logger.exception("DB error in count")
            raise

    async def acknowledge(
        self, id: UUID, acknowledged_by: str, acknowledged_at: datetime
    ) -> Alert:
        try:
            stmt = (
                update(AlertModel)
                .where(AlertModel.id == str(id))
                .values(
                    is_acknowledged=True,
                    acknowledged_by=acknowledged_by,
                    acknowledged_at=acknowledged_at,
                )
            )
            await self._session.execute(stmt)
            await self._session.flush()
            return await self.get_by_id(id)
        except SQLAlchemyError:
            logger.exception("DB error acknowledging alert %s", id)
            raise

    async def get_unacknowledged(self) -> list[Alert]:
        try:
            stmt = (
                select(AlertModel)
                .where(AlertModel.is_acknowledged == False)  # noqa: E712
                .order_by(AlertModel.created_at.desc())
            )
            result = await self._session.execute(stmt)
            return [self._to_domain(m) for m in result.scalars().all()]
        except SQLAlchemyError:
            logger.exception("DB error in get_unacknowledged")
            raise

    async def get_summary(self) -> AlertSummary:
        try:
            stmt = select(
                AlertModel.level,
                func.count(AlertModel.id),
            ).group_by(AlertModel.level)
            result = await self._session.execute(stmt)
            counts = {level: cnt for level, cnt in result.all()}

            # Count unacknowledged
            unack_stmt = select(func.count(AlertModel.id)).where(
                AlertModel.is_acknowledged == False  # noqa: E712
            )
            unack_result = await self._session.execute(unack_stmt)
            unack = unack_result.scalar_one()

            return AlertSummary(
                info=counts.get("INFO", 0),
                warning=counts.get("WARNING", 0),
                critical=counts.get("CRITICAL", 0),
                emergency=counts.get("EMERGENCY", 0),
                total=sum(counts.values()),
                unacknowledged=unack,
            )
        except SQLAlchemyError:
            logger.exception("DB error in get_summary")
            raise

    # ── Extended queries for AlertService ──

    async def get_active_alert_for_sensor(
        self, sensor_id: UUID, alert_type: str
    ) -> Alert | None:
        """Get an existing unacknowledged alert of the same type for a sensor.

        Used for duplicate prevention.
        """
        try:
            stmt = (
                select(AlertModel)
                .where(
                    AlertModel.sensor_id == str(sensor_id),
                    AlertModel.is_acknowledged == False,  # noqa: E712
                    AlertModel.title == alert_type,
                )
                .order_by(AlertModel.created_at.desc())
                .limit(1)
            )
            result = await self._session.execute(stmt)
            model = result.scalar_one_or_none()
            return self._to_domain(model) if model else None
        except SQLAlchemyError:
            logger.exception("DB error in get_active_alert_for_sensor")
            raise

    async def resolve_alerts_for_sensor(self, sensor_id: UUID, resolved_at: datetime) -> int:
        """Acknowledge all unacknowledged alerts for a sensor (auto-resolve).

        Returns the number of alerts resolved.
        """
        try:
            stmt = (
                update(AlertModel)
                .where(
                    AlertModel.sensor_id == str(sensor_id),
                    AlertModel.is_acknowledged == False,  # noqa: E712
                )
                .values(
                    is_acknowledged=True,
                    acknowledged_by="system:auto_resolve",
                    acknowledged_at=resolved_at,
                )
            )
            result = await self._session.execute(stmt)
            await self._session.flush()
            return result.rowcount
        except SQLAlchemyError:
            logger.exception("DB error resolving alerts for sensor %s", sensor_id)
            raise
