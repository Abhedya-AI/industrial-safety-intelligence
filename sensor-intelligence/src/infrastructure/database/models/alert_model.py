"""SQLAlchemy ORM model for the alerts table."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.database.base import Base


class AlertModel(Base):
    """ORM model representing an actionable alert for operators.

    Maps to the 'alerts' table.
    """

    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    anomaly_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("anomalies.id"), nullable=True
    )
    sensor_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sensors.id"), nullable=False
    )
    level: Mapped[str] = mapped_column(String(20), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    is_acknowledged: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    acknowledged_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Relationships
    anomaly = relationship("AnomalyModel", back_populates="alerts")
    sensor = relationship("SensorModel", back_populates="alerts")

    # Index for dashboard query: unacknowledged alerts sorted by recency
    __table_args__ = (
        Index("ix_alerts_acknowledged_created", "is_acknowledged", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<AlertModel(level={self.level}, title={self.title})>"
