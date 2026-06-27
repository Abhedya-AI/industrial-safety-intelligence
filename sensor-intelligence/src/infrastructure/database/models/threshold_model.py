"""SQLAlchemy ORM model for the thresholds table."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.database.base import Base


class ThresholdModel(Base):
    """ORM model representing configurable safety thresholds.

    Maps to the 'thresholds' table.
    Thresholds can be set per-sensor (sensor_id not null) or
    per-sensor-type (sensor_id null, sensor_type set).
    """

    __tablename__ = "thresholds"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    sensor_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("sensors.id"), nullable=True
    )
    sensor_type: Mapped[str] = mapped_column(String(20), nullable=False)
    warning_min: Mapped[float] = mapped_column(Float, nullable=False)
    warning_max: Mapped[float] = mapped_column(Float, nullable=False)
    critical_min: Mapped[float] = mapped_column(Float, nullable=False)
    critical_max: Mapped[float] = mapped_column(Float, nullable=False)
    emergency_min: Mapped[float] = mapped_column(Float, nullable=False)
    emergency_max: Mapped[float] = mapped_column(Float, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    sensor = relationship("SensorModel", back_populates="thresholds")

    # Index for threshold lookups
    __table_args__ = (
        Index("ix_thresholds_sensor_active", "sensor_id", "is_active"),
        Index("ix_thresholds_type_active", "sensor_type", "is_active"),
    )

    def __repr__(self) -> str:
        return (
            f"<ThresholdModel(type={self.sensor_type}, "
            f"active={self.is_active})>"
        )
