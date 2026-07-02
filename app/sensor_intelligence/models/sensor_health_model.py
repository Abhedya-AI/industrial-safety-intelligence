"""SQLAlchemy ORM model for the sensor_health table.

Persists the latest computed health assessment for each sensor.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.shared.database.base import Base


class SensorHealthModel(Base):
    """ORM model representing the latest health assessment for a sensor."""

    __tablename__ = "sensor_health"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    sensor_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sensors.id"), nullable=False, unique=True
    )

    # Health score 0-100 and classification
    health_score: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    health_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="EXCELLENT"
    )

    # Individual factor scores (0-100 each)
    calibration_score: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    uptime_score: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    stability_score: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    missing_readings_score: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)

    # Metadata
    total_readings: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    anomaly_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    missing_reading_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    sensor = relationship("SensorModel", backref="health")

    __table_args__ = (
        Index("ix_sensor_health_sensor_id", "sensor_id"),
        Index("ix_sensor_health_status", "health_status"),
    )

    def __repr__(self) -> str:
        return (
            f"<SensorHealthModel(sensor_id={self.sensor_id}, "
            f"score={self.health_score}, status={self.health_status})>"
        )
