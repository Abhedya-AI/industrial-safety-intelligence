"""SQLAlchemy ORM model for the sensor_readings table."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.database.base import Base


class ReadingModel(Base):
    """ORM model representing a single sensor measurement.

    Maps to the 'sensor_readings' table.
    This is the high-volume table — indexed on (sensor_id, timestamp DESC).
    """

    __tablename__ = "sensor_readings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    sensor_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sensors.id"), nullable=False
    )
    value: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    raw_metadata: Mapped[str | None] = mapped_column(Text, nullable=True)
    received_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Relationships
    sensor = relationship("SensorModel", back_populates="readings")
    anomalies = relationship("AnomalyModel", back_populates="reading", lazy="select")

    # Composite index for efficient time-range queries per sensor
    __table_args__ = (
        Index("ix_readings_sensor_timestamp", "sensor_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<ReadingModel(sensor_id={self.sensor_id}, value={self.value})>"
