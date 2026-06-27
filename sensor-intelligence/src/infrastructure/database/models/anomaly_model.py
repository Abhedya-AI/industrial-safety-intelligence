"""SQLAlchemy ORM model for the anomalies table."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.database.base import Base


class AnomalyModel(Base):
    """ORM model representing a detected anomaly in sensor data.

    Maps to the 'anomalies' table.
    """

    __tablename__ = "anomalies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    reading_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sensor_readings.id"), nullable=False
    )
    sensor_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sensors.id"), nullable=False
    )
    anomaly_type: Mapped[str] = mapped_column(String(20), nullable=False)
    severity_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    is_resolved: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    detected_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    reading = relationship("ReadingModel", back_populates="anomalies")
    sensor = relationship("SensorModel", back_populates="anomalies")
    alerts = relationship("AlertModel", back_populates="anomaly", lazy="select")

    # Index for querying unresolved anomalies per sensor
    __table_args__ = (
        Index("ix_anomalies_sensor_detected", "sensor_id", "detected_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<AnomalyModel(type={self.anomaly_type}, "
            f"severity={self.severity_score})>"
        )
