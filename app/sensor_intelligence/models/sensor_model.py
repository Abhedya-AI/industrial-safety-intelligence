"""SQLAlchemy ORM model for the sensors table."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, Float, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.shared.database.base import Base


class SensorModel(Base):
    """ORM model representing a physical IoT sensor."""

    __tablename__ = "sensors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    sensor_id: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    sensor_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    location_zone: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    equipment_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="NORMAL")
    unit: Mapped[str] = mapped_column(String(20), nullable=False)
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    calibration_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    accuracy_rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    readings = relationship("ReadingModel", back_populates="sensor", lazy="dynamic")
    anomalies = relationship("AnomalyModel", back_populates="sensor", lazy="dynamic")
    alerts = relationship("AlertModel", back_populates="sensor", lazy="dynamic")
    thresholds = relationship("ThresholdModel", back_populates="sensor", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<SensorModel(sensor_id={self.sensor_id}, type={self.sensor_type})>"
