"""SQLAlchemy ORM model for the sensors table.

Field names aligned to the finalized API specification
(PS1_Detailed_API_Specifications_V2, endpoints 6 & 7).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, Float, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.shared.database.base import Base


def generate_uuid() -> str:
    """Generate a UUID4 string for use as a primary key."""
    return str(uuid.uuid4())


class SensorModel(Base):
    """ORM model representing a physical IoT sensor."""

    __tablename__ = "sensors"

    # ── Internal Primary Key (UUID) ──
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )

    # ── Business Identity (matches API spec field "sensor_id") ──
    sensor_id: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    sensor_name: Mapped[str] = mapped_column(String(150), nullable=False)

    # ── Classification ──
    sensor_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="NORMAL", index=True
    )

    # ── Location & Equipment ──
    location_zone: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, index=True
    )
    equipment_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, index=True
    )

    # ── Hardware Metadata ──
    manufacturer: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # ── Measurement Spec ──
    unit: Mapped[str] = mapped_column(String(20), nullable=False)
    min_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Calibration & Quality ──
    accuracy_rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    installation_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    last_calibration: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    next_calibration_due: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    # ── Audit ──
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # ── Relationships ──
    readings = relationship("ReadingModel", back_populates="sensor", lazy="dynamic")
    anomalies = relationship("AnomalyModel", back_populates="sensor", lazy="dynamic")
    alerts = relationship("AlertModel", back_populates="sensor", lazy="dynamic")
    thresholds = relationship("ThresholdModel", back_populates="sensor", lazy="dynamic")

    def __repr__(self) -> str:
        return (
            f"<SensorModel(id={self.id}, sensor_id={self.sensor_id}, "
            f"type={self.sensor_type}, status={self.status})>"
        )
