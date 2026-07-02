"""SQLAlchemy ORM model for the sensor_baselines table.

Stores learned normal operating statistics for each sensor.
Used by the anomaly detection module for comparison.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.shared.database.base import Base


class SensorBaselineModel(Base):
    """ORM model representing a learned baseline for a sensor."""

    __tablename__ = "sensor_baselines"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    sensor_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sensors.id"), nullable=False, unique=True
    )

    # Core statistics
    mean: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    median: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    std_dev: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    variance: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    min_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Normal operating range (mean ± n*std_dev)
    normal_range_low: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    normal_range_high: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Rolling averages (JSON-encoded list)
    rolling_avg_5: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rolling_avg_10: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Seasonal pattern (JSON: per-hour-of-day mean values, 24 entries)
    hourly_pattern: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Trend
    trend_direction: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    trend_slope: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Metadata
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    window_days: Mapped[int] = mapped_column(Integer, nullable=False, default=7)
    sigma_multiplier: Mapped[float] = mapped_column(Float, nullable=False, default=2.0)

    # Timestamps
    learned_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    sensor = relationship("SensorModel", backref="baselines")

    __table_args__ = (
        Index("ix_sensor_baselines_sensor_id", "sensor_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<SensorBaselineModel(sensor_id={self.sensor_id}, "
            f"mean={self.mean}, std_dev={self.std_dev})>"
        )
