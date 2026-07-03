"""SQLAlchemy ORM model for the risk_predictions table.

Fields are aligned with the API specification (GET /risk/current)
and the architecture document (RiskPredictionService).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.shared.database.base import Base


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class RiskPredictionModel(Base):
    """ORM model representing a single risk prediction result.

    Each row captures one prediction event: the features that went in,
    the resulting scores, the model that produced it, and its metadata.
    """

    __tablename__ = "risk_predictions"

    # ── Primary Key ──
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_uuid,
    )

    # ── Context ──
    sensor_id: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, index=True,
        comment="Source sensor business ID (if prediction is sensor-scoped)",
    )
    equipment_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, index=True,
        comment="Equipment associated with this prediction",
    )
    zone_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, index=True,
        comment="Zone where the prediction applies",
    )

    # ── Prediction Output ──
    prediction_timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False,
        comment="When the prediction was computed",
    )
    accident_probability: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Raw probability of accident (0.0–1.0)",
    )
    predicted_risk_score: Mapped[int] = mapped_column(
        Integer, nullable=False,
        comment="Normalised risk score (0–100)",
    )
    risk_level: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True,
        comment="Classified level: LOW | MEDIUM | HIGH | CRITICAL",
    )
    confidence_score: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Model confidence in the prediction (0.0–1.0)",
    )

    # ── Model Provenance ──
    model_name: Mapped[str] = mapped_column(
        String(100), nullable=False,
        comment="Name of the model (e.g. xgboost_ensemble)",
    )
    model_version: Mapped[str] = mapped_column(
        String(50), nullable=False,
        comment="Semantic version of the model that produced this prediction",
    )

    # Provide Python-level defaults via __init__
    def __init__(self, **kwargs):
        kwargs.setdefault("model_version", "1.0.0")
        kwargs.setdefault("status", "COMPLETED")
        super().__init__(**kwargs)

    # ── Risk Breakdown (JSON) ──
    risk_factors: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="JSON-encoded risk factor breakdown",
    )
    top_contributing_factors: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="JSON-encoded top contributing features",
    )
    explanation: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Human-readable risk explanation",
    )

    # ── Status ──
    status: Mapped[str] = mapped_column(
        String(20), nullable=False,
        comment="PENDING | COMPLETED | FAILED | STALE",
    )

    # ── Audit ──
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(),
    )

    # ── Indexes ──
    __table_args__ = (
        Index("ix_risk_pred_zone_ts", "zone_id", "prediction_timestamp"),
        Index("ix_risk_pred_sensor_ts", "sensor_id", "prediction_timestamp"),
        Index("ix_risk_pred_level_ts", "risk_level", "prediction_timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<RiskPredictionModel("
            f"id={self.id}, "
            f"risk_level={self.risk_level}, "
            f"score={self.predicted_risk_score}, "
            f"model={self.model_name}"
            f")>"
        )
