"""SQLAlchemy ORM model for the compound_risk_analyses table.

Fields are aligned with:
  - PS-1 API Specification: POST /risk/compound-analysis
  - PS-1 Common Domain Names v2.0 (enum values, naming conventions)

Naming follows the domain conventions:
  - snake_case for DB fields
  - ISO 8601 UTC timestamps
  - UPPERCASE enum values (LOW | MEDIUM | HIGH | CRITICAL)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.shared.database.base import Base


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class CompoundRiskModel(Base):
    """ORM model representing a single compound risk analysis result.

    Each row captures one compound risk evaluation: the input scores,
    computed compound risk, contributing factors, recommendations,
    and metadata.
    """

    __tablename__ = "compound_risk_analyses"

    # ── Primary Key ──
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_uuid,
    )

    # ── Context ──
    equipment_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, index=True,
        comment="Equipment ID (e.g. EQ001, EQ_BOILER_1)",
    )
    zone_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, index=True,
        comment="Zone ID (e.g. ZONE_A, ZONE_BOILER)",
    )

    # ── Input Scores ──
    anomaly_score: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Anomaly score from anomaly detection (0.0–1.0)",
    )
    accident_probability: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Accident probability from risk prediction (0.0–1.0)",
    )
    risk_score: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Risk score from risk prediction (0–100)",
    )
    sensor_health_score: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Sensor health score (0.0–100.0)",
    )

    # ── Output ──
    compound_risk_score: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Computed compound risk score (0.0–1.0)",
    )
    risk_level: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True,
        comment="Classified level: LOW | MEDIUM | HIGH | CRITICAL",
    )
    confidence_score: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Analysis confidence (0.0–1.0)",
    )

    # ── Breakdown (JSON) ──
    contributing_factors: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="JSON-encoded contributing factor breakdown",
    )
    recommendation: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="JSON-encoded recommended actions or text recommendation",
    )

    # ── Audit ──
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(),
    )

    # ── Indexes ──
    __table_args__ = (
        Index("ix_compound_risk_zone_ts", "zone_id", "created_at"),
        Index("ix_compound_risk_equip_ts", "equipment_id", "created_at"),
        Index("ix_compound_risk_level_ts", "risk_level", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<CompoundRiskModel("
            f"id={self.id}, "
            f"zone_id={self.zone_id}, "
            f"risk_level={self.risk_level}, "
            f"compound_risk_score={self.compound_risk_score}"
            f")>"
        )
