"""SQLAlchemy ORM model for the zone_state_snapshots table.

Each row captures one zone's state at the time a facility snapshot
was taken. Linked to the facility snapshot by snapshot_id.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.shared.database.base import Base


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class ZoneStateModel(Base):
    """ORM model representing one zone's state within a snapshot."""

    __tablename__ = "zone_state_snapshots"

    # ── Primary Key ──
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_uuid,
    )

    # ── Snapshot Reference ──
    snapshot_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True,
        comment="References facility_snapshots.snapshot_id",
    )

    # ── Zone Identity ──
    zone_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True,
    )

    # ── Metrics ──
    risk_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Overall risk score at snapshot time",
    )
    compound_risk_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
    )
    hazard_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    anomaly_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    equipment_health: Mapped[float] = mapped_column(
        Float, nullable=False, default=100.0,
    )
    worker_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )

    # ── Full Zone State (JSON) ──
    state_payload: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="JSON-encoded full zone state",
    )

    # ── Audit ──
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(),
    )

    # ── Indexes ──
    __table_args__ = (
        Index("ix_zone_state_snapshot_zone", "snapshot_id", "zone_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<ZoneStateModel("
            f"snapshot_id={self.snapshot_id}, "
            f"zone_id={self.zone_id}, "
            f"risk_score={self.risk_score}"
            f")>"
        )
