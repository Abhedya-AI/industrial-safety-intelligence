"""SQLAlchemy ORM model for the facility_snapshots table.

Captures a point-in-time snapshot of the entire facility state,
including the aggregated health score and top-level metrics.

The full zone-level detail is stored as a JSON payload for
flexible cold-start recovery without needing to join.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.shared.database.base import Base


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class FacilitySnapshotModel(Base):
    """ORM model representing a facility-wide snapshot.

    Each row is one snapshot — created manually via API, or
    automatically when certain trigger conditions are met
    (critical hazard, health change, threshold exceedance).
    """

    __tablename__ = "facility_snapshots"

    # ── Primary Key ──
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_uuid,
    )

    # ── Identity ──
    snapshot_id: Mapped[str] = mapped_column(
        String(36), unique=True, nullable=False, index=True,
        default=_generate_uuid,
        comment="Unique identifier for this snapshot",
    )

    # ── Audit ──
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(),
        index=True,
        comment="When this snapshot was taken (UTC)",
    )

    # ── Aggregated Metrics ──
    facility_health: Mapped[float] = mapped_column(
        Float, nullable=False, default=100.0,
        comment="Overall facility health (0-100)",
    )
    total_zones: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    active_hazards: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    critical_zones: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    workers_at_risk: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    events_processed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )

    # ── Full Payload (JSON) ──
    snapshot_payload: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="JSON-encoded full facility + zone state",
    )

    # ── Trigger Reason ──
    trigger_reason: Mapped[str] = mapped_column(
        String(100), nullable=False, default="manual",
        comment="What triggered this snapshot: manual | critical_hazard | "
        "health_change | compound_risk_threshold | startup",
    )

    def __repr__(self) -> str:
        return (
            f"<FacilitySnapshotModel("
            f"snapshot_id={self.snapshot_id}, "
            f"facility_health={self.facility_health}, "
            f"trigger={self.trigger_reason}"
            f")>"
        )
