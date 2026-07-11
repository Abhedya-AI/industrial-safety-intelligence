"""Pydantic schemas for the Digital Twin snapshot REST APIs.

Covers:
  - GET /twin/snapshots          (list)
  - GET /twin/snapshots/{id}     (detail)
  - POST /twin/snapshot          (create)
  - DELETE /twin/snapshots/{id}  (delete)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# List entry (lightweight)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SnapshotSummarySchema(BaseModel):
    """Lightweight summary for snapshot list responses."""

    snapshot_id: str
    created_at: str = ""
    facility_health: float = 100.0
    total_zones: int = 0
    active_hazards: int = 0
    critical_zones: int = 0
    workers_at_risk: int = 0
    events_processed: int = 0
    trigger_reason: str = "manual"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Zone state within a snapshot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ZoneStateSnapshotSchema(BaseModel):
    """Zone-level data within a snapshot."""

    zone_id: str
    risk_score: float = 0.0
    compound_risk_score: float = 0.0
    hazard_count: int = 0
    anomaly_count: int = 0
    equipment_health: float = 100.0
    worker_count: int = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API responses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SnapshotListResponse(BaseModel):
    """Response for GET /twin/snapshots."""

    success: bool = True
    timestamp: str = ""
    total: int = 0
    offset: int = 0
    limit: int = 50
    snapshots: List[SnapshotSummarySchema] = Field(
        default_factory=list,
    )


class SnapshotDetailResponse(BaseModel):
    """Response for GET /twin/snapshots/{snapshot_id}."""

    success: bool = True
    timestamp: str = ""
    snapshot: Optional[SnapshotSummarySchema] = None
    zone_states: List[ZoneStateSnapshotSchema] = Field(
        default_factory=list,
    )
    snapshot_payload: Optional[Dict[str, Any]] = None


class SnapshotCreateResponse(BaseModel):
    """Response for POST /twin/snapshot."""

    success: bool = True
    timestamp: str = ""
    message: str = ""
    snapshot: Optional[SnapshotSummarySchema] = None


class SnapshotDeleteResponse(BaseModel):
    """Response for DELETE /twin/snapshots/{snapshot_id}."""

    success: bool = True
    timestamp: str = ""
    message: str = ""
    deleted: bool = False
