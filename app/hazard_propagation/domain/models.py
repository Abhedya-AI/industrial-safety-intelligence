"""Domain models for the Hazard Propagation Engine.

Pure domain entities — no ORM coupling, no framework dependencies.
These are used by the propagation simulation engine and graph layer.

Models:
  - Hazard              — a detected hazard event
  - PropagationPath     — a single path through which a hazard spreads
  - ZoneRiskState       — current risk state of a zone during propagation
  - HazardPropagation   — complete propagation simulation result

All field names follow PS-1 v2.0 snake_case convention (§5).
All IDs follow UPPERCASE_SUFFIX convention (§6.1 Rule #2).
All timestamps are ISO 8601 UTC (§6.1 Rule #3).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.hazard_propagation.domain.value_objects import (
    HazardType,
    PropagationLevel,
    PropagationStatus,
    RiskLevel,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hazard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class Hazard:
    """A detected hazard event at a specific location.

    Attributes:
        hazard_id:    Unique identifier (UUID).
        hazard_type:  Type from PS-1 §4.6 HazardType enum.
        origin_zone:  Zone ID where the hazard originated.
        severity:     Risk level of the hazard at origin.
        detected_at:  When the hazard was first detected (ISO 8601 UTC).
        description:  Optional human-readable description.
        sensor_id:    Sensor that triggered the detection (if any).
        equipment_id: Equipment involved (if any).
        is_active:    Whether the hazard is still ongoing.
        metadata:     Additional key-value data.
    """

    hazard_type: HazardType
    origin_zone: str
    severity: RiskLevel = RiskLevel.HIGH
    hazard_id: str = field(default_factory=_new_id)
    detected_at: datetime = field(default_factory=_utc_now)
    description: str = ""
    sensor_id: Optional[str] = None
    equipment_id: Optional[str] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_critical(self) -> bool:
        """True if severity is CRITICAL."""
        return self.severity == RiskLevel.CRITICAL

    def deactivate(self) -> None:
        """Mark hazard as no longer active."""
        self.is_active = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PropagationPath
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class PropagationPath:
    """A single path through which a hazard spreads between zones.

    Represents an edge in the propagation graph: the hazard can travel
    from ``from_zone`` to ``to_zone`` with a given probability and
    estimated time delay.

    Attributes:
        from_zone:       Source zone ID.
        to_zone:         Destination zone ID.
        probability:     Likelihood of propagation (0.0–1.0).
        estimated_time_minutes: Estimated time for hazard to reach
                                the destination zone (minutes).
        path_type:       Type of connection (e.g. "CONNECTED_TO",
                         "VENTILATION", "PIPE").
        distance_meters: Physical distance between zones (if known).
        blocked:         True if the path is blocked (e.g. fire door closed).
    """

    from_zone: str
    to_zone: str
    probability: float = 1.0
    estimated_time_minutes: float = 5.0
    path_type: str = "CONNECTED_TO"
    distance_meters: Optional[float] = None
    blocked: bool = False

    def __post_init__(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"probability must be 0.0–1.0, got {self.probability}"
            )
        if self.estimated_time_minutes < 0:
            raise ValueError(
                f"estimated_time_minutes must be >= 0, got "
                f"{self.estimated_time_minutes}"
            )

    @property
    def is_passable(self) -> bool:
        """True if the path is not blocked and has non-zero probability."""
        return not self.blocked and self.probability > 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ZoneRiskState
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ZoneRiskState:
    """Current risk state of a zone during hazard propagation.

    Tracks how a zone is affected by a propagating hazard, including
    the cumulative risk score and the time at which the hazard is
    estimated to arrive.

    Attributes:
        zone_id:              Zone identifier.
        zone_name:            Human-readable name.
        risk_level:           Current risk classification.
        risk_score:           Numerical score (0–100).
        is_origin:            True if this is the hazard origin zone.
        is_affected:          True if the hazard has reached this zone.
        arrival_time_minutes: Estimated time for hazard arrival (0 if origin).
        propagation_probability: Cumulative probability of being affected.
        worker_count:         Number of workers currently in the zone.
        equipment_count:      Number of equipment items in the zone.
        active_hazards:       List of active hazard types in this zone.
    """

    zone_id: str
    zone_name: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0
    is_origin: bool = False
    is_affected: bool = False
    arrival_time_minutes: float = 0.0
    propagation_probability: float = 0.0
    worker_count: int = 0
    equipment_count: int = 0
    active_hazards: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not 0.0 <= self.risk_score <= 100.0:
            raise ValueError(
                f"risk_score must be 0–100, got {self.risk_score}"
            )

    @property
    def has_workers_at_risk(self) -> bool:
        """True if workers are present in an affected zone."""
        return self.is_affected and self.worker_count > 0

    @property
    def requires_evacuation(self) -> bool:
        """True if the zone risk warrants evacuation."""
        return self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) and self.is_affected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HazardPropagation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class HazardPropagation:
    """Complete result of a hazard propagation simulation.

    Aggregates the hazard source, all affected zones, propagation paths,
    and recommended actions into a single response object.

    Attributes:
        propagation_id:           Unique ID for this simulation.
        hazard:                   The source hazard being propagated.
        status:                   Simulation status.
        propagation_level:        Overall severity classification.
        affected_zones:           List of ZoneRiskState for each zone.
        propagation_paths:        List of paths the hazard can travel.
        affected_workers:         Worker IDs in the impact area.
        impact_radius_meters:     Estimated radius of impact.
        time_to_critical_minutes: Estimated time until critical threshold.
        recommended_action:       Human-readable recommended response.
        created_at:               When the simulation was run.
        completed_at:             When the simulation finished.
        metadata:                 Additional context.
    """

    hazard: Hazard
    propagation_id: str = field(default_factory=_new_id)
    status: PropagationStatus = PropagationStatus.PENDING
    propagation_level: PropagationLevel = PropagationLevel.CONTAINED
    affected_zones: List[ZoneRiskState] = field(default_factory=list)
    propagation_paths: List[PropagationPath] = field(default_factory=list)
    affected_workers: List[str] = field(default_factory=list)
    impact_radius_meters: float = 0.0
    time_to_critical_minutes: float = 0.0
    recommended_action: str = ""
    created_at: datetime = field(default_factory=_utc_now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Computed properties ──

    @property
    def total_affected_zones(self) -> int:
        """Number of zones affected by the hazard."""
        return sum(1 for z in self.affected_zones if z.is_affected)

    @property
    def total_workers_at_risk(self) -> int:
        """Total number of workers in affected zones."""
        return sum(
            z.worker_count for z in self.affected_zones if z.is_affected
        )

    @property
    def is_completed(self) -> bool:
        return self.status == PropagationStatus.COMPLETED

    @property
    def affected_zone_ids(self) -> List[str]:
        """List of zone IDs that are affected."""
        return [z.zone_id for z in self.affected_zones if z.is_affected]

    @property
    def origin_zone(self) -> Optional[ZoneRiskState]:
        """The origin zone state, if present."""
        for z in self.affected_zones:
            if z.is_origin:
                return z
        return None

    # ── Mutations ──

    def mark_completed(self) -> None:
        """Mark the simulation as completed."""
        self.status = PropagationStatus.COMPLETED
        self.completed_at = _utc_now()

    def mark_failed(self, reason: str = "") -> None:
        """Mark the simulation as failed."""
        self.status = PropagationStatus.FAILED
        self.completed_at = _utc_now()
        if reason:
            self.metadata["failure_reason"] = reason

    def add_affected_zone(self, zone: ZoneRiskState) -> None:
        """Add a zone to the affected list."""
        self.affected_zones.append(zone)

    def add_propagation_path(self, path: PropagationPath) -> None:
        """Add a propagation path."""
        self.propagation_paths.append(path)
