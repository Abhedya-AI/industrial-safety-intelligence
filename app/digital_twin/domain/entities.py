"""Digital Twin domain entities.

Zone-centric state model. Each zone aggregates live data from all
upstream modules:

  - Sensor Intelligence  → sensor health, anomalies, readings
  - Risk Prediction      → predicted risk score, risk level
  - Compound Risk        → compound risk score, contributing factors
  - Hazard Propagation   → active hazards, affected neighbors

All entities are plain dataclasses — no ORM coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.digital_twin.domain.enums import HeatmapColor, RiskLevel


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sensor state (per sensor)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class SensorReading:
    """Latest reading from a single sensor."""

    sensor_id: str
    sensor_type: str = ""
    value: float = 0.0
    unit: str = ""
    anomaly_score: float = 0.0
    is_anomalous: bool = False
    health_score: float = 100.0
    status: str = "ACTIVE"
    last_updated: str = field(default_factory=_now)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Equipment state (per equipment)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class EquipmentState:
    """Live state of a single piece of equipment."""

    equipment_id: str
    equipment_type: str = ""
    zone_id: str = ""
    operational_status: str = "ACTIVE"
    health_score: float = 100.0
    risk_score: float = 0.0
    sensor_count: int = 0
    last_updated: str = field(default_factory=_now)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Active hazard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ActiveHazard:
    """An active hazard affecting a zone."""

    hazard_id: str
    hazard_type: str
    severity: str = "HIGH"
    origin_zone: str = ""
    propagation_level: str = "CONTAINED"
    affected_zones: List[str] = field(default_factory=list)
    detected_at: str = field(default_factory=_now)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Zone state (the core twin entity)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ZoneState:
    """Complete live state of a single facility zone.

    This is the central entity of the Digital Twin — one ZoneState
    per physical zone, continuously updated by Kafka events.
    """

    zone_id: str
    zone_name: str = ""

    # ── Sensor Intelligence ──
    sensor_health: float = 100.0
    anomaly_count: int = 0
    sensor_count: int = 0
    latest_sensor_readings: Dict[str, SensorReading] = field(
        default_factory=dict,
    )

    # ── Risk Prediction ──
    predicted_risk_score: float = 0.0
    risk_level: str = "LOW"
    accident_probability: float = 0.0

    # ── Compound Risk Intelligence ──
    compound_risk_score: float = 0.0
    compound_risk_level: str = "LOW"
    compound_risk_confidence: float = 0.0
    contributing_factors: Dict[str, float] = field(default_factory=dict)

    # ── Hazard Propagation ──
    active_hazards: List[ActiveHazard] = field(default_factory=list)
    affected_neighbors: List[str] = field(default_factory=list)

    # ── Derived / Context ──
    workers_at_risk: int = 0
    worker_capacity: int = 0
    current_worker_count: int = 0
    equipment: List[EquipmentState] = field(default_factory=list)
    connected_zones: List[str] = field(default_factory=list)

    # ── Metadata ──
    last_updated: str = field(default_factory=_now)
    event_count: int = 0

    # ── Computed properties ──

    @property
    def active_hazard_count(self) -> int:
        return len(self.active_hazards)

    @property
    def is_critical(self) -> bool:
        return (
            self.risk_level == "CRITICAL"
            or self.compound_risk_level == "CRITICAL"
            or self.active_hazard_count > 0
        )

    @property
    def overall_risk_score(self) -> float:
        """Blended risk score from all sources."""
        scores = [self.predicted_risk_score]
        if self.compound_risk_score > 0:
            scores.append(self.compound_risk_score)
        return max(scores) if scores else 0.0

    @property
    def heatmap_color(self) -> str:
        return HeatmapColor.from_score(self.overall_risk_score).value

    def touch(self) -> None:
        """Update the last_updated timestamp and increment event count."""
        self.last_updated = _now()
        self.event_count += 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Facility-wide snapshot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class FacilityState:
    """Facility-wide aggregated state derived from all zone states."""

    facility_health: float = 100.0
    total_zones: int = 0
    active_hazards: int = 0
    critical_zones: int = 0
    workers_at_risk: int = 0
    total_workers: int = 0
    total_equipment: int = 0
    total_sensors: int = 0
    total_anomalies: int = 0
    average_risk_score: float = 0.0
    max_risk_score: float = 0.0
    events_processed: int = 0
    last_updated: str = field(default_factory=_now)
    zone_ids: List[str] = field(default_factory=list)
