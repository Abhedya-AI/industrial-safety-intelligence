"""Graph entities for the Hazard Propagation Engine.

Implements the entity-relationship model from PS-1 SentinelAI Common
Domain Names v2.0, §2 (Entity Relationships).

Graph nodes:
  - Zone       — a physical zone in the facility
  - Equipment  — equipment located in a zone
  - Sensor     — sensor monitoring equipment or zone
  - HazardNode — a hazard affecting zones

Relationships (from PS-1 §2, Table 4):
  - Zone CONNECTED_TO Zone
  - Zone CONTAINS Equipment
  - Equipment HAS_SENSOR Sensor
  - Hazard AFFECTS Zone
  - Hazard CAUSES Incident

All IDs follow UPPERCASE_SUFFIX convention (§6.1 Rule #2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Relationship type constants (from PS-1 §2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RelationshipType:
    """Standardised relationship names from PS-1 v2.0, §2."""

    # Zone relationships
    CONNECTED_TO = "CONNECTED_TO"       # Zone → Zone
    CONTAINS = "CONTAINS"               # Zone → Equipment
    HAS_SENSOR = "HAS_SENSOR"           # Zone/Equipment → Sensor
    HAS_INCIDENT = "HAS_INCIDENT"       # Zone → Incident

    # Equipment relationships
    LOCATED_IN = "LOCATED_IN"           # Equipment → Zone
    HAS_MAINTENANCE = "HAS_MAINTENANCE" # Equipment → Maintenance

    # Sensor relationships
    MONITORS = "MONITORS"               # Sensor → Equipment

    # Hazard relationships
    AFFECTS = "AFFECTS"                 # Hazard → Zone
    CAUSES = "CAUSES"                   # Hazard → Incident
    EXISTS_IN = "EXISTS_IN"             # Hazard → Zone (from architecture)
    CAN_CAUSE = "CAN_CAUSE"            # Equipment → Hazard


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph node: Sensor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class SensorNode:
    """A sensor in the facility graph.

    Fields aligned with PS-1 §1.4 (Sensor entity):
      sensor_id, sensor_type, unit_of_measurement, sensor_status
    """

    sensor_id: str
    sensor_type: str = ""
    unit_of_measurement: str = ""
    equipment_id: Optional[str] = None
    zone_id: Optional[str] = None
    sensor_status: str = "ACTIVE"
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    accuracy_rating: float = 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph node: Equipment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class EquipmentNode:
    """Equipment in the facility graph.

    Fields aligned with PS-1 §1.3 (Equipment entity):
      equipment_id, equipment_type, location_zone_id, operational_status
    """

    equipment_id: str
    equipment_type: str = ""
    manufacturer: str = ""
    location_zone_id: str = ""
    operational_status: str = "ACTIVE"
    health_score: float = 100.0
    sensors: List[SensorNode] = field(default_factory=list)

    def add_sensor(self, sensor: SensorNode) -> None:
        """Add a sensor to this equipment (HAS_SENSOR relationship)."""
        sensor.equipment_id = self.equipment_id
        self.sensors.append(sensor)

    @property
    def sensor_count(self) -> int:
        return len(self.sensors)

    @property
    def is_operational(self) -> bool:
        return self.operational_status == "ACTIVE"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph node: Zone
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ZoneNode:
    """A physical zone in the facility graph.

    Fields aligned with PS-1 §1.2 (Zone entity):
      zone_id, zone_name, risk_level_baseline, current_risk_score

    Relationships implemented:
      - CONNECTED_TO → connected_zones (Zone ↔ Zone)
      - CONTAINS     → equipment (Zone → Equipment)
      - HAS_SENSOR   → sensors (Zone → Sensor, via equipment)
    """

    zone_id: str
    zone_name: str = ""
    risk_level_baseline: str = "LOW"
    current_risk_score: float = 0.0
    coordinates: Optional[tuple] = None
    worker_capacity: int = 0
    is_restricted: bool = False

    # Graph relationships
    connected_zones: List[str] = field(default_factory=list)
    equipment: List[EquipmentNode] = field(default_factory=list)

    # Runtime state
    current_worker_count: int = 0
    active_hazards: List[str] = field(default_factory=list)

    # ── Relationship mutations ──

    def connect_to(self, zone_id: str) -> None:
        """Add a CONNECTED_TO relationship to another zone."""
        if zone_id not in self.connected_zones and zone_id != self.zone_id:
            self.connected_zones.append(zone_id)

    def add_equipment(self, equip: EquipmentNode) -> None:
        """Add a CONTAINS relationship (Zone → Equipment)."""
        equip.location_zone_id = self.zone_id
        self.equipment.append(equip)

    # ── Computed properties ──

    @property
    def equipment_count(self) -> int:
        return len(self.equipment)

    @property
    def all_sensors(self) -> List[SensorNode]:
        """All sensors in this zone (across all equipment)."""
        sensors = []
        for eq in self.equipment:
            sensors.extend(eq.sensors)
        return sensors

    @property
    def sensor_count(self) -> int:
        return len(self.all_sensors)

    @property
    def has_active_hazards(self) -> bool:
        return len(self.active_hazards) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph node: Hazard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class HazardNode:
    """A hazard in the facility graph.

    Relationships implemented:
      - AFFECTS → affected_zones (Hazard → Zone)
      - CAUSES  → caused_incidents (Hazard → Incident)
    """

    hazard_id: str
    hazard_type: str
    severity: str = "HIGH"

    # AFFECTS relationship
    affected_zones: List[str] = field(default_factory=list)

    # CAUSES relationship
    caused_incidents: List[str] = field(default_factory=list)

    def affects(self, zone_id: str) -> None:
        """Add an AFFECTS relationship (Hazard → Zone)."""
        if zone_id not in self.affected_zones:
            self.affected_zones.append(zone_id)

    def causes(self, incident_id: str) -> None:
        """Add a CAUSES relationship (Hazard → Incident)."""
        if incident_id not in self.caused_incidents:
            self.caused_incidents.append(incident_id)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph edge
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class GraphEdge:
    """A typed, directed edge in the facility graph.

    Used for generic graph traversal and propagation path computation.
    """

    from_id: str
    to_id: str
    relationship: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
