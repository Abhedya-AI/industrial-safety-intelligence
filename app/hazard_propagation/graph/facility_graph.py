"""Facility graph — in-memory graph of zones, equipment, and sensors.

Provides the topological structure needed for hazard propagation
simulation. Zones are connected via CONNECTED_TO edges; equipment
and sensors are nested within zones.

This is a pure data structure — no IO, no persistence. The graph
is built programmatically by the service layer or from external data.

Graph operations:
  - Add/remove zones, equipment, sensors
  - Connect zones (CONNECTED_TO)
  - Query adjacency and paths
  - Get all entities in a zone's neighborhood
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from app.hazard_propagation.domain.exceptions import ZoneNotFoundError
from app.hazard_propagation.graph.entities import (
    EquipmentNode,
    GraphEdge,
    HazardNode,
    RelationshipType,
    SensorNode,
    ZoneNode,
)

logger = logging.getLogger(__name__)


class FacilityGraph:
    """In-memory facility graph for hazard propagation.

    Stores zones, equipment, sensors, and hazards with their
    PS-1 §2 relationships. Supports BFS/DFS traversal for
    propagation path computation.

    Usage:
        graph = FacilityGraph()
        graph.add_zone(ZoneNode(zone_id="ZONE_A", zone_name="Zone A"))
        graph.add_zone(ZoneNode(zone_id="ZONE_B", zone_name="Zone B"))
        graph.connect_zones("ZONE_A", "ZONE_B")
        neighbors = graph.get_connected_zones("ZONE_A")
    """

    def __init__(self) -> None:
        self._zones: Dict[str, ZoneNode] = {}
        self._hazards: Dict[str, HazardNode] = {}
        self._edges: List[GraphEdge] = []

    # ── Properties ──

    @property
    def zone_count(self) -> int:
        return len(self._zones)

    @property
    def hazard_count(self) -> int:
        return len(self._hazards)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def zone_ids(self) -> List[str]:
        return list(self._zones.keys())

    # ── Zone operations ──

    def add_zone(self, zone: ZoneNode) -> None:
        """Add a zone to the graph."""
        self._zones[zone.zone_id] = zone
        logger.debug("Added zone: %s", zone.zone_id)

    def get_zone(self, zone_id: str) -> Optional[ZoneNode]:
        """Get a zone by ID, or None if not found."""
        return self._zones.get(zone_id)

    def get_zone_or_raise(self, zone_id: str) -> ZoneNode:
        """Get a zone by ID, or raise ZoneNotFoundError."""
        zone = self._zones.get(zone_id)
        if zone is None:
            raise ZoneNotFoundError(zone_id)
        return zone

    def has_zone(self, zone_id: str) -> bool:
        return zone_id in self._zones

    def remove_zone(self, zone_id: str) -> None:
        """Remove a zone and all its edges."""
        if zone_id in self._zones:
            del self._zones[zone_id]
            # Remove edges involving this zone
            self._edges = [
                e for e in self._edges
                if e.from_id != zone_id and e.to_id != zone_id
            ]
            # Remove from other zones' connected lists
            for zone in self._zones.values():
                if zone_id in zone.connected_zones:
                    zone.connected_zones.remove(zone_id)

    # ── Zone connections ──

    def connect_zones(
        self,
        zone_a: str,
        zone_b: str,
        weight: float = 1.0,
        bidirectional: bool = True,
    ) -> None:
        """Create a CONNECTED_TO relationship between two zones.

        Bidirectional by default (hazards can spread in both directions).
        """
        a = self.get_zone_or_raise(zone_a)
        b = self.get_zone_or_raise(zone_b)

        a.connect_to(zone_b)
        self._edges.append(GraphEdge(
            from_id=zone_a, to_id=zone_b,
            relationship=RelationshipType.CONNECTED_TO,
            weight=weight,
        ))

        if bidirectional:
            b.connect_to(zone_a)
            self._edges.append(GraphEdge(
                from_id=zone_b, to_id=zone_a,
                relationship=RelationshipType.CONNECTED_TO,
                weight=weight,
            ))

    def get_connected_zones(self, zone_id: str) -> List[ZoneNode]:
        """Get all zones directly connected to the given zone."""
        zone = self.get_zone_or_raise(zone_id)
        return [
            self._zones[zid]
            for zid in zone.connected_zones
            if zid in self._zones
        ]

    def get_connected_zone_ids(self, zone_id: str) -> List[str]:
        """Get IDs of all zones connected to the given zone."""
        zone = self.get_zone_or_raise(zone_id)
        return [zid for zid in zone.connected_zones if zid in self._zones]

    # ── Equipment operations ──

    def add_equipment_to_zone(
        self, zone_id: str, equipment: EquipmentNode,
    ) -> None:
        """Add equipment to a zone (CONTAINS relationship)."""
        zone = self.get_zone_or_raise(zone_id)
        zone.add_equipment(equipment)
        self._edges.append(GraphEdge(
            from_id=zone_id, to_id=equipment.equipment_id,
            relationship=RelationshipType.CONTAINS,
        ))

    def get_equipment_in_zone(self, zone_id: str) -> List[EquipmentNode]:
        """Get all equipment in a zone."""
        zone = self.get_zone_or_raise(zone_id)
        return zone.equipment

    # ── Sensor operations ──

    def add_sensor_to_equipment(
        self, equipment_id: str, sensor: SensorNode,
    ) -> None:
        """Add a sensor to equipment (HAS_SENSOR relationship).

        Searches all zones for the equipment.
        """
        for zone in self._zones.values():
            for eq in zone.equipment:
                if eq.equipment_id == equipment_id:
                    eq.add_sensor(sensor)
                    sensor.zone_id = zone.zone_id
                    self._edges.append(GraphEdge(
                        from_id=equipment_id, to_id=sensor.sensor_id,
                        relationship=RelationshipType.HAS_SENSOR,
                    ))
                    return
        raise ValueError(f"Equipment not found: {equipment_id}")

    def get_sensors_in_zone(self, zone_id: str) -> List[SensorNode]:
        """Get all sensors in a zone (via equipment)."""
        zone = self.get_zone_or_raise(zone_id)
        return zone.all_sensors

    # ── Hazard operations ──

    def add_hazard(self, hazard: HazardNode) -> None:
        """Add a hazard to the graph."""
        self._hazards[hazard.hazard_id] = hazard
        # Create AFFECTS edges
        for zone_id in hazard.affected_zones:
            self._edges.append(GraphEdge(
                from_id=hazard.hazard_id, to_id=zone_id,
                relationship=RelationshipType.AFFECTS,
            ))

    def get_hazard(self, hazard_id: str) -> Optional[HazardNode]:
        return self._hazards.get(hazard_id)

    # ── Graph traversal ──

    def get_zones_within_hops(
        self, origin_zone: str, max_hops: int,
    ) -> Dict[str, int]:
        """BFS: find all reachable zones within max_hops.

        Returns a dict of {zone_id: hop_distance}.
        """
        self.get_zone_or_raise(origin_zone)

        visited: Dict[str, int] = {origin_zone: 0}
        frontier: List[str] = [origin_zone]

        for hop in range(1, max_hops + 1):
            next_frontier: List[str] = []
            for current in frontier:
                for neighbor_id in self.get_connected_zone_ids(current):
                    if neighbor_id not in visited:
                        visited[neighbor_id] = hop
                        next_frontier.append(neighbor_id)
            frontier = next_frontier
            if not frontier:
                break

        return visited

    def get_all_paths(
        self,
        origin: str,
        max_depth: int = 3,
    ) -> List[List[str]]:
        """Find all simple paths from origin up to max_depth.

        Returns a list of paths, where each path is a list of zone IDs.
        """
        self.get_zone_or_raise(origin)
        all_paths: List[List[str]] = []
        self._dfs_paths(origin, [origin], set(), max_depth, all_paths)
        return all_paths

    def _dfs_paths(
        self,
        current: str,
        path: List[str],
        visited: Set[str],
        remaining_depth: int,
        results: List[List[str]],
    ) -> None:
        """Recursive DFS to enumerate all simple paths."""
        visited.add(current)

        for neighbor_id in self.get_connected_zone_ids(current):
            if neighbor_id not in visited and remaining_depth > 0:
                new_path = path + [neighbor_id]
                results.append(new_path)
                self._dfs_paths(
                    neighbor_id, new_path,
                    visited.copy(), remaining_depth - 1, results,
                )

    # ── Serialization ──

    def to_dict(self) -> dict:
        """Serialize the graph to a dictionary."""
        return {
            "zones": {
                zid: {
                    "zone_id": z.zone_id,
                    "zone_name": z.zone_name,
                    "risk_level_baseline": z.risk_level_baseline,
                    "connected_zones": z.connected_zones,
                    "equipment_count": z.equipment_count,
                    "sensor_count": z.sensor_count,
                }
                for zid, z in self._zones.items()
            },
            "hazards": {
                hid: {
                    "hazard_id": h.hazard_id,
                    "hazard_type": h.hazard_type,
                    "severity": h.severity,
                    "affected_zones": h.affected_zones,
                }
                for hid, h in self._hazards.items()
            },
            "edge_count": self.edge_count,
        }
