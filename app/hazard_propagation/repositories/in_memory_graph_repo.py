"""In-memory implementation of GraphRepository.

Backed by the ``FacilityGraph`` data structure. Fully functional
without any external database — suitable for development, testing,
and environments where Neo4j is not available.

Thread-safety: NOT thread-safe. For concurrent access wrap
calls in an asyncio lock or use the Neo4j implementation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.hazard_propagation.domain.exceptions import ZoneNotFoundError
from app.hazard_propagation.graph.entities import (
    EquipmentNode,
    HazardNode,
    SensorNode,
    ZoneNode,
)
from app.hazard_propagation.graph.facility_graph import FacilityGraph
from app.hazard_propagation.repositories.graph_repository import GraphRepository

logger = logging.getLogger(__name__)


class InMemoryGraphRepository(GraphRepository):
    """Graph repository backed by an in-memory FacilityGraph.

    Usage:
        repo = InMemoryGraphRepository()
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        await repo.create_connection("ZONE_A", "ZONE_B")
    """

    def __init__(self, graph: Optional[FacilityGraph] = None) -> None:
        self._graph = graph or FacilityGraph()

    @property
    def graph(self) -> FacilityGraph:
        """Expose the underlying graph for direct access (testing)."""
        return self._graph

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Zone operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def create_zone(self, zone: ZoneNode) -> ZoneNode:
        if self._graph.has_zone(zone.zone_id):
            raise ValueError(f"Zone already exists: {zone.zone_id}")
        self._graph.add_zone(zone)
        logger.info("Created zone: %s", zone.zone_id)
        return zone

    async def get_zone(self, zone_id: str) -> Optional[ZoneNode]:
        return self._graph.get_zone(zone_id)

    async def get_all_zones(self) -> List[ZoneNode]:
        return [
            self._graph.get_zone(zid)
            for zid in self._graph.zone_ids
        ]

    async def delete_zone(self, zone_id: str) -> bool:
        if not self._graph.has_zone(zone_id):
            return False
        self._graph.remove_zone(zone_id)
        logger.info("Deleted zone: %s", zone_id)
        return True

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Equipment operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def create_equipment(
        self, zone_id: str, equipment: EquipmentNode,
    ) -> EquipmentNode:
        self._graph.add_equipment_to_zone(zone_id, equipment)
        logger.info(
            "Created equipment %s in zone %s",
            equipment.equipment_id, zone_id,
        )
        return equipment

    async def get_equipment_in_zone(
        self, zone_id: str,
    ) -> List[EquipmentNode]:
        return self._graph.get_equipment_in_zone(zone_id)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Sensor operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def create_sensor(
        self, equipment_id: str, sensor: SensorNode,
    ) -> SensorNode:
        self._graph.add_sensor_to_equipment(equipment_id, sensor)
        logger.info(
            "Created sensor %s on equipment %s",
            sensor.sensor_id, equipment_id,
        )
        return sensor

    async def get_sensors_in_zone(self, zone_id: str) -> List[SensorNode]:
        return self._graph.get_sensors_in_zone(zone_id)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Connection operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def create_connection(
        self,
        zone_a: str,
        zone_b: str,
        weight: float = 1.0,
        bidirectional: bool = True,
    ) -> None:
        self._graph.connect_zones(
            zone_a, zone_b,
            weight=weight, bidirectional=bidirectional,
        )
        direction = "↔" if bidirectional else "→"
        logger.info(
            "Connected %s %s %s (weight=%.2f)",
            zone_a, direction, zone_b, weight,
        )

    async def get_connected_zones(self, zone_id: str) -> List[ZoneNode]:
        return self._graph.get_connected_zones(zone_id)

    async def get_neighbors(
        self, zone_id: str, max_hops: int = 1,
    ) -> Dict[str, int]:
        return self._graph.get_zones_within_hops(zone_id, max_hops)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Hazard operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def create_hazard(self, hazard: HazardNode) -> HazardNode:
        self._graph.add_hazard(hazard)
        logger.info(
            "Created hazard %s (%s) affecting %s",
            hazard.hazard_id, hazard.hazard_type,
            hazard.affected_zones,
        )
        return hazard

    async def get_hazard(self, hazard_id: str) -> Optional[HazardNode]:
        return self._graph.get_hazard(hazard_id)

    async def get_hazard_paths(
        self, origin_zone: str, max_depth: int = 3,
    ) -> List[List[str]]:
        return self._graph.get_all_paths(origin_zone, max_depth)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Graph metadata
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_graph_stats(self) -> Dict[str, int]:
        total_equipment = 0
        total_sensors = 0
        for zid in self._graph.zone_ids:
            zone = self._graph.get_zone(zid)
            if zone:
                total_equipment += zone.equipment_count
                total_sensors += zone.sensor_count
        return {
            "zones": self._graph.zone_count,
            "equipment": total_equipment,
            "sensors": total_sensors,
            "edges": self._graph.edge_count,
            "hazards": self._graph.hazard_count,
        }
