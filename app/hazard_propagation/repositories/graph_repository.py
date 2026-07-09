"""Graph repository interface (port) for the Hazard Propagation Engine.

Defines the abstract contract for graph persistence. Concrete
implementations (InMemory, Neo4j) must implement every method.

Follows the same ABC-based ports-and-adapters pattern used by
CompoundRiskRepository and RiskPredictionRepository.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from app.hazard_propagation.graph.entities import (
    EquipmentNode,
    GraphEdge,
    HazardNode,
    SensorNode,
    ZoneNode,
)


class GraphRepository(ABC):
    """Abstract interface for facility graph persistence.

    All methods are async to support both in-memory and database-backed
    implementations (Neo4j uses async driver).
    """

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Zone operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @abstractmethod
    async def create_zone(self, zone: ZoneNode) -> ZoneNode:
        """Persist a new zone node.

        Returns the created zone.
        Raises if a zone with the same zone_id already exists.
        """
        ...

    @abstractmethod
    async def get_zone(self, zone_id: str) -> Optional[ZoneNode]:
        """Retrieve a zone by ID, or None if not found."""
        ...

    @abstractmethod
    async def get_all_zones(self) -> List[ZoneNode]:
        """Retrieve all zones in the graph."""
        ...

    @abstractmethod
    async def delete_zone(self, zone_id: str) -> bool:
        """Delete a zone and its connections.

        Returns True if deleted, False if not found.
        """
        ...

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Equipment operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @abstractmethod
    async def create_equipment(
        self, zone_id: str, equipment: EquipmentNode,
    ) -> EquipmentNode:
        """Add equipment to a zone (CONTAINS relationship).

        Returns the created equipment.
        Raises if the zone does not exist.
        """
        ...

    @abstractmethod
    async def get_equipment_in_zone(
        self, zone_id: str,
    ) -> List[EquipmentNode]:
        """Get all equipment in a zone."""
        ...

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Sensor operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @abstractmethod
    async def create_sensor(
        self, equipment_id: str, sensor: SensorNode,
    ) -> SensorNode:
        """Add a sensor to equipment (HAS_SENSOR relationship).

        Returns the created sensor.
        Raises if the equipment does not exist.
        """
        ...

    @abstractmethod
    async def get_sensors_in_zone(self, zone_id: str) -> List[SensorNode]:
        """Get all sensors in a zone (via equipment)."""
        ...

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Connection operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @abstractmethod
    async def create_connection(
        self,
        zone_a: str,
        zone_b: str,
        weight: float = 1.0,
        bidirectional: bool = True,
    ) -> None:
        """Create a CONNECTED_TO relationship between two zones.

        Bidirectional by default.
        Raises if either zone does not exist.
        """
        ...

    @abstractmethod
    async def get_connected_zones(self, zone_id: str) -> List[ZoneNode]:
        """Get all zones directly connected to the given zone."""
        ...

    @abstractmethod
    async def get_neighbors(
        self, zone_id: str, max_hops: int = 1,
    ) -> Dict[str, int]:
        """BFS: find all reachable zones within max_hops.

        Returns {zone_id: hop_distance}.
        """
        ...

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Hazard operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @abstractmethod
    async def create_hazard(self, hazard: HazardNode) -> HazardNode:
        """Persist a hazard node with its AFFECTS relationships.

        Returns the created hazard.
        """
        ...

    @abstractmethod
    async def get_hazard(self, hazard_id: str) -> Optional[HazardNode]:
        """Retrieve a hazard by ID."""
        ...

    @abstractmethod
    async def get_hazard_paths(
        self, origin_zone: str, max_depth: int = 3,
    ) -> List[List[str]]:
        """Find all simple paths from origin up to max_depth.

        Returns a list of paths, each a list of zone IDs.
        """
        ...

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Graph metadata
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @abstractmethod
    async def get_graph_stats(self) -> Dict[str, int]:
        """Return counts of zones, equipment, sensors, edges, hazards."""
        ...
