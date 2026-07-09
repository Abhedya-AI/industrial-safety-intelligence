"""Neo4j implementation of GraphRepository.

Uses the Neo4j Python async driver to persist the facility graph
in a Neo4j database. Cypher queries follow the PS-1 §2 relationship
naming conventions.

Prerequisites:
  - ``neo4j`` Python driver (``pip install neo4j``)
  - Neo4j 5.x instance running at the configured URI

If ``neo4j`` is not installed, this module logs a warning at import
time and the class raises ``GraphNotInitializedError`` on instantiation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.hazard_propagation.domain.exceptions import (
    GraphNotInitializedError,
    ZoneNotFoundError,
)
from app.hazard_propagation.graph.entities import (
    EquipmentNode,
    HazardNode,
    SensorNode,
    ZoneNode,
)
from app.hazard_propagation.repositories.graph_repository import GraphRepository

logger = logging.getLogger(__name__)

# Attempt to import the Neo4j driver
try:
    from neo4j import AsyncGraphDatabase  # type: ignore[import-untyped]

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning(
        "neo4j driver not installed — Neo4jGraphRepository unavailable. "
        "Install with: pip install neo4j"
    )


class Neo4jGraphRepository(GraphRepository):
    """Graph repository backed by a Neo4j database.

    Cypher queries use the exact relationship names from PS-1 §2:
      - (:Zone)-[:CONNECTED_TO]->(:Zone)
      - (:Zone)-[:CONTAINS]->(:Equipment)
      - (:Equipment)-[:HAS_SENSOR]->(:Sensor)
      - (:Hazard)-[:AFFECTS]->(:Zone)
      - (:Hazard)-[:CAUSES]->(:Incident)

    Usage:
        repo = Neo4jGraphRepository(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
        )
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ) -> None:
        if not NEO4J_AVAILABLE:
            raise GraphNotInitializedError(
                "neo4j driver not installed. "
                "Install with: pip install neo4j"
            )
        self._driver = AsyncGraphDatabase.driver(
            uri, auth=(username, password),
        )
        self._database = database
        logger.info("Neo4j graph repository initialised: %s", uri)

    async def close(self) -> None:
        """Close the Neo4j driver connection."""
        await self._driver.close()

    async def _run(self, query: str, **params: Any) -> List[Dict]:
        """Execute a Cypher query and return all records as dicts."""
        async with self._driver.session(
            database=self._database,
        ) as session:
            result = await session.run(query, **params)
            return [record.data() async for record in result]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Zone operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def create_zone(self, zone: ZoneNode) -> ZoneNode:
        await self._run(
            """
            CREATE (z:Zone {
                zone_id: $zone_id,
                zone_name: $zone_name,
                risk_level_baseline: $risk_level_baseline,
                current_risk_score: $current_risk_score,
                worker_capacity: $worker_capacity,
                current_worker_count: $current_worker_count,
                is_restricted: $is_restricted
            })
            """,
            zone_id=zone.zone_id,
            zone_name=zone.zone_name,
            risk_level_baseline=zone.risk_level_baseline,
            current_risk_score=zone.current_risk_score,
            worker_capacity=zone.worker_capacity,
            current_worker_count=zone.current_worker_count,
            is_restricted=zone.is_restricted,
        )
        logger.info("Neo4j: created zone %s", zone.zone_id)
        return zone

    async def get_zone(self, zone_id: str) -> Optional[ZoneNode]:
        records = await self._run(
            "MATCH (z:Zone {zone_id: $zone_id}) RETURN z",
            zone_id=zone_id,
        )
        if not records:
            return None
        props = records[0]["z"]
        return ZoneNode(
            zone_id=props["zone_id"],
            zone_name=props.get("zone_name", ""),
            risk_level_baseline=props.get("risk_level_baseline", "LOW"),
            current_risk_score=props.get("current_risk_score", 0.0),
            worker_capacity=props.get("worker_capacity", 0),
            current_worker_count=props.get("current_worker_count", 0),
            is_restricted=props.get("is_restricted", False),
        )

    async def get_all_zones(self) -> List[ZoneNode]:
        records = await self._run("MATCH (z:Zone) RETURN z")
        zones = []
        for rec in records:
            props = rec["z"]
            zones.append(ZoneNode(
                zone_id=props["zone_id"],
                zone_name=props.get("zone_name", ""),
                risk_level_baseline=props.get("risk_level_baseline", "LOW"),
                current_risk_score=props.get("current_risk_score", 0.0),
                worker_capacity=props.get("worker_capacity", 0),
                current_worker_count=props.get("current_worker_count", 0),
                is_restricted=props.get("is_restricted", False),
            ))
        return zones

    async def delete_zone(self, zone_id: str) -> bool:
        records = await self._run(
            """
            MATCH (z:Zone {zone_id: $zone_id})
            DETACH DELETE z
            RETURN count(z) AS deleted
            """,
            zone_id=zone_id,
        )
        deleted = records[0]["deleted"] if records else 0
        return deleted > 0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Equipment operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def create_equipment(
        self, zone_id: str, equipment: EquipmentNode,
    ) -> EquipmentNode:
        await self._run(
            """
            MATCH (z:Zone {zone_id: $zone_id})
            CREATE (e:Equipment {
                equipment_id: $equipment_id,
                equipment_type: $equipment_type,
                manufacturer: $manufacturer,
                operational_status: $operational_status,
                health_score: $health_score
            })
            CREATE (z)-[:CONTAINS]->(e)
            CREATE (e)-[:LOCATED_IN]->(z)
            """,
            zone_id=zone_id,
            equipment_id=equipment.equipment_id,
            equipment_type=equipment.equipment_type,
            manufacturer=equipment.manufacturer,
            operational_status=equipment.operational_status,
            health_score=equipment.health_score,
        )
        equipment.location_zone_id = zone_id
        return equipment

    async def get_equipment_in_zone(
        self, zone_id: str,
    ) -> List[EquipmentNode]:
        records = await self._run(
            """
            MATCH (z:Zone {zone_id: $zone_id})-[:CONTAINS]->(e:Equipment)
            RETURN e
            """,
            zone_id=zone_id,
        )
        return [
            EquipmentNode(
                equipment_id=rec["e"]["equipment_id"],
                equipment_type=rec["e"].get("equipment_type", ""),
                location_zone_id=zone_id,
                operational_status=rec["e"].get("operational_status", "ACTIVE"),
                health_score=rec["e"].get("health_score", 100.0),
            )
            for rec in records
        ]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Sensor operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def create_sensor(
        self, equipment_id: str, sensor: SensorNode,
    ) -> SensorNode:
        await self._run(
            """
            MATCH (e:Equipment {equipment_id: $equipment_id})
            CREATE (s:Sensor {
                sensor_id: $sensor_id,
                sensor_type: $sensor_type,
                unit_of_measurement: $unit_of_measurement,
                sensor_status: $sensor_status
            })
            CREATE (e)-[:HAS_SENSOR]->(s)
            CREATE (s)-[:MONITORS]->(e)
            """,
            equipment_id=equipment_id,
            sensor_id=sensor.sensor_id,
            sensor_type=sensor.sensor_type,
            unit_of_measurement=sensor.unit_of_measurement,
            sensor_status=sensor.sensor_status,
        )
        sensor.equipment_id = equipment_id
        return sensor

    async def get_sensors_in_zone(self, zone_id: str) -> List[SensorNode]:
        records = await self._run(
            """
            MATCH (z:Zone {zone_id: $zone_id})-[:CONTAINS]->
                  (e:Equipment)-[:HAS_SENSOR]->(s:Sensor)
            RETURN s, e.equipment_id AS equipment_id
            """,
            zone_id=zone_id,
        )
        return [
            SensorNode(
                sensor_id=rec["s"]["sensor_id"],
                sensor_type=rec["s"].get("sensor_type", ""),
                equipment_id=rec["equipment_id"],
                zone_id=zone_id,
            )
            for rec in records
        ]

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
        if bidirectional:
            await self._run(
                """
                MATCH (a:Zone {zone_id: $zone_a}), (b:Zone {zone_id: $zone_b})
                MERGE (a)-[:CONNECTED_TO {weight: $weight}]->(b)
                MERGE (b)-[:CONNECTED_TO {weight: $weight}]->(a)
                """,
                zone_a=zone_a, zone_b=zone_b, weight=weight,
            )
        else:
            await self._run(
                """
                MATCH (a:Zone {zone_id: $zone_a}), (b:Zone {zone_id: $zone_b})
                MERGE (a)-[:CONNECTED_TO {weight: $weight}]->(b)
                """,
                zone_a=zone_a, zone_b=zone_b, weight=weight,
            )

    async def get_connected_zones(self, zone_id: str) -> List[ZoneNode]:
        records = await self._run(
            """
            MATCH (z:Zone {zone_id: $zone_id})-[:CONNECTED_TO]->(n:Zone)
            RETURN n
            """,
            zone_id=zone_id,
        )
        return [
            ZoneNode(
                zone_id=rec["n"]["zone_id"],
                zone_name=rec["n"].get("zone_name", ""),
            )
            for rec in records
        ]

    async def get_neighbors(
        self, zone_id: str, max_hops: int = 1,
    ) -> Dict[str, int]:
        records = await self._run(
            """
            MATCH path = (start:Zone {zone_id: $zone_id})
                         -[:CONNECTED_TO*1..$max_hops]->(end:Zone)
            RETURN end.zone_id AS zone_id,
                   min(length(path)) AS hops
            """,
            zone_id=zone_id, max_hops=max_hops,
        )
        result: Dict[str, int] = {zone_id: 0}
        for rec in records:
            result[rec["zone_id"]] = rec["hops"]
        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Hazard operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def create_hazard(self, hazard: HazardNode) -> HazardNode:
        await self._run(
            """
            CREATE (h:Hazard {
                hazard_id: $hazard_id,
                hazard_type: $hazard_type,
                severity: $severity
            })
            """,
            hazard_id=hazard.hazard_id,
            hazard_type=hazard.hazard_type,
            severity=hazard.severity,
        )
        # Create AFFECTS relationships
        for zone_id in hazard.affected_zones:
            await self._run(
                """
                MATCH (h:Hazard {hazard_id: $hazard_id}),
                      (z:Zone {zone_id: $zone_id})
                CREATE (h)-[:AFFECTS]->(z)
                """,
                hazard_id=hazard.hazard_id, zone_id=zone_id,
            )
        return hazard

    async def get_hazard(self, hazard_id: str) -> Optional[HazardNode]:
        records = await self._run(
            """
            MATCH (h:Hazard {hazard_id: $hazard_id})
            OPTIONAL MATCH (h)-[:AFFECTS]->(z:Zone)
            RETURN h, collect(z.zone_id) AS affected_zones
            """,
            hazard_id=hazard_id,
        )
        if not records:
            return None
        props = records[0]["h"]
        return HazardNode(
            hazard_id=props["hazard_id"],
            hazard_type=props.get("hazard_type", ""),
            severity=props.get("severity", "HIGH"),
            affected_zones=records[0].get("affected_zones", []),
        )

    async def get_hazard_paths(
        self, origin_zone: str, max_depth: int = 3,
    ) -> List[List[str]]:
        records = await self._run(
            """
            MATCH path = (start:Zone {zone_id: $origin})
                         -[:CONNECTED_TO*1..$max_depth]->(end:Zone)
            WHERE ALL(n IN nodes(path) WHERE
                      single(x IN nodes(path) WHERE x = n))
            RETURN [n IN nodes(path) | n.zone_id] AS zone_path
            """,
            origin=origin_zone, max_depth=max_depth,
        )
        return [rec["zone_path"] for rec in records]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Graph metadata
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_graph_stats(self) -> Dict[str, int]:
        records = await self._run(
            """
            MATCH (z:Zone) WITH count(z) AS zones
            MATCH (e:Equipment) WITH zones, count(e) AS equipment
            MATCH (s:Sensor) WITH zones, equipment, count(s) AS sensors
            MATCH (h:Hazard) WITH zones, equipment, sensors,
                  count(h) AS hazards
            MATCH ()-[r]->() WITH zones, equipment, sensors, hazards,
                  count(r) AS edges
            RETURN zones, equipment, sensors, edges, hazards
            """
        )
        if records:
            return records[0]
        return {
            "zones": 0, "equipment": 0, "sensors": 0,
            "edges": 0, "hazards": 0,
        }
