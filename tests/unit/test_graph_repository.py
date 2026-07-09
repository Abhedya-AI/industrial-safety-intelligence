"""Unit tests for Hazard Propagation graph repository layer.

Tests the InMemoryGraphRepository against the GraphRepository interface.
All tests are async since the repository contract is async.

Test categories:
  1. Zone CRUD (create, get, get_all, delete)
  2. Equipment operations (create, get in zone)
  3. Sensor operations (create, get in zone)
  4. Zone connections (create, get connected, bidirectional/unidirectional)
  5. Neighbor queries (BFS with max_hops)
  6. Hazard operations (create, get, paths)
  7. Graph stats
  8. Error handling (duplicates, not-found)
  9. Complex topologies (chains, stars, disconnected)
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from app.hazard_propagation.domain.exceptions import ZoneNotFoundError
from app.hazard_propagation.graph.entities import (
    EquipmentNode,
    HazardNode,
    SensorNode,
    ZoneNode,
)
from app.hazard_propagation.repositories.graph_repository import GraphRepository
from app.hazard_propagation.repositories.in_memory_graph_repo import (
    InMemoryGraphRepository,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest_asyncio.fixture
async def repo() -> InMemoryGraphRepository:
    return InMemoryGraphRepository()


@pytest_asyncio.fixture
async def populated_repo() -> InMemoryGraphRepository:
    """Repository with 4 zones in a linear topology:

    ZONE_A ── ZONE_B ── ZONE_C ── ZONE_D
    """
    r = InMemoryGraphRepository()
    await r.create_zone(ZoneNode(zone_id="ZONE_A", zone_name="Zone A"))
    await r.create_zone(ZoneNode(zone_id="ZONE_B", zone_name="Zone B"))
    await r.create_zone(ZoneNode(zone_id="ZONE_C", zone_name="Zone C"))
    await r.create_zone(ZoneNode(zone_id="ZONE_D", zone_name="Zone D"))
    await r.create_connection("ZONE_A", "ZONE_B")
    await r.create_connection("ZONE_B", "ZONE_C")
    await r.create_connection("ZONE_C", "ZONE_D")
    return r


@pytest_asyncio.fixture
async def star_repo() -> InMemoryGraphRepository:
    """Repository with a star topology:

    ZONE_B ─┐
    ZONE_C ─┤── ZONE_A
    ZONE_D ─┘
    """
    r = InMemoryGraphRepository()
    await r.create_zone(ZoneNode(zone_id="ZONE_A", zone_name="Hub"))
    await r.create_zone(ZoneNode(zone_id="ZONE_B"))
    await r.create_zone(ZoneNode(zone_id="ZONE_C"))
    await r.create_zone(ZoneNode(zone_id="ZONE_D"))
    await r.create_connection("ZONE_A", "ZONE_B")
    await r.create_connection("ZONE_A", "ZONE_C")
    await r.create_connection("ZONE_A", "ZONE_D")
    return r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Zone CRUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestZoneCRUD:
    async def test_create_zone(self, repo):
        zone = await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        assert zone.zone_id == "ZONE_A"

    async def test_create_zone_returns_node(self, repo):
        zone = await repo.create_zone(
            ZoneNode(zone_id="ZONE_A", zone_name="Zone A"),
        )
        assert isinstance(zone, ZoneNode)
        assert zone.zone_name == "Zone A"

    async def test_create_duplicate_raises(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        with pytest.raises(ValueError, match="already exists"):
            await repo.create_zone(ZoneNode(zone_id="ZONE_A"))

    async def test_get_zone(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        zone = await repo.get_zone("ZONE_A")
        assert zone is not None
        assert zone.zone_id == "ZONE_A"

    async def test_get_zone_not_found(self, repo):
        zone = await repo.get_zone("ZONE_Z")
        assert zone is None

    async def test_get_all_zones(self, populated_repo):
        zones = await populated_repo.get_all_zones()
        assert len(zones) == 4
        ids = {z.zone_id for z in zones}
        assert ids == {"ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D"}

    async def test_get_all_zones_empty(self, repo):
        zones = await repo.get_all_zones()
        assert zones == []

    async def test_delete_zone(self, populated_repo):
        deleted = await populated_repo.delete_zone("ZONE_D")
        assert deleted is True
        zone = await populated_repo.get_zone("ZONE_D")
        assert zone is None

    async def test_delete_zone_not_found(self, repo):
        deleted = await repo.delete_zone("ZONE_Z")
        assert deleted is False

    async def test_delete_zone_removes_connections(self, populated_repo):
        await populated_repo.delete_zone("ZONE_B")
        # ZONE_A should no longer list ZONE_B as connected
        connected = await populated_repo.get_connected_zones("ZONE_A")
        connected_ids = {z.zone_id for z in connected}
        assert "ZONE_B" not in connected_ids

    async def test_create_zone_with_all_fields(self, repo):
        zone = await repo.create_zone(ZoneNode(
            zone_id="ZONE_A",
            zone_name="Zone Alpha",
            risk_level_baseline="HIGH",
            current_risk_score=75.0,
            worker_capacity=20,
            is_restricted=True,
        ))
        assert zone.risk_level_baseline == "HIGH"
        assert zone.current_risk_score == 75.0
        assert zone.worker_capacity == 20
        assert zone.is_restricted is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Equipment operations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEquipmentOperations:
    async def test_create_equipment(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        eq = await repo.create_equipment(
            "ZONE_A", EquipmentNode(equipment_id="EQ001"),
        )
        assert eq.equipment_id == "EQ001"
        assert eq.location_zone_id == "ZONE_A"

    async def test_create_equipment_with_type(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        eq = await repo.create_equipment(
            "ZONE_A",
            EquipmentNode(
                equipment_id="EQ001",
                equipment_type="Boiler",
                manufacturer="Acme",
            ),
        )
        assert eq.equipment_type == "Boiler"
        assert eq.manufacturer == "Acme"

    async def test_get_equipment_in_zone(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        await repo.create_equipment(
            "ZONE_A", EquipmentNode(equipment_id="EQ001"),
        )
        await repo.create_equipment(
            "ZONE_A", EquipmentNode(equipment_id="EQ002"),
        )
        equipment = await repo.get_equipment_in_zone("ZONE_A")
        assert len(equipment) == 2
        ids = {e.equipment_id for e in equipment}
        assert ids == {"EQ001", "EQ002"}

    async def test_get_equipment_empty_zone(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        equipment = await repo.get_equipment_in_zone("ZONE_A")
        assert equipment == []

    async def test_create_equipment_nonexistent_zone(self, repo):
        with pytest.raises(ZoneNotFoundError):
            await repo.create_equipment(
                "ZONE_Z", EquipmentNode(equipment_id="EQ001"),
            )

    async def test_multiple_zones_equipment_isolation(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        await repo.create_zone(ZoneNode(zone_id="ZONE_B"))
        await repo.create_equipment(
            "ZONE_A", EquipmentNode(equipment_id="EQ001"),
        )
        await repo.create_equipment(
            "ZONE_B", EquipmentNode(equipment_id="EQ002"),
        )
        eq_a = await repo.get_equipment_in_zone("ZONE_A")
        eq_b = await repo.get_equipment_in_zone("ZONE_B")
        assert len(eq_a) == 1
        assert eq_a[0].equipment_id == "EQ001"
        assert len(eq_b) == 1
        assert eq_b[0].equipment_id == "EQ002"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Sensor operations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSensorOperations:
    async def test_create_sensor(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        await repo.create_equipment(
            "ZONE_A", EquipmentNode(equipment_id="EQ001"),
        )
        sensor = await repo.create_sensor(
            "EQ001", SensorNode(sensor_id="S001", sensor_type="GAS"),
        )
        assert sensor.sensor_id == "S001"
        assert sensor.equipment_id == "EQ001"

    async def test_get_sensors_in_zone(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        await repo.create_equipment(
            "ZONE_A", EquipmentNode(equipment_id="EQ001"),
        )
        await repo.create_sensor(
            "EQ001", SensorNode(sensor_id="S001"),
        )
        await repo.create_sensor(
            "EQ001", SensorNode(sensor_id="S002"),
        )
        sensors = await repo.get_sensors_in_zone("ZONE_A")
        assert len(sensors) == 2

    async def test_create_sensor_nonexistent_equipment(self, repo):
        with pytest.raises(ValueError, match="Equipment not found"):
            await repo.create_sensor(
                "EQ_MISSING", SensorNode(sensor_id="S001"),
            )

    async def test_sensor_zone_assignment(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        await repo.create_equipment(
            "ZONE_A", EquipmentNode(equipment_id="EQ001"),
        )
        sensor = await repo.create_sensor(
            "EQ001", SensorNode(sensor_id="S001"),
        )
        assert sensor.zone_id == "ZONE_A"

    async def test_sensors_across_equipment(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        await repo.create_equipment(
            "ZONE_A", EquipmentNode(equipment_id="EQ001"),
        )
        await repo.create_equipment(
            "ZONE_A", EquipmentNode(equipment_id="EQ002"),
        )
        await repo.create_sensor(
            "EQ001", SensorNode(sensor_id="S001"),
        )
        await repo.create_sensor(
            "EQ002", SensorNode(sensor_id="S002"),
        )
        sensors = await repo.get_sensors_in_zone("ZONE_A")
        assert len(sensors) == 2
        ids = {s.sensor_id for s in sensors}
        assert ids == {"S001", "S002"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Zone connections
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestZoneConnections:
    async def test_create_connection(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        await repo.create_zone(ZoneNode(zone_id="ZONE_B"))
        await repo.create_connection("ZONE_A", "ZONE_B")
        connected = await repo.get_connected_zones("ZONE_A")
        assert len(connected) == 1
        assert connected[0].zone_id == "ZONE_B"

    async def test_bidirectional_connection(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        await repo.create_zone(ZoneNode(zone_id="ZONE_B"))
        await repo.create_connection("ZONE_A", "ZONE_B", bidirectional=True)

        from_a = await repo.get_connected_zones("ZONE_A")
        from_b = await repo.get_connected_zones("ZONE_B")
        assert any(z.zone_id == "ZONE_B" for z in from_a)
        assert any(z.zone_id == "ZONE_A" for z in from_b)

    async def test_unidirectional_connection(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        await repo.create_zone(ZoneNode(zone_id="ZONE_B"))
        await repo.create_connection(
            "ZONE_A", "ZONE_B", bidirectional=False,
        )

        from_a = await repo.get_connected_zones("ZONE_A")
        from_b = await repo.get_connected_zones("ZONE_B")
        assert any(z.zone_id == "ZONE_B" for z in from_a)
        assert not any(z.zone_id == "ZONE_A" for z in from_b)

    async def test_connection_nonexistent_zone(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        with pytest.raises(ZoneNotFoundError):
            await repo.create_connection("ZONE_A", "ZONE_Z")

    async def test_get_connected_zones_empty(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        connected = await repo.get_connected_zones("ZONE_A")
        assert connected == []

    async def test_weighted_connection(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        await repo.create_zone(ZoneNode(zone_id="ZONE_B"))
        await repo.create_connection("ZONE_A", "ZONE_B", weight=0.5)
        connected = await repo.get_connected_zones("ZONE_A")
        assert len(connected) == 1

    async def test_linear_chain(self, populated_repo):
        """ZONE_A ── ZONE_B ── ZONE_C ── ZONE_D"""
        connected_a = await populated_repo.get_connected_zones("ZONE_A")
        connected_b = await populated_repo.get_connected_zones("ZONE_B")
        assert len(connected_a) == 1  # Only ZONE_B
        assert len(connected_b) == 2  # ZONE_A and ZONE_C


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Neighbor queries (BFS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNeighborQueries:
    async def test_neighbors_1_hop(self, populated_repo):
        result = await populated_repo.get_neighbors("ZONE_A", max_hops=1)
        assert result == {"ZONE_A": 0, "ZONE_B": 1}

    async def test_neighbors_2_hops(self, populated_repo):
        result = await populated_repo.get_neighbors("ZONE_A", max_hops=2)
        assert result == {"ZONE_A": 0, "ZONE_B": 1, "ZONE_C": 2}

    async def test_neighbors_3_hops(self, populated_repo):
        result = await populated_repo.get_neighbors("ZONE_A", max_hops=3)
        assert result == {
            "ZONE_A": 0, "ZONE_B": 1, "ZONE_C": 2, "ZONE_D": 3,
        }

    async def test_neighbors_star(self, star_repo):
        result = await star_repo.get_neighbors("ZONE_A", max_hops=1)
        assert len(result) == 4  # Hub + 3 spokes
        assert result["ZONE_A"] == 0
        assert result["ZONE_B"] == 1
        assert result["ZONE_C"] == 1
        assert result["ZONE_D"] == 1

    async def test_neighbors_from_spoke(self, star_repo):
        result = await star_repo.get_neighbors("ZONE_B", max_hops=2)
        assert result["ZONE_B"] == 0
        assert result["ZONE_A"] == 1
        assert "ZONE_C" in result  # Via hub at hop 2
        assert "ZONE_D" in result

    async def test_neighbors_nonexistent(self, repo):
        with pytest.raises(ZoneNotFoundError):
            await repo.get_neighbors("ZONE_Z")

    async def test_neighbors_isolated_zone(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_ISOLATED"))
        result = await repo.get_neighbors("ZONE_ISOLATED", max_hops=5)
        assert result == {"ZONE_ISOLATED": 0}

    async def test_neighbors_0_hops(self, populated_repo):
        """max_hops=0 should return only the origin zone itself."""
        # The BFS range(1, max_hops+1) is empty for max_hops=0
        result = await populated_repo.get_neighbors("ZONE_B", max_hops=0)
        assert result == {"ZONE_B": 0}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Hazard operations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHazardOperations:
    async def test_create_hazard(self, populated_repo):
        hazard = await populated_repo.create_hazard(
            HazardNode(
                hazard_id="H001",
                hazard_type="GAS_LEAK",
                affected_zones=["ZONE_A"],
            ),
        )
        assert hazard.hazard_id == "H001"
        assert hazard.hazard_type == "GAS_LEAK"

    async def test_get_hazard(self, populated_repo):
        await populated_repo.create_hazard(
            HazardNode(
                hazard_id="H001",
                hazard_type="FIRE",
                affected_zones=["ZONE_A", "ZONE_B"],
            ),
        )
        hazard = await populated_repo.get_hazard("H001")
        assert hazard is not None
        assert hazard.hazard_type == "FIRE"
        assert set(hazard.affected_zones) == {"ZONE_A", "ZONE_B"}

    async def test_get_hazard_not_found(self, repo):
        hazard = await repo.get_hazard("H999")
        assert hazard is None

    async def test_hazard_paths_linear(self, populated_repo):
        """ZONE_A ── ZONE_B ── ZONE_C ── ZONE_D"""
        paths = await populated_repo.get_hazard_paths("ZONE_A", max_depth=3)
        all_destinations = set()
        for path in paths:
            all_destinations.add(path[-1])
        assert "ZONE_B" in all_destinations
        assert "ZONE_C" in all_destinations
        assert "ZONE_D" in all_destinations

    async def test_hazard_paths_max_depth_1(self, populated_repo):
        paths = await populated_repo.get_hazard_paths("ZONE_A", max_depth=1)
        for path in paths:
            assert len(path) <= 2

    async def test_hazard_paths_star(self, star_repo):
        paths = await star_repo.get_hazard_paths("ZONE_A", max_depth=1)
        destinations = {p[-1] for p in paths}
        assert "ZONE_B" in destinations
        assert "ZONE_C" in destinations
        assert "ZONE_D" in destinations

    async def test_hazard_paths_nonexistent(self, repo):
        with pytest.raises(ZoneNotFoundError):
            await repo.get_hazard_paths("ZONE_Z")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Graph stats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGraphStats:
    async def test_empty_stats(self, repo):
        stats = await repo.get_graph_stats()
        assert stats["zones"] == 0
        assert stats["equipment"] == 0
        assert stats["sensors"] == 0
        assert stats["edges"] == 0
        assert stats["hazards"] == 0

    async def test_populated_stats(self, populated_repo):
        stats = await populated_repo.get_graph_stats()
        assert stats["zones"] == 4
        assert stats["edges"] == 6  # 3 bidirectional connections = 6 edges

    async def test_stats_with_equipment_and_sensors(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        await repo.create_equipment(
            "ZONE_A", EquipmentNode(equipment_id="EQ001"),
        )
        await repo.create_sensor(
            "EQ001", SensorNode(sensor_id="S001"),
        )
        await repo.create_sensor(
            "EQ001", SensorNode(sensor_id="S002"),
        )
        stats = await repo.get_graph_stats()
        assert stats["zones"] == 1
        assert stats["equipment"] == 1
        assert stats["sensors"] == 2

    async def test_stats_with_hazards(self, repo):
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        await repo.create_hazard(
            HazardNode(hazard_id="H001", hazard_type="FIRE"),
        )
        stats = await repo.get_graph_stats()
        assert stats["hazards"] == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Interface compliance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInterfaceCompliance:
    def test_implements_interface(self):
        assert issubclass(InMemoryGraphRepository, GraphRepository)

    async def test_graph_property(self, repo):
        """InMemoryGraphRepository exposes graph for testing."""
        assert repo.graph is not None

    async def test_custom_graph_injection(self):
        from app.hazard_propagation.graph.facility_graph import FacilityGraph
        custom_graph = FacilityGraph()
        custom_graph.add_zone(ZoneNode(zone_id="PRE_EXISTING"))
        repo = InMemoryGraphRepository(graph=custom_graph)
        zone = await repo.get_zone("PRE_EXISTING")
        assert zone is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Complex topologies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestComplexTopologies:
    async def test_disconnected_zones(self, repo):
        """Two disconnected subgraphs."""
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        await repo.create_zone(ZoneNode(zone_id="ZONE_B"))
        await repo.create_zone(ZoneNode(zone_id="ZONE_C"))
        await repo.create_connection("ZONE_A", "ZONE_B")
        # ZONE_C is isolated
        neighbors_a = await repo.get_neighbors("ZONE_A", max_hops=5)
        assert "ZONE_C" not in neighbors_a

    async def test_diamond_topology(self, repo):
        """
        ZONE_A ── ZONE_B
           │         │
        ZONE_C ── ZONE_D
        """
        await repo.create_zone(ZoneNode(zone_id="ZONE_A"))
        await repo.create_zone(ZoneNode(zone_id="ZONE_B"))
        await repo.create_zone(ZoneNode(zone_id="ZONE_C"))
        await repo.create_zone(ZoneNode(zone_id="ZONE_D"))
        await repo.create_connection("ZONE_A", "ZONE_B")
        await repo.create_connection("ZONE_A", "ZONE_C")
        await repo.create_connection("ZONE_B", "ZONE_D")
        await repo.create_connection("ZONE_C", "ZONE_D")

        neighbors = await repo.get_neighbors("ZONE_A", max_hops=2)
        assert len(neighbors) == 4  # All zones reachable in 2 hops

    async def test_large_linear_chain(self, repo):
        """Chain of 10 zones."""
        for i in range(10):
            await repo.create_zone(ZoneNode(zone_id=f"ZONE_{i}"))
        for i in range(9):
            await repo.create_connection(f"ZONE_{i}", f"ZONE_{i+1}")

        # From ZONE_0, all zones reachable within 9 hops
        neighbors = await repo.get_neighbors("ZONE_0", max_hops=9)
        assert len(neighbors) == 10

        # From ZONE_0, only 4 zones reachable within 3 hops
        neighbors_3 = await repo.get_neighbors("ZONE_0", max_hops=3)
        assert len(neighbors_3) == 4  # ZONE_0..ZONE_3

    async def test_full_pipeline(self, repo):
        """End-to-end: zones + equipment + sensors + connections + hazard."""
        # Create zones
        await repo.create_zone(ZoneNode(zone_id="ZONE_A", zone_name="Alpha"))
        await repo.create_zone(ZoneNode(zone_id="ZONE_B", zone_name="Beta"))

        # Connect zones
        await repo.create_connection("ZONE_A", "ZONE_B")

        # Add equipment
        eq = await repo.create_equipment(
            "ZONE_A",
            EquipmentNode(equipment_id="EQ001", equipment_type="Boiler"),
        )
        assert eq.location_zone_id == "ZONE_A"

        # Add sensors
        s1 = await repo.create_sensor(
            "EQ001",
            SensorNode(sensor_id="S001", sensor_type="TEMPERATURE"),
        )
        s2 = await repo.create_sensor(
            "EQ001",
            SensorNode(sensor_id="S002", sensor_type="PRESSURE"),
        )
        assert s1.zone_id == "ZONE_A"
        assert s2.equipment_id == "EQ001"

        # Add hazard
        hazard = await repo.create_hazard(HazardNode(
            hazard_id="H001",
            hazard_type="FIRE",
            affected_zones=["ZONE_A"],
        ))
        assert hazard.affected_zones == ["ZONE_A"]

        # Verify stats
        stats = await repo.get_graph_stats()
        assert stats["zones"] == 2
        assert stats["equipment"] == 1
        assert stats["sensors"] == 2
        assert stats["hazards"] == 1
        assert stats["edges"] > 0

        # Verify paths
        paths = await repo.get_hazard_paths("ZONE_A", max_depth=1)
        assert any("ZONE_B" in p for p in paths)
