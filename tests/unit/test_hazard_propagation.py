"""Unit tests for Hazard Propagation Engine — domain and graph layers.

Tests:
  1. Value objects / enums
  2. Domain models (Hazard, PropagationPath, ZoneRiskState, HazardPropagation)
  3. Graph entities (ZoneNode, EquipmentNode, SensorNode, HazardNode)
  4. FacilityGraph (zones, connections, equipment, sensors, traversal)
  5. Schemas (request validation, response serialization)
  6. Domain exceptions
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.hazard_propagation.domain.exceptions import (
    CyclicPropagationError,
    GraphNotInitializedError,
    HazardPropagationError,
    InvalidHazardError,
    PropagationSimulationError,
    ZoneNotFoundError,
)
from app.hazard_propagation.domain.models import (
    Hazard,
    HazardPropagation,
    PropagationPath,
    ZoneRiskState,
)
from app.hazard_propagation.domain.value_objects import (
    HazardType,
    PropagationLevel,
    PropagationStatus,
    RiskLevel,
)
from app.hazard_propagation.graph.entities import (
    EquipmentNode,
    GraphEdge,
    HazardNode,
    RelationshipType,
    SensorNode,
    ZoneNode,
)
from app.hazard_propagation.graph.facility_graph import FacilityGraph
from app.hazard_propagation.schemas.hazard_schemas import (
    HazardPathResponse,
    HazardPropagationRequest,
    HazardPropagationResponse,
    ZoneRiskStateResponse,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Value Objects / Enums
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHazardType:
    def test_all_ps1_hazard_types(self):
        """All 9 hazard types from PS-1 §4.6."""
        expected = {
            "GAS_LEAK", "FIRE", "SMOKE", "CHEMICAL_SPILL",
            "PPE_VIOLATION", "FALL_DETECTED", "ELECTRICAL_FAULT",
            "TEMPERATURE_ANOMALY", "PRESSURE_ANOMALY",
        }
        actual = {h.value for h in HazardType}
        assert actual == expected

    def test_hazard_type_count(self):
        assert len(HazardType) == 9

    def test_hazard_type_is_string_enum(self):
        assert isinstance(HazardType.GAS_LEAK, str)
        assert HazardType.FIRE == "FIRE"


class TestPropagationStatus:
    def test_all_statuses(self):
        expected = {"PENDING", "RUNNING", "COMPLETED", "FAILED"}
        actual = {s.value for s in PropagationStatus}
        assert actual == expected


class TestPropagationLevel:
    def test_all_levels(self):
        expected = {"CONTAINED", "SPREADING", "CRITICAL", "EMERGENCY"}
        actual = {l.value for l in PropagationLevel}
        assert actual == expected


class TestRiskLevel:
    def test_all_risk_levels(self):
        """PS-1 §4.1 — LOW | MEDIUM | HIGH | CRITICAL."""
        expected = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        actual = {r.value for r in RiskLevel}
        assert actual == expected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Domain Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHazard:
    def test_create_with_defaults(self):
        h = Hazard(hazard_type=HazardType.GAS_LEAK, origin_zone="ZONE_A")
        assert h.hazard_type == HazardType.GAS_LEAK
        assert h.origin_zone == "ZONE_A"
        assert h.severity == RiskLevel.HIGH
        assert h.is_active is True
        assert h.hazard_id  # UUID generated

    def test_create_with_all_fields(self):
        h = Hazard(
            hazard_type=HazardType.FIRE,
            origin_zone="ZONE_B",
            severity=RiskLevel.CRITICAL,
            description="Fire in boiler room",
            sensor_id="S001",
            equipment_id="EQ001",
        )
        assert h.severity == RiskLevel.CRITICAL
        assert h.description == "Fire in boiler room"
        assert h.sensor_id == "S001"
        assert h.equipment_id == "EQ001"

    def test_is_critical(self):
        h = Hazard(
            hazard_type=HazardType.FIRE, origin_zone="ZONE_A",
            severity=RiskLevel.CRITICAL,
        )
        assert h.is_critical is True

    def test_not_critical(self):
        h = Hazard(
            hazard_type=HazardType.SMOKE, origin_zone="ZONE_A",
            severity=RiskLevel.MEDIUM,
        )
        assert h.is_critical is False

    def test_deactivate(self):
        h = Hazard(hazard_type=HazardType.GAS_LEAK, origin_zone="ZONE_A")
        assert h.is_active is True
        h.deactivate()
        assert h.is_active is False

    def test_detected_at_is_utc(self):
        h = Hazard(hazard_type=HazardType.GAS_LEAK, origin_zone="ZONE_A")
        assert h.detected_at.tzinfo is not None

    def test_metadata_defaults_to_empty(self):
        h = Hazard(hazard_type=HazardType.GAS_LEAK, origin_zone="ZONE_A")
        assert h.metadata == {}


class TestPropagationPath:
    def test_create_with_defaults(self):
        p = PropagationPath(from_zone="ZONE_A", to_zone="ZONE_B")
        assert p.from_zone == "ZONE_A"
        assert p.to_zone == "ZONE_B"
        assert p.probability == 1.0
        assert p.estimated_time_minutes == 5.0
        assert p.path_type == "CONNECTED_TO"
        assert p.blocked is False

    def test_is_passable(self):
        p = PropagationPath(from_zone="ZONE_A", to_zone="ZONE_B")
        assert p.is_passable is True

    def test_blocked_not_passable(self):
        p = PropagationPath(
            from_zone="ZONE_A", to_zone="ZONE_B", blocked=True,
        )
        assert p.is_passable is False

    def test_zero_probability_not_passable(self):
        p = PropagationPath(
            from_zone="ZONE_A", to_zone="ZONE_B", probability=0.0,
        )
        assert p.is_passable is False

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError, match="probability"):
            PropagationPath(
                from_zone="ZONE_A", to_zone="ZONE_B", probability=1.5,
            )

    def test_negative_probability_raises(self):
        with pytest.raises(ValueError, match="probability"):
            PropagationPath(
                from_zone="ZONE_A", to_zone="ZONE_B", probability=-0.1,
            )

    def test_negative_time_raises(self):
        with pytest.raises(ValueError, match="estimated_time_minutes"):
            PropagationPath(
                from_zone="ZONE_A", to_zone="ZONE_B",
                estimated_time_minutes=-1.0,
            )


class TestZoneRiskState:
    def test_create_with_defaults(self):
        z = ZoneRiskState(zone_id="ZONE_A")
        assert z.zone_id == "ZONE_A"
        assert z.risk_level == RiskLevel.LOW
        assert z.risk_score == 0.0
        assert z.is_origin is False
        assert z.is_affected is False

    def test_has_workers_at_risk(self):
        z = ZoneRiskState(
            zone_id="ZONE_A", is_affected=True, worker_count=5,
        )
        assert z.has_workers_at_risk is True

    def test_no_workers_at_risk_when_not_affected(self):
        z = ZoneRiskState(
            zone_id="ZONE_A", is_affected=False, worker_count=5,
        )
        assert z.has_workers_at_risk is False

    def test_requires_evacuation_high_risk(self):
        z = ZoneRiskState(
            zone_id="ZONE_A", risk_level=RiskLevel.HIGH, is_affected=True,
        )
        assert z.requires_evacuation is True

    def test_requires_evacuation_critical(self):
        z = ZoneRiskState(
            zone_id="ZONE_A", risk_level=RiskLevel.CRITICAL, is_affected=True,
        )
        assert z.requires_evacuation is True

    def test_no_evacuation_low_risk(self):
        z = ZoneRiskState(
            zone_id="ZONE_A", risk_level=RiskLevel.LOW, is_affected=True,
        )
        assert z.requires_evacuation is False

    def test_invalid_risk_score_raises(self):
        with pytest.raises(ValueError, match="risk_score"):
            ZoneRiskState(zone_id="ZONE_A", risk_score=101.0)

    def test_active_hazards(self):
        z = ZoneRiskState(
            zone_id="ZONE_A",
            active_hazards=["GAS_LEAK", "FIRE"],
        )
        assert len(z.active_hazards) == 2


class TestHazardPropagation:
    def _make_propagation(self) -> HazardPropagation:
        hazard = Hazard(hazard_type=HazardType.GAS_LEAK, origin_zone="ZONE_A")
        return HazardPropagation(
            hazard=hazard,
            affected_zones=[
                ZoneRiskState(
                    zone_id="ZONE_A", is_origin=True, is_affected=True,
                    risk_level=RiskLevel.CRITICAL, worker_count=3,
                ),
                ZoneRiskState(
                    zone_id="ZONE_B", is_affected=True,
                    risk_level=RiskLevel.HIGH, worker_count=5,
                ),
                ZoneRiskState(
                    zone_id="ZONE_C", is_affected=False, worker_count=2,
                ),
            ],
        )

    def test_total_affected_zones(self):
        p = self._make_propagation()
        assert p.total_affected_zones == 2

    def test_total_workers_at_risk(self):
        p = self._make_propagation()
        assert p.total_workers_at_risk == 8  # 3 + 5

    def test_affected_zone_ids(self):
        p = self._make_propagation()
        assert set(p.affected_zone_ids) == {"ZONE_A", "ZONE_B"}

    def test_origin_zone(self):
        p = self._make_propagation()
        assert p.origin_zone is not None
        assert p.origin_zone.zone_id == "ZONE_A"

    def test_mark_completed(self):
        p = self._make_propagation()
        assert p.status == PropagationStatus.PENDING
        p.mark_completed()
        assert p.status == PropagationStatus.COMPLETED
        assert p.completed_at is not None

    def test_mark_failed(self):
        p = self._make_propagation()
        p.mark_failed("Timeout")
        assert p.status == PropagationStatus.FAILED
        assert p.metadata["failure_reason"] == "Timeout"

    def test_is_completed(self):
        p = self._make_propagation()
        assert p.is_completed is False
        p.mark_completed()
        assert p.is_completed is True

    def test_add_affected_zone(self):
        p = self._make_propagation()
        initial = len(p.affected_zones)
        p.add_affected_zone(
            ZoneRiskState(zone_id="ZONE_D", is_affected=True),
        )
        assert len(p.affected_zones) == initial + 1

    def test_add_propagation_path(self):
        p = self._make_propagation()
        p.add_propagation_path(
            PropagationPath(from_zone="ZONE_A", to_zone="ZONE_B"),
        )
        assert len(p.propagation_paths) == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Graph Entities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestZoneNode:
    def test_create(self):
        z = ZoneNode(zone_id="ZONE_A", zone_name="Zone A")
        assert z.zone_id == "ZONE_A"
        assert z.zone_name == "Zone A"
        assert z.connected_zones == []
        assert z.equipment == []

    def test_connect_to(self):
        z = ZoneNode(zone_id="ZONE_A")
        z.connect_to("ZONE_B")
        assert "ZONE_B" in z.connected_zones

    def test_connect_to_self_ignored(self):
        z = ZoneNode(zone_id="ZONE_A")
        z.connect_to("ZONE_A")
        assert z.connected_zones == []

    def test_connect_to_duplicate_ignored(self):
        z = ZoneNode(zone_id="ZONE_A")
        z.connect_to("ZONE_B")
        z.connect_to("ZONE_B")
        assert z.connected_zones == ["ZONE_B"]

    def test_add_equipment(self):
        z = ZoneNode(zone_id="ZONE_A")
        eq = EquipmentNode(equipment_id="EQ001")
        z.add_equipment(eq)
        assert z.equipment_count == 1
        assert eq.location_zone_id == "ZONE_A"

    def test_all_sensors(self):
        z = ZoneNode(zone_id="ZONE_A")
        eq = EquipmentNode(equipment_id="EQ001")
        eq.add_sensor(SensorNode(sensor_id="S001"))
        eq.add_sensor(SensorNode(sensor_id="S002"))
        z.add_equipment(eq)
        assert z.sensor_count == 2

    def test_has_active_hazards(self):
        z = ZoneNode(zone_id="ZONE_A", active_hazards=["GAS_LEAK"])
        assert z.has_active_hazards is True

    def test_no_active_hazards(self):
        z = ZoneNode(zone_id="ZONE_A")
        assert z.has_active_hazards is False


class TestEquipmentNode:
    def test_create(self):
        eq = EquipmentNode(equipment_id="EQ001", equipment_type="Boiler")
        assert eq.equipment_id == "EQ001"
        assert eq.equipment_type == "Boiler"
        assert eq.is_operational is True

    def test_add_sensor(self):
        eq = EquipmentNode(equipment_id="EQ001")
        s = SensorNode(sensor_id="S001")
        eq.add_sensor(s)
        assert eq.sensor_count == 1
        assert s.equipment_id == "EQ001"

    def test_not_operational(self):
        eq = EquipmentNode(
            equipment_id="EQ001", operational_status="FAULTY",
        )
        assert eq.is_operational is False


class TestSensorNode:
    def test_create(self):
        s = SensorNode(
            sensor_id="S001", sensor_type="GAS",
            unit_of_measurement="ppm",
        )
        assert s.sensor_id == "S001"
        assert s.sensor_type == "GAS"
        assert s.sensor_status == "ACTIVE"


class TestHazardNode:
    def test_create(self):
        h = HazardNode(hazard_id="H001", hazard_type="GAS_LEAK")
        assert h.hazard_id == "H001"
        assert h.hazard_type == "GAS_LEAK"

    def test_affects(self):
        h = HazardNode(hazard_id="H001", hazard_type="FIRE")
        h.affects("ZONE_A")
        h.affects("ZONE_B")
        assert h.affected_zones == ["ZONE_A", "ZONE_B"]

    def test_affects_duplicate_ignored(self):
        h = HazardNode(hazard_id="H001", hazard_type="FIRE")
        h.affects("ZONE_A")
        h.affects("ZONE_A")
        assert h.affected_zones == ["ZONE_A"]

    def test_causes(self):
        h = HazardNode(hazard_id="H001", hazard_type="FIRE")
        h.causes("INC-001")
        assert h.caused_incidents == ["INC-001"]

    def test_causes_duplicate_ignored(self):
        h = HazardNode(hazard_id="H001", hazard_type="FIRE")
        h.causes("INC-001")
        h.causes("INC-001")
        assert h.caused_incidents == ["INC-001"]


class TestRelationshipType:
    def test_connected_to(self):
        assert RelationshipType.CONNECTED_TO == "CONNECTED_TO"

    def test_contains(self):
        assert RelationshipType.CONTAINS == "CONTAINS"

    def test_has_sensor(self):
        assert RelationshipType.HAS_SENSOR == "HAS_SENSOR"

    def test_affects(self):
        assert RelationshipType.AFFECTS == "AFFECTS"

    def test_causes(self):
        assert RelationshipType.CAUSES == "CAUSES"


class TestGraphEdge:
    def test_create(self):
        e = GraphEdge(
            from_id="ZONE_A", to_id="ZONE_B",
            relationship="CONNECTED_TO",
        )
        assert e.from_id == "ZONE_A"
        assert e.to_id == "ZONE_B"
        assert e.weight == 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. FacilityGraph
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def sample_graph() -> FacilityGraph:
    """A sample 4-zone graph:

    ZONE_A ── ZONE_B ── ZONE_D
      │
    ZONE_C
    """
    g = FacilityGraph()
    g.add_zone(ZoneNode(zone_id="ZONE_A", zone_name="Zone A"))
    g.add_zone(ZoneNode(zone_id="ZONE_B", zone_name="Zone B"))
    g.add_zone(ZoneNode(zone_id="ZONE_C", zone_name="Zone C"))
    g.add_zone(ZoneNode(zone_id="ZONE_D", zone_name="Zone D"))
    g.connect_zones("ZONE_A", "ZONE_B")
    g.connect_zones("ZONE_A", "ZONE_C")
    g.connect_zones("ZONE_B", "ZONE_D")
    return g


class TestFacilityGraphZones:
    def test_add_zone(self):
        g = FacilityGraph()
        g.add_zone(ZoneNode(zone_id="ZONE_A"))
        assert g.zone_count == 1

    def test_get_zone(self, sample_graph):
        z = sample_graph.get_zone("ZONE_A")
        assert z is not None
        assert z.zone_id == "ZONE_A"

    def test_get_zone_not_found(self, sample_graph):
        assert sample_graph.get_zone("ZONE_Z") is None

    def test_get_zone_or_raise(self, sample_graph):
        z = sample_graph.get_zone_or_raise("ZONE_A")
        assert z.zone_id == "ZONE_A"

    def test_get_zone_or_raise_not_found(self, sample_graph):
        with pytest.raises(ZoneNotFoundError, match="ZONE_Z"):
            sample_graph.get_zone_or_raise("ZONE_Z")

    def test_has_zone(self, sample_graph):
        assert sample_graph.has_zone("ZONE_A") is True
        assert sample_graph.has_zone("ZONE_Z") is False

    def test_zone_count(self, sample_graph):
        assert sample_graph.zone_count == 4

    def test_zone_ids(self, sample_graph):
        assert set(sample_graph.zone_ids) == {
            "ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D",
        }

    def test_remove_zone(self, sample_graph):
        sample_graph.remove_zone("ZONE_C")
        assert sample_graph.zone_count == 3
        assert not sample_graph.has_zone("ZONE_C")
        # ZONE_A should no longer list ZONE_C as connected
        a = sample_graph.get_zone("ZONE_A")
        assert "ZONE_C" not in a.connected_zones


class TestFacilityGraphConnections:
    def test_connect_zones(self, sample_graph):
        connected = sample_graph.get_connected_zone_ids("ZONE_A")
        assert set(connected) == {"ZONE_B", "ZONE_C"}

    def test_bidirectional(self, sample_graph):
        """Connections are bidirectional by default."""
        connected = sample_graph.get_connected_zone_ids("ZONE_B")
        assert "ZONE_A" in connected

    def test_unidirectional(self):
        g = FacilityGraph()
        g.add_zone(ZoneNode(zone_id="ZONE_A"))
        g.add_zone(ZoneNode(zone_id="ZONE_B"))
        g.connect_zones("ZONE_A", "ZONE_B", bidirectional=False)
        assert "ZONE_B" in g.get_connected_zone_ids("ZONE_A")
        assert "ZONE_A" not in g.get_connected_zone_ids("ZONE_B")

    def test_connect_nonexistent_zone_raises(self):
        g = FacilityGraph()
        g.add_zone(ZoneNode(zone_id="ZONE_A"))
        with pytest.raises(ZoneNotFoundError):
            g.connect_zones("ZONE_A", "ZONE_Z")

    def test_get_connected_zones_returns_nodes(self, sample_graph):
        connected = sample_graph.get_connected_zones("ZONE_A")
        assert len(connected) == 2
        assert all(isinstance(z, ZoneNode) for z in connected)


class TestFacilityGraphEquipment:
    def test_add_equipment(self, sample_graph):
        eq = EquipmentNode(equipment_id="EQ001", equipment_type="Boiler")
        sample_graph.add_equipment_to_zone("ZONE_A", eq)
        zone = sample_graph.get_zone("ZONE_A")
        assert zone.equipment_count == 1

    def test_get_equipment_in_zone(self, sample_graph):
        eq1 = EquipmentNode(equipment_id="EQ001")
        eq2 = EquipmentNode(equipment_id="EQ002")
        sample_graph.add_equipment_to_zone("ZONE_B", eq1)
        sample_graph.add_equipment_to_zone("ZONE_B", eq2)
        result = sample_graph.get_equipment_in_zone("ZONE_B")
        assert len(result) == 2

    def test_add_equipment_sets_zone(self, sample_graph):
        eq = EquipmentNode(equipment_id="EQ001")
        sample_graph.add_equipment_to_zone("ZONE_C", eq)
        assert eq.location_zone_id == "ZONE_C"


class TestFacilityGraphSensors:
    def test_add_sensor_to_equipment(self, sample_graph):
        eq = EquipmentNode(equipment_id="EQ001")
        sample_graph.add_equipment_to_zone("ZONE_A", eq)
        sensor = SensorNode(sensor_id="S001", sensor_type="GAS")
        sample_graph.add_sensor_to_equipment("EQ001", sensor)
        assert sensor.equipment_id == "EQ001"
        assert sensor.zone_id == "ZONE_A"

    def test_get_sensors_in_zone(self, sample_graph):
        eq = EquipmentNode(equipment_id="EQ001")
        sample_graph.add_equipment_to_zone("ZONE_A", eq)
        sample_graph.add_sensor_to_equipment(
            "EQ001", SensorNode(sensor_id="S001"),
        )
        sample_graph.add_sensor_to_equipment(
            "EQ001", SensorNode(sensor_id="S002"),
        )
        sensors = sample_graph.get_sensors_in_zone("ZONE_A")
        assert len(sensors) == 2

    def test_add_sensor_nonexistent_equipment(self, sample_graph):
        with pytest.raises(ValueError, match="Equipment not found"):
            sample_graph.add_sensor_to_equipment(
                "EQ_NONEXIST", SensorNode(sensor_id="S999"),
            )


class TestFacilityGraphHazards:
    def test_add_hazard(self, sample_graph):
        h = HazardNode(
            hazard_id="H001", hazard_type="GAS_LEAK",
            affected_zones=["ZONE_A"],
        )
        sample_graph.add_hazard(h)
        assert sample_graph.hazard_count == 1

    def test_get_hazard(self, sample_graph):
        h = HazardNode(hazard_id="H001", hazard_type="FIRE")
        sample_graph.add_hazard(h)
        assert sample_graph.get_hazard("H001") is not None
        assert sample_graph.get_hazard("H999") is None


class TestFacilityGraphTraversal:
    def test_zones_within_1_hop(self, sample_graph):
        result = sample_graph.get_zones_within_hops("ZONE_A", 1)
        assert result == {"ZONE_A": 0, "ZONE_B": 1, "ZONE_C": 1}

    def test_zones_within_2_hops(self, sample_graph):
        result = sample_graph.get_zones_within_hops("ZONE_A", 2)
        assert result == {
            "ZONE_A": 0, "ZONE_B": 1, "ZONE_C": 1, "ZONE_D": 2,
        }

    def test_zones_within_0_hops(self, sample_graph):
        result = sample_graph.get_zones_within_hops("ZONE_A", 0)
        # max_hops=0 → only the range(1,1) body runs (empty), so only origin
        assert result == {"ZONE_A": 0}

    def test_zones_within_hops_nonexistent_raises(self, sample_graph):
        with pytest.raises(ZoneNotFoundError):
            sample_graph.get_zones_within_hops("ZONE_Z", 1)

    def test_zones_within_hops_isolated(self):
        g = FacilityGraph()
        g.add_zone(ZoneNode(zone_id="ZONE_X"))
        result = g.get_zones_within_hops("ZONE_X", 3)
        assert result == {"ZONE_X": 0}

    def test_all_paths_from_origin(self, sample_graph):
        paths = sample_graph.get_all_paths("ZONE_A", max_depth=3)
        # Should include paths to B, C, D
        all_destinations = set()
        for path in paths:
            all_destinations.add(path[-1])
        assert "ZONE_B" in all_destinations
        assert "ZONE_C" in all_destinations
        assert "ZONE_D" in all_destinations

    def test_all_paths_max_depth_1(self, sample_graph):
        paths = sample_graph.get_all_paths("ZONE_A", max_depth=1)
        for path in paths:
            assert len(path) <= 2  # origin + 1 hop

    def test_all_paths_nonexistent_raises(self, sample_graph):
        with pytest.raises(ZoneNotFoundError):
            sample_graph.get_all_paths("ZONE_Z")


class TestFacilityGraphSerialization:
    def test_to_dict(self, sample_graph):
        data = sample_graph.to_dict()
        assert "zones" in data
        assert "hazards" in data
        assert "edge_count" in data
        assert len(data["zones"]) == 4

    def test_to_dict_zone_fields(self, sample_graph):
        data = sample_graph.to_dict()
        zone_a = data["zones"]["ZONE_A"]
        assert zone_a["zone_id"] == "ZONE_A"
        assert zone_a["zone_name"] == "Zone A"
        assert set(zone_a["connected_zones"]) == {"ZONE_B", "ZONE_C"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHazardPropagationRequest:
    def test_valid_request(self):
        req = HazardPropagationRequest(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert req.hazard_type == "GAS_LEAK"
        assert req.origin_zone == "ZONE_A"
        assert req.max_propagation_depth == 3
        assert req.include_paths is True

    def test_case_insensitive_hazard_type(self):
        req = HazardPropagationRequest(
            hazard_type="gas_leak", origin_zone="ZONE_A",
        )
        assert req.hazard_type == "GAS_LEAK"

    def test_invalid_hazard_type_raises(self):
        with pytest.raises(Exception):
            HazardPropagationRequest(
                hazard_type="INVALID_TYPE", origin_zone="ZONE_A",
            )

    def test_all_valid_hazard_types(self):
        valid_types = [
            "GAS_LEAK", "FIRE", "SMOKE", "CHEMICAL_SPILL",
            "PPE_VIOLATION", "FALL_DETECTED", "ELECTRICAL_FAULT",
            "TEMPERATURE_ANOMALY", "PRESSURE_ANOMALY",
        ]
        for ht in valid_types:
            req = HazardPropagationRequest(
                hazard_type=ht, origin_zone="ZONE_A",
            )
            assert req.hazard_type == ht

    def test_custom_depth(self):
        req = HazardPropagationRequest(
            hazard_type="FIRE", origin_zone="ZONE_A",
            max_propagation_depth=5,
        )
        assert req.max_propagation_depth == 5

    def test_depth_too_high(self):
        with pytest.raises(Exception):
            HazardPropagationRequest(
                hazard_type="FIRE", origin_zone="ZONE_A",
                max_propagation_depth=15,
            )

    def test_severity_override(self):
        req = HazardPropagationRequest(
            hazard_type="FIRE", origin_zone="ZONE_A",
            severity="CRITICAL",
        )
        assert req.severity == "CRITICAL"


class TestHazardPathResponse:
    def test_create(self):
        r = HazardPathResponse(
            from_zone="ZONE_A", to_zone="ZONE_B",
            probability=0.8, estimated_time_minutes=10.0,
        )
        assert r.from_zone == "ZONE_A"
        assert r.probability == 0.8

    def test_invalid_probability(self):
        with pytest.raises(Exception):
            HazardPathResponse(
                from_zone="ZONE_A", to_zone="ZONE_B",
                probability=1.5, estimated_time_minutes=10.0,
            )


class TestZoneRiskStateResponse:
    def test_create(self):
        r = ZoneRiskStateResponse(
            zone_id="ZONE_A", risk_level="HIGH", risk_score=80.0,
            is_origin=True, is_affected=True,
        )
        assert r.zone_id == "ZONE_A"
        assert r.risk_score == 80.0


class TestHazardPropagationResponse:
    def test_create(self):
        r = HazardPropagationResponse(
            propagation_id="P001",
            hazard_type="GAS_LEAK",
            origin_zone="ZONE_A",
            propagation_level="SPREADING",
            affected_zones=["ZONE_A", "ZONE_B"],
            affected_workers=["W001", "W002", "W010"],
            impact_radius_meters=75.0,
            time_to_critical_minutes=15.0,
            recommended_action="Evacuate Zone A and restrict access to Zone B",
        )
        assert r.success is True
        assert r.hazard_type == "GAS_LEAK"
        assert len(r.affected_zones) == 2
        assert len(r.affected_workers) == 3
        assert r.impact_radius_meters == 75.0
        assert r.time_to_critical_minutes == 15.0

    def test_response_matches_api_spec(self):
        """Verify response matches the API spec §21."""
        r = HazardPropagationResponse(
            propagation_id="P001",
            hazard_type="GAS_LEAK",
            origin_zone="ZONE_A",
            propagation_level="SPREADING",
            affected_zones=["ZONE_A", "ZONE_B"],
            affected_workers=["W001", "W002", "W010"],
            impact_radius_meters=75,
            time_to_critical_minutes=15,
            recommended_action="Evacuate Zone A and restrict access to Zone B",
        )
        data = r.model_dump()
        assert "success" in data
        assert "affected_zones" in data
        assert "affected_workers" in data
        assert "impact_radius_meters" in data
        assert "time_to_critical_minutes" in data
        assert "recommended_action" in data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Exceptions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestExceptions:
    def test_base_exception(self):
        with pytest.raises(HazardPropagationError):
            raise HazardPropagationError("test")

    def test_invalid_hazard(self):
        with pytest.raises(InvalidHazardError):
            raise InvalidHazardError("bad input")

    def test_zone_not_found(self):
        err = ZoneNotFoundError("ZONE_X")
        assert err.zone_id == "ZONE_X"
        assert "ZONE_X" in str(err)

    def test_graph_not_initialized(self):
        with pytest.raises(GraphNotInitializedError):
            raise GraphNotInitializedError()

    def test_propagation_simulation_error(self):
        with pytest.raises(PropagationSimulationError):
            raise PropagationSimulationError("engine error")

    def test_cyclic_propagation(self):
        with pytest.raises(CyclicPropagationError):
            raise CyclicPropagationError("cycle detected")

    def test_all_inherit_from_base(self):
        assert issubclass(InvalidHazardError, HazardPropagationError)
        assert issubclass(ZoneNotFoundError, HazardPropagationError)
        assert issubclass(GraphNotInitializedError, HazardPropagationError)
        assert issubclass(PropagationSimulationError, HazardPropagationError)
        assert issubclass(CyclicPropagationError, HazardPropagationError)
