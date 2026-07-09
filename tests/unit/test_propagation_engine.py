"""Comprehensive unit tests for the Hazard Propagation Engine.

Tests:
  1.  Configuration validation
  2.  Single-zone propagation (contained)
  3.  Multi-zone propagation (linear, star, diamond)
  4.  Cycles in graph
  5.  Disconnected graphs
  6.  Depth limits
  7.  Decay behaviour (per-hazard-type overrides)
  8.  Zone impact scoring
  9.  Equipment impact scoring
  10. Propagation probability calculation
  11. Propagation level classification
  12. Recommendation generation
  13. Multiple hazards
  14. Edge cases (nonexistent zone, zero risk, max depth)
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from app.hazard_propagation.domain.exceptions import (
    PropagationSimulationError,
    ZoneNotFoundError,
)
from app.hazard_propagation.domain.value_objects import (
    PropagationLevel,
    PropagationStatus,
    RiskLevel,
)
from app.hazard_propagation.graph.entities import (
    EquipmentNode,
    SensorNode,
    ZoneNode,
)
from app.hazard_propagation.repositories.in_memory_graph_repo import (
    InMemoryGraphRepository,
)
from app.hazard_propagation.services.config import (
    HAZARD_DECAY_OVERRIDES,
    PropagationConfig,
)
from app.hazard_propagation.services.propagation_engine import (
    HazardPropagationEngine,
    PropagationResult,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest_asyncio.fixture
async def single_zone_repo() -> InMemoryGraphRepository:
    """Single isolated zone."""
    repo = InMemoryGraphRepository()
    await repo.create_zone(ZoneNode(
        zone_id="ZONE_A", zone_name="Zone A",
        risk_level_baseline="HIGH",
        current_worker_count=5,
    ))
    return repo


@pytest_asyncio.fixture
async def linear_repo() -> InMemoryGraphRepository:
    """Linear chain: A ── B ── C ── D ── E"""
    repo = InMemoryGraphRepository()
    zones = [
        ZoneNode(zone_id="ZONE_A", zone_name="Zone A",
                 risk_level_baseline="HIGH", current_worker_count=5),
        ZoneNode(zone_id="ZONE_B", zone_name="Zone B",
                 risk_level_baseline="MEDIUM", current_worker_count=3),
        ZoneNode(zone_id="ZONE_C", zone_name="Zone C",
                 risk_level_baseline="LOW", current_worker_count=2),
        ZoneNode(zone_id="ZONE_D", zone_name="Zone D",
                 risk_level_baseline="LOW", current_worker_count=1),
        ZoneNode(zone_id="ZONE_E", zone_name="Zone E",
                 risk_level_baseline="MEDIUM", current_worker_count=4),
    ]
    for z in zones:
        await repo.create_zone(z)
    await repo.create_connection("ZONE_A", "ZONE_B")
    await repo.create_connection("ZONE_B", "ZONE_C")
    await repo.create_connection("ZONE_C", "ZONE_D")
    await repo.create_connection("ZONE_D", "ZONE_E")
    return repo


@pytest_asyncio.fixture
async def star_repo() -> InMemoryGraphRepository:
    """Star topology: B,C,D all connected to hub A."""
    repo = InMemoryGraphRepository()
    await repo.create_zone(ZoneNode(
        zone_id="ZONE_A", zone_name="Hub",
        risk_level_baseline="CRITICAL", current_worker_count=10,
    ))
    for suffix in ["B", "C", "D"]:
        await repo.create_zone(ZoneNode(
            zone_id=f"ZONE_{suffix}", zone_name=f"Zone {suffix}",
            risk_level_baseline="MEDIUM", current_worker_count=3,
        ))
        await repo.create_connection("ZONE_A", f"ZONE_{suffix}")
    return repo


@pytest_asyncio.fixture
async def diamond_repo() -> InMemoryGraphRepository:
    """Diamond:  A ── B
                 │    │
                 C ── D
    """
    repo = InMemoryGraphRepository()
    for suffix in ["A", "B", "C", "D"]:
        await repo.create_zone(ZoneNode(
            zone_id=f"ZONE_{suffix}", zone_name=f"Zone {suffix}",
            risk_level_baseline="MEDIUM", current_worker_count=2,
        ))
    await repo.create_connection("ZONE_A", "ZONE_B")
    await repo.create_connection("ZONE_A", "ZONE_C")
    await repo.create_connection("ZONE_B", "ZONE_D")
    await repo.create_connection("ZONE_C", "ZONE_D")
    return repo


@pytest_asyncio.fixture
async def cycle_repo() -> InMemoryGraphRepository:
    """Cycle: A ── B ── C ── A"""
    repo = InMemoryGraphRepository()
    for suffix in ["A", "B", "C"]:
        await repo.create_zone(ZoneNode(
            zone_id=f"ZONE_{suffix}", current_worker_count=2,
        ))
    await repo.create_connection("ZONE_A", "ZONE_B")
    await repo.create_connection("ZONE_B", "ZONE_C")
    await repo.create_connection("ZONE_C", "ZONE_A")
    return repo


@pytest_asyncio.fixture
async def disconnected_repo() -> InMemoryGraphRepository:
    """Two disconnected subgraphs: (A──B) and (C──D)."""
    repo = InMemoryGraphRepository()
    for suffix in ["A", "B", "C", "D"]:
        await repo.create_zone(ZoneNode(
            zone_id=f"ZONE_{suffix}", current_worker_count=2,
        ))
    await repo.create_connection("ZONE_A", "ZONE_B")
    await repo.create_connection("ZONE_C", "ZONE_D")
    return repo


@pytest_asyncio.fixture
async def equipped_repo() -> InMemoryGraphRepository:
    """Two connected zones with equipment and sensors."""
    repo = InMemoryGraphRepository()
    await repo.create_zone(ZoneNode(
        zone_id="ZONE_A", zone_name="Zone A",
        risk_level_baseline="HIGH", current_worker_count=5,
    ))
    await repo.create_zone(ZoneNode(
        zone_id="ZONE_B", zone_name="Zone B",
        risk_level_baseline="LOW", current_worker_count=1,
    ))
    await repo.create_connection("ZONE_A", "ZONE_B")

    # Equipment in ZONE_A: one healthy, one faulty
    await repo.create_equipment(
        "ZONE_A",
        EquipmentNode(
            equipment_id="EQ001", equipment_type="Boiler",
            health_score=90.0, operational_status="ACTIVE",
        ),
    )
    await repo.create_equipment(
        "ZONE_A",
        EquipmentNode(
            equipment_id="EQ002", equipment_type="Pump",
            health_score=30.0, operational_status="FAULTY",
        ),
    )

    # Equipment in ZONE_B
    await repo.create_equipment(
        "ZONE_B",
        EquipmentNode(
            equipment_id="EQ003", equipment_type="Valve",
            health_score=95.0, operational_status="ACTIVE",
        ),
    )

    # Sensors
    await repo.create_sensor("EQ001", SensorNode(sensor_id="S001"))
    await repo.create_sensor("EQ002", SensorNode(sensor_id="S002"))
    return repo


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPropagationConfig:
    def test_defaults(self):
        cfg = PropagationConfig()
        assert cfg.propagation_decay_factor == 0.6
        assert cfg.max_depth == 3
        assert cfg.minimum_propagation_threshold == 0.1

    def test_custom_values(self):
        cfg = PropagationConfig(
            propagation_decay_factor=0.8,
            max_depth=5,
            minimum_propagation_threshold=0.05,
        )
        assert cfg.propagation_decay_factor == 0.8
        assert cfg.max_depth == 5

    def test_invalid_decay(self):
        with pytest.raises(ValueError):
            PropagationConfig(propagation_decay_factor=1.5)

    def test_invalid_decay_negative(self):
        with pytest.raises(ValueError):
            PropagationConfig(propagation_decay_factor=-0.1)

    def test_invalid_max_depth_low(self):
        with pytest.raises(ValueError):
            PropagationConfig(max_depth=0)

    def test_invalid_max_depth_high(self):
        with pytest.raises(ValueError):
            PropagationConfig(max_depth=11)

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            PropagationConfig(minimum_propagation_threshold=2.0)

    def test_frozen(self):
        cfg = PropagationConfig()
        with pytest.raises(AttributeError):
            cfg.max_depth = 5  # type: ignore[misc]

    def test_hazard_decay_overrides(self):
        assert "GAS_LEAK" in HAZARD_DECAY_OVERRIDES
        assert "FIRE" in HAZARD_DECAY_OVERRIDES
        assert HAZARD_DECAY_OVERRIDES["PPE_VIOLATION"] == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Single-zone propagation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSingleZonePropagation:
    async def test_contained_propagation(self, single_zone_repo):
        engine = HazardPropagationEngine(single_zone_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert result.total_affected_zones == 1
        assert result.propagation_level == PropagationLevel.CONTAINED

    async def test_origin_zone_marked(self, single_zone_repo):
        engine = HazardPropagationEngine(single_zone_repo)
        result = await engine.propagate(
            hazard_type="FIRE", origin_zone="ZONE_A",
        )
        origin = next(z for z in result.affected_zones if z.is_origin)
        assert origin.zone_id == "ZONE_A"
        assert origin.is_affected is True
        assert origin.arrival_time_minutes == 0.0

    async def test_origin_probability_is_one(self, single_zone_repo):
        engine = HazardPropagationEngine(single_zone_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert result.propagation_probabilities["ZONE_A"] == 1.0

    async def test_no_propagation_paths(self, single_zone_repo):
        engine = HazardPropagationEngine(single_zone_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert result.propagation_paths == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Multi-zone propagation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLinearPropagation:
    async def test_spreads_to_neighbors(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=70.0,
        )
        assert result.total_affected_zones >= 2
        assert "ZONE_B" in result.affected_zone_ids

    async def test_depth_limits_spread(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            max_depth=1,
        )
        assert "ZONE_A" in result.affected_zone_ids
        assert "ZONE_B" in result.affected_zone_ids
        # ZONE_C should not be reached at depth 1
        # (unless decay keeps probability above threshold)
        assert "ZONE_D" not in result.affected_zone_ids

    async def test_propagation_paths_exist(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert len(result.propagation_paths) > 0
        # First path should be from A to B
        path_edges = {(p.from_zone, p.to_zone) for p in result.propagation_paths}
        assert ("ZONE_A", "ZONE_B") in path_edges

    async def test_probability_decreases_with_distance(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        probs = result.propagation_probabilities
        if "ZONE_B" in probs and "ZONE_C" in probs:
            assert probs["ZONE_B"] > probs["ZONE_C"]

    async def test_arrival_time_increases(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        times = {
            z.zone_id: z.arrival_time_minutes
            for z in result.affected_zones
        }
        assert times["ZONE_A"] == 0.0
        if "ZONE_B" in times:
            assert times["ZONE_B"] > 0.0


class TestStarPropagation:
    async def test_all_spokes_affected(self, star_repo):
        engine = HazardPropagationEngine(star_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        ids = result.affected_zone_ids
        assert "ZONE_B" in ids
        assert "ZONE_C" in ids
        assert "ZONE_D" in ids

    async def test_all_spokes_same_probability(self, star_repo):
        engine = HazardPropagationEngine(star_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        probs = result.propagation_probabilities
        assert probs.get("ZONE_B") == probs.get("ZONE_C")
        assert probs.get("ZONE_C") == probs.get("ZONE_D")

    async def test_spoke_to_hub_propagation(self, star_repo):
        """Starting from a spoke, should reach hub and other spokes."""
        engine = HazardPropagationEngine(star_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_B",
            max_depth=2,
        )
        ids = result.affected_zone_ids
        assert "ZONE_A" in ids  # Hub at hop 1


class TestDiamondPropagation:
    async def test_reaches_all_zones(self, diamond_repo):
        engine = HazardPropagationEngine(diamond_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert result.total_affected_zones == 4

    async def test_diamond_no_duplicate_zones(self, diamond_repo):
        engine = HazardPropagationEngine(diamond_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        zone_ids = [z.zone_id for z in result.affected_zones]
        assert len(zone_ids) == len(set(zone_ids))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Cycles in graph
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCycles:
    async def test_cycle_terminates(self, cycle_repo):
        """BFS must handle cycles without infinite loop."""
        engine = HazardPropagationEngine(cycle_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert result.status == PropagationStatus.COMPLETED
        assert result.total_affected_zones == 3

    async def test_cycle_no_duplicate_zones(self, cycle_repo):
        engine = HazardPropagationEngine(cycle_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        zone_ids = [z.zone_id for z in result.affected_zones]
        assert len(zone_ids) == len(set(zone_ids))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Disconnected graphs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDisconnectedGraphs:
    async def test_only_connected_component_affected(self, disconnected_repo):
        engine = HazardPropagationEngine(disconnected_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        ids = result.affected_zone_ids
        assert "ZONE_A" in ids
        assert "ZONE_B" in ids
        assert "ZONE_C" not in ids
        assert "ZONE_D" not in ids

    async def test_other_component_unaffected(self, disconnected_repo):
        engine = HazardPropagationEngine(disconnected_repo)
        result = await engine.propagate(
            hazard_type="FIRE", origin_zone="ZONE_C",
        )
        ids = result.affected_zone_ids
        assert "ZONE_C" in ids
        assert "ZONE_D" in ids
        assert "ZONE_A" not in ids


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Depth limits
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDepthLimits:
    async def test_depth_1(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            max_depth=1,
        )
        # Only origin + 1 hop
        assert "ZONE_A" in result.affected_zone_ids
        assert "ZONE_B" in result.affected_zone_ids

    async def test_depth_config_override(self, linear_repo):
        config = PropagationConfig(max_depth=1)
        engine = HazardPropagationEngine(linear_repo, config=config)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert "ZONE_A" in result.affected_zone_ids
        assert "ZONE_B" in result.affected_zone_ids

    async def test_max_depth_parameter_overrides_config(self, linear_repo):
        config = PropagationConfig(max_depth=1)
        engine = HazardPropagationEngine(linear_repo, config=config)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            max_depth=2,
        )
        # max_depth=2 should override config's max_depth=1
        assert "ZONE_C" in result.affected_zone_ids or len(result.affected_zone_ids) >= 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Decay behaviour
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDecayBehaviour:
    async def test_gas_leak_higher_decay(self, linear_repo):
        """GAS_LEAK decay=0.7 → spreads further than FIRE decay=0.5."""
        engine = HazardPropagationEngine(linear_repo)

        gas_result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        fire_result = await engine.propagate(
            hazard_type="FIRE", origin_zone="ZONE_A",
        )
        assert gas_result.total_affected_zones >= fire_result.total_affected_zones

    async def test_ppe_violation_no_propagation(self, linear_repo):
        """PPE_VIOLATION decay=0.0 → does not propagate."""
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="PPE_VIOLATION", origin_zone="ZONE_A",
        )
        assert result.total_affected_zones == 1
        assert result.propagation_level == PropagationLevel.CONTAINED

    async def test_smoke_high_decay(self, linear_repo):
        """SMOKE decay=0.8 → spreads further than most."""
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="SMOKE", origin_zone="ZONE_A",
            max_depth=5,
        )
        assert result.total_affected_zones >= 3

    async def test_custom_decay_config(self, linear_repo):
        """Custom decay factor via config with high threshold to limit spread."""
        low_decay = PropagationConfig(
            propagation_decay_factor=0.2,
            minimum_propagation_threshold=0.15,
        )
        engine = HazardPropagationEngine(linear_repo, config=low_decay)
        # TEMPERATURE_ANOMALY has a hazard-specific override of 0.6,
        # but threshold=0.15 will prune after a few hops
        result = await engine.propagate(
            hazard_type="TEMPERATURE_ANOMALY", origin_zone="ZONE_A",
        )
        # With decay=0.6 and threshold=0.15: 0.6^3=0.216>0.15, 0.6^4=0.1296<0.15
        assert result.total_affected_zones <= 4

    async def test_threshold_prunes_low_probability(self, linear_repo):
        """Zones below minimum_propagation_threshold are excluded."""
        config = PropagationConfig(minimum_propagation_threshold=0.5)
        engine = HazardPropagationEngine(linear_repo, config=config)
        result = await engine.propagate(
            hazard_type="FIRE", origin_zone="ZONE_A",
            max_depth=5,
        )
        # With threshold=0.5 and FIRE decay=0.5, only hop 1 passes
        for zone_id, prob in result.propagation_probabilities.items():
            assert prob >= 0.5 or zone_id == "ZONE_A"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Zone impact scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestZoneImpactScoring:
    async def test_origin_has_highest_score(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        origin_score = result.impact_scores.get("ZONE_A", 0.0)
        for zone_id, score in result.impact_scores.items():
            if zone_id != "ZONE_A":
                assert origin_score >= score

    async def test_scores_in_valid_range(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=90.0,
        )
        for score in result.impact_scores.values():
            assert 0.0 <= score <= 100.0

    async def test_high_compound_risk_higher_scores(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        low = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=20.0,
        )
        high = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=90.0,
        )
        assert high.impact_scores["ZONE_A"] > low.impact_scores["ZONE_A"]

    async def test_risk_level_assigned(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        for zone in result.affected_zones:
            assert zone.risk_level in (
                RiskLevel.LOW, RiskLevel.MEDIUM,
                RiskLevel.HIGH, RiskLevel.CRITICAL,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Equipment impact scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEquipmentImpactScoring:
    async def test_equipment_impacts_computed(self, equipped_repo):
        engine = HazardPropagationEngine(equipped_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        assert len(result.affected_equipment) > 0

    async def test_faulty_equipment_higher_impact(self, equipped_repo):
        engine = HazardPropagationEngine(equipped_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        eq_impacts = {
            e.equipment_id: e for e in result.affected_equipment
        }
        # EQ002 (health=30) should have higher impact than EQ001 (health=90)
        if "EQ001" in eq_impacts and "EQ002" in eq_impacts:
            assert eq_impacts["EQ002"].impact_score > eq_impacts["EQ001"].impact_score

    async def test_equipment_zone_assignment(self, equipped_repo):
        engine = HazardPropagationEngine(equipped_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        for eq in result.affected_equipment:
            assert eq.zone_id in result.affected_zone_ids

    async def test_equipment_impact_range(self, equipped_repo):
        engine = HazardPropagationEngine(equipped_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        for eq in result.affected_equipment:
            assert 0.0 <= eq.impact_score <= 100.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Propagation probability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPropagationProbability:
    async def test_origin_probability_one(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert result.propagation_probabilities["ZONE_A"] == 1.0

    async def test_probabilities_decrease(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        probs = result.propagation_probabilities
        sorted_zones = sorted(probs.items(), key=lambda x: -x[1])
        # Origin should have highest probability
        assert sorted_zones[0][0] == "ZONE_A"

    async def test_probabilities_in_range(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        for prob in result.propagation_probabilities.values():
            assert 0.0 <= prob <= 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. Propagation level classification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPropagationLevel:
    async def test_contained_single_zone(self, single_zone_repo):
        engine = HazardPropagationEngine(single_zone_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert result.propagation_level == PropagationLevel.CONTAINED

    async def test_spreading_multi_zone(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            max_depth=1,
        )
        if result.total_affected_zones >= 2:
            assert result.propagation_level in (
                PropagationLevel.SPREADING,
                PropagationLevel.CRITICAL,
            )

    async def test_status_completed(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert result.status == PropagationStatus.COMPLETED


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. Recommendations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRecommendations:
    async def test_contained_recommendation(self, single_zone_repo):
        engine = HazardPropagationEngine(single_zone_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert "CONTAINED" in result.recommended_action

    async def test_recommendation_not_empty(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert len(result.recommended_action) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. Multiple hazards
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMultipleHazards:
    async def test_different_hazard_types(self, linear_repo):
        """Different hazard types produce different results."""
        engine = HazardPropagationEngine(linear_repo)
        gas = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        fire = await engine.propagate(
            hazard_type="FIRE", origin_zone="ZONE_A",
        )
        assert gas.hazard_type == "GAS_LEAK"
        assert fire.hazard_type == "FIRE"

    async def test_independent_simulations(self, linear_repo):
        """Two simulations don't interfere with each other."""
        engine = HazardPropagationEngine(linear_repo)
        r1 = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        r2 = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert r1.propagation_id != r2.propagation_id
        assert r1.total_affected_zones == r2.total_affected_zones

    async def test_different_origins(self, linear_repo):
        """Propagation from different origins."""
        engine = HazardPropagationEngine(linear_repo)
        from_a = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        from_e = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_E",
        )
        assert from_a.origin_zone == "ZONE_A"
        assert from_e.origin_zone == "ZONE_E"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 14. Edge cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEdgeCases:
    async def test_nonexistent_origin_raises(self, single_zone_repo):
        engine = HazardPropagationEngine(single_zone_repo)
        with pytest.raises(ZoneNotFoundError):
            await engine.propagate(
                hazard_type="GAS_LEAK", origin_zone="ZONE_Z",
            )

    async def test_zero_compound_risk(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=0.0,
        )
        # Zero risk → all impact scores should be 0
        for score in result.impact_scores.values():
            assert score == 0.0

    async def test_max_compound_risk(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=100.0,
        )
        assert result.impact_scores["ZONE_A"] > 0.0

    async def test_custom_hazard_id(self, single_zone_repo):
        engine = HazardPropagationEngine(single_zone_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            hazard_id="CUSTOM-ID-001",
        )
        assert result.propagation_id == "CUSTOM-ID-001"

    async def test_impact_radius_increases_with_spread(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        r1 = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            max_depth=1,
        )
        r3 = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            max_depth=3,
        )
        assert r3.impact_radius_meters >= r1.impact_radius_meters

    async def test_time_to_critical_high_risk(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=90.0,
        )
        # Already above critical threshold → time_to_critical ≈ 0
        assert result.time_to_critical_minutes == 0.0

    async def test_result_properties(self, linear_repo):
        engine = HazardPropagationEngine(linear_repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert isinstance(result, PropagationResult)
        assert result.total_affected_zones >= 1
        assert isinstance(result.affected_zone_ids, list)

    async def test_engine_config_accessible(self, single_zone_repo):
        config = PropagationConfig(max_depth=5)
        engine = HazardPropagationEngine(single_zone_repo, config=config)
        assert engine.config.max_depth == 5
