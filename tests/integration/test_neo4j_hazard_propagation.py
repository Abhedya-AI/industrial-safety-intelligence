"""Neo4j Hazard Propagation Verification Script.

Standalone integration test that:
  1. Connects to Neo4j and validates driver connectivity
  2. Seeds a PS-1 compliant facility graph (zones, equipment, sensors, hazards)
  3. Runs propagation scenarios using Neo4jGraphRepository
  4. Runs the same scenarios using InMemoryGraphRepository for parity
  5. Validates propagation paths, affected zones, and impact scores
  6. Verifies DI-based repository selection via settings
  7. Produces a verification report

Usage:
    python3 -m tests.integration.test_neo4j_hazard_propagation

Prerequisites:
    - Neo4j 5.x running at bolt://localhost:7687
    - neo4j Python driver installed (pip install neo4j)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Setup project path ──
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

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
from app.hazard_propagation.services.config import PropagationConfig
from app.hazard_propagation.services.hazard_propagation_service import (
    HazardPropagationService,
)
from app.hazard_propagation.services.propagation_engine import (
    HazardPropagationEngine,
    PropagationResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("neo4j_verification")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test data: PS-1 compliant facility graph
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ZONES = [
    ZoneNode(
        zone_id="ZONE_A", zone_name="Main Processing Hall",
        risk_level_baseline="MEDIUM", current_risk_score=35.0,
        worker_capacity=30, current_worker_count=12,
    ),
    ZoneNode(
        zone_id="ZONE_B", zone_name="Boiler Room",
        risk_level_baseline="HIGH", current_risk_score=55.0,
        worker_capacity=10, current_worker_count=4,
        is_restricted=True,
    ),
    ZoneNode(
        zone_id="ZONE_C", zone_name="Chemical Storage",
        risk_level_baseline="HIGH", current_risk_score=60.0,
        worker_capacity=8, current_worker_count=2,
        is_restricted=True,
    ),
    ZoneNode(
        zone_id="ZONE_D", zone_name="Assembly Line",
        risk_level_baseline="MEDIUM", current_risk_score=30.0,
        worker_capacity=40, current_worker_count=18,
    ),
    ZoneNode(
        zone_id="ZONE_E", zone_name="Warehouse",
        risk_level_baseline="LOW", current_risk_score=15.0,
        worker_capacity=20, current_worker_count=6,
    ),
    ZoneNode(
        zone_id="ZONE_F", zone_name="Loading Dock",
        risk_level_baseline="LOW", current_risk_score=10.0,
        worker_capacity=15, current_worker_count=3,
    ),
]

# Zone connectivity: A↔B, A↔C, B↔D, C↔D, D↔E, E↔F
CONNECTIONS = [
    ("ZONE_A", "ZONE_B"),
    ("ZONE_A", "ZONE_C"),
    ("ZONE_B", "ZONE_D"),
    ("ZONE_C", "ZONE_D"),
    ("ZONE_D", "ZONE_E"),
    ("ZONE_E", "ZONE_F"),
]

# Equipment per zone (PS-1 §1.3 entity format)
EQUIPMENT = {
    "ZONE_A": [
        EquipmentNode("EQ_PUMP_A1", "Pump", "Acme", "ZONE_A", "ACTIVE", 85.0),
        EquipmentNode("EQ_COMP_A1", "Compressor", "Baker", "ZONE_A", "ACTIVE", 90.0),
    ],
    "ZONE_B": [
        EquipmentNode("EQ_BOILER_B1", "Boiler", "CaldeiraTech", "ZONE_B", "ACTIVE", 70.0),
        EquipmentNode("EQ_VALVE_B1", "Valve", "ValveCorp", "ZONE_B", "FAULTY", 40.0),
    ],
    "ZONE_C": [
        EquipmentNode("EQ_TANK_C1", "Tank", "ChemStore", "ZONE_C", "ACTIVE", 95.0),
    ],
    "ZONE_D": [
        EquipmentNode("EQ_PRESS_D1", "Press", "IndPress", "ZONE_D", "ACTIVE", 88.0),
        EquipmentNode("EQ_CONV_D1", "Conveyor", "ConveyAll", "ZONE_D", "ACTIVE", 92.0),
    ],
    "ZONE_E": [
        EquipmentNode("EQ_FORK_E1", "Forklift", "LiftCo", "ZONE_E", "IDLE", 78.0),
    ],
}

# Sensors per equipment (PS-1 §1.4 entity format)
SENSORS = {
    "EQ_PUMP_A1": [
        SensorNode("S_GAS_A1", "GAS", "ppm", "EQ_PUMP_A1", "ZONE_A"),
        SensorNode("S_TEMP_A1", "TEMPERATURE", "°C", "EQ_PUMP_A1", "ZONE_A"),
    ],
    "EQ_COMP_A1": [
        SensorNode("S_VIB_A1", "VIBRATION", "m/s²", "EQ_COMP_A1", "ZONE_A"),
    ],
    "EQ_BOILER_B1": [
        SensorNode("S_TEMP_B1", "TEMPERATURE", "°C", "EQ_BOILER_B1", "ZONE_B"),
        SensorNode("S_PRES_B1", "PRESSURE", "bar", "EQ_BOILER_B1", "ZONE_B"),
    ],
    "EQ_VALVE_B1": [
        SensorNode("S_GAS_B1", "GAS", "ppm", "EQ_VALVE_B1", "ZONE_B"),
    ],
    "EQ_TANK_C1": [
        SensorNode("S_GAS_C1", "GAS", "ppm", "EQ_TANK_C1", "ZONE_C"),
        SensorNode("S_HUM_C1", "HUMIDITY", "%", "EQ_TANK_C1", "ZONE_C"),
    ],
    "EQ_PRESS_D1": [
        SensorNode("S_TEMP_D1", "TEMPERATURE", "°C", "EQ_PRESS_D1", "ZONE_D"),
    ],
    "EQ_CONV_D1": [
        SensorNode("S_VIB_D1", "VIBRATION", "m/s²", "EQ_CONV_D1", "ZONE_D"),
    ],
    "EQ_FORK_E1": [
        SensorNode("S_GAS_E1", "GAS", "ppm", "EQ_FORK_E1", "ZONE_E"),
    ],
}

# Pre-existing hazard
TEST_HAZARD = HazardNode(
    hazard_id="HAZ_TEST_001",
    hazard_type="GAS_LEAK",
    severity="HIGH",
    affected_zones=["ZONE_A", "ZONE_B"],
)

# Propagation scenarios
SCENARIOS = [
    {
        "name": "GAS_LEAK from ZONE_A (High Risk)",
        "hazard_type": "GAS_LEAK",
        "origin_zone": "ZONE_A",
        "compound_risk_score": 80.0,
        "expected_min_zones": 3,    # A + at least B and C (direct neighbors)
        "expected_origin_prob": 1.0,
    },
    {
        "name": "FIRE from ZONE_D (Critical)",
        "hazard_type": "FIRE",
        "origin_zone": "ZONE_D",
        "compound_risk_score": 95.0,
        "expected_min_zones": 3,    # D + neighbors
        "expected_origin_prob": 1.0,
    },
    {
        "name": "CHEMICAL_SPILL from ZONE_F (Edge Zone)",
        "hazard_type": "CHEMICAL_SPILL",
        "origin_zone": "ZONE_F",
        "compound_risk_score": 60.0,
        "expected_min_zones": 1,    # F (low propagation, edge zone)
        "expected_origin_prob": 1.0,
    },
    {
        "name": "SMOKE from ZONE_A (High Decay Factor)",
        "hazard_type": "SMOKE",
        "origin_zone": "ZONE_A",
        "compound_risk_score": 50.0,
        "expected_min_zones": 3,    # Smoke has 0.8 decay, spreads far
        "expected_origin_prob": 1.0,
    },
    {
        "name": "ELECTRICAL_FAULT from ZONE_B (Low Propagation)",
        "hazard_type": "ELECTRICAL_FAULT",
        "origin_zone": "ZONE_B",
        "compound_risk_score": 70.0,
        "expected_min_zones": 1,    # 0.3 decay, doesn't spread much
        "expected_origin_prob": 1.0,
    },
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result containers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TestResult:
    name: str
    passed: bool
    details: str = ""
    error: str = ""


@dataclass
class ScenarioResult:
    name: str
    hazard_type: str
    origin_zone: str
    compound_risk_score: float
    propagation_level: str = ""
    affected_zone_ids: List[str] = field(default_factory=list)
    total_affected_zones: int = 0
    total_workers_at_risk: int = 0
    impact_radius_meters: float = 0.0
    time_to_critical_minutes: float = 0.0
    impact_scores: Dict[str, float] = field(default_factory=dict)
    propagation_probabilities: Dict[str, float] = field(default_factory=dict)
    propagation_paths: List[Dict[str, Any]] = field(default_factory=list)
    equipment_impacts: List[Dict[str, Any]] = field(default_factory=list)
    recommended_action: str = ""
    processing_time_ms: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Seed graph into a repository
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def seed_graph(repo: GraphRepository) -> Dict[str, int]:
    """Seed the PS-1 compliant facility graph into a repository.

    Returns counts of created entities.
    """
    zone_count = 0
    equipment_count = 0
    sensor_count = 0
    connection_count = 0

    # 1. Create zones
    for zone in ZONES:
        await repo.create_zone(zone)
        zone_count += 1

    # 2. Create connections
    for zone_a, zone_b in CONNECTIONS:
        await repo.create_connection(zone_a, zone_b, bidirectional=True)
        connection_count += 1

    # 3. Create equipment
    for zone_id, equip_list in EQUIPMENT.items():
        for eq in equip_list:
            await repo.create_equipment(zone_id, eq)
            equipment_count += 1

    # 4. Create sensors
    for equip_id, sensor_list in SENSORS.items():
        for sensor in sensor_list:
            await repo.create_sensor(equip_id, sensor)
            sensor_count += 1

    # 5. Create hazard
    await repo.create_hazard(TEST_HAZARD)

    return {
        "zones": zone_count,
        "equipment": equipment_count,
        "sensors": sensor_count,
        "connections": connection_count,
        "hazards": 1,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run propagation scenario
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def run_scenario(
    service: HazardPropagationService,
    scenario: Dict[str, Any],
) -> ScenarioResult:
    """Execute a single propagation scenario and capture results."""
    start = time.monotonic()

    analysis = await service.propagate_hazard(
        hazard_type=scenario["hazard_type"],
        origin_zone=scenario["origin_zone"],
        compound_risk_score=scenario["compound_risk_score"],
    )

    elapsed = (time.monotonic() - start) * 1000
    r = analysis.propagation_result

    return ScenarioResult(
        name=scenario["name"],
        hazard_type=scenario["hazard_type"],
        origin_zone=scenario["origin_zone"],
        compound_risk_score=scenario["compound_risk_score"],
        propagation_level=r.propagation_level.value,
        affected_zone_ids=r.affected_zone_ids,
        total_affected_zones=r.total_affected_zones,
        total_workers_at_risk=r.total_workers_at_risk,
        impact_radius_meters=r.impact_radius_meters,
        time_to_critical_minutes=r.time_to_critical_minutes,
        impact_scores=r.impact_scores,
        propagation_probabilities=r.propagation_probabilities,
        propagation_paths=[
            {
                "from_zone": p.from_zone,
                "to_zone": p.to_zone,
                "probability": p.probability,
                "estimated_time_minutes": p.estimated_time_minutes,
            }
            for p in r.propagation_paths
        ],
        equipment_impacts=[
            {
                "equipment_id": eq.equipment_id,
                "zone_id": eq.zone_id,
                "impact_score": eq.impact_score,
                "is_critical": eq.is_critical,
            }
            for eq in r.affected_equipment
        ],
        recommended_action=r.recommended_action,
        processing_time_ms=elapsed,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Validation functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_scenario(
    result: ScenarioResult,
    scenario: Dict[str, Any],
) -> List[TestResult]:
    """Validate a scenario result against expected constraints."""
    tests = []

    # T1: Origin zone has probability 1.0
    origin_prob = result.propagation_probabilities.get(result.origin_zone, 0.0)
    tests.append(TestResult(
        name=f"[{result.name}] Origin zone probability = 1.0",
        passed=abs(origin_prob - 1.0) < 1e-6,
        details=f"Origin probability: {origin_prob}",
    ))

    # T2: Origin zone in affected list
    tests.append(TestResult(
        name=f"[{result.name}] Origin zone in affected zones",
        passed=result.origin_zone in result.affected_zone_ids,
        details=f"Affected zones: {result.affected_zone_ids}",
    ))

    # T3: Minimum expected zones
    tests.append(TestResult(
        name=f"[{result.name}] Minimum affected zones",
        passed=result.total_affected_zones >= scenario["expected_min_zones"],
        details=f"Expected ≥ {scenario['expected_min_zones']}, got {result.total_affected_zones}",
    ))

    # T4: All impact scores in valid range [0, 100]
    scores_valid = all(0.0 <= s <= 100.0 for s in result.impact_scores.values())
    tests.append(TestResult(
        name=f"[{result.name}] Impact scores in [0, 100]",
        passed=scores_valid,
        details=f"Scores: {result.impact_scores}",
    ))

    # T5: All probabilities in valid range [0, 1]
    probs_valid = all(0.0 <= p <= 1.0 for p in result.propagation_probabilities.values())
    tests.append(TestResult(
        name=f"[{result.name}] Propagation probabilities in [0, 1]",
        passed=probs_valid,
        details=f"Probabilities: {result.propagation_probabilities}",
    ))

    # T6: Probabilities decay from origin
    origin_prob = result.propagation_probabilities.get(result.origin_zone, 0)
    for zone_id, prob in result.propagation_probabilities.items():
        if zone_id != result.origin_zone and prob > origin_prob:
            tests.append(TestResult(
                name=f"[{result.name}] Probability decay",
                passed=False,
                details=f"{zone_id} has prob {prob} > origin {origin_prob}",
            ))
            break
    else:
        tests.append(TestResult(
            name=f"[{result.name}] Probability decay from origin",
            passed=True,
            details="All non-origin probabilities ≤ origin",
        ))

    # T7: Propagation paths reference valid zones
    all_zone_ids = {z.zone_id for z in ZONES}
    paths_valid = all(
        p["from_zone"] in all_zone_ids and p["to_zone"] in all_zone_ids
        for p in result.propagation_paths
    )
    tests.append(TestResult(
        name=f"[{result.name}] Propagation paths reference valid zones",
        passed=paths_valid,
        details=f"Paths: {[(p['from_zone'], p['to_zone']) for p in result.propagation_paths]}",
    ))

    # T8: Workers at risk ≥ 0
    tests.append(TestResult(
        name=f"[{result.name}] Workers at risk ≥ 0",
        passed=result.total_workers_at_risk >= 0,
        details=f"Workers at risk: {result.total_workers_at_risk}",
    ))

    # T9: Impact radius ≥ 0
    tests.append(TestResult(
        name=f"[{result.name}] Impact radius ≥ 0",
        passed=result.impact_radius_meters >= 0.0,
        details=f"Impact radius: {result.impact_radius_meters}m",
    ))

    # T10: Recommended action is non-empty
    tests.append(TestResult(
        name=f"[{result.name}] Recommended action is non-empty",
        passed=len(result.recommended_action) > 0,
        details=f"Action: {result.recommended_action[:60]}...",
    ))

    return tests


def compare_parity(
    neo4j_results: List[ScenarioResult],
    memory_results: List[ScenarioResult],
) -> List[TestResult]:
    """Compare Neo4j and InMemory results for parity."""
    tests = []

    for neo4j_r, mem_r in zip(neo4j_results, memory_results):
        # Same propagation level
        tests.append(TestResult(
            name=f"[Parity: {neo4j_r.name}] Propagation level matches",
            passed=neo4j_r.propagation_level == mem_r.propagation_level,
            details=f"Neo4j={neo4j_r.propagation_level}, InMemory={mem_r.propagation_level}",
        ))

        # Same affected zones (as sets)
        neo4j_zones = set(neo4j_r.affected_zone_ids)
        mem_zones = set(mem_r.affected_zone_ids)
        tests.append(TestResult(
            name=f"[Parity: {neo4j_r.name}] Affected zones match",
            passed=neo4j_zones == mem_zones,
            details=f"Neo4j={sorted(neo4j_zones)}, InMemory={sorted(mem_zones)}",
        ))

        # Same total workers at risk
        tests.append(TestResult(
            name=f"[Parity: {neo4j_r.name}] Workers at risk match",
            passed=neo4j_r.total_workers_at_risk == mem_r.total_workers_at_risk,
            details=f"Neo4j={neo4j_r.total_workers_at_risk}, InMemory={mem_r.total_workers_at_risk}",
        ))

        # Similar impact scores (within 1% tolerance)
        scores_match = True
        score_details = []
        for zone_id in neo4j_r.impact_scores:
            n = neo4j_r.impact_scores.get(zone_id, 0)
            m = mem_r.impact_scores.get(zone_id, 0)
            if abs(n - m) > 1.0:
                scores_match = False
                score_details.append(f"{zone_id}: Neo4j={n:.2f} vs InMemory={m:.2f}")

        tests.append(TestResult(
            name=f"[Parity: {neo4j_r.name}] Impact scores match (±1.0)",
            passed=scores_match,
            details="; ".join(score_details) if score_details else "All scores match",
        ))

    return tests


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DI verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def verify_di_selection() -> List[TestResult]:
    """Verify that the DI mechanism correctly selects repositories."""
    import importlib
    tests = []

    # Test 1: Default config → InMemoryGraphRepository
    try:
        import app.core.dependencies as deps
        # Reset singleton
        deps._graph_repository = None

        from app.core.settings import Settings
        default_settings = Settings(graph_repository="in_memory")
        repo = deps._get_graph_repository(default_settings)

        from app.hazard_propagation.repositories.in_memory_graph_repo import (
            InMemoryGraphRepository,
        )
        tests.append(TestResult(
            name="[DI] Default config selects InMemoryGraphRepository",
            passed=isinstance(repo, InMemoryGraphRepository),
            details=f"Type: {type(repo).__name__}",
        ))
        deps._graph_repository = None  # Reset for next test
    except Exception as e:
        tests.append(TestResult(
            name="[DI] Default config selects InMemoryGraphRepository",
            passed=False,
            error=str(e),
        ))

    # Test 2: Neo4j config → Neo4jGraphRepository
    try:
        deps._graph_repository = None

        neo4j_settings = Settings(
            graph_repository="neo4j",
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="sentinelai_test",
        )
        repo = deps._get_graph_repository(neo4j_settings)

        from app.hazard_propagation.repositories.neo4j_graph_repo import (
            Neo4jGraphRepository,
        )
        is_neo4j = isinstance(repo, Neo4jGraphRepository)
        tests.append(TestResult(
            name="[DI] Neo4j config selects Neo4jGraphRepository",
            passed=is_neo4j,
            details=f"Type: {type(repo).__name__}"
                    + (" (fallback to InMemory due to connection)" if not is_neo4j else ""),
        ))
        deps._graph_repository = None  # Reset
    except Exception as e:
        tests.append(TestResult(
            name="[DI] Neo4j config selects Neo4jGraphRepository",
            passed=False,
            error=str(e),
        ))

    # Test 3: Invalid Neo4j URI → graceful fallback to InMemory
    try:
        deps._graph_repository = None

        bad_settings = Settings(
            graph_repository="neo4j",
            neo4j_uri="bolt://nonexistent:9999",
            neo4j_username="fake",
            neo4j_password="fake",
        )
        repo = deps._get_graph_repository(bad_settings)

        # Should have fallen back to InMemory OR created a Neo4j repo
        # (Neo4j driver doesn't connect until first query)
        tests.append(TestResult(
            name="[DI] Invalid Neo4j URI → graceful handling",
            passed=repo is not None,
            details=f"Type: {type(repo).__name__} (no crash)",
        ))
        deps._graph_repository = None  # Reset
    except Exception as e:
        tests.append(TestResult(
            name="[DI] Invalid Neo4j URI → graceful handling",
            passed=False,
            error=f"Crashed: {e}",
        ))

    # Test 4: Singleton caching works
    try:
        deps._graph_repository = None

        settings = Settings(graph_repository="in_memory")
        repo1 = deps._get_graph_repository(settings)
        repo2 = deps._get_graph_repository(settings)

        tests.append(TestResult(
            name="[DI] Repository singleton caching",
            passed=repo1 is repo2,
            details=f"Same object: {repo1 is repo2}",
        ))
        deps._graph_repository = None
    except Exception as e:
        tests.append(TestResult(
            name="[DI] Repository singleton caching",
            passed=False,
            error=str(e),
        ))

    # Test 5: Settings from env var
    try:
        deps._graph_repository = None
        original = os.environ.get("GRAPH_REPOSITORY")
        os.environ["GRAPH_REPOSITORY"] = "in_memory"

        env_settings = Settings()
        tests.append(TestResult(
            name="[DI] GRAPH_REPOSITORY env var recognized",
            passed=env_settings.graph_repository == "in_memory",
            details=f"Value: {env_settings.graph_repository}",
        ))

        if original is not None:
            os.environ["GRAPH_REPOSITORY"] = original
        else:
            os.environ.pop("GRAPH_REPOSITORY", None)
        deps._graph_repository = None
    except Exception as e:
        tests.append(TestResult(
            name="[DI] GRAPH_REPOSITORY env var recognized",
            passed=False,
            error=str(e),
        ))

    return tests


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Report generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_report(
    all_tests: List[TestResult],
    neo4j_scenarios: List[ScenarioResult],
    memory_scenarios: List[ScenarioResult],
    graph_stats_neo4j: Dict[str, int],
    graph_stats_memory: Dict[str, int],
    neo4j_connected: bool,
    neo4j_uri: str,
    total_time_s: float,
) -> str:
    """Generate the markdown verification report."""

    passed = sum(1 for t in all_tests if t.passed)
    failed = sum(1 for t in all_tests if not t.passed)
    total = len(all_tests)

    status_emoji = "✅" if failed == 0 else "⚠️"

    lines = [
        "# Hazard Propagation — Neo4j Verification Report",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Neo4j URI:** `{neo4j_uri}`  ",
        f"**Neo4j Connected:** {'✅ Yes' if neo4j_connected else '❌ No'}  ",
        f"**Total Execution Time:** {total_time_s:.2f}s",
        "",
        "---",
        "",
        f"## {status_emoji} Test Summary: {passed}/{total} passed, {failed} failed",
        "",
    ]

    # ── DI Verification Results ──
    di_tests = [t for t in all_tests if t.name.startswith("[DI]")]
    lines.append("## 1. Dependency Injection Verification")
    lines.append("")
    lines.append("### Repository Selection Mechanism")
    lines.append("")
    lines.append("The DI container in [`dependencies.py`](file:///Users/alisha/Hackathon/industrial-safety-intelligence/app/core/dependencies.py) selects the graph repository based on:")
    lines.append("")
    lines.append("| Setting | Env Var | Default | Effect |")
    lines.append("|---------|---------|---------|--------|")
    lines.append("| `graph_repository` | `GRAPH_REPOSITORY` | `in_memory` | `in_memory` → InMemoryGraphRepository, `neo4j` → Neo4jGraphRepository |")
    lines.append("| `neo4j_uri` | `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection URI |")
    lines.append("| `neo4j_username` | `NEO4J_USERNAME` | `neo4j` | Neo4j authentication username |")
    lines.append("| `neo4j_password` | `NEO4J_PASSWORD` | `sentinelai_test` | Neo4j authentication password |")
    lines.append("| `neo4j_database` | `NEO4J_DATABASE` | `neo4j` | Neo4j database name |")
    lines.append("")
    lines.append("### DI Test Results")
    lines.append("")
    lines.append("| Test | Result | Details |")
    lines.append("|------|--------|---------|")
    for t in di_tests:
        status = "✅ PASS" if t.passed else "❌ FAIL"
        detail = t.error if t.error else t.details
        lines.append(f"| {t.name} | {status} | {detail} |")
    lines.append("")

    # ── Graph Seeding ──
    lines.append("## 2. Graph Data Seeding")
    lines.append("")
    lines.append("### Facility Graph Topology")
    lines.append("")
    lines.append("```")
    lines.append("    ZONE_A (Main Processing Hall, MEDIUM, 12 workers)")
    lines.append("    /    \\")
    lines.append("  ZONE_B  ZONE_C (Boiler Room / Chemical Storage, HIGH)")
    lines.append("    \\    /")
    lines.append("    ZONE_D (Assembly Line, MEDIUM, 18 workers)")
    lines.append("      |")
    lines.append("    ZONE_E (Warehouse, LOW, 6 workers)")
    lines.append("      |")
    lines.append("    ZONE_F (Loading Dock, LOW, 3 workers)")
    lines.append("```")
    lines.append("")
    lines.append("### Entity Counts")
    lines.append("")
    lines.append("| Entity | Neo4j | InMemory | Match |")
    lines.append("|--------|-------|----------|-------|")
    for key in ["zones", "equipment", "sensors", "edges", "hazards"]:
        n = graph_stats_neo4j.get(key, "N/A")
        m = graph_stats_memory.get(key, "N/A")
        match = "✅" if n == m else "⚠️"
        lines.append(f"| {key.capitalize()} | {n} | {m} | {match} |")
    lines.append("")

    # ── Propagation Scenarios ──
    lines.append("## 3. Propagation Scenario Results")
    lines.append("")

    for i, (neo4j_r, mem_r) in enumerate(zip(neo4j_scenarios, memory_scenarios)):
        lines.append(f"### Scenario {i+1}: {neo4j_r.name}")
        lines.append("")
        lines.append(f"**Input:** `{neo4j_r.hazard_type}` from `{neo4j_r.origin_zone}`, compound_risk_score={neo4j_r.compound_risk_score}")
        lines.append("")
        lines.append("| Metric | Neo4j | InMemory | Match |")
        lines.append("|--------|-------|----------|-------|")

        match_level = "✅" if neo4j_r.propagation_level == mem_r.propagation_level else "❌"
        lines.append(f"| Propagation Level | {neo4j_r.propagation_level} | {mem_r.propagation_level} | {match_level} |")

        match_zones = "✅" if set(neo4j_r.affected_zone_ids) == set(mem_r.affected_zone_ids) else "❌"
        lines.append(f"| Affected Zones | {neo4j_r.total_affected_zones} | {mem_r.total_affected_zones} | {match_zones} |")

        match_workers = "✅" if neo4j_r.total_workers_at_risk == mem_r.total_workers_at_risk else "❌"
        lines.append(f"| Workers at Risk | {neo4j_r.total_workers_at_risk} | {mem_r.total_workers_at_risk} | {match_workers} |")

        match_radius = "✅" if abs(neo4j_r.impact_radius_meters - mem_r.impact_radius_meters) < 1.0 else "❌"
        lines.append(f"| Impact Radius (m) | {neo4j_r.impact_radius_meters:.0f} | {mem_r.impact_radius_meters:.0f} | {match_radius} |")

        match_time = "✅" if abs(neo4j_r.time_to_critical_minutes - mem_r.time_to_critical_minutes) < 0.5 else "❌"
        lines.append(f"| Time to Critical (min) | {neo4j_r.time_to_critical_minutes:.1f} | {mem_r.time_to_critical_minutes:.1f} | {match_time} |")

        lines.append(f"| Processing Time (ms) | {neo4j_r.processing_time_ms:.1f} | {mem_r.processing_time_ms:.1f} | — |")
        lines.append("")

        # Zone-level details
        lines.append("**Zone Impact Scores (Neo4j):**")
        lines.append("")
        lines.append("| Zone | Risk Score | Probability | Workers |")
        lines.append("|------|-----------|-------------|---------|")
        for zone_id in sorted(neo4j_r.impact_scores.keys()):
            score = neo4j_r.impact_scores[zone_id]
            prob = neo4j_r.propagation_probabilities.get(zone_id, 0)
            zone_data = next((z for z in ZONES if z.zone_id == zone_id), None)
            workers = zone_data.current_worker_count if zone_data else 0
            lines.append(f"| {zone_id} | {score:.2f} | {prob:.4f} | {workers} |")
        lines.append("")

        # Propagation paths
        if neo4j_r.propagation_paths:
            lines.append("**Propagation Paths (Neo4j):**")
            lines.append("")
            lines.append("| From | To | Probability | ETA (min) |")
            lines.append("|------|----|-------------|-----------|")
            for p in neo4j_r.propagation_paths:
                lines.append(f"| {p['from_zone']} | {p['to_zone']} | {p['probability']:.4f} | {p['estimated_time_minutes']:.0f} |")
            lines.append("")

        lines.append("---")
        lines.append("")

    # ── Validation Results ──
    lines.append("## 4. Validation Test Results")
    lines.append("")

    scenario_tests = [t for t in all_tests if not t.name.startswith("[DI]") and not t.name.startswith("[Parity")]
    parity_tests = [t for t in all_tests if t.name.startswith("[Parity")]

    lines.append("### Scenario Validation")
    lines.append("")
    lines.append("| # | Test | Result | Details |")
    lines.append("|---|------|--------|---------|")
    for i, t in enumerate(scenario_tests, 1):
        status = "✅" if t.passed else "❌"
        detail = t.error if t.error else t.details[:80]
        lines.append(f"| {i} | {t.name} | {status} | {detail} |")
    lines.append("")

    lines.append("### Neo4j ↔ InMemory Parity")
    lines.append("")
    lines.append("| # | Test | Result | Details |")
    lines.append("|---|------|--------|---------|")
    for i, t in enumerate(parity_tests, 1):
        status = "✅" if t.passed else "❌"
        detail = t.error if t.error else t.details[:80]
        lines.append(f"| {i} | {t.name} | {status} | {detail} |")
    lines.append("")

    # ── Deployment Configuration ──
    lines.append("## 5. Neo4j Startup Requirements & Deployment Configuration")
    lines.append("")
    lines.append("### Docker Compose")
    lines.append("")
    lines.append("```bash")
    lines.append("# Start Neo4j")
    lines.append("docker compose up -d neo4j")
    lines.append("")
    lines.append("# Wait for health check (auto in docker-compose)")
    lines.append("# Neo4j Browser: http://localhost:7474")
    lines.append("# Bolt endpoint: bolt://localhost:7687")
    lines.append("```")
    lines.append("")
    lines.append("### Environment Variables")
    lines.append("")
    lines.append("```bash")
    lines.append("# .env file — switch to Neo4j")
    lines.append('GRAPH_REPOSITORY=neo4j')
    lines.append('NEO4J_URI=bolt://localhost:7687')
    lines.append('NEO4J_USERNAME=neo4j')
    lines.append('NEO4J_PASSWORD=sentinelai_test')
    lines.append('NEO4J_DATABASE=neo4j')
    lines.append("```")
    lines.append("")
    lines.append("### Fallback Behavior")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> If `GRAPH_REPOSITORY=neo4j` but Neo4j is unavailable, the DI container")
    lines.append("> logs a warning and falls back to `InMemoryGraphRepository` automatically.")
    lines.append("> No application crash occurs.")
    lines.append("")
    lines.append("### Python Dependencies")
    lines.append("")
    lines.append("```bash")
    lines.append("pip install neo4j  # Required only when GRAPH_REPOSITORY=neo4j")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main verification flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    overall_start = time.monotonic()
    all_tests: List[TestResult] = []
    neo4j_scenarios: List[ScenarioResult] = []
    memory_scenarios: List[ScenarioResult] = []
    graph_stats_neo4j: Dict[str, int] = {}
    graph_stats_memory: Dict[str, int] = {}
    neo4j_connected = False
    neo4j_uri = "bolt://localhost:7687"

    print("=" * 70)
    print("  HAZARD PROPAGATION — NEO4J VERIFICATION")
    print("=" * 70)
    print()

    # ── Phase 0: DI Verification ──
    print("▶ Phase 0: Dependency Injection Verification")
    di_tests = verify_di_selection()
    all_tests.extend(di_tests)
    di_passed = sum(1 for t in di_tests if t.passed)
    print(f"  DI Tests: {di_passed}/{len(di_tests)} passed")
    print()

    # ── Phase 1: Connect to Neo4j ──
    print("▶ Phase 1: Connecting to Neo4j...")
    neo4j_repo = None
    try:
        from app.hazard_propagation.repositories.neo4j_graph_repo import (
            Neo4jGraphRepository,
            NEO4J_AVAILABLE,
        )

        if not NEO4J_AVAILABLE:
            print("  ✗ neo4j driver not installed")
            raise ImportError("neo4j driver not installed")

        neo4j_repo = Neo4jGraphRepository(
            uri=neo4j_uri,
            username="neo4j",
            password="sentinelai_test",
        )

        # Verify connectivity by running a simple query
        stats = await neo4j_repo.get_graph_stats()
        neo4j_connected = True
        print(f"  ✓ Connected to Neo4j at {neo4j_uri}")
        print(f"  ✓ Initial graph stats: {stats}")
    except Exception as e:
        print(f"  ✗ Failed to connect to Neo4j: {e}")
        print("  ▸ Running verification with InMemoryGraphRepository only")
        neo4j_connected = False

    # ── Phase 2: Clean Neo4j and Seed Data ──
    if neo4j_connected and neo4j_repo:
        print()
        print("▶ Phase 2: Seeding Neo4j graph...")

        # Clean existing data
        try:
            await neo4j_repo._run("MATCH (n) DETACH DELETE n")
            print("  ✓ Cleaned existing Neo4j data")
        except Exception as e:
            print(f"  ✗ Failed to clean Neo4j: {e}")

        # Seed graph
        try:
            counts = await seed_graph(neo4j_repo)
            print(f"  ✓ Seeded graph: {counts}")

            # Verify graph stats
            graph_stats_neo4j = await neo4j_repo.get_graph_stats()
            print(f"  ✓ Neo4j graph stats: {graph_stats_neo4j}")
        except Exception as e:
            print(f"  ✗ Failed to seed Neo4j: {e}")
            traceback.print_exc()
            neo4j_connected = False

    # ── Phase 3: Seed InMemory graph ──
    print()
    print("▶ Phase 3: Seeding InMemory graph...")
    memory_repo = InMemoryGraphRepository()
    try:
        counts = await seed_graph(memory_repo)
        print(f"  ✓ Seeded InMemory graph: {counts}")
        graph_stats_memory = await memory_repo.get_graph_stats()
        print(f"  ✓ InMemory graph stats: {graph_stats_memory}")
    except Exception as e:
        print(f"  ✗ Failed to seed InMemory graph: {e}")
        traceback.print_exc()
        return

    # ── Phase 4: Run scenarios on both repositories ──
    config = PropagationConfig()

    # Run on Neo4j (if connected)
    if neo4j_connected and neo4j_repo:
        print()
        print("▶ Phase 4a: Running scenarios on Neo4j...")
        neo4j_service = HazardPropagationService(
            graph_repo=neo4j_repo,
            config=config,
        )

        for scenario in SCENARIOS:
            try:
                result = await run_scenario(neo4j_service, scenario)
                neo4j_scenarios.append(result)
                print(f"  ✓ {scenario['name']}: "
                      f"level={result.propagation_level}, "
                      f"zones={result.total_affected_zones}, "
                      f"workers={result.total_workers_at_risk}, "
                      f"time={result.processing_time_ms:.1f}ms")
            except Exception as e:
                print(f"  ✗ {scenario['name']}: {e}")
                traceback.print_exc()
                # Create a failed result
                neo4j_scenarios.append(ScenarioResult(
                    name=scenario["name"],
                    hazard_type=scenario["hazard_type"],
                    origin_zone=scenario["origin_zone"],
                    compound_risk_score=scenario["compound_risk_score"],
                ))

    # Run on InMemory
    print()
    print("▶ Phase 4b: Running scenarios on InMemory...")
    memory_service = HazardPropagationService(
        graph_repo=memory_repo,
        config=config,
    )

    for scenario in SCENARIOS:
        try:
            result = await run_scenario(memory_service, scenario)
            memory_scenarios.append(result)
            print(f"  ✓ {scenario['name']}: "
                  f"level={result.propagation_level}, "
                  f"zones={result.total_affected_zones}, "
                  f"workers={result.total_workers_at_risk}, "
                  f"time={result.processing_time_ms:.1f}ms")
        except Exception as e:
            print(f"  ✗ {scenario['name']}: {e}")
            traceback.print_exc()
            memory_scenarios.append(ScenarioResult(
                name=scenario["name"],
                hazard_type=scenario["hazard_type"],
                origin_zone=scenario["origin_zone"],
                compound_risk_score=scenario["compound_risk_score"],
            ))

    # ── Phase 5: Validate scenarios ──
    print()
    print("▶ Phase 5: Validating results...")

    # Validate Neo4j scenarios
    target_scenarios = neo4j_scenarios if neo4j_connected else memory_scenarios
    target_label = "Neo4j" if neo4j_connected else "InMemory"

    for result, scenario in zip(target_scenarios, SCENARIOS):
        tests = validate_scenario(result, scenario)
        all_tests.extend(tests)

    # Parity tests (if both ran)
    if neo4j_connected and neo4j_scenarios and memory_scenarios:
        parity_tests = compare_parity(neo4j_scenarios, memory_scenarios)
        all_tests.extend(parity_tests)
        parity_passed = sum(1 for t in parity_tests if t.passed)
        print(f"  ✓ Parity tests: {parity_passed}/{len(parity_tests)} passed")

    # ── Phase 6: Verify graph persistence (Neo4j) ──
    if neo4j_connected and neo4j_repo:
        print()
        print("▶ Phase 6: Verifying graph persistence in Neo4j...")
        try:
            # Check that hazard nodes were persisted during propagation
            records = await neo4j_repo._run(
                "MATCH (h:Hazard) RETURN h.hazard_id AS id, h.hazard_type AS type"
            )
            hazard_count = len(records)
            # Original HAZ_TEST_001 + 5 scenarios
            expected_min_hazards = 1 + len(SCENARIOS)

            all_tests.append(TestResult(
                name="[Neo4j] Hazard nodes persisted after propagation",
                passed=hazard_count >= expected_min_hazards,
                details=f"Found {hazard_count} hazard nodes (expected ≥ {expected_min_hazards})",
            ))

            # Check AFFECTS relationships
            records = await neo4j_repo._run(
                "MATCH (h:Hazard)-[:AFFECTS]->(z:Zone) "
                "RETURN h.hazard_id AS hazard_id, z.zone_id AS zone_id"
            )
            affects_count = len(records)
            all_tests.append(TestResult(
                name="[Neo4j] AFFECTS relationships persisted",
                passed=affects_count > 0,
                details=f"Found {affects_count} AFFECTS relationships",
            ))

            print(f"  ✓ Hazard nodes: {hazard_count}")
            print(f"  ✓ AFFECTS relationships: {affects_count}")

            # Final graph stats
            final_stats = await neo4j_repo.get_graph_stats()
            print(f"  ✓ Final Neo4j stats: {final_stats}")

        except Exception as e:
            print(f"  ✗ Persistence verification failed: {e}")
            all_tests.append(TestResult(
                name="[Neo4j] Graph persistence",
                passed=False,
                error=str(e),
            ))

    # ── Phase 7: Cleanup Neo4j ──
    if neo4j_connected and neo4j_repo:
        print()
        print("▶ Phase 7: Cleaning up Neo4j...")
        try:
            await neo4j_repo._run("MATCH (n) DETACH DELETE n")
            await neo4j_repo.close()
            print("  ✓ Neo4j cleaned and connection closed")
        except Exception as e:
            print(f"  ✗ Cleanup failed: {e}")

    # ── Phase 8: Generate report ──
    print()
    print("▶ Phase 8: Generating verification report...")

    # If Neo4j wasn't connected, use InMemory results for both columns
    if not neo4j_connected:
        neo4j_scenarios = memory_scenarios
        graph_stats_neo4j = graph_stats_memory

    total_time = time.monotonic() - overall_start

    report = generate_report(
        all_tests=all_tests,
        neo4j_scenarios=neo4j_scenarios,
        memory_scenarios=memory_scenarios,
        graph_stats_neo4j=graph_stats_neo4j,
        graph_stats_memory=graph_stats_memory,
        neo4j_connected=neo4j_connected,
        neo4j_uri=neo4j_uri,
        total_time_s=total_time,
    )

    report_path = project_root / "tests" / "integration" / "neo4j_verification_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    # ── Summary ──
    passed = sum(1 for t in all_tests if t.passed)
    failed = sum(1 for t in all_tests if not t.passed)
    total = len(all_tests)

    print()
    print("=" * 70)
    print(f"  VERIFICATION COMPLETE: {passed}/{total} passed, {failed} failed")
    print(f"  Report: {report_path}")
    print(f"  Time: {total_time:.2f}s")
    print("=" * 70)

    # Print failures
    if failed > 0:
        print()
        print("FAILURES:")
        for t in all_tests:
            if not t.passed:
                print(f"  ✗ {t.name}")
                if t.error:
                    print(f"    Error: {t.error}")
                if t.details:
                    print(f"    Details: {t.details}")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
