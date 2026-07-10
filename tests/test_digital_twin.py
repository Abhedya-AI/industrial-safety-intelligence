"""Comprehensive tests for the Digital Twin module.

Covers:
  - TwinStateManager: state updates, facility health, heatmap, queries
  - DigitalTwinEventHandler: event routing, dedup, error handling
  - DigitalTwinConsumerSetup: topic registration
  - REST API endpoints: /twin/facility, /twin/zones, /twin/zones/{id}, /twin/heatmap
  - Domain entities: enums, exceptions, computed properties
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.digital_twin.domain.entities import (
    ActiveHazard,
    EquipmentState,
    FacilityState,
    SensorReading,
    ZoneState,
)
from app.digital_twin.domain.enums import HeatmapColor, RiskLevel, TwinUpdateType
from app.digital_twin.domain.exceptions import (
    TwinNotInitializedError,
    TwinStateError,
    ZoneNotFoundInTwinError,
)
from app.digital_twin.messaging.consumer import (
    DIGITAL_TWIN_SUBSCRIBED_TOPICS,
    DigitalTwinConsumerSetup,
)
from app.digital_twin.messaging.handler import DigitalTwinEventHandler
from app.digital_twin.services.twin_state_manager import TwinStateManager
from app.hazard_propagation.graph.entities import (
    EquipmentNode,
    SensorNode,
    ZoneNode,
)
from app.hazard_propagation.repositories.in_memory_graph_repo import (
    InMemoryGraphRepository,
)
from app.shared.messaging.topics import KafkaTopics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def graph_repo() -> InMemoryGraphRepository:
    """Provide a fresh in-memory graph repository."""
    return InMemoryGraphRepository()


@pytest.fixture
def populated_graph_repo(graph_repo: InMemoryGraphRepository):
    """Graph repo pre-loaded with facility topology."""

    async def _populate():
        zone_a = ZoneNode(
            zone_id="ZONE_A",
            zone_name="Production Area 1",
            worker_capacity=20,
            current_worker_count=12,
        )
        zone_b = ZoneNode(
            zone_id="ZONE_B",
            zone_name="Production Area 2",
            worker_capacity=15,
            current_worker_count=8,
        )
        await graph_repo.create_zone(zone_a)
        await graph_repo.create_zone(zone_b)
        await graph_repo.create_connection("ZONE_A", "ZONE_B")

        eq = EquipmentNode(
            equipment_id="EQ001",
            equipment_type="boiler",
            operational_status="ACTIVE",
            health_score=85.0,
        )
        await graph_repo.create_equipment("ZONE_A", eq)

        sensor = SensorNode(
            sensor_id="S001",
            sensor_type="temperature",
            unit_of_measurement="celsius",
        )
        await graph_repo.create_sensor("EQ001", sensor)

    asyncio.get_event_loop().run_until_complete(_populate())
    return graph_repo


@pytest.fixture
def twin_manager(graph_repo: InMemoryGraphRepository) -> TwinStateManager:
    """Provide a TwinStateManager with an empty graph."""
    manager = TwinStateManager(graph_repo=graph_repo)
    asyncio.get_event_loop().run_until_complete(manager.initialize())
    return manager


@pytest.fixture
def populated_twin(
    populated_graph_repo: InMemoryGraphRepository,
) -> TwinStateManager:
    """Provide a TwinStateManager pre-loaded from graph topology."""
    manager = TwinStateManager(graph_repo=populated_graph_repo)
    asyncio.get_event_loop().run_until_complete(manager.initialize())
    return manager


@pytest.fixture
def handler(twin_manager: TwinStateManager) -> DigitalTwinEventHandler:
    """Provide a handler wired to an empty twin."""
    return DigitalTwinEventHandler(state_manager=twin_manager)


@pytest.fixture
def populated_handler(
    populated_twin: TwinStateManager,
) -> DigitalTwinEventHandler:
    """Provide a handler wired to a pre-populated twin."""
    return DigitalTwinEventHandler(state_manager=populated_twin)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Domain Entity Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEnums:
    """Test domain enumerations."""

    def test_heatmap_color_green(self):
        assert HeatmapColor.from_score(0) == HeatmapColor.GREEN
        assert HeatmapColor.from_score(25) == HeatmapColor.GREEN

    def test_heatmap_color_yellow(self):
        assert HeatmapColor.from_score(26) == HeatmapColor.YELLOW
        assert HeatmapColor.from_score(50) == HeatmapColor.YELLOW

    def test_heatmap_color_orange(self):
        assert HeatmapColor.from_score(51) == HeatmapColor.ORANGE
        assert HeatmapColor.from_score(75) == HeatmapColor.ORANGE

    def test_heatmap_color_red(self):
        assert HeatmapColor.from_score(76) == HeatmapColor.RED
        assert HeatmapColor.from_score(100) == HeatmapColor.RED

    def test_risk_level_values(self):
        assert RiskLevel.LOW.value == "LOW"
        assert RiskLevel.CRITICAL.value == "CRITICAL"

    def test_twin_update_type_values(self):
        assert TwinUpdateType.SENSOR_READING.value == "SENSOR_READING"
        assert TwinUpdateType.HAZARD_PROPAGATED.value == "HAZARD_PROPAGATED"


class TestExceptions:
    """Test domain exceptions."""

    def test_zone_not_found(self):
        exc = ZoneNotFoundInTwinError("ZONE_X")
        assert exc.zone_id == "ZONE_X"
        assert "ZONE_X" in str(exc)

    def test_twin_not_initialized(self):
        exc = TwinNotInitializedError()
        assert "not initialized" in str(exc)

    def test_inheritance(self):
        assert issubclass(ZoneNotFoundInTwinError, TwinStateError)
        assert issubclass(TwinNotInitializedError, TwinStateError)


class TestZoneState:
    """Test ZoneState computed properties."""

    def test_active_hazard_count(self):
        zone = ZoneState(zone_id="Z1")
        assert zone.active_hazard_count == 0
        zone.active_hazards.append(
            ActiveHazard(hazard_id="H1", hazard_type="GAS_LEAK")
        )
        assert zone.active_hazard_count == 1

    def test_is_critical_by_risk_level(self):
        zone = ZoneState(zone_id="Z1", risk_level="CRITICAL")
        assert zone.is_critical is True

    def test_is_critical_by_compound_risk(self):
        zone = ZoneState(zone_id="Z1", compound_risk_level="CRITICAL")
        assert zone.is_critical is True

    def test_is_critical_by_hazard(self):
        zone = ZoneState(zone_id="Z1")
        zone.active_hazards.append(
            ActiveHazard(hazard_id="H1", hazard_type="FIRE")
        )
        assert zone.is_critical is True

    def test_not_critical(self):
        zone = ZoneState(zone_id="Z1")
        assert zone.is_critical is False

    def test_overall_risk_score(self):
        zone = ZoneState(zone_id="Z1", predicted_risk_score=50.0)
        assert zone.overall_risk_score == 50.0

    def test_overall_risk_score_with_compound(self):
        zone = ZoneState(
            zone_id="Z1",
            predicted_risk_score=50.0,
            compound_risk_score=80.0,
        )
        assert zone.overall_risk_score == 80.0

    def test_heatmap_color_property(self):
        zone = ZoneState(zone_id="Z1", predicted_risk_score=80.0)
        assert zone.heatmap_color == "red"

    def test_touch_updates_metadata(self):
        zone = ZoneState(zone_id="Z1")
        initial_count = zone.event_count
        zone.touch()
        assert zone.event_count == initial_count + 1
        assert zone.last_updated != ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TwinStateManager Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTwinStateManagerInit:
    """Test initialization and graph loading."""

    def test_empty_init(self, twin_manager: TwinStateManager):
        assert twin_manager.is_initialized is True
        assert twin_manager.zone_count == 0

    def test_populated_init(self, populated_twin: TwinStateManager):
        assert populated_twin.is_initialized is True
        assert populated_twin.zone_count == 2

    def test_populated_zones_have_topology(
        self, populated_twin: TwinStateManager,
    ):
        zone_a = populated_twin.get_zone("ZONE_A")
        assert zone_a.zone_name == "Production Area 1"
        assert "ZONE_B" in zone_a.connected_zones
        assert len(zone_a.equipment) == 1
        assert zone_a.equipment[0].equipment_id == "EQ001"

    def test_populated_zones_have_sensors(
        self, populated_twin: TwinStateManager,
    ):
        zone_a = populated_twin.get_zone("ZONE_A")
        assert "S001" in zone_a.latest_sensor_readings
        assert zone_a.sensor_count == 1

    def test_zone_not_found(self, twin_manager: TwinStateManager):
        with pytest.raises(ZoneNotFoundInTwinError):
            twin_manager.get_zone("NONEXISTENT")


class TestSensorUpdates:
    """Test sensor event processing."""

    def test_sensor_anomaly(self, twin_manager: TwinStateManager):
        twin_manager.update_sensor_anomaly(
            zone_id="ZONE_A",
            sensor_id="S001",
            sensor_type="gas",
            value=150.0,
            unit="ppm",
            anomaly_score=-0.85,
        )
        zone = twin_manager.get_zone("ZONE_A")
        assert zone.anomaly_count == 1
        assert zone.sensor_count == 1
        reading = zone.latest_sensor_readings["S001"]
        assert reading.value == 150.0
        assert reading.is_anomalous is True
        assert reading.anomaly_score == -0.85

    def test_sensor_status(self, twin_manager: TwinStateManager):
        twin_manager.update_sensor_status(
            zone_id="ZONE_A", sensor_id="S001", status="FAULTY",
        )
        zone = twin_manager.get_zone("ZONE_A")
        assert zone.latest_sensor_readings["S001"].status == "FAULTY"

    def test_sensor_health(self, twin_manager: TwinStateManager):
        twin_manager.update_sensor_health(
            zone_id="ZONE_A", sensor_id="S001", health_score=75.0,
        )
        twin_manager.update_sensor_health(
            zone_id="ZONE_A", sensor_id="S002", health_score=85.0,
        )
        zone = twin_manager.get_zone("ZONE_A")
        assert zone.sensor_health == 80.0  # (75 + 85) / 2

    def test_multiple_anomalies_counted(
        self, twin_manager: TwinStateManager,
    ):
        twin_manager.update_sensor_anomaly(
            zone_id="ZONE_A", sensor_id="S001", value=100,
        )
        twin_manager.update_sensor_anomaly(
            zone_id="ZONE_A", sensor_id="S002", value=200,
        )
        zone = twin_manager.get_zone("ZONE_A")
        assert zone.anomaly_count == 2


class TestRiskUpdates:
    """Test risk prediction event processing."""

    def test_risk_assessment(self, twin_manager: TwinStateManager):
        twin_manager.update_risk_assessment(
            zone_id="ZONE_A",
            risk_score=72.0,
            risk_level="HIGH",
            accident_probability=0.35,
        )
        zone = twin_manager.get_zone("ZONE_A")
        assert zone.predicted_risk_score == 72.0
        assert zone.risk_level == "HIGH"
        assert zone.accident_probability == 0.35

    def test_risk_score_update(self, twin_manager: TwinStateManager):
        twin_manager.update_risk_score(
            zone_id="ZONE_A", risk_score=55.0,
        )
        zone = twin_manager.get_zone("ZONE_A")
        assert zone.predicted_risk_score == 55.0
        assert zone.risk_level == "HIGH"  # 51-75 = HIGH

    def test_risk_threshold_exceeded(
        self, twin_manager: TwinStateManager,
    ):
        twin_manager.update_risk_threshold_exceeded(
            zone_id="ZONE_A",
            threshold_type="gas_level",
            current_value=150.0,
        )
        zone = twin_manager.get_zone("ZONE_A")
        assert zone.risk_level == "CRITICAL"
        assert zone.predicted_risk_score >= 76.0

    def test_risk_level_auto_calculated(
        self, twin_manager: TwinStateManager,
    ):
        twin_manager.update_risk_score(
            zone_id="ZONE_A", risk_score=10.0,
        )
        assert twin_manager.get_zone("ZONE_A").risk_level == "LOW"

        twin_manager.update_risk_score(
            zone_id="ZONE_A", risk_score=30.0,
        )
        assert twin_manager.get_zone("ZONE_A").risk_level == "MEDIUM"


class TestCompoundRiskUpdates:
    """Test compound risk event processing."""

    def test_compound_risk_update(self, twin_manager: TwinStateManager):
        twin_manager.update_compound_risk(
            zone_id="ZONE_A",
            compound_risk_score=85.0,
            risk_level="CRITICAL",
            confidence_score=0.92,
            contributing_factors={"gas_risk": 0.7, "temp_risk": 0.3},
        )
        zone = twin_manager.get_zone("ZONE_A")
        assert zone.compound_risk_score == 85.0
        assert zone.compound_risk_level == "CRITICAL"
        assert zone.compound_risk_confidence == 0.92
        assert zone.contributing_factors["gas_risk"] == 0.7

    def test_workers_at_risk_high_compound(
        self, twin_manager: TwinStateManager,
    ):
        # First set worker count by initializing via sensor event
        twin_manager._zones["ZONE_A"] = ZoneState(
            zone_id="ZONE_A", current_worker_count=10,
        )
        twin_manager.update_compound_risk(
            zone_id="ZONE_A", compound_risk_score=75.0,
        )
        zone = twin_manager.get_zone("ZONE_A")
        assert zone.workers_at_risk == 10

    def test_workers_not_at_risk_low_compound(
        self, twin_manager: TwinStateManager,
    ):
        twin_manager._zones["ZONE_A"] = ZoneState(
            zone_id="ZONE_A", current_worker_count=10,
        )
        twin_manager.update_compound_risk(
            zone_id="ZONE_A", compound_risk_score=20.0,
        )
        zone = twin_manager.get_zone("ZONE_A")
        assert zone.workers_at_risk == 0


class TestHazardUpdates:
    """Test hazard propagation event processing."""

    def test_hazard_detected(self, twin_manager: TwinStateManager):
        twin_manager.update_hazard_detected(
            zone_id="ZONE_A",
            hazard_id="HAZ-001",
            hazard_type="GAS_LEAK",
            severity="HIGH",
        )
        zone = twin_manager.get_zone("ZONE_A")
        assert zone.active_hazard_count == 1
        assert zone.active_hazards[0].hazard_type == "GAS_LEAK"

    def test_no_duplicate_hazards(self, twin_manager: TwinStateManager):
        for _ in range(3):
            twin_manager.update_hazard_detected(
                zone_id="ZONE_A",
                hazard_id="HAZ-001",
                hazard_type="GAS_LEAK",
            )
        zone = twin_manager.get_zone("ZONE_A")
        assert zone.active_hazard_count == 1

    def test_hazard_propagated(self, twin_manager: TwinStateManager):
        twin_manager.update_hazard_propagated(
            origin_zone="ZONE_A",
            hazard_type="GAS_LEAK",
            propagation_level="SPREADING",
            affected_zones=["ZONE_A", "ZONE_B", "ZONE_C"],
            propagation_id="PROP-001",
        )
        zone_a = twin_manager.get_zone("ZONE_A")
        assert zone_a.active_hazard_count == 1
        assert zone_a.affected_neighbors == ["ZONE_A", "ZONE_B", "ZONE_C"]

        zone_b = twin_manager.get_zone("ZONE_B")
        assert zone_b.active_hazard_count == 1

        zone_c = twin_manager.get_zone("ZONE_C")
        assert zone_c.active_hazard_count == 1


class TestFacilityHealth:
    """Test facility health score calculation."""

    def test_empty_facility_health(self, twin_manager: TwinStateManager):
        state = twin_manager.get_facility_state()
        assert state.facility_health == 100.0

    def test_healthy_facility(self, populated_twin: TwinStateManager):
        state = populated_twin.get_facility_state()
        # No hazards, no risk → should be high health
        assert state.facility_health >= 90.0

    def test_degraded_facility_with_risk(
        self, populated_twin: TwinStateManager,
    ):
        populated_twin.update_risk_score(
            zone_id="ZONE_A", risk_score=80.0,
        )
        populated_twin.update_risk_score(
            zone_id="ZONE_B", risk_score=60.0,
        )
        state = populated_twin.get_facility_state()
        # avg_risk = 70, so risk_component = 100-70 = 30
        # This should bring facility health down
        assert state.facility_health < 90.0

    def test_facility_with_hazards(
        self, populated_twin: TwinStateManager,
    ):
        populated_twin.update_hazard_detected(
            zone_id="ZONE_A",
            hazard_id="HAZ-001",
            hazard_type="FIRE",
        )
        state = populated_twin.get_facility_state()
        assert state.active_hazards == 1
        assert state.facility_health < 100.0

    def test_facility_health_bounded(
        self, populated_twin: TwinStateManager,
    ):
        # Push everything to worst case
        populated_twin.update_risk_score(
            zone_id="ZONE_A", risk_score=100.0,
        )
        populated_twin.update_risk_score(
            zone_id="ZONE_B", risk_score=100.0,
        )
        populated_twin.update_hazard_detected(
            zone_id="ZONE_A", hazard_id="H1", hazard_type="FIRE",
        )
        populated_twin.update_hazard_detected(
            zone_id="ZONE_B", hazard_id="H2", hazard_type="GAS_LEAK",
        )
        state = populated_twin.get_facility_state()
        assert 0 <= state.facility_health <= 100

    def test_facility_state_aggregation(
        self, populated_twin: TwinStateManager,
    ):
        populated_twin.update_sensor_anomaly(
            zone_id="ZONE_A", sensor_id="S001", value=100,
        )
        populated_twin.update_risk_score(
            zone_id="ZONE_A", risk_score=65.0,
        )
        state = populated_twin.get_facility_state()
        assert state.total_zones == 2
        assert state.total_anomalies == 1
        assert state.zone_ids == ["ZONE_A", "ZONE_B"]
        assert state.events_processed >= 2


class TestHeatmap:
    """Test heatmap generation."""

    def test_empty_heatmap(self, twin_manager: TwinStateManager):
        heatmap = twin_manager.get_heatmap()
        assert heatmap == []

    def test_heatmap_colors(self, populated_twin: TwinStateManager):
        populated_twin.update_risk_score(
            zone_id="ZONE_A", risk_score=80.0,
        )
        populated_twin.update_risk_score(
            zone_id="ZONE_B", risk_score=10.0,
        )
        heatmap = populated_twin.get_heatmap()
        assert len(heatmap) == 2

        zone_a_entry = next(
            h for h in heatmap if h["zone_id"] == "ZONE_A"
        )
        zone_b_entry = next(
            h for h in heatmap if h["zone_id"] == "ZONE_B"
        )
        assert zone_a_entry["color"] == "red"
        assert zone_b_entry["color"] == "green"

    def test_heatmap_all_colors(self, twin_manager: TwinStateManager):
        # Create zones with different risk levels
        twin_manager.update_risk_score(zone_id="Z1", risk_score=10)
        twin_manager.update_risk_score(zone_id="Z2", risk_score=40)
        twin_manager.update_risk_score(zone_id="Z3", risk_score=60)
        twin_manager.update_risk_score(zone_id="Z4", risk_score=90)

        heatmap = twin_manager.get_heatmap()
        colors = {h["zone_id"]: h["color"] for h in heatmap}
        assert colors["Z1"] == "green"
        assert colors["Z2"] == "yellow"
        assert colors["Z3"] == "orange"
        assert colors["Z4"] == "red"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event Handler Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEventHandler:
    """Test the DigitalTwinEventHandler."""

    @pytest.mark.asyncio
    async def test_sensor_anomaly_event(
        self, handler: DigitalTwinEventHandler,
    ):
        event = {
            "event_type": "sensor.reading.anomaly",
            "event_id": "evt-001",
            "data": {
                "sensor_id": "S001",
                "zone_id": "ZONE_A",
                "sensor_type": "gas",
                "value": 150.0,
                "unit": "ppm",
                "anomaly_score": -0.9,
            },
        }
        await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY, event,
        )
        assert handler.events_processed == 1
        assert handler.events_failed == 0

    @pytest.mark.asyncio
    async def test_sensor_status_event(
        self, handler: DigitalTwinEventHandler,
    ):
        event = {
            "event_type": "sensor.status.changed",
            "event_id": "evt-002",
            "data": {
                "sensor_id": "S001",
                "zone_id": "ZONE_A",
                "status": "FAULTY",
            },
        }
        await handler.handle_event(
            KafkaTopics.SENSOR_STATUS_CHANGED, event,
        )
        assert handler.events_processed == 1

    @pytest.mark.asyncio
    async def test_sensor_health_event(
        self, handler: DigitalTwinEventHandler,
    ):
        event = {
            "event_type": "sensor.health.updated",
            "event_id": "evt-003",
            "data": {
                "sensor_id": "S001",
                "zone_id": "ZONE_A",
                "health_score": 75.0,
            },
        }
        await handler.handle_event(
            KafkaTopics.SENSOR_HEALTH_UPDATED, event,
        )
        assert handler.events_processed == 1

    @pytest.mark.asyncio
    async def test_risk_assessment_event(
        self, handler: DigitalTwinEventHandler,
    ):
        event = {
            "event_type": "risk.assessment.generated",
            "event_id": "evt-004",
            "data": {
                "zone_id": "ZONE_A",
                "risk_score": 65.0,
                "risk_level": "HIGH",
                "accident_probability": 0.4,
                "equipment_id": "EQ001",
            },
        }
        await handler.handle_event(
            KafkaTopics.RISK_ASSESSMENT_GENERATED, event,
        )
        assert handler.events_processed == 1

    @pytest.mark.asyncio
    async def test_risk_score_event(
        self, handler: DigitalTwinEventHandler,
    ):
        event = {
            "event_type": "risk.score.updated",
            "event_id": "evt-005",
            "data": {
                "zone_id": "ZONE_A",
                "risk_score": 72.0,
                "risk_level": "HIGH",
            },
        }
        await handler.handle_event(
            KafkaTopics.RISK_SCORE_UPDATED, event,
        )
        assert handler.events_processed == 1

    @pytest.mark.asyncio
    async def test_risk_threshold_event(
        self, handler: DigitalTwinEventHandler,
    ):
        event = {
            "event_type": "risk.threshold.exceeded",
            "event_id": "evt-006",
            "data": {
                "zone_id": "ZONE_A",
                "threshold_type": "gas_level",
                "current_value": 200.0,
                "threshold_value": 100.0,
            },
        }
        await handler.handle_event(
            KafkaTopics.RISK_THRESHOLD_EXCEEDED, event,
        )
        assert handler.events_processed == 1

    @pytest.mark.asyncio
    async def test_compound_risk_event(
        self, handler: DigitalTwinEventHandler,
    ):
        event = {
            "event_type": "compound.risk.detected",
            "event_id": "evt-007",
            "data": {
                "zone_id": "ZONE_A",
                "compound_risk_score": 88.0,
                "risk_level": "CRITICAL",
                "confidence_score": 0.95,
                "contributing_factors": {"gas": 0.6, "temp": 0.4},
            },
        }
        await handler.handle_event(
            KafkaTopics.COMPOUND_RISK_DETECTED, event,
        )
        assert handler.events_processed == 1

    @pytest.mark.asyncio
    async def test_hazard_detected_event(
        self, handler: DigitalTwinEventHandler,
    ):
        event = {
            "event_type": "hazard.detected",
            "event_id": "evt-008",
            "data": {
                "hazard_id": "HAZ-001",
                "zone_id": "ZONE_A",
                "hazard_type": "GAS_LEAK",
                "severity": "HIGH",
            },
        }
        await handler.handle_event(
            KafkaTopics.HAZARD_DETECTED, event,
        )
        assert handler.events_processed == 1

    @pytest.mark.asyncio
    async def test_hazard_propagated_event(
        self, handler: DigitalTwinEventHandler,
    ):
        event = {
            "event_type": "hazard.propagated",
            "event_id": "evt-009",
            "data": {
                "propagation_id": "PROP-001",
                "origin_zone": "ZONE_A",
                "hazard_type": "GAS_LEAK",
                "propagation_level": "SPREADING",
                "affected_zones": ["ZONE_A", "ZONE_B"],
                "severity": "HIGH",
            },
        }
        await handler.handle_event(
            KafkaTopics.HAZARD_PROPAGATED, event,
        )
        assert handler.events_processed == 1

    @pytest.mark.asyncio
    async def test_duplicate_event_skipped(
        self, handler: DigitalTwinEventHandler,
    ):
        event = {
            "event_type": "risk.score.updated",
            "event_id": "duplicate-001",
            "data": {"zone_id": "ZONE_A", "risk_score": 50.0},
        }
        await handler.handle_event(
            KafkaTopics.RISK_SCORE_UPDATED, event,
        )
        await handler.handle_event(
            KafkaTopics.RISK_SCORE_UPDATED, event,
        )
        assert handler.events_processed == 1
        assert handler.events_skipped == 1

    @pytest.mark.asyncio
    async def test_unknown_topic_skipped(
        self, handler: DigitalTwinEventHandler,
    ):
        await handler.handle_event("unknown.topic", {"data": {}})
        assert handler.events_skipped == 1
        assert handler.events_processed == 0

    @pytest.mark.asyncio
    async def test_event_error_counted(
        self, handler: DigitalTwinEventHandler,
    ):
        """Malformed event should be counted as failed, not crash."""
        event = {
            "event_type": "risk.score.updated",
            "event_id": "bad-001",
            "data": None,  # Will cause AttributeError
        }
        # Patch state manager to raise
        with patch.object(
            handler._state, "update_risk_score",
            side_effect=TypeError("bad data"),
        ):
            await handler.handle_event(
                KafkaTopics.RISK_SCORE_UPDATED, event,
            )
        assert handler.events_failed == 1

    @pytest.mark.asyncio
    async def test_end_to_end_state_update(
        self, handler: DigitalTwinEventHandler,
    ):
        """Verify event flows through handler to state manager."""
        event = {
            "event_type": "risk.score.updated",
            "event_id": "e2e-001",
            "data": {"zone_id": "ZONE_TEST", "risk_score": 65.0},
        }
        await handler.handle_event(
            KafkaTopics.RISK_SCORE_UPDATED, event,
        )
        zone = handler._state.get_zone("ZONE_TEST")
        assert zone.predicted_risk_score == 65.0
        assert zone.risk_level == "HIGH"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Consumer Setup Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConsumerSetup:
    """Test DigitalTwinConsumerSetup."""

    def test_subscribed_topics_count(self):
        assert len(DIGITAL_TWIN_SUBSCRIBED_TOPICS) == 9

    def test_subscribed_topics_include_all_modules(self):
        topics = set(DIGITAL_TWIN_SUBSCRIBED_TOPICS)
        # Sensor Intelligence
        assert KafkaTopics.SENSOR_READING_ANOMALY in topics
        assert KafkaTopics.SENSOR_STATUS_CHANGED in topics
        assert KafkaTopics.SENSOR_HEALTH_UPDATED in topics
        # Risk Prediction
        assert KafkaTopics.RISK_ASSESSMENT_GENERATED in topics
        assert KafkaTopics.RISK_SCORE_UPDATED in topics
        assert KafkaTopics.RISK_THRESHOLD_EXCEEDED in topics
        # Compound Risk
        assert KafkaTopics.COMPOUND_RISK_DETECTED in topics
        # Hazard Propagation
        assert KafkaTopics.HAZARD_DETECTED in topics
        assert KafkaTopics.HAZARD_PROPAGATED in topics

    def test_register_once(self, twin_manager: TwinStateManager):
        consumer = MagicMock()
        handler = DigitalTwinEventHandler(state_manager=twin_manager)
        setup = DigitalTwinConsumerSetup(consumer, handler)

        setup.register()
        assert setup.is_registered is True
        assert consumer.register_handler.call_count == 9

        # Second call is a no-op
        setup.register()
        assert consumer.register_handler.call_count == 9

    def test_subscribed_topics_property(
        self, twin_manager: TwinStateManager,
    ):
        consumer = MagicMock()
        handler = DigitalTwinEventHandler(state_manager=twin_manager)
        setup = DigitalTwinConsumerSetup(consumer, handler)
        assert len(setup.subscribed_topics) == 9


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REST API Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def api_client(populated_twin: TwinStateManager):
    """TestClient with DT service injected."""
    from app.core.dependencies import get_digital_twin_service
    from app.main import app

    app.dependency_overrides[get_digital_twin_service] = lambda: populated_twin
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


class TestFacilityEndpoint:
    """Test GET /twin/facility."""

    def test_get_facility(self, api_client: TestClient):
        resp = api_client.get("/api/v1/twin/facility")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["total_zones"] == 2
        assert "facility_health" in data
        assert "zone_ids" in data

    def test_facility_health_range(
        self, api_client: TestClient, populated_twin: TwinStateManager,
    ):
        resp = api_client.get("/api/v1/twin/facility")
        health = resp.json()["facility_health"]
        assert 0 <= health <= 100

    def test_facility_with_events(
        self, api_client: TestClient, populated_twin: TwinStateManager,
    ):
        populated_twin.update_risk_score(
            zone_id="ZONE_A", risk_score=85.0,
        )
        populated_twin.update_hazard_detected(
            zone_id="ZONE_A", hazard_id="H1", hazard_type="FIRE",
        )
        resp = api_client.get("/api/v1/twin/facility")
        data = resp.json()
        assert data["active_hazards"] >= 1
        assert data["critical_zones"] >= 1
        assert data["events_processed"] >= 2


class TestZonesEndpoint:
    """Test GET /twin/zones."""

    def test_get_all_zones(self, api_client: TestClient):
        resp = api_client.get("/api/v1/twin/zones")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["total"] == 2
        assert len(data["zones"]) == 2

    def test_zone_has_full_schema(self, api_client: TestClient):
        resp = api_client.get("/api/v1/twin/zones")
        zone = resp.json()["zones"][0]
        # Check all expected fields
        expected_fields = [
            "zone_id", "zone_name", "sensor_health", "anomaly_count",
            "predicted_risk_score", "risk_level", "compound_risk_score",
            "active_hazards", "affected_neighbors", "workers_at_risk",
            "equipment", "connected_zones", "overall_risk_score",
            "heatmap_color", "is_critical",
        ]
        for field in expected_fields:
            assert field in zone, f"Missing field: {field}"


class TestZoneDetailEndpoint:
    """Test GET /twin/zones/{zone_id}."""

    def test_get_zone_detail(self, api_client: TestClient):
        resp = api_client.get("/api/v1/twin/zones/ZONE_A")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["zone"]["zone_id"] == "ZONE_A"
        assert data["zone"]["zone_name"] == "Production Area 1"

    def test_zone_has_equipment(self, api_client: TestClient):
        resp = api_client.get("/api/v1/twin/zones/ZONE_A")
        zone = resp.json()["zone"]
        assert len(zone["equipment"]) == 1
        assert zone["equipment"][0]["equipment_id"] == "EQ001"

    def test_zone_has_sensors(self, api_client: TestClient):
        resp = api_client.get("/api/v1/twin/zones/ZONE_A")
        zone = resp.json()["zone"]
        assert "S001" in zone["latest_sensor_readings"]

    def test_zone_not_found(self, api_client: TestClient):
        resp = api_client.get("/api/v1/twin/zones/NONEXISTENT")
        assert resp.status_code == 404

    def test_zone_with_updates(
        self, api_client: TestClient, populated_twin: TwinStateManager,
    ):
        populated_twin.update_risk_score(
            zone_id="ZONE_A", risk_score=80.0,
        )
        populated_twin.update_compound_risk(
            zone_id="ZONE_A",
            compound_risk_score=90.0,
            risk_level="CRITICAL",
        )
        resp = api_client.get("/api/v1/twin/zones/ZONE_A")
        zone = resp.json()["zone"]
        assert zone["predicted_risk_score"] == 80.0
        assert zone["compound_risk_score"] == 90.0
        assert zone["overall_risk_score"] == 90.0
        assert zone["is_critical"] is True
        assert zone["heatmap_color"] == "red"


class TestHeatmapEndpoint:
    """Test GET /twin/heatmap."""

    def test_get_heatmap(self, api_client: TestClient):
        resp = api_client.get("/api/v1/twin/heatmap")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["total"] == 2

    def test_heatmap_entries_have_color(
        self, api_client: TestClient,
    ):
        resp = api_client.get("/api/v1/twin/heatmap")
        for entry in resp.json()["heatmap"]:
            assert entry["color"] in ["green", "yellow", "orange", "red"]

    def test_heatmap_reflects_risk(
        self, api_client: TestClient, populated_twin: TwinStateManager,
    ):
        populated_twin.update_risk_score(
            zone_id="ZONE_A", risk_score=90.0,
        )
        populated_twin.update_risk_score(
            zone_id="ZONE_B", risk_score=15.0,
        )
        resp = api_client.get("/api/v1/twin/heatmap")
        entries = {
            e["zone_id"]: e for e in resp.json()["heatmap"]
        }
        assert entries["ZONE_A"]["color"] == "red"
        assert entries["ZONE_A"]["risk_score"] == 90.0
        assert entries["ZONE_B"]["color"] == "green"
        assert entries["ZONE_B"]["risk_score"] == 15.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEventToApiIntegration:
    """End-to-end: Kafka event → handler → state → API response."""

    @pytest.mark.asyncio
    async def test_full_pipeline(
        self, populated_twin: TwinStateManager,
    ):
        handler = DigitalTwinEventHandler(state_manager=populated_twin)

        # 1. Sensor anomaly
        await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY,
            {
                "event_id": "int-001",
                "data": {
                    "sensor_id": "S001",
                    "zone_id": "ZONE_A",
                    "value": 180.0,
                    "anomaly_score": -0.95,
                },
            },
        )

        # 2. Risk assessment
        await handler.handle_event(
            KafkaTopics.RISK_SCORE_UPDATED,
            {
                "event_id": "int-002",
                "data": {
                    "zone_id": "ZONE_A",
                    "risk_score": 78.0,
                    "risk_level": "CRITICAL",
                },
            },
        )

        # 3. Compound risk
        await handler.handle_event(
            KafkaTopics.COMPOUND_RISK_DETECTED,
            {
                "event_id": "int-003",
                "data": {
                    "zone_id": "ZONE_A",
                    "compound_risk_score": 92.0,
                    "risk_level": "CRITICAL",
                    "confidence_score": 0.97,
                },
            },
        )

        # 4. Hazard propagation
        await handler.handle_event(
            KafkaTopics.HAZARD_PROPAGATED,
            {
                "event_id": "int-004",
                "data": {
                    "propagation_id": "PROP-INT-001",
                    "origin_zone": "ZONE_A",
                    "hazard_type": "GAS_LEAK",
                    "propagation_level": "CRITICAL",
                    "affected_zones": ["ZONE_A", "ZONE_B"],
                },
            },
        )

        # Verify state
        zone_a = populated_twin.get_zone("ZONE_A")
        assert zone_a.anomaly_count == 1
        assert zone_a.predicted_risk_score == 78.0
        assert zone_a.compound_risk_score == 92.0
        assert zone_a.active_hazard_count >= 1
        assert zone_a.is_critical is True
        assert zone_a.overall_risk_score == 92.0
        assert zone_a.heatmap_color == "red"

        # Verify facility state
        facility = populated_twin.get_facility_state()
        assert facility.active_hazards >= 1
        assert facility.critical_zones >= 1
        assert facility.events_processed >= 4
        assert facility.facility_health < 100.0

        # Verify heatmap
        heatmap = populated_twin.get_heatmap()
        zone_a_hm = next(h for h in heatmap if h["zone_id"] == "ZONE_A")
        assert zone_a_hm["color"] == "red"
        assert zone_a_hm["risk_score"] == 92.0


class TestAutoZoneCreation:
    """Test that zones are auto-created on first event."""

    @pytest.mark.asyncio
    async def test_event_creates_zone(
        self, handler: DigitalTwinEventHandler,
    ):
        await handler.handle_event(
            KafkaTopics.RISK_SCORE_UPDATED,
            {
                "event_id": "auto-001",
                "data": {
                    "zone_id": "NEW_ZONE",
                    "risk_score": 42.0,
                },
            },
        )
        zone = handler._state.get_zone("NEW_ZONE")
        assert zone.predicted_risk_score == 42.0
        assert zone.risk_level == "MEDIUM"
