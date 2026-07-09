"""Comprehensive unit tests for the HazardPropagationService.

Tests:
  1.  Initialization and configuration
  2.  Full propagation analysis (propagate_hazard)
  3.  Simulation mode (no persist / publish)
  4.  Validation (hazard type, risk score)
  5.  Zone not found handling
  6.  Graph repository failure handling
  7.  Propagation engine failure handling
  8.  Kafka publish failure handling
  9.  Hazard persistence failure handling
  10. Summary and explanation generation
  11. Recommendation generation per propagation level
  12. Metrics tracking
  13. Query methods (graph stats, neighbors, zone risk, paths)
  14. Configurable parameters (decay, depth, threshold)
  15. End-to-end scenarios
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from app.hazard_propagation.domain.exceptions import (
    HazardPropagationError,
    InvalidHazardError,
    PropagationSimulationError,
    ZoneNotFoundError,
)
from app.hazard_propagation.domain.value_objects import (
    HazardType,
    PropagationLevel,
    PropagationStatus,
)
from app.hazard_propagation.graph.entities import (
    EquipmentNode,
    SensorNode,
    ZoneNode,
)
from app.hazard_propagation.messaging.publisher import (
    HazardPropagationPublisher,
)
from app.hazard_propagation.repositories.in_memory_graph_repo import (
    InMemoryGraphRepository,
)
from app.hazard_propagation.services.config import PropagationConfig
from app.hazard_propagation.services.hazard_propagation_service import (
    HazardPropagationService,
    PropagationAnalysisResult,
)
from app.hazard_propagation.services.propagation_engine import (
    HazardPropagationEngine,
)
from app.shared.messaging.events import BaseEvent
from app.shared.messaging.producer import NoopEventProducer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def publish_log() -> List[Dict[str, Any]]:
    return []


@pytest.fixture
def tracking_producer(publish_log) -> NoopEventProducer:
    prod = NoopEventProducer()
    original = prod.publish

    def tracking(topic, data, **kw):
        event = original(topic, data, **kw)
        publish_log.append({"topic": topic, "data": data, "event": event})
        return event

    prod.publish = tracking
    return prod


@pytest.fixture
def publisher(tracking_producer) -> HazardPropagationPublisher:
    return HazardPropagationPublisher(tracking_producer)


@pytest_asyncio.fixture
async def graph_repo() -> InMemoryGraphRepository:
    """Graph with 3 connected zones + equipment + sensors."""
    repo = InMemoryGraphRepository()

    # Zones
    await repo.create_zone(ZoneNode(
        zone_id="ZONE_A", zone_name="Zone Alpha",
        risk_level_baseline="HIGH", current_worker_count=5,
        worker_capacity=20,
    ))
    await repo.create_zone(ZoneNode(
        zone_id="ZONE_B", zone_name="Zone Bravo",
        risk_level_baseline="MEDIUM", current_worker_count=3,
        worker_capacity=15,
    ))
    await repo.create_zone(ZoneNode(
        zone_id="ZONE_C", zone_name="Zone Charlie",
        risk_level_baseline="LOW", current_worker_count=2,
        worker_capacity=10,
    ))

    # Connections
    await repo.create_connection("ZONE_A", "ZONE_B")
    await repo.create_connection("ZONE_B", "ZONE_C")

    # Equipment in ZONE_A
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
    await repo.create_sensor(
        "EQ001", SensorNode(sensor_id="S001", sensor_type="TEMPERATURE"),
    )
    await repo.create_sensor(
        "EQ002", SensorNode(sensor_id="S002", sensor_type="GAS"),
    )

    return repo


@pytest_asyncio.fixture
async def service(graph_repo, publisher) -> HazardPropagationService:
    """Service with default config and publisher."""
    return HazardPropagationService(
        graph_repo=graph_repo,
        publisher=publisher,
    )


@pytest_asyncio.fixture
async def service_no_publisher(graph_repo) -> HazardPropagationService:
    """Service without Kafka publisher."""
    return HazardPropagationService(
        graph_repo=graph_repo,
        publisher=None,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Initialization and configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInitialization:
    async def test_default_config(self, service):
        assert service.config.propagation_decay_factor == 0.6
        assert service.config.max_depth == 3
        assert service.config.minimum_propagation_threshold == 0.1

    async def test_custom_config(self, graph_repo):
        config = PropagationConfig(
            propagation_decay_factor=0.8,
            max_depth=5,
            minimum_propagation_threshold=0.05,
        )
        svc = HazardPropagationService(
            graph_repo=graph_repo, config=config,
        )
        assert svc.config.propagation_decay_factor == 0.8
        assert svc.config.max_depth == 5

    async def test_initial_metrics_zero(self, service):
        assert service.total_propagations == 0
        assert service.failed_propagations == 0
        assert service.total_zones_affected == 0
        assert service.total_workers_at_risk == 0

    async def test_publisher_optional(self, service_no_publisher):
        """Service works without a publisher."""
        result = await service_no_publisher.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=70.0,
        )
        assert result.propagation_result.status == PropagationStatus.COMPLETED


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Full propagation analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPropagate:
    async def test_basic_propagation(self, service, publish_log):
        result = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        assert isinstance(result, PropagationAnalysisResult)
        assert result.propagation_result.status == PropagationStatus.COMPLETED
        assert result.propagation_result.total_affected_zones >= 1
        assert result.processing_time_ms > 0

    async def test_publishes_event(self, service, publish_log):
        await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        assert len(publish_log) == 1
        assert publish_log[0]["topic"] == "hazard.propagated"

    async def test_returns_summary(self, service):
        result = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        assert "GAS_LEAK" in result.summary
        assert "ZONE_A" in result.summary
        assert len(result.summary) > 0

    async def test_returns_explanation(self, service):
        result = await service.propagate_hazard(
            hazard_type="FIRE", origin_zone="ZONE_A",
            compound_risk_score=75.0,
        )
        assert "FIRE" in result.explanation
        assert "ZONE_A" in result.explanation
        assert len(result.explanation) > 20

    async def test_returns_recommendations(self, service):
        result = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        assert len(result.recommendations) > 0

    async def test_custom_hazard_id(self, service):
        result = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            hazard_id="HAZ-CUSTOM-001",
        )
        assert result.propagation_result.propagation_id == "HAZ-CUSTOM-001"

    async def test_custom_max_depth(self, service):
        result_d1 = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0, max_depth=1,
        )
        result_d3 = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0, max_depth=3,
        )
        assert (
            result_d3.propagation_result.total_affected_zones
            >= result_d1.propagation_result.total_affected_zones
        )

    async def test_correlation_id(self, service, publish_log):
        await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
            correlation_id="CORR-001",
        )
        assert len(publish_log) == 1

    async def test_to_dict_serialization(self, service):
        result = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        d = result.to_dict()
        assert "propagation_id" in d
        assert "hazard_type" in d
        assert "origin_zone" in d
        assert "propagation_level" in d
        assert "affected_zones" in d
        assert "summary" in d
        assert "explanation" in d
        assert "recommendations" in d
        assert "processing_time_ms" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Simulation mode
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSimulate:
    async def test_simulate_returns_dict(self, service):
        result = await service.simulate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        assert isinstance(result, dict)
        assert "propagation_id" in result
        assert "affected_zones" in result
        assert "summary" in result

    async def test_simulate_no_publish(self, service, publish_log):
        await service.simulate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        # Simulate should NOT publish events
        assert len(publish_log) == 0

    async def test_simulate_no_metrics(self, service):
        await service.simulate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        # Simulate should NOT increment metrics
        assert service.total_propagations == 0

    async def test_simulate_custom_depth(self, service):
        r1 = await service.simulate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            max_depth=1,
        )
        r3 = await service.simulate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            max_depth=3,
        )
        assert r3["total_affected_zones"] >= r1["total_affected_zones"]

    async def test_simulate_invalid_zone_raises(self, service):
        with pytest.raises(ZoneNotFoundError):
            await service.simulate(
                hazard_type="GAS_LEAK", origin_zone="ZONE_NOPE",
            )

    async def test_simulate_invalid_hazard_raises(self, service):
        with pytest.raises(InvalidHazardError):
            await service.simulate(
                hazard_type="INVALID", origin_zone="ZONE_A",
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestValidation:
    async def test_invalid_hazard_type_raises(self, service):
        with pytest.raises(InvalidHazardError, match="Invalid hazard type"):
            await service.propagate_hazard(
                hazard_type="EARTHQUAKE", origin_zone="ZONE_A",
            )

    async def test_all_valid_hazard_types_accepted(self, service):
        for ht in HazardType:
            result = await service.propagate_hazard(
                hazard_type=ht.value, origin_zone="ZONE_A",
                compound_risk_score=50.0,
            )
            assert result.propagation_result.status == PropagationStatus.COMPLETED

    async def test_negative_risk_score_raises(self, service):
        with pytest.raises(InvalidHazardError, match="compound_risk_score"):
            await service.propagate_hazard(
                hazard_type="GAS_LEAK", origin_zone="ZONE_A",
                compound_risk_score=-10.0,
            )

    async def test_risk_score_above_100_raises(self, service):
        with pytest.raises(InvalidHazardError, match="compound_risk_score"):
            await service.propagate_hazard(
                hazard_type="GAS_LEAK", origin_zone="ZONE_A",
                compound_risk_score=150.0,
            )

    async def test_boundary_risk_scores(self, service):
        """Boundary values 0.0 and 100.0 are valid."""
        r0 = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=0.0,
        )
        r100 = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=100.0,
        )
        assert r0.propagation_result.status == PropagationStatus.COMPLETED
        assert r100.propagation_result.status == PropagationStatus.COMPLETED


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Zone not found
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestZoneNotFound:
    async def test_unknown_zone_raises(self, service):
        with pytest.raises(ZoneNotFoundError):
            await service.propagate_hazard(
                hazard_type="GAS_LEAK", origin_zone="NONEXISTENT",
            )

    async def test_failed_propagation_increments(self, service):
        assert service.failed_propagations == 0
        with pytest.raises(ZoneNotFoundError):
            await service.propagate_hazard(
                hazard_type="GAS_LEAK", origin_zone="NONEXISTENT",
            )
        assert service.failed_propagations == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Graph repository failure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGraphRepoFailure:
    async def test_get_zone_failure_raises(self, publisher):
        mock_repo = AsyncMock()
        mock_repo.get_zone = AsyncMock(
            side_effect=RuntimeError("Neo4j down"),
        )
        svc = HazardPropagationService(
            graph_repo=mock_repo, publisher=publisher,
        )
        with pytest.raises(PropagationSimulationError):
            await svc.propagate_hazard(
                hazard_type="GAS_LEAK", origin_zone="ZONE_A",
                compound_risk_score=80.0,
            )
        assert svc.failed_propagations == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Propagation engine failure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEngineFailure:
    async def test_engine_failure_raises(self, graph_repo, publisher):
        mock_engine = AsyncMock()
        mock_engine.propagate = AsyncMock(
            side_effect=PropagationSimulationError("BFS exploded"),
        )
        svc = HazardPropagationService(
            graph_repo=graph_repo,
            engine=mock_engine,
            publisher=publisher,
        )
        with pytest.raises(PropagationSimulationError):
            await svc.propagate_hazard(
                hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            )
        assert svc.failed_propagations == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Kafka publish failure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPublishFailure:
    async def test_publish_failure_does_not_crash(self, graph_repo):
        """Kafka publish failure should not crash propagation."""
        failing_producer = NoopEventProducer()
        failing_producer.publish = MagicMock(
            side_effect=RuntimeError("Kafka unavailable"),
        )
        failing_publisher = HazardPropagationPublisher(failing_producer)
        svc = HazardPropagationService(
            graph_repo=graph_repo, publisher=failing_publisher,
        )
        # Should NOT raise
        result = await svc.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        assert result.propagation_result.status == PropagationStatus.COMPLETED
        assert svc.total_propagations == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Hazard persistence failure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPersistenceFailure:
    async def test_persist_failure_does_not_crash(
        self, publisher, publish_log,
    ):
        """Graph persistence failure should not crash propagation."""
        repo = InMemoryGraphRepository()
        await repo.create_zone(ZoneNode(
            zone_id="ZONE_A", current_worker_count=3,
        ))

        # Patch create_hazard to fail
        original_create = repo.create_hazard
        repo.create_hazard = AsyncMock(
            side_effect=RuntimeError("Neo4j write failure"),
        )

        svc = HazardPropagationService(
            graph_repo=repo, publisher=publisher,
        )
        result = await svc.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert result.propagation_result.status == PropagationStatus.COMPLETED
        assert svc.total_propagations == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Summary and explanation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSummaryAndExplanation:
    async def test_summary_contains_key_info(self, service):
        result = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        assert "GAS_LEAK" in result.summary
        assert "ZONE_A" in result.summary
        assert "zone(s)" in result.summary
        assert "worker(s)" in result.summary
        assert "radius" in result.summary

    async def test_explanation_has_zone_assessments(self, service):
        result = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        assert "Zone Impact Assessment" in result.explanation
        assert "ZONE_A" in result.explanation

    async def test_explanation_has_equipment_section(self, service):
        result = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        assert "Equipment Impact Assessment" in result.explanation

    async def test_explanation_has_propagation_paths(self, service):
        result = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        if result.propagation_result.propagation_paths:
            assert "Propagation Paths" in result.explanation


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. Recommendations per propagation level
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRecommendations:
    async def test_contained_recommendations(self, service_no_publisher):
        """PPE_VIOLATION doesn't propagate → CONTAINED."""
        result = await service_no_publisher.propagate_hazard(
            hazard_type="PPE_VIOLATION", origin_zone="ZONE_A",
            compound_risk_score=50.0,
        )
        assert result.propagation_result.propagation_level == PropagationLevel.CONTAINED
        assert any("MONITOR" in r for r in result.recommendations)

    async def test_recommendations_not_empty(self, service):
        result = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        assert len(result.recommendations) > 0

    async def test_high_risk_has_urgent_recommendations(self, service):
        result = await service.propagate_hazard(
            hazard_type="SMOKE", origin_zone="ZONE_A",
            compound_risk_score=95.0, max_depth=5,
        )
        level = result.propagation_result.propagation_level
        if level in (PropagationLevel.CRITICAL, PropagationLevel.EMERGENCY):
            combined = " ".join(result.recommendations)
            assert "URGENT" in combined or "IMMEDIATE" in combined


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. Metrics tracking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMetrics:
    async def test_total_propagations_increments(self, service):
        assert service.total_propagations == 0
        await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert service.total_propagations == 1
        await service.propagate_hazard(
            hazard_type="FIRE", origin_zone="ZONE_A",
        )
        assert service.total_propagations == 2

    async def test_zones_affected_accumulates(self, service):
        await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        assert service.total_zones_affected >= 1

    async def test_workers_at_risk_accumulates(self, service):
        await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        assert service.total_workers_at_risk >= 1

    async def test_failed_propagations_increments(self, service):
        assert service.failed_propagations == 0
        with pytest.raises(ZoneNotFoundError):
            await service.propagate_hazard(
                hazard_type="GAS_LEAK", origin_zone="NO_ZONE",
            )
        assert service.failed_propagations == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. Query methods
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestQueries:
    async def test_get_graph_stats(self, service):
        stats = await service.get_graph_stats()
        assert isinstance(stats, dict)
        assert stats.get("zones", 0) >= 3
        assert stats.get("equipment", 0) >= 3
        assert stats.get("sensors", 0) >= 2

    async def test_get_zone_neighbors(self, service):
        neighbors = await service.get_zone_neighbors("ZONE_A", max_hops=2)
        assert isinstance(neighbors, dict)
        assert "ZONE_A" in neighbors
        assert "ZONE_B" in neighbors

    async def test_get_zone_neighbors_not_found(self, service):
        with pytest.raises(ZoneNotFoundError):
            await service.get_zone_neighbors("NO_ZONE")

    async def test_get_zone_risk_assessment(self, service):
        assessment = await service.get_zone_risk_assessment("ZONE_A")
        assert assessment["zone_id"] == "ZONE_A"
        assert assessment["zone_name"] == "Zone Alpha"
        assert assessment["risk_level_baseline"] == "HIGH"
        assert assessment["worker_count"] == 5
        assert assessment["equipment_count"] == 2
        assert assessment["sensor_count"] == 2
        assert assessment["connected_zones"] >= 1

    async def test_get_zone_risk_assessment_not_found(self, service):
        with pytest.raises(ZoneNotFoundError):
            await service.get_zone_risk_assessment("NO_ZONE")

    async def test_get_hazard_paths(self, service):
        paths = await service.get_hazard_paths("ZONE_A", max_depth=3)
        assert isinstance(paths, list)
        # Should find at least one path
        assert len(paths) >= 1

    async def test_get_hazard_paths_not_found(self, service):
        with pytest.raises(ZoneNotFoundError):
            await service.get_hazard_paths("NO_ZONE")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 14. Configurable parameters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConfigurableParameters:
    async def test_high_decay_more_spread(self, graph_repo, publisher):
        """Higher decay → more zones affected."""
        config_high = PropagationConfig(propagation_decay_factor=0.9)
        config_low = PropagationConfig(propagation_decay_factor=0.3)

        svc_high = HazardPropagationService(
            graph_repo=graph_repo, config=config_high, publisher=publisher,
        )
        svc_low = HazardPropagationService(
            graph_repo=graph_repo, config=config_low, publisher=publisher,
        )

        # Use type NOT in hazard overrides
        r_high = await svc_high.simulate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        r_low = await svc_low.simulate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        assert r_high["total_affected_zones"] >= r_low["total_affected_zones"]

    async def test_high_threshold_less_spread(self, graph_repo, publisher):
        """Higher threshold → fewer zones pass the filter."""
        config = PropagationConfig(minimum_propagation_threshold=0.9)
        svc = HazardPropagationService(
            graph_repo=graph_repo, config=config, publisher=publisher,
        )
        result = await svc.simulate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        # With threshold=0.9, only origin (prob=1.0) should pass
        assert result["total_affected_zones"] == 1

    async def test_max_depth_limits_reach(self, graph_repo, publisher):
        config = PropagationConfig(max_depth=1)
        svc = HazardPropagationService(
            graph_repo=graph_repo, config=config, publisher=publisher,
        )
        result = await svc.simulate(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        assert "ZONE_C" not in result["affected_zones"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 15. End-to-end scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEndToEnd:
    async def test_full_pipeline(self, service, publish_log):
        """Full: validate → propagate → persist → summarize → publish."""
        result = await service.propagate_hazard(
            hazard_type="FIRE", origin_zone="ZONE_A",
            compound_risk_score=85.0,
            correlation_id="E2E-001",
        )

        # Propagation result
        pr = result.propagation_result
        assert pr.status == PropagationStatus.COMPLETED
        assert pr.hazard_type == "FIRE"
        assert pr.origin_zone == "ZONE_A"
        assert pr.total_affected_zones >= 1

        # Summary
        assert "FIRE" in result.summary
        assert len(result.explanation) > 0
        assert len(result.recommendations) > 0

        # Kafka event
        assert len(publish_log) == 1
        data = publish_log[0]["data"]
        assert data["hazard_type"] == "FIRE"
        assert data["origin_zone"] == "ZONE_A"

        # Metrics
        assert service.total_propagations == 1

    async def test_multiple_sequential_propagations(self, service, publish_log):
        """Multiple propagations should work independently."""
        r1 = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=70.0,
        )
        r2 = await service.propagate_hazard(
            hazard_type="FIRE", origin_zone="ZONE_B",
            compound_risk_score=60.0,
        )
        assert r1.propagation_result.propagation_id != r2.propagation_result.propagation_id
        assert service.total_propagations == 2
        assert len(publish_log) == 2

    async def test_different_hazard_types_different_results(self, service):
        """Different hazard types produce different decay behaviours."""
        gas = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        ppe = await service.propagate_hazard(
            hazard_type="PPE_VIOLATION", origin_zone="ZONE_A",
        )
        assert (
            gas.propagation_result.total_affected_zones
            >= ppe.propagation_result.total_affected_zones
        )

    async def test_propagation_from_different_zones(self, service):
        """Same hazard from different origins."""
        from_a = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
        )
        from_c = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_C",
        )
        assert from_a.propagation_result.origin_zone == "ZONE_A"
        assert from_c.propagation_result.origin_zone == "ZONE_C"

    async def test_to_dict_complete(self, service):
        result = await service.propagate_hazard(
            hazard_type="GAS_LEAK", origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
        d = result.to_dict()

        # All expected keys present
        expected = [
            "propagation_id", "hazard_type", "origin_zone",
            "propagation_level", "status", "affected_zones",
            "total_affected_zones", "total_workers_at_risk",
            "impact_radius_meters", "time_to_critical_minutes",
            "impact_scores", "propagation_probabilities",
            "affected_equipment", "propagation_paths",
            "recommended_action", "summary", "explanation",
            "recommendations", "processing_time_ms",
        ]
        for key in expected:
            assert key in d, f"Missing key: {key}"
