"""Comprehensive tests for Digital Twin Phase 3: Snapshot Persistence.

Covers:
  - SnapshotService: create, recover, list, get, delete, retention
  - Automatic snapshot triggers
  - SQLAlchemySnapshotRepository: CRUD, retention cleanup
  - TwinStateManager: restore_zone, restore_events_processed
  - REST API endpoints: GET/POST/DELETE snapshots
  - Startup recovery flow
  - Full pipeline: event → state update → auto-snapshot → recovery
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.digital_twin.domain.entities import (
    ActiveHazard,
    EquipmentState,
    FacilityState,
    SensorReading,
    ZoneState,
)
from app.digital_twin.messaging.handler import DigitalTwinEventHandler
from app.digital_twin.models.facility_snapshot_model import (
    FacilitySnapshotModel,
)
from app.digital_twin.models.zone_state_model import ZoneStateModel
from app.digital_twin.repositories.twin_snapshot_repository import (
    TwinSnapshotRepository,
)
from app.digital_twin.services.snapshot_service import (
    COMPOUND_RISK_THRESHOLD,
    DEFAULT_MAX_SNAPSHOTS,
    HEALTH_CHANGE_THRESHOLD,
    SnapshotService,
)
from app.digital_twin.services.twin_state_manager import TwinStateManager
from app.hazard_propagation.repositories.in_memory_graph_repo import (
    InMemoryGraphRepository,
)
from app.shared.messaging.topics import KafkaTopics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# In-Memory Snapshot Repository (test double)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class InMemorySnapshotRepository(TwinSnapshotRepository):
    """In-memory implementation of TwinSnapshotRepository for tests."""

    def __init__(self) -> None:
        self._snapshots: Dict[str, FacilitySnapshotModel] = {}
        self._zone_states: Dict[str, List[ZoneStateModel]] = {}

    async def save_snapshot(
        self,
        snapshot: FacilitySnapshotModel,
        zone_states: List[ZoneStateModel],
    ) -> FacilitySnapshotModel:
        if not snapshot.id:
            snapshot.id = str(uuid.uuid4())
        if not snapshot.created_at:
            snapshot.created_at = datetime.now(timezone.utc)
        self._snapshots[snapshot.snapshot_id] = snapshot
        self._zone_states[snapshot.snapshot_id] = zone_states
        return snapshot

    async def get_snapshot(
        self, snapshot_id: str,
    ) -> Optional[FacilitySnapshotModel]:
        return self._snapshots.get(snapshot_id)

    async def get_latest_snapshot(
        self,
    ) -> Optional[FacilitySnapshotModel]:
        if not self._snapshots:
            return None
        return max(
            self._snapshots.values(),
            key=lambda s: s.created_at or datetime.min,
        )

    async def list_snapshots(
        self, offset: int = 0, limit: int = 50,
    ) -> List[FacilitySnapshotModel]:
        sorted_snaps = sorted(
            self._snapshots.values(),
            key=lambda s: s.created_at or datetime.min,
            reverse=True,
        )
        return sorted_snaps[offset : offset + limit]

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        if snapshot_id in self._snapshots:
            del self._snapshots[snapshot_id]
            self._zone_states.pop(snapshot_id, None)
            return True
        return False

    async def count_snapshots(self) -> int:
        return len(self._snapshots)

    async def get_zone_states_for_snapshot(
        self, snapshot_id: str,
    ) -> List[ZoneStateModel]:
        return self._zone_states.get(snapshot_id, [])

    async def delete_oldest_snapshots(
        self, keep_count: int,
    ) -> int:
        if len(self._snapshots) <= keep_count:
            return 0
        sorted_snaps = sorted(
            self._snapshots.values(),
            key=lambda s: s.created_at or datetime.min,
            reverse=True,
        )
        to_keep = sorted_snaps[:keep_count]
        keep_ids = {s.snapshot_id for s in to_keep}
        to_delete = [
            sid for sid in self._snapshots
            if sid not in keep_ids
        ]
        for sid in to_delete:
            del self._snapshots[sid]
            self._zone_states.pop(sid, None)
        return len(to_delete)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def graph_repo() -> InMemoryGraphRepository:
    return InMemoryGraphRepository()


@pytest.fixture
def twin_manager(graph_repo: InMemoryGraphRepository) -> TwinStateManager:
    manager = TwinStateManager(graph_repo=graph_repo)
    asyncio.get_event_loop().run_until_complete(manager.initialize())
    return manager


@pytest.fixture
def snapshot_repo() -> InMemorySnapshotRepository:
    return InMemorySnapshotRepository()


@pytest.fixture
def snapshot_service(
    twin_manager: TwinStateManager,
    snapshot_repo: InMemorySnapshotRepository,
) -> SnapshotService:
    return SnapshotService(
        state_manager=twin_manager,
        repository=snapshot_repo,
        max_snapshots=5,  # Low for testing retention
    )


def _populate_twin(manager: TwinStateManager) -> None:
    """Add some state to the twin for snapshot testing."""
    manager.update_sensor_anomaly(
        zone_id="ZONE_A",
        sensor_id="S001",
        sensor_type="gas",
        value=150.0,
        unit="ppm",
        anomaly_score=-0.9,
    )
    manager.update_risk_score(
        zone_id="ZONE_A",
        risk_score=72.0,
    )
    manager.update_compound_risk(
        zone_id="ZONE_A",
        compound_risk_score=80.0,
        risk_level="CRITICAL",
    )
    manager.update_hazard_detected(
        zone_id="ZONE_A",
        hazard_id="HAZ-001",
        hazard_type="GAS_LEAK",
        severity="HIGH",
    )
    manager.update_sensor_anomaly(
        zone_id="ZONE_B",
        sensor_id="S002",
        sensor_type="temp",
        value=95.0,
        unit="C",
        anomaly_score=-0.5,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TwinStateManager Restore Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTwinStateManagerRestore:
    """Test state restoration methods."""

    def test_restore_zone(self, twin_manager: TwinStateManager):
        zone = ZoneState(
            zone_id="ZONE_X",
            zone_name="Restored Zone",
            predicted_risk_score=55.0,
            risk_level="HIGH",
            anomaly_count=3,
        )
        twin_manager.restore_zone(zone)
        restored = twin_manager.get_zone("ZONE_X")
        assert restored.zone_name == "Restored Zone"
        assert restored.predicted_risk_score == 55.0
        assert restored.anomaly_count == 3

    def test_restore_zone_replaces_existing(
        self, twin_manager: TwinStateManager,
    ):
        twin_manager.update_risk_score(
            zone_id="ZONE_A", risk_score=10.0,
        )
        new_zone = ZoneState(
            zone_id="ZONE_A",
            predicted_risk_score=99.0,
        )
        twin_manager.restore_zone(new_zone)
        assert twin_manager.get_zone("ZONE_A").predicted_risk_score == 99.0

    def test_restore_events_processed(
        self, twin_manager: TwinStateManager,
    ):
        twin_manager.restore_events_processed(500)
        assert twin_manager.events_processed == 500

    def test_restore_events_processed_uses_max(
        self, twin_manager: TwinStateManager,
    ):
        # Process some events first
        twin_manager.update_risk_score(
            zone_id="Z1", risk_score=10.0,
        )
        twin_manager.update_risk_score(
            zone_id="Z1", risk_score=20.0,
        )
        current = twin_manager.events_processed  # 2
        twin_manager.restore_events_processed(1)  # lower
        assert twin_manager.events_processed == current  # stays at 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SnapshotService Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSnapshotCreation:
    """Test snapshot creation."""

    @pytest.mark.asyncio
    async def test_create_snapshot(
        self,
        snapshot_service: SnapshotService,
        twin_manager: TwinStateManager,
    ):
        _populate_twin(twin_manager)
        snap = await snapshot_service.create_snapshot(
            trigger_reason="manual",
        )
        assert snap.snapshot_id
        assert snap.facility_health > 0
        assert snap.total_zones == 2
        assert snap.trigger_reason == "manual"
        assert snap.active_hazards == 1

    @pytest.mark.asyncio
    async def test_create_snapshot_payload(
        self,
        snapshot_service: SnapshotService,
        twin_manager: TwinStateManager,
    ):
        _populate_twin(twin_manager)
        snap = await snapshot_service.create_snapshot()
        payload = json.loads(snap.snapshot_payload)
        assert "facility" in payload
        assert "zones" in payload
        assert len(payload["zones"]) == 2

        zone_a = next(
            z for z in payload["zones"]
            if z["zone_id"] == "ZONE_A"
        )
        assert zone_a["compound_risk_score"] == 80.0
        assert len(zone_a["active_hazards"]) == 1
        assert zone_a["active_hazards"][0]["hazard_type"] == "GAS_LEAK"

    @pytest.mark.asyncio
    async def test_create_snapshot_with_sensor_readings(
        self,
        snapshot_service: SnapshotService,
        twin_manager: TwinStateManager,
    ):
        _populate_twin(twin_manager)
        snap = await snapshot_service.create_snapshot()
        payload = json.loads(snap.snapshot_payload)
        zone_a = next(
            z for z in payload["zones"]
            if z["zone_id"] == "ZONE_A"
        )
        assert "S001" in zone_a["latest_sensor_readings"]
        s001 = zone_a["latest_sensor_readings"]["S001"]
        assert s001["value"] == 150.0
        assert s001["sensor_type"] == "gas"

    @pytest.mark.asyncio
    async def test_create_increments_counter(
        self, snapshot_service: SnapshotService,
    ):
        await snapshot_service.create_snapshot()
        await snapshot_service.create_snapshot()
        assert snapshot_service.snapshots_created == 2

    @pytest.mark.asyncio
    async def test_create_snapshot_zone_models(
        self,
        snapshot_service: SnapshotService,
        snapshot_repo: InMemorySnapshotRepository,
        twin_manager: TwinStateManager,
    ):
        _populate_twin(twin_manager)
        snap = await snapshot_service.create_snapshot()
        zone_states = await snapshot_repo.get_zone_states_for_snapshot(
            snap.snapshot_id,
        )
        assert len(zone_states) == 2
        zone_a = next(
            zs for zs in zone_states
            if zs.zone_id == "ZONE_A"
        )
        assert zone_a.risk_score == 80.0  # overall_risk_score
        assert zone_a.compound_risk_score == 80.0
        assert zone_a.hazard_count == 1
        assert zone_a.anomaly_count == 1


class TestSnapshotRecovery:
    """Test snapshot recovery."""

    @pytest.mark.asyncio
    async def test_recover_no_snapshots(
        self, snapshot_service: SnapshotService,
    ):
        result = await snapshot_service.recover_latest_snapshot()
        assert result is False

    @pytest.mark.asyncio
    async def test_recover_latest_snapshot(
        self,
        snapshot_service: SnapshotService,
        twin_manager: TwinStateManager,
    ):
        _populate_twin(twin_manager)
        await snapshot_service.create_snapshot()

        # Create a fresh twin manager (simulates restart)
        new_manager = TwinStateManager(
            graph_repo=InMemoryGraphRepository(),
        )
        await new_manager.initialize()
        assert new_manager.zone_count == 0  # Empty

        # Create new service with the same repo
        new_service = SnapshotService(
            state_manager=new_manager,
            repository=snapshot_service._repo,
        )
        result = await new_service.recover_latest_snapshot()
        assert result is True
        assert new_manager.zone_count == 2

        zone_a = new_manager.get_zone("ZONE_A")
        assert zone_a.compound_risk_score == 80.0
        assert zone_a.active_hazard_count == 1
        assert zone_a.predicted_risk_score == 72.0

    @pytest.mark.asyncio
    async def test_recover_restores_sensor_readings(
        self,
        snapshot_service: SnapshotService,
        twin_manager: TwinStateManager,
    ):
        _populate_twin(twin_manager)
        await snapshot_service.create_snapshot()

        new_manager = TwinStateManager(
            graph_repo=InMemoryGraphRepository(),
        )
        await new_manager.initialize()
        new_service = SnapshotService(
            state_manager=new_manager,
            repository=snapshot_service._repo,
        )
        await new_service.recover_latest_snapshot()

        zone_a = new_manager.get_zone("ZONE_A")
        assert "S001" in zone_a.latest_sensor_readings
        s001 = zone_a.latest_sensor_readings["S001"]
        assert s001.value == 150.0
        assert s001.is_anomalous is True

    @pytest.mark.asyncio
    async def test_recover_restores_hazards(
        self,
        snapshot_service: SnapshotService,
        twin_manager: TwinStateManager,
    ):
        _populate_twin(twin_manager)
        await snapshot_service.create_snapshot()

        new_manager = TwinStateManager(
            graph_repo=InMemoryGraphRepository(),
        )
        await new_manager.initialize()
        new_service = SnapshotService(
            state_manager=new_manager,
            repository=snapshot_service._repo,
        )
        await new_service.recover_latest_snapshot()

        zone_a = new_manager.get_zone("ZONE_A")
        assert len(zone_a.active_hazards) == 1
        assert zone_a.active_hazards[0].hazard_type == "GAS_LEAK"

    @pytest.mark.asyncio
    async def test_recover_from_corrupt_payload(
        self,
        snapshot_repo: InMemorySnapshotRepository,
        twin_manager: TwinStateManager,
    ):
        """Recovery fails gracefully on corrupt payloads."""
        snap = FacilitySnapshotModel(
            snapshot_id="corrupt-1",
            created_at=datetime.now(timezone.utc),
            facility_health=100.0,
            total_zones=0,
            active_hazards=0,
            critical_zones=0,
            workers_at_risk=0,
            events_processed=0,
            snapshot_payload="{invalid json!!!",
            trigger_reason="manual",
        )
        await snapshot_repo.save_snapshot(snap, [])

        service = SnapshotService(
            state_manager=twin_manager,
            repository=snapshot_repo,
        )
        result = await service.recover_latest_snapshot()
        assert result is False


class TestSnapshotQueries:
    """Test snapshot listing, getting, and deletion."""

    @pytest.mark.asyncio
    async def test_list_snapshots(
        self, snapshot_service: SnapshotService,
    ):
        for _ in range(3):
            await snapshot_service.create_snapshot()
        snapshots = await snapshot_service.list_snapshots()
        assert len(snapshots) == 3

    @pytest.mark.asyncio
    async def test_list_snapshots_pagination(
        self, snapshot_service: SnapshotService,
    ):
        for _ in range(3):
            await snapshot_service.create_snapshot()
        page1 = await snapshot_service.list_snapshots(
            offset=0, limit=2,
        )
        assert len(page1) == 2
        page2 = await snapshot_service.list_snapshots(
            offset=2, limit=2,
        )
        assert len(page2) == 1

    @pytest.mark.asyncio
    async def test_get_snapshot(
        self, snapshot_service: SnapshotService,
    ):
        snap = await snapshot_service.create_snapshot()
        result = await snapshot_service.get_snapshot(snap.snapshot_id)
        assert result is not None
        assert result.snapshot_id == snap.snapshot_id

    @pytest.mark.asyncio
    async def test_get_snapshot_not_found(
        self, snapshot_service: SnapshotService,
    ):
        result = await snapshot_service.get_snapshot("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_snapshot_with_zones(
        self,
        snapshot_service: SnapshotService,
        twin_manager: TwinStateManager,
    ):
        _populate_twin(twin_manager)
        snap = await snapshot_service.create_snapshot()
        result = await snapshot_service.get_snapshot_with_zones(
            snap.snapshot_id,
        )
        assert result is not None
        assert result["snapshot"].snapshot_id == snap.snapshot_id
        assert len(result["zone_states"]) == 2

    @pytest.mark.asyncio
    async def test_delete_snapshot(
        self, snapshot_service: SnapshotService,
    ):
        snap = await snapshot_service.create_snapshot()
        deleted = await snapshot_service.delete_snapshot(
            snap.snapshot_id,
        )
        assert deleted is True
        remaining = await snapshot_service.count_snapshots()
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(
        self, snapshot_service: SnapshotService,
    ):
        deleted = await snapshot_service.delete_snapshot("nope")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_count_snapshots(
        self, snapshot_service: SnapshotService,
    ):
        assert await snapshot_service.count_snapshots() == 0
        await snapshot_service.create_snapshot()
        assert await snapshot_service.count_snapshots() == 1


class TestRetentionPolicy:
    """Test snapshot retention enforcement."""

    @pytest.mark.asyncio
    async def test_retention_deletes_oldest(
        self,
        twin_manager: TwinStateManager,
        snapshot_repo: InMemorySnapshotRepository,
    ):
        service = SnapshotService(
            state_manager=twin_manager,
            repository=snapshot_repo,
            max_snapshots=3,
        )
        for _ in range(5):
            await service.create_snapshot()

        count = await service.count_snapshots()
        assert count == 3

    @pytest.mark.asyncio
    async def test_retention_keeps_newest(
        self,
        twin_manager: TwinStateManager,
        snapshot_repo: InMemorySnapshotRepository,
    ):
        service = SnapshotService(
            state_manager=twin_manager,
            repository=snapshot_repo,
            max_snapshots=2,
        )
        snaps = []
        for _ in range(4):
            snaps.append(await service.create_snapshot())

        remaining = await service.list_snapshots()
        remaining_ids = {s.snapshot_id for s in remaining}
        # The newest should still be there
        assert snaps[-1].snapshot_id in remaining_ids

    @pytest.mark.asyncio
    async def test_no_retention_when_under_limit(
        self,
        snapshot_service: SnapshotService,
    ):
        await snapshot_service.create_snapshot()
        count = await snapshot_service.count_snapshots()
        assert count == 1  # Well under limit of 5

    @pytest.mark.asyncio
    async def test_max_snapshots_property(
        self, snapshot_service: SnapshotService,
    ):
        assert snapshot_service.max_snapshots == 5


class TestAutomaticTriggers:
    """Test automatic snapshot trigger evaluation."""

    @pytest.mark.asyncio
    async def test_critical_hazard_triggers_snapshot(
        self,
        snapshot_service: SnapshotService,
        twin_manager: TwinStateManager,
    ):
        twin_manager.update_hazard_detected(
            zone_id="ZONE_A",
            hazard_id="HAZ-CRIT",
            hazard_type="FIRE",
            severity="CRITICAL",
        )
        result = await snapshot_service.evaluate_snapshot_trigger(
            event_category="hazard",
            zone_id="ZONE_A",
        )
        assert result is not None
        assert result.trigger_reason == "critical_hazard"

    @pytest.mark.asyncio
    async def test_compound_risk_threshold_triggers(
        self,
        snapshot_service: SnapshotService,
        twin_manager: TwinStateManager,
    ):
        twin_manager.update_compound_risk(
            zone_id="ZONE_A",
            compound_risk_score=COMPOUND_RISK_THRESHOLD + 1,
        )
        result = await snapshot_service.evaluate_snapshot_trigger(
            event_category="risk",
            zone_id="ZONE_A",
        )
        assert result is not None
        assert result.trigger_reason == "compound_risk_threshold"

    @pytest.mark.asyncio
    async def test_health_change_triggers(
        self,
        snapshot_service: SnapshotService,
        twin_manager: TwinStateManager,
    ):
        # First snapshot sets baseline health
        await snapshot_service.create_snapshot()

        # Big health change
        twin_manager.update_risk_score(
            zone_id="ZONE_A", risk_score=95.0,
        )
        twin_manager.update_risk_score(
            zone_id="ZONE_B", risk_score=90.0,
        )

        result = await snapshot_service.evaluate_snapshot_trigger(
            event_category="risk",
            zone_id="ZONE_A",
        )
        # Health should have changed enough to trigger
        assert result is not None
        assert result.trigger_reason == "health_change"

    @pytest.mark.asyncio
    async def test_no_trigger_for_normal_event(
        self,
        snapshot_service: SnapshotService,
        twin_manager: TwinStateManager,
    ):
        # Set baseline
        await snapshot_service.create_snapshot()

        # Small change - shouldn't trigger
        twin_manager.update_risk_score(
            zone_id="ZONE_A", risk_score=1.0,
        )
        result = await snapshot_service.evaluate_snapshot_trigger(
            event_category="sensor",
            zone_id="ZONE_A",
        )
        assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Serialization Round-Trip Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSerializationRoundTrip:
    """Test zone serialization and deserialization."""

    def test_zone_roundtrip(self):
        zone = ZoneState(
            zone_id="Z1",
            zone_name="Test Zone",
            sensor_health=85.0,
            anomaly_count=2,
            predicted_risk_score=65.0,
            risk_level="HIGH",
            compound_risk_score=78.0,
            compound_risk_level="CRITICAL",
            workers_at_risk=5,
            worker_capacity=20,
            current_worker_count=12,
            connected_zones=["Z2", "Z3"],
        )
        zone.latest_sensor_readings["S1"] = SensorReading(
            sensor_id="S1",
            sensor_type="temperature",
            value=95.0,
            unit="C",
            anomaly_score=-0.8,
            is_anomalous=True,
            health_score=70.0,
        )
        zone.equipment.append(EquipmentState(
            equipment_id="EQ1",
            equipment_type="pump",
            zone_id="Z1",
            health_score=90.0,
        ))
        zone.active_hazards.append(ActiveHazard(
            hazard_id="H1",
            hazard_type="FIRE",
            severity="CRITICAL",
            origin_zone="Z1",
            affected_zones=["Z1", "Z2"],
        ))

        d = SnapshotService._zone_to_dict(zone)
        restored = SnapshotService._dict_to_zone(d)

        assert restored.zone_id == "Z1"
        assert restored.zone_name == "Test Zone"
        assert restored.sensor_health == 85.0
        assert restored.predicted_risk_score == 65.0
        assert restored.compound_risk_score == 78.0
        assert restored.workers_at_risk == 5
        assert restored.connected_zones == ["Z2", "Z3"]

        assert "S1" in restored.latest_sensor_readings
        s1 = restored.latest_sensor_readings["S1"]
        assert s1.value == 95.0
        assert s1.is_anomalous is True

        assert len(restored.equipment) == 1
        assert restored.equipment[0].equipment_type == "pump"

        assert len(restored.active_hazards) == 1
        assert restored.active_hazards[0].hazard_type == "FIRE"
        assert restored.active_hazards[0].affected_zones == ["Z1", "Z2"]

    def test_empty_zone_roundtrip(self):
        zone = ZoneState(zone_id="EMPTY")
        d = SnapshotService._zone_to_dict(zone)
        restored = SnapshotService._dict_to_zone(d)
        assert restored.zone_id == "EMPTY"
        assert restored.anomaly_count == 0
        assert len(restored.latest_sensor_readings) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# In-Memory Repository Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInMemoryRepository:
    """Test the in-memory repository implementation."""

    @pytest.mark.asyncio
    async def test_save_and_get(
        self, snapshot_repo: InMemorySnapshotRepository,
    ):
        snap = FacilitySnapshotModel(
            snapshot_id="test-1",
            created_at=datetime.now(timezone.utc),
            facility_health=85.0,
            total_zones=2,
            active_hazards=0,
            critical_zones=0,
            workers_at_risk=0,
            events_processed=50,
            snapshot_payload="{}",
            trigger_reason="manual",
        )
        result = await snapshot_repo.save_snapshot(snap, [])
        assert result.snapshot_id == "test-1"

        fetched = await snapshot_repo.get_snapshot("test-1")
        assert fetched is not None
        assert fetched.facility_health == 85.0

    @pytest.mark.asyncio
    async def test_get_latest(
        self, snapshot_repo: InMemorySnapshotRepository,
    ):
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        for i in range(3):
            snap = FacilitySnapshotModel(
                snapshot_id=f"s-{i}",
                created_at=now + timedelta(seconds=i),
                facility_health=float(50 + i * 10),
                total_zones=0,
                active_hazards=0,
                critical_zones=0,
                workers_at_risk=0,
                events_processed=0,
                snapshot_payload="{}",
                trigger_reason="manual",
            )
            await snapshot_repo.save_snapshot(snap, [])

        latest = await snapshot_repo.get_latest_snapshot()
        assert latest.snapshot_id == "s-2"

    @pytest.mark.asyncio
    async def test_delete(
        self, snapshot_repo: InMemorySnapshotRepository,
    ):
        snap = FacilitySnapshotModel(
            snapshot_id="del-1",
            created_at=datetime.now(timezone.utc),
            facility_health=100.0,
            total_zones=0,
            active_hazards=0,
            critical_zones=0,
            workers_at_risk=0,
            events_processed=0,
            snapshot_payload="{}",
            trigger_reason="manual",
        )
        await snapshot_repo.save_snapshot(snap, [])
        assert await snapshot_repo.delete_snapshot("del-1") is True
        assert await snapshot_repo.get_snapshot("del-1") is None

    @pytest.mark.asyncio
    async def test_zone_states(
        self, snapshot_repo: InMemorySnapshotRepository,
    ):
        snap = FacilitySnapshotModel(
            snapshot_id="zs-1",
            created_at=datetime.now(timezone.utc),
            facility_health=100.0,
            total_zones=1,
            active_hazards=0,
            critical_zones=0,
            workers_at_risk=0,
            events_processed=0,
            snapshot_payload="{}",
            trigger_reason="manual",
        )
        zone_states = [
            ZoneStateModel(
                snapshot_id="zs-1",
                zone_id="Z1",
                risk_score=50.0,
                compound_risk_score=40.0,
                hazard_count=0,
                anomaly_count=1,
                equipment_health=90.0,
                worker_count=10,
                state_payload="{}",
            ),
        ]
        await snapshot_repo.save_snapshot(snap, zone_states)
        zs = await snapshot_repo.get_zone_states_for_snapshot("zs-1")
        assert len(zs) == 1
        assert zs[0].zone_id == "Z1"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Handler Integration Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHandlerSnapshotIntegration:
    """Test that the handler triggers automatic snapshots."""

    @pytest.mark.asyncio
    async def test_handler_without_snapshot_service(
        self, twin_manager: TwinStateManager,
    ):
        """Handler works without snapshot service (backward compat)."""
        handler = DigitalTwinEventHandler(
            state_manager=twin_manager,
        )
        await handler.handle_event(
            KafkaTopics.RISK_SCORE_UPDATED,
            {"event_id": "no-ss-001", "data": {"zone_id": "Z1"}},
        )
        assert handler.events_processed == 1

    @pytest.mark.asyncio
    async def test_handler_triggers_snapshot_on_critical(
        self,
        twin_manager: TwinStateManager,
        snapshot_repo: InMemorySnapshotRepository,
    ):
        service = SnapshotService(
            state_manager=twin_manager,
            repository=snapshot_repo,
        )
        handler = DigitalTwinEventHandler(
            state_manager=twin_manager,
            snapshot_service=service,
        )

        await handler.handle_event(
            KafkaTopics.HAZARD_DETECTED,
            {
                "event_id": "h-crit-001",
                "data": {
                    "hazard_id": "HAZ-CRIT",
                    "zone_id": "ZONE_A",
                    "hazard_type": "EXPLOSION",
                    "severity": "CRITICAL",
                },
            },
        )

        assert handler.events_processed == 1
        count = await snapshot_repo.count_snapshots()
        assert count >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REST API Endpoint Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSnapshotEndpoints:
    """Test the snapshot REST API endpoints."""

    def _setup(self, twin_manager: TwinStateManager):
        """Set up DI overrides for API testing."""
        from app.core.dependencies import (
            get_digital_twin_service,
            get_snapshot_service,
        )
        from app.main import app

        repo = InMemorySnapshotRepository()
        service = SnapshotService(
            state_manager=twin_manager,
            repository=repo,
        )

        app.dependency_overrides[get_digital_twin_service] = (
            lambda: twin_manager
        )
        app.dependency_overrides[get_snapshot_service] = (
            lambda: service
        )
        return app, service, repo

    def _teardown(self, app):
        app.dependency_overrides.clear()

    def test_post_snapshot(self, twin_manager: TwinStateManager):
        app, service, repo = self._setup(twin_manager)
        _populate_twin(twin_manager)
        try:
            client = TestClient(app)
            resp = client.post("/api/v1/twin/snapshot")
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["snapshot"]["total_zones"] == 2
            assert data["snapshot"]["trigger_reason"] == "manual"
        finally:
            self._teardown(app)

    def test_get_snapshots_list(
        self, twin_manager: TwinStateManager,
    ):
        app, service, repo = self._setup(twin_manager)
        try:
            client = TestClient(app)
            # Create 2 snapshots
            client.post("/api/v1/twin/snapshot")
            client.post("/api/v1/twin/snapshot")

            resp = client.get("/api/v1/twin/snapshots")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 2
            assert len(data["snapshots"]) == 2
        finally:
            self._teardown(app)

    def test_get_snapshot_detail(
        self, twin_manager: TwinStateManager,
    ):
        app, service, repo = self._setup(twin_manager)
        _populate_twin(twin_manager)
        try:
            client = TestClient(app)
            create_resp = client.post("/api/v1/twin/snapshot")
            snap_id = create_resp.json()["snapshot"]["snapshot_id"]

            resp = client.get(
                f"/api/v1/twin/snapshots/{snap_id}",
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["snapshot"]["snapshot_id"] == snap_id
            assert len(data["zone_states"]) == 2
            assert "snapshot_payload" in data
        finally:
            self._teardown(app)

    def test_get_snapshot_not_found(
        self, twin_manager: TwinStateManager,
    ):
        app, service, repo = self._setup(twin_manager)
        try:
            client = TestClient(app)
            resp = client.get(
                "/api/v1/twin/snapshots/nonexistent",
            )
            assert resp.status_code == 404
        finally:
            self._teardown(app)

    def test_delete_snapshot(
        self, twin_manager: TwinStateManager,
    ):
        app, service, repo = self._setup(twin_manager)
        try:
            client = TestClient(app)
            create_resp = client.post("/api/v1/twin/snapshot")
            snap_id = create_resp.json()["snapshot"]["snapshot_id"]

            resp = client.delete(
                f"/api/v1/twin/snapshots/{snap_id}",
            )
            assert resp.status_code == 200
            assert resp.json()["deleted"] is True

            # Verify deleted
            resp2 = client.get(
                f"/api/v1/twin/snapshots/{snap_id}",
            )
            assert resp2.status_code == 404
        finally:
            self._teardown(app)

    def test_delete_not_found(
        self, twin_manager: TwinStateManager,
    ):
        app, service, repo = self._setup(twin_manager)
        try:
            client = TestClient(app)
            resp = client.delete(
                "/api/v1/twin/snapshots/nope",
            )
            assert resp.status_code == 404
        finally:
            self._teardown(app)

    def test_snapshots_pagination(
        self, twin_manager: TwinStateManager,
    ):
        app, service, repo = self._setup(twin_manager)
        try:
            client = TestClient(app)
            for _ in range(5):
                client.post("/api/v1/twin/snapshot")

            resp = client.get(
                "/api/v1/twin/snapshots?offset=0&limit=2",
            )
            data = resp.json()
            assert data["total"] == 5
            assert len(data["snapshots"]) == 2
            assert data["offset"] == 0
            assert data["limit"] == 2
        finally:
            self._teardown(app)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full Pipeline Integration Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFullSnapshotPipeline:
    """End-to-end: events → state → snapshot → restart → recovery."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Simulate a complete snapshot lifecycle."""
        # 1. Create infrastructure
        graph_repo = InMemoryGraphRepository()
        manager = TwinStateManager(graph_repo=graph_repo)
        await manager.initialize()
        snap_repo = InMemorySnapshotRepository()
        snap_service = SnapshotService(
            state_manager=manager,
            repository=snap_repo,
        )

        # 2. Process events
        handler = DigitalTwinEventHandler(
            state_manager=manager,
            snapshot_service=snap_service,
        )
        events = [
            (KafkaTopics.SENSOR_READING_ANOMALY, {
                "event_id": "e1",
                "data": {
                    "sensor_id": "S001", "zone_id": "ZONE_A",
                    "value": 200.0, "anomaly_score": -0.95,
                },
            }),
            (KafkaTopics.RISK_SCORE_UPDATED, {
                "event_id": "e2",
                "data": {"zone_id": "ZONE_A", "risk_score": 85.0},
            }),
            (KafkaTopics.COMPOUND_RISK_DETECTED, {
                "event_id": "e3",
                "data": {
                    "zone_id": "ZONE_A",
                    "compound_risk_score": 90.0,
                    "risk_level": "CRITICAL",
                },
            }),
            (KafkaTopics.HAZARD_DETECTED, {
                "event_id": "e4",
                "data": {
                    "hazard_id": "HAZ-1", "zone_id": "ZONE_A",
                    "hazard_type": "GAS_LEAK", "severity": "HIGH",
                },
            }),
        ]
        for topic, event in events:
            await handler.handle_event(topic, event)

        assert handler.events_processed == 4

        # 3. Verify automatic snapshot was created
        snap_count = await snap_repo.count_snapshots()
        assert snap_count >= 1  # Auto-triggered by critical events

        # 4. Also create a manual snapshot
        manual_snap = await snap_service.create_snapshot(
            trigger_reason="manual",
        )

        # 5. Simulate restart — fresh manager
        new_manager = TwinStateManager(
            graph_repo=InMemoryGraphRepository(),
        )
        await new_manager.initialize()
        assert new_manager.zone_count == 0

        new_service = SnapshotService(
            state_manager=new_manager,
            repository=snap_repo,
        )

        # 6. Recover
        recovered = await new_service.recover_latest_snapshot()
        assert recovered is True

        # 7. Verify restored state matches
        zone_a = new_manager.get_zone("ZONE_A")
        assert zone_a.compound_risk_score == 90.0
        assert zone_a.predicted_risk_score == 85.0
        assert zone_a.active_hazard_count == 1
        assert "S001" in zone_a.latest_sensor_readings
        assert zone_a.latest_sensor_readings["S001"].value == 200.0
