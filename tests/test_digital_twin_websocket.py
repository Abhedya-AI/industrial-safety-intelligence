"""Comprehensive tests for Digital Twin Phase 2: WebSocket streaming.

Covers:
  - WebSocketBroadcaster: connect/disconnect, broadcast, multi-client,
    fault tolerance, facility snapshot, typed updates
  - WebSocket endpoint: /ws/twin connection lifecycle
  - Handler integration: event → broadcast pipeline
"""

from __future__ import annotations

import asyncio
import json
import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import WebSocket
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteTestClient
from starlette.websockets import WebSocketDisconnect

from app.digital_twin.domain.entities import ZoneState
from app.digital_twin.messaging.handler import DigitalTwinEventHandler
from app.digital_twin.services.twin_state_manager import TwinStateManager
from app.digital_twin.services.websocket_broadcaster import (
    WebSocketBroadcaster,
)
from app.hazard_propagation.graph.entities import ZoneNode
from app.hazard_propagation.repositories.in_memory_graph_repo import (
    InMemoryGraphRepository,
)
from app.shared.messaging.topics import KafkaTopics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def graph_repo() -> InMemoryGraphRepository:
    return InMemoryGraphRepository()


@pytest.fixture
def populated_graph_repo(graph_repo: InMemoryGraphRepository):
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

    asyncio.get_event_loop().run_until_complete(_populate())
    return graph_repo


@pytest.fixture
def twin_manager(graph_repo: InMemoryGraphRepository) -> TwinStateManager:
    manager = TwinStateManager(graph_repo=graph_repo)
    asyncio.get_event_loop().run_until_complete(manager.initialize())
    return manager


@pytest.fixture
def populated_twin(
    populated_graph_repo: InMemoryGraphRepository,
) -> TwinStateManager:
    manager = TwinStateManager(graph_repo=populated_graph_repo)
    asyncio.get_event_loop().run_until_complete(manager.initialize())
    return manager


@pytest.fixture
def broadcaster(twin_manager: TwinStateManager) -> WebSocketBroadcaster:
    return WebSocketBroadcaster(state_manager=twin_manager)


@pytest.fixture
def populated_broadcaster(
    populated_twin: TwinStateManager,
) -> WebSocketBroadcaster:
    return WebSocketBroadcaster(state_manager=populated_twin)


def _make_mock_ws(accept_ok: bool = True) -> AsyncMock:
    """Create a mock WebSocket."""
    ws = AsyncMock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    return ws


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WebSocketBroadcaster Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBroadcasterConnect:
    """Test connection management."""

    @pytest.mark.asyncio
    async def test_connect_accepts_websocket(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        ws.accept.assert_awaited_once()
        assert broadcaster.connection_count == 1

    @pytest.mark.asyncio
    async def test_connect_sends_snapshot(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        # Should have sent facility_snapshot
        ws.send_json.assert_awaited_once()
        snapshot = ws.send_json.call_args[0][0]
        assert snapshot["type"] == "facility_snapshot"
        assert "data" in snapshot
        assert "timestamp" in snapshot

    @pytest.mark.asyncio
    async def test_connect_snapshot_has_zones(
        self, populated_broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await populated_broadcaster.connect(ws)
        snapshot = ws.send_json.call_args[0][0]
        assert snapshot["data"]["total_zones"] == 2
        assert len(snapshot["data"]["zones"]) == 2

    @pytest.mark.asyncio
    async def test_multiple_connects(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws1 = _make_mock_ws()
        ws2 = _make_mock_ws()
        ws3 = _make_mock_ws()
        await broadcaster.connect(ws1)
        await broadcaster.connect(ws2)
        await broadcaster.connect(ws3)
        assert broadcaster.connection_count == 3
        assert broadcaster.total_connections == 3

    @pytest.mark.asyncio
    async def test_connect_increments_total(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        assert broadcaster.total_connections == 1


class TestBroadcasterDisconnect:
    """Test disconnection handling."""

    @pytest.mark.asyncio
    async def test_disconnect_removes_client(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        assert broadcaster.connection_count == 1
        await broadcaster.disconnect(ws)
        assert broadcaster.connection_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_unknown_client_safe(
        self, broadcaster: WebSocketBroadcaster,
    ):
        """Disconnecting an unknown client doesn't crash."""
        ws = _make_mock_ws()
        await broadcaster.disconnect(ws)  # Not connected
        assert broadcaster.connection_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_one_keeps_others(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws1 = _make_mock_ws()
        ws2 = _make_mock_ws()
        await broadcaster.connect(ws1)
        await broadcaster.connect(ws2)
        await broadcaster.disconnect(ws1)
        assert broadcaster.connection_count == 1


class TestBroadcast:
    """Test broadcast delivery."""

    @pytest.mark.asyncio
    async def test_broadcast_to_all_clients(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws1 = _make_mock_ws()
        ws2 = _make_mock_ws()
        await broadcaster.connect(ws1)
        await broadcaster.connect(ws2)

        # Reset send_json (connect sends snapshot)
        ws1.send_json.reset_mock()
        ws2.send_json.reset_mock()

        await broadcaster.broadcast({"type": "test", "data": "hello"})
        ws1.send_json.assert_awaited_once()
        ws2.send_json.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_broadcast_adds_timestamp(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await broadcaster.broadcast({"type": "test"})
        sent = ws.send_json.call_args[0][0]
        assert "timestamp" in sent

    @pytest.mark.asyncio
    async def test_broadcast_no_clients_is_noop(
        self, broadcaster: WebSocketBroadcaster,
    ):
        # Should not raise
        await broadcaster.broadcast({"type": "test"})
        assert broadcaster.messages_sent == 0

    @pytest.mark.asyncio
    async def test_broadcast_increments_messages_sent(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        initial = broadcaster.messages_sent  # snapshot message
        ws.send_json.reset_mock()

        await broadcaster.broadcast({"type": "test"})
        assert broadcaster.messages_sent == initial + 1


class TestBroadcastFaultTolerance:
    """Test that failing clients are removed without affecting others."""

    @pytest.mark.asyncio
    async def test_failing_client_auto_removed(
        self, broadcaster: WebSocketBroadcaster,
    ):
        from app.digital_twin.services.websocket_broadcaster import (
            _ClientState,
        )
        ws_good = _make_mock_ws()
        ws_bad = _make_mock_ws()
        ws_bad.send_json = AsyncMock(
            side_effect=RuntimeError("connection lost"),
        )

        # Connect both (ws_bad will fail on snapshot too, let it pass)
        await broadcaster.connect(ws_good)
        # Manually add ws_bad after connect
        broadcaster._clients[ws_bad] = _ClientState(websocket=ws_bad)
        assert broadcaster.connection_count == 2

        await broadcaster.broadcast({"type": "test"})

        # ws_bad should be removed, ws_good stays
        assert broadcaster.connection_count == 1
        assert broadcaster.messages_failed > 0

    @pytest.mark.asyncio
    async def test_good_client_still_receives_after_bad_removed(
        self, broadcaster: WebSocketBroadcaster,
    ):
        from app.digital_twin.services.websocket_broadcaster import (
            _ClientState,
        )
        ws_good = _make_mock_ws()
        ws_bad = _make_mock_ws()
        ws_bad.send_json = AsyncMock(
            side_effect=RuntimeError("disconnect"),
        )

        await broadcaster.connect(ws_good)
        broadcaster._clients[ws_bad] = _ClientState(websocket=ws_bad)
        ws_good.send_json.reset_mock()

        await broadcaster.broadcast({"type": "test_msg"})
        ws_good.send_json.assert_awaited_once()
        sent = ws_good.send_json.call_args[0][0]
        assert sent["type"] == "test_msg"


class TestBroadcastZoneUpdate:
    """Test zone update broadcasts."""

    @pytest.mark.asyncio
    async def test_zone_update_message(
        self, populated_broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await populated_broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await populated_broadcaster.broadcast_zone_update("ZONE_A")

        ws.send_json.assert_awaited_once()
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "zone_update"
        assert msg["zone_id"] == "ZONE_A"
        assert "data" in msg
        assert msg["data"]["zone_id"] == "ZONE_A"

    @pytest.mark.asyncio
    async def test_zone_update_with_risk(
        self, populated_broadcaster: WebSocketBroadcaster,
    ):
        populated_broadcaster._state.update_risk_score(
            zone_id="ZONE_A", risk_score=85.0,
        )
        ws = _make_mock_ws()
        await populated_broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await populated_broadcaster.broadcast_zone_update("ZONE_A")
        msg = ws.send_json.call_args[0][0]
        assert msg["data"]["predicted_risk_score"] == 85.0
        assert msg["data"]["heatmap_color"] == "red"
        assert msg["data"]["is_critical"] is True


class TestBroadcastFacilityUpdate:
    """Test facility update broadcasts."""

    @pytest.mark.asyncio
    async def test_facility_update_message(
        self, populated_broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await populated_broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await populated_broadcaster.broadcast_facility_update()

        ws.send_json.assert_awaited_once()
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "facility_update"
        assert "data" in msg
        assert msg["data"]["total_zones"] == 2


class TestBroadcastSensorUpdate:
    """Test sensor-specific broadcasts."""

    @pytest.mark.asyncio
    async def test_sensor_update_message(
        self, broadcaster: WebSocketBroadcaster,
    ):
        broadcaster._state.update_sensor_anomaly(
            zone_id="ZONE_A",
            sensor_id="S001",
            sensor_type="gas",
            value=150.0,
            unit="ppm",
            anomaly_score=-0.9,
        )
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await broadcaster.broadcast_sensor_update("ZONE_A", "S001")

        ws.send_json.assert_awaited_once()
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "sensor_update"
        assert msg["zone_id"] == "ZONE_A"
        assert msg["data"]["sensor_id"] == "S001"
        assert msg["data"]["value"] == 150.0
        assert msg["data"]["is_anomalous"] is True


class TestBroadcastRiskUpdate:
    """Test risk-specific broadcasts."""

    @pytest.mark.asyncio
    async def test_risk_update_message(
        self, broadcaster: WebSocketBroadcaster,
    ):
        broadcaster._state.update_risk_score(
            zone_id="ZONE_A", risk_score=72.0,
        )
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await broadcaster.broadcast_risk_update("ZONE_A")

        ws.send_json.assert_awaited_once()
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "risk_update"
        assert msg["zone_id"] == "ZONE_A"
        assert msg["data"]["predicted_risk_score"] == 72.0
        assert msg["data"]["risk_level"] == "HIGH"


class TestBroadcastHazardUpdate:
    """Test hazard-specific broadcasts."""

    @pytest.mark.asyncio
    async def test_hazard_update_message(
        self, broadcaster: WebSocketBroadcaster,
    ):
        broadcaster._state.update_hazard_detected(
            zone_id="ZONE_A",
            hazard_id="HAZ-001",
            hazard_type="GAS_LEAK",
            severity="HIGH",
        )
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await broadcaster.broadcast_hazard_update("ZONE_A")

        ws.send_json.assert_awaited_once()
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "hazard_update"
        assert msg["zone_id"] == "ZONE_A"
        assert msg["data"]["active_hazard_count"] == 1
        assert len(msg["data"]["active_hazards"]) == 1
        assert msg["data"]["active_hazards"][0]["hazard_type"] == "GAS_LEAK"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Handler + Broadcaster Integration Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHandlerBroadcastIntegration:
    """Test that events processed by the handler trigger broadcasts."""

    @pytest.mark.asyncio
    async def test_handler_without_broadcaster_still_works(
        self, twin_manager: TwinStateManager,
    ):
        """Phase 1 compatibility: handler works without broadcaster."""
        handler = DigitalTwinEventHandler(state_manager=twin_manager)
        await handler.handle_event(
            KafkaTopics.RISK_SCORE_UPDATED,
            {
                "event_id": "no-bc-001",
                "data": {"zone_id": "Z1", "risk_score": 50.0},
            },
        )
        assert handler.events_processed == 1

    @pytest.mark.asyncio
    async def test_handler_with_broadcaster_broadcasts(
        self, twin_manager: TwinStateManager,
    ):
        bc = WebSocketBroadcaster(state_manager=twin_manager)
        ws = _make_mock_ws()
        await bc.connect(ws)
        ws.send_json.reset_mock()

        handler = DigitalTwinEventHandler(
            state_manager=twin_manager, broadcaster=bc,
        )
        await handler.handle_event(
            KafkaTopics.RISK_SCORE_UPDATED,
            {
                "event_id": "bc-001",
                "data": {"zone_id": "ZONE_A", "risk_score": 65.0},
            },
        )
        assert handler.events_processed == 1
        # Should have broadcast risk_update + zone_update + facility_update
        assert ws.send_json.await_count >= 3

    @pytest.mark.asyncio
    async def test_sensor_event_broadcasts_sensor_update(
        self, twin_manager: TwinStateManager,
    ):
        bc = WebSocketBroadcaster(state_manager=twin_manager)
        ws = _make_mock_ws()
        await bc.connect(ws)
        ws.send_json.reset_mock()

        handler = DigitalTwinEventHandler(
            state_manager=twin_manager, broadcaster=bc,
        )
        await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY,
            {
                "event_id": "s-001",
                "data": {
                    "sensor_id": "S001",
                    "zone_id": "ZONE_A",
                    "value": 200.0,
                    "anomaly_score": -0.9,
                },
            },
        )

        # Collect all sent message types
        types = [
            call[0][0]["type"]
            for call in ws.send_json.call_args_list
        ]
        assert "sensor_update" in types
        assert "zone_update" in types
        assert "facility_update" in types

    @pytest.mark.asyncio
    async def test_hazard_event_broadcasts_hazard_update(
        self, twin_manager: TwinStateManager,
    ):
        bc = WebSocketBroadcaster(state_manager=twin_manager)
        ws = _make_mock_ws()
        await bc.connect(ws)
        ws.send_json.reset_mock()

        handler = DigitalTwinEventHandler(
            state_manager=twin_manager, broadcaster=bc,
        )
        await handler.handle_event(
            KafkaTopics.HAZARD_DETECTED,
            {
                "event_id": "h-001",
                "data": {
                    "hazard_id": "HAZ-001",
                    "zone_id": "ZONE_A",
                    "hazard_type": "FIRE",
                    "severity": "CRITICAL",
                },
            },
        )

        types = [
            call[0][0]["type"]
            for call in ws.send_json.call_args_list
        ]
        assert "hazard_update" in types
        assert "zone_update" in types
        assert "facility_update" in types

    @pytest.mark.asyncio
    async def test_compound_risk_broadcasts_risk_update(
        self, twin_manager: TwinStateManager,
    ):
        bc = WebSocketBroadcaster(state_manager=twin_manager)
        ws = _make_mock_ws()
        await bc.connect(ws)
        ws.send_json.reset_mock()

        handler = DigitalTwinEventHandler(
            state_manager=twin_manager, broadcaster=bc,
        )
        await handler.handle_event(
            KafkaTopics.COMPOUND_RISK_DETECTED,
            {
                "event_id": "cr-001",
                "data": {
                    "zone_id": "ZONE_A",
                    "compound_risk_score": 88.0,
                    "risk_level": "CRITICAL",
                },
            },
        )

        types = [
            call[0][0]["type"]
            for call in ws.send_json.call_args_list
        ]
        assert "risk_update" in types

    @pytest.mark.asyncio
    async def test_no_broadcast_when_zero_clients(
        self, twin_manager: TwinStateManager,
    ):
        bc = WebSocketBroadcaster(state_manager=twin_manager)
        # No clients connected
        handler = DigitalTwinEventHandler(
            state_manager=twin_manager, broadcaster=bc,
        )
        await handler.handle_event(
            KafkaTopics.RISK_SCORE_UPDATED,
            {
                "event_id": "no-client-001",
                "data": {"zone_id": "Z1", "risk_score": 50.0},
            },
        )
        # Processed but no broadcast
        assert handler.events_processed == 1
        assert bc.messages_sent == 0

    @pytest.mark.asyncio
    async def test_broadcaster_setter(
        self, twin_manager: TwinStateManager,
    ):
        handler = DigitalTwinEventHandler(state_manager=twin_manager)
        assert handler.broadcaster is None

        bc = WebSocketBroadcaster(state_manager=twin_manager)
        handler.broadcaster = bc
        assert handler.broadcaster is bc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WebSocket Endpoint Tests (via TestClient)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestWebSocketEndpoint:
    """Test the WS /ws/twin endpoint via FastAPI TestClient."""

    def test_websocket_connect_and_receive_snapshot(
        self, populated_twin: TwinStateManager,
    ):
        from app.digital_twin.api.websocket_endpoints import set_broadcaster
        from app.main import app

        bc = WebSocketBroadcaster(state_manager=populated_twin)
        set_broadcaster(bc)

        client = TestClient(app)
        with client.websocket_connect("/api/v1/ws/twin") as ws:
            data = ws.receive_json()
            assert data["type"] == "facility_snapshot"
            assert data["data"]["total_zones"] == 2
            assert len(data["data"]["zones"]) == 2
            assert "facility_health" in data["data"]

        # Reset
        set_broadcaster(None)

    def test_websocket_snapshot_has_zone_details(
        self, populated_twin: TwinStateManager,
    ):
        from app.digital_twin.api.websocket_endpoints import set_broadcaster
        from app.main import app

        populated_twin.update_risk_score(
            zone_id="ZONE_A", risk_score=75.0,
        )

        bc = WebSocketBroadcaster(state_manager=populated_twin)
        set_broadcaster(bc)

        client = TestClient(app)
        with client.websocket_connect("/api/v1/ws/twin") as ws:
            snapshot = ws.receive_json()
            zone_a = next(
                z for z in snapshot["data"]["zones"]
                if z["zone_id"] == "ZONE_A"
            )
            assert zone_a["predicted_risk_score"] == 75.0
            assert zone_a["heatmap_color"] == "orange"

        set_broadcaster(None)

    def test_websocket_multiple_clients(
        self, populated_twin: TwinStateManager,
    ):
        from app.digital_twin.api.websocket_endpoints import set_broadcaster
        from app.main import app

        bc = WebSocketBroadcaster(state_manager=populated_twin)
        set_broadcaster(bc)

        client = TestClient(app)
        with client.websocket_connect("/api/v1/ws/twin") as ws1:
            snapshot1 = ws1.receive_json()
            assert snapshot1["type"] == "facility_snapshot"

            with client.websocket_connect("/api/v1/ws/twin") as ws2:
                snapshot2 = ws2.receive_json()
                assert snapshot2["type"] == "facility_snapshot"
                assert bc.connection_count == 2

            # ws2 disconnected
            # Give async disconnect a moment
            assert bc.connection_count >= 1

        set_broadcaster(None)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full Pipeline Integration Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFullPipeline:
    """End-to-end: Kafka event → handler → state → broadcaster → WS."""

    @pytest.mark.asyncio
    async def test_event_to_websocket_pipeline(
        self, populated_twin: TwinStateManager,
    ):
        """Simulate a complete event flow from Kafka to WebSocket."""
        bc = WebSocketBroadcaster(state_manager=populated_twin)
        handler = DigitalTwinEventHandler(
            state_manager=populated_twin, broadcaster=bc,
        )

        # Connect a mock client
        ws = _make_mock_ws()
        await bc.connect(ws)
        ws.send_json.reset_mock()

        # Simulate sequential events
        events = [
            (KafkaTopics.SENSOR_READING_ANOMALY, {
                "event_id": "pipe-001",
                "data": {
                    "sensor_id": "S001", "zone_id": "ZONE_A",
                    "value": 180.0, "anomaly_score": -0.95,
                },
            }),
            (KafkaTopics.RISK_SCORE_UPDATED, {
                "event_id": "pipe-002",
                "data": {
                    "zone_id": "ZONE_A", "risk_score": 82.0,
                },
            }),
            (KafkaTopics.COMPOUND_RISK_DETECTED, {
                "event_id": "pipe-003",
                "data": {
                    "zone_id": "ZONE_A",
                    "compound_risk_score": 90.0,
                    "risk_level": "CRITICAL",
                },
            }),
            (KafkaTopics.HAZARD_PROPAGATED, {
                "event_id": "pipe-004",
                "data": {
                    "propagation_id": "P-001",
                    "origin_zone": "ZONE_A",
                    "hazard_type": "GAS_LEAK",
                    "propagation_level": "SPREADING",
                    "affected_zones": ["ZONE_A", "ZONE_B"],
                },
            }),
        ]

        for topic, event in events:
            await handler.handle_event(topic, event)

        assert handler.events_processed == 4

        # Verify all message types were broadcast
        all_types = set()
        for call in ws.send_json.call_args_list:
            msg = call[0][0]
            all_types.add(msg["type"])

        assert "sensor_update" in all_types
        assert "risk_update" in all_types
        assert "hazard_update" in all_types
        assert "zone_update" in all_types
        assert "facility_update" in all_types

        # Verify final state is correct
        zone_a = populated_twin.get_zone("ZONE_A")
        assert zone_a.compound_risk_score == 90.0
        assert zone_a.is_critical is True

        # Verify WebSocket client received many messages
        assert ws.send_json.await_count >= 12  # 4 events × 3 broadcasts each
