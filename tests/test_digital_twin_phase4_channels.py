"""Comprehensive tests for Digital Twin Phase 4: Channel Subscriptions & Enhanced WebSocket.

Covers:
  - Channel-based subscriptions (subscribe / unsubscribe)
  - Subscription filtering (only subscribed channels receive events)
  - Broadcast delivery per channel
  - Disconnect cleanup (subscriptions removed)
  - Malformed message handling
  - Heartbeat / ping-pong
  - Concurrent clients with different subscriptions
  - Client message handling (all actions)
  - get_subscriptions action
  - Channel metadata in broadcast messages
  - Default subscription (all channels)
  - Handler integration with channel-aware broadcasts
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

from app.digital_twin.domain.entities import ZoneState
from app.digital_twin.domain.enums import WsChannel
from app.digital_twin.messaging.handler import DigitalTwinEventHandler
from app.digital_twin.services.twin_state_manager import TwinStateManager
from app.digital_twin.services.websocket_broadcaster import (
    CATEGORY_TO_CHANNEL,
    CHANNELS,
    DEFAULT_CHANNELS,
    TYPE_TO_CHANNEL,
    WebSocketBroadcaster,
    _ClientState,
)
from app.hazard_propagation.repositories.in_memory_graph_repo import (
    InMemoryGraphRepository,
)
from app.shared.messaging.topics import KafkaTopics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_mock_ws() -> MagicMock:
    """Create a mock WebSocket that behaves like a real one."""
    ws = MagicMock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    ws.receive_text = AsyncMock(return_value='{"action":"ping"}')
    return ws


def _get_sent_messages(ws: MagicMock) -> List[Dict[str, Any]]:
    """Extract all messages sent to a mock WebSocket."""
    return [call[0][0] for call in ws.send_json.call_args_list]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def graph_repo() -> InMemoryGraphRepository:
    return InMemoryGraphRepository()


@pytest.fixture
def twin_manager(graph_repo):
    manager = TwinStateManager(graph_repo=graph_repo)
    asyncio.get_event_loop().run_until_complete(manager.initialize())
    return manager


@pytest.fixture
def broadcaster(twin_manager) -> WebSocketBroadcaster:
    return WebSocketBroadcaster(state_manager=twin_manager)


@pytest.fixture
def populated_broadcaster(twin_manager) -> WebSocketBroadcaster:
    """Broadcaster with pre-populated state."""
    twin_manager.update_sensor_anomaly(
        zone_id="ZONE_A", sensor_id="S001",
        sensor_type="gas", value=150.0,
        unit="ppm", anomaly_score=-0.9,
    )
    twin_manager.update_risk_score(
        zone_id="ZONE_A", risk_score=75.0,
    )
    twin_manager.update_compound_risk(
        zone_id="ZONE_A",
        compound_risk_score=80.0,
        risk_level="CRITICAL",
    )
    twin_manager.update_hazard_detected(
        zone_id="ZONE_A",
        hazard_id="HAZ-001",
        hazard_type="GAS_LEAK",
        severity="HIGH",
    )
    return WebSocketBroadcaster(state_manager=twin_manager)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Channel Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestChannelConstants:
    """Test channel configuration constants."""

    def test_all_channels_defined(self):
        assert "risk-updates" in CHANNELS
        assert "sensor-data" in CHANNELS
        assert "hazard-events" in CHANNELS
        assert "twin-state" in CHANNELS
        assert len(CHANNELS) == 4

    def test_default_channels_include_all(self):
        assert DEFAULT_CHANNELS == CHANNELS

    def test_category_to_channel_mapping(self):
        assert CATEGORY_TO_CHANNEL["sensor"] == "sensor-data"
        assert CATEGORY_TO_CHANNEL["risk"] == "risk-updates"
        assert CATEGORY_TO_CHANNEL["hazard"] == "hazard-events"

    def test_type_to_channel_mapping(self):
        assert TYPE_TO_CHANNEL["sensor_update"] == "sensor-data"
        assert TYPE_TO_CHANNEL["risk_update"] == "risk-updates"
        assert TYPE_TO_CHANNEL["hazard_update"] == "hazard-events"
        assert TYPE_TO_CHANNEL["zone_update"] == "twin-state"
        assert TYPE_TO_CHANNEL["facility_update"] == "twin-state"

    def test_ws_channel_enum(self):
        assert WsChannel.RISK_UPDATES.value == "risk-updates"
        assert WsChannel.SENSOR_DATA.value == "sensor-data"
        assert WsChannel.HAZARD_EVENTS.value == "hazard-events"
        assert WsChannel.TWIN_STATE.value == "twin-state"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Default Subscription
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDefaultSubscription:
    """Test that clients are subscribed to all channels by default."""

    @pytest.mark.asyncio
    async def test_new_client_subscribed_to_all(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        subs = broadcaster.get_subscriptions(ws)
        assert subs == set(CHANNELS)

    @pytest.mark.asyncio
    async def test_default_receives_all_channels(
        self, populated_broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await populated_broadcaster.connect(ws)
        ws.send_json.reset_mock()

        # Sensor
        await populated_broadcaster.broadcast_sensor_update(
            "ZONE_A", "S001",
        )
        # Risk
        await populated_broadcaster.broadcast_risk_update("ZONE_A")
        # Hazard
        await populated_broadcaster.broadcast_hazard_update("ZONE_A")
        # Twin state
        await populated_broadcaster.broadcast_zone_update("ZONE_A")

        assert ws.send_json.await_count == 4


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Subscribe / Unsubscribe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSubscribeUnsubscribe:
    """Test channel subscription management."""

    @pytest.mark.asyncio
    async def test_subscribe_adds_channels(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        # Unsubscribe from all first
        await broadcaster.unsubscribe(ws, list(CHANNELS))
        assert broadcaster.get_subscriptions(ws) == set()

        # Subscribe to specific channels
        resp = await broadcaster.subscribe(
            ws, ["risk-updates", "sensor-data"],
        )
        assert resp["type"] == "subscribed"
        assert set(resp["channels"]) == {
            "risk-updates", "sensor-data",
        }
        assert broadcaster.get_subscriptions(ws) == {
            "risk-updates", "sensor-data",
        }

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_channels(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        resp = await broadcaster.unsubscribe(
            ws, ["hazard-events", "twin-state"],
        )
        assert resp["type"] == "unsubscribed"
        assert broadcaster.get_subscriptions(ws) == {
            "risk-updates", "sensor-data",
        }

    @pytest.mark.asyncio
    async def test_subscribe_invalid_channel_ignored(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        await broadcaster.unsubscribe(ws, list(CHANNELS))

        resp = await broadcaster.subscribe(
            ws, ["invalid-channel", "risk-updates"],
        )
        assert resp["type"] == "subscribed"
        assert broadcaster.get_subscriptions(ws) == {"risk-updates"}

    @pytest.mark.asyncio
    async def test_subscribe_unknown_client(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        resp = await broadcaster.subscribe(ws, ["risk-updates"])
        assert resp["type"] == "error"

    @pytest.mark.asyncio
    async def test_unsubscribe_unknown_client(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        resp = await broadcaster.unsubscribe(ws, ["risk-updates"])
        assert resp["type"] == "error"

    @pytest.mark.asyncio
    async def test_get_subscriptions_unknown_client(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        subs = broadcaster.get_subscriptions(ws)
        assert subs == set()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Subscription Filtering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSubscriptionFiltering:
    """Test that broadcasts are filtered by subscription."""

    @pytest.mark.asyncio
    async def test_unsubscribed_channel_not_received(
        self, populated_broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await populated_broadcaster.connect(ws)
        await populated_broadcaster.unsubscribe(
            ws, ["sensor-data"],
        )
        ws.send_json.reset_mock()

        # Sensor update should be filtered out
        await populated_broadcaster.broadcast_sensor_update(
            "ZONE_A", "S001",
        )
        ws.send_json.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_subscribed_channel_received(
        self, populated_broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await populated_broadcaster.connect(ws)
        await populated_broadcaster.unsubscribe(ws, list(CHANNELS))
        await populated_broadcaster.subscribe(ws, ["risk-updates"])
        ws.send_json.reset_mock()

        await populated_broadcaster.broadcast_risk_update("ZONE_A")
        assert ws.send_json.await_count == 1
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "risk_update"
        assert msg["channel"] == "risk-updates"

    @pytest.mark.asyncio
    async def test_mixed_subscriptions_two_clients(
        self, populated_broadcaster: WebSocketBroadcaster,
    ):
        """Two clients with different subscriptions."""
        ws_risk = _make_mock_ws()
        ws_sensor = _make_mock_ws()

        await populated_broadcaster.connect(ws_risk)
        await populated_broadcaster.connect(ws_sensor)

        # ws_risk: only risk-updates
        await populated_broadcaster.unsubscribe(
            ws_risk, list(CHANNELS),
        )
        await populated_broadcaster.subscribe(
            ws_risk, ["risk-updates"],
        )

        # ws_sensor: only sensor-data
        await populated_broadcaster.unsubscribe(
            ws_sensor, list(CHANNELS),
        )
        await populated_broadcaster.subscribe(
            ws_sensor, ["sensor-data"],
        )

        ws_risk.send_json.reset_mock()
        ws_sensor.send_json.reset_mock()

        # Risk broadcast → only ws_risk
        await populated_broadcaster.broadcast_risk_update("ZONE_A")
        assert ws_risk.send_json.await_count == 1
        assert ws_sensor.send_json.await_count == 0

        ws_risk.send_json.reset_mock()

        # Sensor broadcast → only ws_sensor
        await populated_broadcaster.broadcast_sensor_update(
            "ZONE_A", "S001",
        )
        assert ws_risk.send_json.await_count == 0
        assert ws_sensor.send_json.await_count == 1

    @pytest.mark.asyncio
    async def test_broadcast_without_channel_goes_to_all(
        self, broadcaster: WebSocketBroadcaster,
    ):
        """broadcast(event, channel=None) → all clients."""
        ws1 = _make_mock_ws()
        ws2 = _make_mock_ws()
        await broadcaster.connect(ws1)
        await broadcaster.connect(ws2)

        # Unsubscribe ws2 from everything
        await broadcaster.unsubscribe(ws2, list(CHANNELS))

        ws1.send_json.reset_mock()
        ws2.send_json.reset_mock()

        # Broadcast without channel → both receive
        await broadcaster.broadcast({"type": "test"}, channel=None)
        assert ws1.send_json.await_count == 1
        assert ws2.send_json.await_count == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Channel Metadata
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestChannelMetadata:
    """Test that broadcast messages include channel metadata."""

    @pytest.mark.asyncio
    async def test_zone_update_has_channel(
        self, populated_broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await populated_broadcaster.connect(ws)
        ws.send_json.reset_mock()
        await populated_broadcaster.broadcast_zone_update("ZONE_A")
        msg = ws.send_json.call_args[0][0]
        assert msg["channel"] == "twin-state"

    @pytest.mark.asyncio
    async def test_sensor_update_has_channel(
        self, populated_broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await populated_broadcaster.connect(ws)
        ws.send_json.reset_mock()
        await populated_broadcaster.broadcast_sensor_update(
            "ZONE_A", "S001",
        )
        msg = ws.send_json.call_args[0][0]
        assert msg["channel"] == "sensor-data"

    @pytest.mark.asyncio
    async def test_risk_update_has_channel(
        self, populated_broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await populated_broadcaster.connect(ws)
        ws.send_json.reset_mock()
        await populated_broadcaster.broadcast_risk_update("ZONE_A")
        msg = ws.send_json.call_args[0][0]
        assert msg["channel"] == "risk-updates"

    @pytest.mark.asyncio
    async def test_hazard_update_has_channel(
        self, populated_broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await populated_broadcaster.connect(ws)
        ws.send_json.reset_mock()
        await populated_broadcaster.broadcast_hazard_update("ZONE_A")
        msg = ws.send_json.call_args[0][0]
        assert msg["channel"] == "hazard-events"

    @pytest.mark.asyncio
    async def test_facility_update_has_channel(
        self, populated_broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await populated_broadcaster.connect(ws)
        ws.send_json.reset_mock()
        await populated_broadcaster.broadcast_facility_update()
        msg = ws.send_json.call_args[0][0]
        assert msg["channel"] == "twin-state"

    @pytest.mark.asyncio
    async def test_snapshot_has_channel(
        self, populated_broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await populated_broadcaster.connect(ws)
        # Snapshot is the first message
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "facility_snapshot"
        assert msg["channel"] == "twin-state"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Client Message Handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestClientMessageHandling:
    """Test handle_client_message for all actions."""

    @pytest.mark.asyncio
    async def test_ping_returns_pong(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await broadcaster.handle_client_message(
            ws, '{"action": "ping"}',
        )
        ws.send_json.assert_awaited_once()
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "pong"
        assert "timestamp" in msg

    @pytest.mark.asyncio
    async def test_subscribe_via_message(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        await broadcaster.unsubscribe(ws, list(CHANNELS))
        ws.send_json.reset_mock()

        await broadcaster.handle_client_message(
            ws,
            '{"action": "subscribe", "channels": ["risk-updates"]}',
        )
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "subscribed"
        assert "risk-updates" in msg["channels"]

    @pytest.mark.asyncio
    async def test_unsubscribe_via_message(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await broadcaster.handle_client_message(
            ws,
            '{"action": "unsubscribe", "channels": ["sensor-data"]}',
        )
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "unsubscribed"
        assert "sensor-data" not in msg["channels"]

    @pytest.mark.asyncio
    async def test_get_subscriptions_via_message(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await broadcaster.handle_client_message(
            ws, '{"action": "get_subscriptions"}',
        )
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "subscriptions"
        assert set(msg["channels"]) == set(CHANNELS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Malformed Messages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMalformedMessages:
    """Test safe handling of malformed client messages."""

    @pytest.mark.asyncio
    async def test_invalid_json(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await broadcaster.handle_client_message(ws, "not json{{{")
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "error"
        assert "Invalid JSON" in msg["message"]

    @pytest.mark.asyncio
    async def test_non_object_json(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await broadcaster.handle_client_message(ws, "[1,2,3]")
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "error"
        assert "Expected JSON object" in msg["message"]

    @pytest.mark.asyncio
    async def test_missing_action(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await broadcaster.handle_client_message(ws, '{"foo": "bar"}')
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "error"
        assert "Missing" in msg["message"]

    @pytest.mark.asyncio
    async def test_unknown_action(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await broadcaster.handle_client_message(
            ws, '{"action": "explode"}',
        )
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "error"
        assert "Unknown action" in msg["message"]

    @pytest.mark.asyncio
    async def test_subscribe_channels_not_list(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await broadcaster.handle_client_message(
            ws,
            '{"action": "subscribe", "channels": "risk-updates"}',
        )
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "error"
        assert "list" in msg["message"]

    @pytest.mark.asyncio
    async def test_empty_string(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await broadcaster.handle_client_message(ws, "")
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "error"

    @pytest.mark.asyncio
    async def test_null_json(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        ws.send_json.reset_mock()

        await broadcaster.handle_client_message(ws, "null")
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "error"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Disconnect Cleanup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDisconnectCleanup:
    """Test that disconnect removes subscriptions."""

    @pytest.mark.asyncio
    async def test_disconnect_removes_subscriptions(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws = _make_mock_ws()
        await broadcaster.connect(ws)
        assert broadcaster.get_subscriptions(ws) == set(CHANNELS)

        await broadcaster.disconnect(ws)
        assert broadcaster.get_subscriptions(ws) == set()
        assert broadcaster.connection_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_during_broadcast(
        self, broadcaster: WebSocketBroadcaster,
    ):
        ws_good = _make_mock_ws()
        ws_bad = _make_mock_ws()
        ws_bad.send_json = AsyncMock(
            side_effect=RuntimeError("gone"),
        )

        await broadcaster.connect(ws_good)
        broadcaster._clients[ws_bad] = _ClientState(
            websocket=ws_bad,
        )
        assert broadcaster.connection_count == 2

        await broadcaster.broadcast(
            {"type": "test"}, channel="twin-state",
        )
        assert broadcaster.connection_count == 1
        assert broadcaster.get_subscriptions(ws_bad) == set()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Concurrent Clients
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConcurrentClients:
    """Test multiple clients with different subscription sets."""

    @pytest.mark.asyncio
    async def test_three_clients_different_subs(
        self, populated_broadcaster: WebSocketBroadcaster,
    ):
        ws_all = _make_mock_ws()  # All channels
        ws_risk_only = _make_mock_ws()  # risk-updates only
        ws_none = _make_mock_ws()  # No channels

        await populated_broadcaster.connect(ws_all)
        await populated_broadcaster.connect(ws_risk_only)
        await populated_broadcaster.connect(ws_none)

        # Configure subscriptions
        await populated_broadcaster.unsubscribe(
            ws_risk_only, list(CHANNELS),
        )
        await populated_broadcaster.subscribe(
            ws_risk_only, ["risk-updates"],
        )
        await populated_broadcaster.unsubscribe(
            ws_none, list(CHANNELS),
        )

        ws_all.send_json.reset_mock()
        ws_risk_only.send_json.reset_mock()
        ws_none.send_json.reset_mock()

        # Sensor broadcast
        await populated_broadcaster.broadcast_sensor_update(
            "ZONE_A", "S001",
        )
        assert ws_all.send_json.await_count == 1
        assert ws_risk_only.send_json.await_count == 0
        assert ws_none.send_json.await_count == 0

        ws_all.send_json.reset_mock()

        # Risk broadcast
        await populated_broadcaster.broadcast_risk_update("ZONE_A")
        assert ws_all.send_json.await_count == 1
        assert ws_risk_only.send_json.await_count == 1
        assert ws_none.send_json.await_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_subscribe_unsubscribe(
        self, broadcaster: WebSocketBroadcaster,
    ):
        """Rapid subscribe/unsubscribe doesn't corrupt state."""
        ws = _make_mock_ws()
        await broadcaster.connect(ws)

        await broadcaster.unsubscribe(ws, list(CHANNELS))
        await broadcaster.subscribe(ws, ["risk-updates"])
        await broadcaster.subscribe(ws, ["sensor-data"])
        await broadcaster.unsubscribe(ws, ["risk-updates"])
        await broadcaster.subscribe(ws, ["hazard-events"])

        subs = broadcaster.get_subscriptions(ws)
        assert subs == {"sensor-data", "hazard-events"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _ClientState Internal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestClientState:
    """Test _ClientState internals."""

    def test_default_subscriptions(self):
        ws = _make_mock_ws()
        client = _ClientState(websocket=ws)
        assert client.subscriptions == set(DEFAULT_CHANNELS)

    def test_custom_subscriptions(self):
        ws = _make_mock_ws()
        client = _ClientState(
            websocket=ws,
            subscriptions={"risk-updates"},
        )
        assert client.subscriptions == {"risk-updates"}

    def test_connected_at_timestamp(self):
        ws = _make_mock_ws()
        client = _ClientState(websocket=ws)
        assert client.connected_at  # Non-empty


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Handler Integration (Channel-Aware)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHandlerChannelIntegration:
    """Test that handler broadcasts use channels correctly."""

    @pytest.mark.asyncio
    async def test_sensor_event_uses_sensor_channel(
        self, twin_manager: TwinStateManager,
    ):
        broadcaster = WebSocketBroadcaster(state_manager=twin_manager)
        ws = _make_mock_ws()
        await broadcaster.connect(ws)

        # Subscribe only to sensor-data
        await broadcaster.unsubscribe(ws, list(CHANNELS))
        await broadcaster.subscribe(ws, ["sensor-data"])
        ws.send_json.reset_mock()

        handler = DigitalTwinEventHandler(
            state_manager=twin_manager,
            broadcaster=broadcaster,
        )
        await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY,
            {
                "event_id": "ch-001",
                "data": {
                    "sensor_id": "S001", "zone_id": "Z1",
                    "value": 100.0, "anomaly_score": -0.9,
                },
            },
        )

        # Should receive sensor_update
        messages = _get_sent_messages(ws)
        types = {m["type"] for m in messages}
        assert "sensor_update" in types

    @pytest.mark.asyncio
    async def test_risk_event_filtered_for_sensor_only_client(
        self, twin_manager: TwinStateManager,
    ):
        broadcaster = WebSocketBroadcaster(state_manager=twin_manager)
        ws = _make_mock_ws()
        await broadcaster.connect(ws)

        # Subscribe only to sensor-data (not risk-updates or twin-state)
        await broadcaster.unsubscribe(ws, list(CHANNELS))
        await broadcaster.subscribe(ws, ["sensor-data"])
        ws.send_json.reset_mock()

        handler = DigitalTwinEventHandler(
            state_manager=twin_manager,
            broadcaster=broadcaster,
        )
        await handler.handle_event(
            KafkaTopics.RISK_SCORE_UPDATED,
            {
                "event_id": "ch-002",
                "data": {"zone_id": "Z1", "risk_score": 80.0},
            },
        )

        # Should NOT receive risk_update or zone_update
        assert ws.send_json.await_count == 0

    @pytest.mark.asyncio
    async def test_hazard_event_reaches_hazard_subscriber(
        self, twin_manager: TwinStateManager,
    ):
        broadcaster = WebSocketBroadcaster(state_manager=twin_manager)
        ws = _make_mock_ws()
        await broadcaster.connect(ws)

        await broadcaster.unsubscribe(ws, list(CHANNELS))
        await broadcaster.subscribe(ws, ["hazard-events"])
        ws.send_json.reset_mock()

        handler = DigitalTwinEventHandler(
            state_manager=twin_manager,
            broadcaster=broadcaster,
        )
        await handler.handle_event(
            KafkaTopics.HAZARD_DETECTED,
            {
                "event_id": "ch-003",
                "data": {
                    "hazard_id": "H1",
                    "zone_id": "Z1",
                    "hazard_type": "FIRE",
                    "severity": "HIGH",
                },
            },
        )

        messages = _get_sent_messages(ws)
        types = {m["type"] for m in messages}
        assert "hazard_update" in types


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WebSocket Endpoint Tests (starlette TestClient)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestWebSocketEndpointPhase4:
    """Test the /ws/twin endpoint with Phase 4 message handling."""

    def _setup_app(self, twin_manager: TwinStateManager):
        from app.digital_twin.api.websocket_endpoints import (
            set_broadcaster,
        )
        from app.main import app

        broadcaster = WebSocketBroadcaster(state_manager=twin_manager)
        set_broadcaster(broadcaster)
        return app, broadcaster

    def test_connect_receives_snapshot(
        self, twin_manager: TwinStateManager,
    ):
        app, _ = self._setup_app(twin_manager)
        twin_manager.update_risk_score(
            zone_id="Z1", risk_score=50.0,
        )
        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/v1/ws/twin",
            ) as ws:
                data = ws.receive_json()
                assert data["type"] == "facility_snapshot"
                assert data["channel"] == "twin-state"

    def test_ping_pong(self, twin_manager: TwinStateManager):
        app, _ = self._setup_app(twin_manager)
        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/v1/ws/twin",
            ) as ws:
                ws.receive_json()  # consume snapshot
                ws.send_text('{"action": "ping"}')
                resp = ws.receive_json()
                assert resp["type"] == "pong"
                assert "timestamp" in resp

    def test_subscribe_message(
        self, twin_manager: TwinStateManager,
    ):
        app, _ = self._setup_app(twin_manager)
        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/v1/ws/twin",
            ) as ws:
                ws.receive_json()  # snapshot
                ws.send_text(json.dumps({
                    "action": "subscribe",
                    "channels": ["risk-updates"],
                }))
                resp = ws.receive_json()
                assert resp["type"] == "subscribed"
                assert "risk-updates" in resp["channels"]

    def test_unsubscribe_message(
        self, twin_manager: TwinStateManager,
    ):
        app, _ = self._setup_app(twin_manager)
        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/v1/ws/twin",
            ) as ws:
                ws.receive_json()  # snapshot
                ws.send_text(json.dumps({
                    "action": "unsubscribe",
                    "channels": ["sensor-data"],
                }))
                resp = ws.receive_json()
                assert resp["type"] == "unsubscribed"
                assert "sensor-data" not in resp["channels"]

    def test_get_subscriptions(
        self, twin_manager: TwinStateManager,
    ):
        app, _ = self._setup_app(twin_manager)
        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/v1/ws/twin",
            ) as ws:
                ws.receive_json()  # snapshot
                ws.send_text('{"action": "get_subscriptions"}')
                resp = ws.receive_json()
                assert resp["type"] == "subscriptions"
                assert set(resp["channels"]) == set(CHANNELS)

    def test_malformed_json(
        self, twin_manager: TwinStateManager,
    ):
        app, _ = self._setup_app(twin_manager)
        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/v1/ws/twin",
            ) as ws:
                ws.receive_json()  # snapshot
                ws.send_text("not valid json!!!")
                resp = ws.receive_json()
                assert resp["type"] == "error"
                assert "Invalid JSON" in resp["message"]

    def test_unknown_action(
        self, twin_manager: TwinStateManager,
    ):
        app, _ = self._setup_app(twin_manager)
        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/v1/ws/twin",
            ) as ws:
                ws.receive_json()  # snapshot
                ws.send_text('{"action": "destroy"}')
                resp = ws.receive_json()
                assert resp["type"] == "error"
                assert "Unknown action" in resp["message"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full Pipeline: Event → Channel-Filtered Broadcast
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFullChannelPipeline:
    """End-to-end: Kafka events → filtered WS delivery."""

    @pytest.mark.asyncio
    async def test_full_channel_pipeline(self):
        graph_repo = InMemoryGraphRepository()
        manager = TwinStateManager(graph_repo=graph_repo)
        await manager.initialize()

        broadcaster = WebSocketBroadcaster(state_manager=manager)
        handler = DigitalTwinEventHandler(
            state_manager=manager,
            broadcaster=broadcaster,
        )

        # Client A: only risk-updates
        ws_a = _make_mock_ws()
        await broadcaster.connect(ws_a)
        await broadcaster.unsubscribe(ws_a, list(CHANNELS))
        await broadcaster.subscribe(ws_a, ["risk-updates"])
        ws_a.send_json.reset_mock()

        # Client B: only hazard-events
        ws_b = _make_mock_ws()
        await broadcaster.connect(ws_b)
        await broadcaster.unsubscribe(ws_b, list(CHANNELS))
        await broadcaster.subscribe(ws_b, ["hazard-events"])
        ws_b.send_json.reset_mock()

        # Client C: all channels
        ws_c = _make_mock_ws()
        await broadcaster.connect(ws_c)
        ws_c.send_json.reset_mock()

        # Sensor event → should reach only C
        await handler.handle_event(
            KafkaTopics.SENSOR_READING_ANOMALY,
            {
                "event_id": "pipe-001",
                "data": {
                    "sensor_id": "S1", "zone_id": "Z1",
                    "value": 100.0, "anomaly_score": -0.9,
                },
            },
        )
        assert ws_a.send_json.await_count == 0
        assert ws_b.send_json.await_count == 0
        assert ws_c.send_json.await_count > 0

        ws_a.send_json.reset_mock()
        ws_b.send_json.reset_mock()
        ws_c.send_json.reset_mock()

        # Risk event → should reach A (risk-updates) and C (all)
        await handler.handle_event(
            KafkaTopics.RISK_SCORE_UPDATED,
            {
                "event_id": "pipe-002",
                "data": {"zone_id": "Z1", "risk_score": 80.0},
            },
        )
        assert ws_a.send_json.await_count > 0  # risk-updates
        assert ws_b.send_json.await_count == 0
        assert ws_c.send_json.await_count > 0

        ws_a.send_json.reset_mock()
        ws_b.send_json.reset_mock()
        ws_c.send_json.reset_mock()

        # Hazard event → should reach B (hazard-events) and C (all)
        await handler.handle_event(
            KafkaTopics.HAZARD_DETECTED,
            {
                "event_id": "pipe-003",
                "data": {
                    "hazard_id": "H1", "zone_id": "Z1",
                    "hazard_type": "FIRE", "severity": "HIGH",
                },
            },
        )
        assert ws_a.send_json.await_count == 0
        assert ws_b.send_json.await_count > 0  # hazard-events
        assert ws_c.send_json.await_count > 0
