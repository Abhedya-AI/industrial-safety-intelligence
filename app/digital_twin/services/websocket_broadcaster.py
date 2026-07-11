"""WebSocket broadcaster for the Digital Twin.

Manages active WebSocket connections and broadcasts real-time
state updates whenever the twin state changes from a Kafka event.

Phase 4 enhancements:
  - Channel-based subscriptions (clients receive only what they need)
  - Per-client subscription tracking
  - Broadcast filtered by channel membership
  - Change-only delivery (no full facility snapshots per event)

Channels:
  - risk-updates    : risk_update, compound risk changes
  - sensor-data     : sensor_update (anomaly, status, health)
  - hazard-events   : hazard_update (detected, propagated)
  - twin-state      : zone_update, facility_update

Design:
  - Fault-tolerant: a single failing client is disconnected and
    removed automatically without affecting other clients.
  - Thread-safe: connection tracking uses asyncio-safe structures.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket

from app.digital_twin.services.twin_state_manager import TwinStateManager

logger = logging.getLogger(__name__)

# All available subscription channels
CHANNELS = frozenset({
    "risk-updates",
    "sensor-data",
    "hazard-events",
    "twin-state",
})

# Default channels a new client is subscribed to
DEFAULT_CHANNELS = frozenset(CHANNELS)

# Maps event_category (from handler) → channel name
CATEGORY_TO_CHANNEL = {
    "sensor": "sensor-data",
    "risk": "risk-updates",
    "hazard": "hazard-events",
}

# Maps message type → channel
TYPE_TO_CHANNEL = {
    "sensor_update": "sensor-data",
    "risk_update": "risk-updates",
    "hazard_update": "hazard-events",
    "zone_update": "twin-state",
    "facility_update": "twin-state",
    "facility_snapshot": "twin-state",
}


class _ClientState:
    """Per-client connection state."""

    __slots__ = ("websocket", "subscriptions", "connected_at")

    def __init__(
        self,
        websocket: WebSocket,
        subscriptions: Set[str] | None = None,
    ) -> None:
        self.websocket = websocket
        self.subscriptions: Set[str] = (
            set(subscriptions) if subscriptions
            else set(DEFAULT_CHANNELS)
        )
        self.connected_at: str = datetime.now(timezone.utc).isoformat()


class WebSocketBroadcaster:
    """Manages WebSocket connections and broadcasts twin state changes.

    Lifecycle:
        1. Client connects → connect(ws) → send facility_snapshot
        2. Client subscribes/unsubscribes → channel membership updated
        3. Kafka event processed → broadcast_*() → push to subscribed clients
        4. Client disconnects → disconnect(ws) → remove from pool

    Args:
        state_manager: The shared TwinStateManager singleton.
    """

    def __init__(self, state_manager: TwinStateManager) -> None:
        self._state = state_manager
        self._clients: Dict[WebSocket, _ClientState] = {}
        self._lock = asyncio.Lock()

        # Metrics
        self._messages_sent: int = 0
        self._messages_failed: int = 0
        self._total_connections: int = 0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Connection management
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def connect(self, websocket: WebSocket) -> None:
        """Register a new WebSocket connection.

        Accepts the WebSocket, subscribes to all channels by default,
        and sends an initial facility snapshot.
        """
        await websocket.accept()

        client = _ClientState(websocket=websocket)
        async with self._lock:
            self._clients[websocket] = client
            self._total_connections += 1

        logger.info(
            "WebSocket client connected. Active connections: %d",
            self.connection_count,
        )

        # Send initial facility snapshot
        await self._send_facility_snapshot(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection from the pool."""
        async with self._lock:
            self._clients.pop(websocket, None)

        logger.info(
            "WebSocket client disconnected. Active connections: %d",
            self.connection_count,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Subscription management
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def subscribe(
        self, websocket: WebSocket, channels: List[str],
    ) -> Dict[str, Any]:
        """Subscribe a client to specific channels.

        Returns a confirmation message with active subscriptions.
        Invalid channel names are silently ignored.
        """
        valid = [c for c in channels if c in CHANNELS]
        async with self._lock:
            client = self._clients.get(websocket)
            if client:
                client.subscriptions.update(valid)
                return {
                    "type": "subscribed",
                    "channels": sorted(client.subscriptions),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        return {"type": "error", "message": "Client not found"}

    async def unsubscribe(
        self, websocket: WebSocket, channels: List[str],
    ) -> Dict[str, Any]:
        """Unsubscribe a client from specific channels.

        Returns a confirmation message with remaining subscriptions.
        """
        async with self._lock:
            client = self._clients.get(websocket)
            if client:
                client.subscriptions -= set(channels)
                return {
                    "type": "unsubscribed",
                    "channels": sorted(client.subscriptions),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        return {"type": "error", "message": "Client not found"}

    def get_subscriptions(
        self, websocket: WebSocket,
    ) -> Set[str]:
        """Get the current subscriptions for a client."""
        client = self._clients.get(websocket)
        if client:
            return set(client.subscriptions)
        return set()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Properties
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def connection_count(self) -> int:
        """Number of active WebSocket connections."""
        return len(self._clients)

    @property
    def messages_sent(self) -> int:
        return self._messages_sent

    @property
    def messages_failed(self) -> int:
        return self._messages_failed

    @property
    def total_connections(self) -> int:
        return self._total_connections

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Broadcast methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def broadcast(
        self, event: Dict[str, Any], channel: str | None = None,
    ) -> None:
        """Broadcast an event to clients subscribed to the channel.

        If channel is None, sends to all connected clients.
        Failed sends automatically disconnect the failing client.
        """
        if not self._clients:
            return

        event["timestamp"] = datetime.now(timezone.utc).isoformat()

        async with self._lock:
            if channel:
                targets = [
                    cs.websocket for cs in self._clients.values()
                    if channel in cs.subscriptions
                ]
            else:
                targets = [cs.websocket for cs in self._clients.values()]

        disconnected: List[WebSocket] = []

        for ws in targets:
            try:
                await ws.send_json(event)
                self._messages_sent += 1
            except Exception:
                logger.debug(
                    "WebSocket send failed, removing client.",
                )
                disconnected.append(ws)
                self._messages_failed += 1

        for ws in disconnected:
            await self.disconnect(ws)

    async def broadcast_zone_update(self, zone_id: str) -> None:
        """Broadcast a zone state update (twin-state channel)."""
        try:
            zone = self._state.get_zone(zone_id)
            await self.broadcast(
                {
                    "type": "zone_update",
                    "channel": "twin-state",
                    "zone_id": zone_id,
                    "data": {
                        "zone_id": zone.zone_id,
                        "zone_name": zone.zone_name,
                        "sensor_health": round(zone.sensor_health, 2),
                        "anomaly_count": zone.anomaly_count,
                        "predicted_risk_score": round(
                            zone.predicted_risk_score, 2,
                        ),
                        "risk_level": zone.risk_level,
                        "compound_risk_score": round(
                            zone.compound_risk_score, 2,
                        ),
                        "compound_risk_level": zone.compound_risk_level,
                        "active_hazard_count": zone.active_hazard_count,
                        "workers_at_risk": zone.workers_at_risk,
                        "overall_risk_score": round(
                            zone.overall_risk_score, 2,
                        ),
                        "heatmap_color": zone.heatmap_color,
                        "is_critical": zone.is_critical,
                        "event_count": zone.event_count,
                        "last_updated": zone.last_updated,
                    },
                },
                channel="twin-state",
            )
        except Exception:
            logger.debug(
                "Failed to broadcast zone update for %s", zone_id,
            )

    async def broadcast_facility_update(self) -> None:
        """Broadcast facility-wide aggregated state (twin-state channel)."""
        try:
            state = self._state.get_facility_state()
            await self.broadcast(
                {
                    "type": "facility_update",
                    "channel": "twin-state",
                    "data": {
                        "facility_health": state.facility_health,
                        "total_zones": state.total_zones,
                        "active_hazards": state.active_hazards,
                        "critical_zones": state.critical_zones,
                        "workers_at_risk": state.workers_at_risk,
                        "total_workers": state.total_workers,
                        "total_sensors": state.total_sensors,
                        "total_anomalies": state.total_anomalies,
                        "average_risk_score": state.average_risk_score,
                        "max_risk_score": state.max_risk_score,
                        "events_processed": state.events_processed,
                        "last_updated": state.last_updated,
                    },
                },
                channel="twin-state",
            )
        except Exception:
            logger.debug("Failed to broadcast facility update.")

    async def broadcast_sensor_update(
        self, zone_id: str, sensor_id: str,
    ) -> None:
        """Broadcast a sensor-specific update (sensor-data channel)."""
        try:
            zone = self._state.get_zone(zone_id)
            reading = zone.latest_sensor_readings.get(sensor_id)
            if reading:
                await self.broadcast(
                    {
                        "type": "sensor_update",
                        "channel": "sensor-data",
                        "zone_id": zone_id,
                        "data": {
                            "sensor_id": reading.sensor_id,
                            "sensor_type": reading.sensor_type,
                            "value": reading.value,
                            "unit": reading.unit,
                            "anomaly_score": reading.anomaly_score,
                            "is_anomalous": reading.is_anomalous,
                            "health_score": reading.health_score,
                            "status": reading.status,
                            "last_updated": reading.last_updated,
                        },
                    },
                    channel="sensor-data",
                )
        except Exception:
            logger.debug(
                "Failed to broadcast sensor update %s/%s",
                zone_id, sensor_id,
            )

    async def broadcast_risk_update(self, zone_id: str) -> None:
        """Broadcast a risk-specific update (risk-updates channel)."""
        try:
            zone = self._state.get_zone(zone_id)
            await self.broadcast(
                {
                    "type": "risk_update",
                    "channel": "risk-updates",
                    "zone_id": zone_id,
                    "data": {
                        "predicted_risk_score": round(
                            zone.predicted_risk_score, 2,
                        ),
                        "risk_level": zone.risk_level,
                        "compound_risk_score": round(
                            zone.compound_risk_score, 2,
                        ),
                        "compound_risk_level": zone.compound_risk_level,
                        "overall_risk_score": round(
                            zone.overall_risk_score, 2,
                        ),
                        "is_critical": zone.is_critical,
                    },
                },
                channel="risk-updates",
            )
        except Exception:
            logger.debug(
                "Failed to broadcast risk update for %s", zone_id,
            )

    async def broadcast_hazard_update(
        self, zone_id: str,
    ) -> None:
        """Broadcast a hazard-specific update (hazard-events channel)."""
        try:
            zone = self._state.get_zone(zone_id)
            await self.broadcast(
                {
                    "type": "hazard_update",
                    "channel": "hazard-events",
                    "zone_id": zone_id,
                    "data": {
                        "active_hazard_count": zone.active_hazard_count,
                        "active_hazards": [
                            {
                                "hazard_id": h.hazard_id,
                                "hazard_type": h.hazard_type,
                                "severity": h.severity,
                                "origin_zone": h.origin_zone,
                                "propagation_level": h.propagation_level,
                                "affected_zones": h.affected_zones,
                            }
                            for h in zone.active_hazards
                        ],
                        "affected_neighbors": zone.affected_neighbors,
                        "workers_at_risk": zone.workers_at_risk,
                    },
                },
                channel="hazard-events",
            )
        except Exception:
            logger.debug(
                "Failed to broadcast hazard update for %s", zone_id,
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Client message handling
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def handle_client_message(
        self, websocket: WebSocket, raw: str,
    ) -> None:
        """Process an incoming client message.

        Supported actions:
          - {"action": "subscribe", "channels": [...]}
          - {"action": "unsubscribe", "channels": [...]}
          - {"action": "ping"}
          - {"action": "get_subscriptions"}

        Malformed payloads receive an error response but do not
        disconnect the client.
        """
        import json as _json

        try:
            msg = _json.loads(raw)
        except (ValueError, TypeError):
            await self._send_error(
                websocket, "Invalid JSON payload",
            )
            return

        if not isinstance(msg, dict):
            await self._send_error(
                websocket, "Expected JSON object",
            )
            return

        action = msg.get("action", "")

        if action == "subscribe":
            channels = msg.get("channels", [])
            if not isinstance(channels, list):
                await self._send_error(
                    websocket,
                    "channels must be a list",
                )
                return
            resp = await self.subscribe(websocket, channels)
            await self._send_to_client(websocket, resp)

        elif action == "unsubscribe":
            channels = msg.get("channels", [])
            if not isinstance(channels, list):
                await self._send_error(
                    websocket,
                    "channels must be a list",
                )
                return
            resp = await self.unsubscribe(websocket, channels)
            await self._send_to_client(websocket, resp)

        elif action == "ping":
            await self._send_to_client(websocket, {
                "type": "pong",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        elif action == "get_subscriptions":
            subs = self.get_subscriptions(websocket)
            await self._send_to_client(websocket, {
                "type": "subscriptions",
                "channels": sorted(subs),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        else:
            await self._send_error(
                websocket,
                f"Unknown action: {action}" if action
                else "Missing 'action' field",
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _send_to_client(
        self, websocket: WebSocket, data: Dict[str, Any],
    ) -> None:
        """Send data to a single client. Swallows errors."""
        try:
            await websocket.send_json(data)
            self._messages_sent += 1
        except Exception:
            self._messages_failed += 1

    async def _send_error(
        self, websocket: WebSocket, message: str,
    ) -> None:
        """Send an error message to a client."""
        await self._send_to_client(websocket, {
            "type": "error",
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def _send_facility_snapshot(
        self, websocket: WebSocket,
    ) -> None:
        """Send the full facility state to a single client on connect."""
        try:
            facility = self._state.get_facility_state()
            zones = self._state.get_all_zones()

            snapshot = {
                "type": "facility_snapshot",
                "channel": "twin-state",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "facility_health": facility.facility_health,
                    "total_zones": facility.total_zones,
                    "active_hazards": facility.active_hazards,
                    "critical_zones": facility.critical_zones,
                    "workers_at_risk": facility.workers_at_risk,
                    "events_processed": facility.events_processed,
                    "zones": [
                        {
                            "zone_id": z.zone_id,
                            "zone_name": z.zone_name,
                            "sensor_health": round(z.sensor_health, 2),
                            "anomaly_count": z.anomaly_count,
                            "predicted_risk_score": round(
                                z.predicted_risk_score, 2,
                            ),
                            "risk_level": z.risk_level,
                            "compound_risk_score": round(
                                z.compound_risk_score, 2,
                            ),
                            "active_hazard_count": z.active_hazard_count,
                            "overall_risk_score": round(
                                z.overall_risk_score, 2,
                            ),
                            "heatmap_color": z.heatmap_color,
                            "is_critical": z.is_critical,
                        }
                        for z in zones
                    ],
                },
            }
            await websocket.send_json(snapshot)
            self._messages_sent += 1
        except Exception:
            logger.debug("Failed to send facility snapshot to client.")
            self._messages_failed += 1
