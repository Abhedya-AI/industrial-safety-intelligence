"""Digital Twin WebSocket endpoint.

Provides a real-time streaming connection at WS /ws/twin.

Phase 4 enhancements:
  - Subscribe/unsubscribe to specific channels
  - Heartbeat/ping support
  - Malformed payload handling (safe error, no disconnect)
  - Graceful disconnect cleanup

Connection lifecycle:
    1. Client connects → WebSocket accepted
    2. Broadcaster sends facility_snapshot
    3. Client sends subscribe/unsubscribe/ping messages
    4. Broadcaster pushes filtered events by subscription
    5. On disconnect → broadcaster removes client + subscriptions

Supported client messages:
    {"action": "subscribe",   "channels": ["risk-updates", "sensor-data"]}
    {"action": "unsubscribe", "channels": ["hazard-events"]}
    {"action": "ping"}
    {"action": "get_subscriptions"}

The endpoint uses the shared WebSocketBroadcaster singleton
from the Digital Twin DI layer.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.digital_twin.services.websocket_broadcaster import (
    WebSocketBroadcaster,
)

logger = logging.getLogger(__name__)

ws_router = APIRouter()

# Module-level broadcaster reference, set during DI wiring
_broadcaster: WebSocketBroadcaster | None = None


def set_broadcaster(broadcaster: WebSocketBroadcaster) -> None:
    """Set the broadcaster singleton. Called from dependencies.py."""
    global _broadcaster
    _broadcaster = broadcaster


def get_broadcaster() -> WebSocketBroadcaster | None:
    """Get the broadcaster singleton."""
    return _broadcaster


@ws_router.websocket("/ws/twin")
async def websocket_twin_endpoint(websocket: WebSocket) -> None:
    """Real-time Digital Twin WebSocket endpoint.

    On connect:
        - Accepts the WebSocket
        - Registers with the broadcaster (subscribed to all channels)
        - Sends a facility_snapshot with current state

    Message loop:
        - Processes subscribe/unsubscribe/ping messages
        - Malformed payloads receive an error response (no disconnect)

    On disconnect:
        - Removes client and all subscriptions from the broadcaster
    """
    if _broadcaster is None:
        logger.warning(
            "WebSocket connection rejected: broadcaster not initialized",
        )
        await websocket.close(code=1013, reason="Service unavailable")
        return

    await _broadcaster.connect(websocket)

    try:
        while True:
            try:
                raw = await websocket.receive_text()
                # Delegate all message handling to broadcaster
                await _broadcaster.handle_client_message(
                    websocket, raw,
                )
            except WebSocketDisconnect:
                break
    except Exception:
        logger.debug("WebSocket connection error")
    finally:
        await _broadcaster.disconnect(websocket)
