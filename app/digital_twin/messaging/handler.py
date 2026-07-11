"""Digital Twin event handler.

Routes incoming Kafka events to the appropriate TwinStateManager
update method. Follows the same pattern as:
  - HazardPropagationEventHandler
  - CompoundRiskEventHandler

Consumes (9 topics):
  Sensor Intelligence:
    - sensor.reading.anomaly
    - sensor.status.changed
    - sensor.health.updated
  Risk Prediction:
    - risk.assessment.generated
    - risk.score.updated
    - risk.threshold.exceeded
  Compound Risk:
    - compound.risk.detected
  Hazard Propagation:
    - hazard.detected
    - hazard.propagated

Publishes: NOTHING via Kafka (Digital Twin is a pure consumer)
Broadcasts: Real-time WebSocket updates to connected clients
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from app.digital_twin.domain.enums import TwinUpdateType
from app.digital_twin.services.twin_state_manager import TwinStateManager
from app.shared.messaging.topics import KafkaTopics

if TYPE_CHECKING:
    from app.digital_twin.services.snapshot_service import SnapshotService
    from app.digital_twin.services.websocket_broadcaster import (
        WebSocketBroadcaster,
    )

logger = logging.getLogger(__name__)

# Map topic → (handler_method_name, ws_message_type)
_TOPIC_MAP = {
    KafkaTopics.SENSOR_READING_ANOMALY: ("_handle_sensor_anomaly", "sensor"),
    KafkaTopics.SENSOR_STATUS_CHANGED: ("_handle_sensor_status", "sensor"),
    KafkaTopics.SENSOR_HEALTH_UPDATED: ("_handle_sensor_health", "sensor"),
    KafkaTopics.RISK_ASSESSMENT_GENERATED: ("_handle_risk_assessment", "risk"),
    KafkaTopics.RISK_SCORE_UPDATED: ("_handle_risk_score", "risk"),
    KafkaTopics.RISK_THRESHOLD_EXCEEDED: ("_handle_risk_threshold", "risk"),
    KafkaTopics.COMPOUND_RISK_DETECTED: ("_handle_compound_risk", "risk"),
    KafkaTopics.HAZARD_DETECTED: ("_handle_hazard_detected", "hazard"),
    KafkaTopics.HAZARD_PROPAGATED: ("_handle_hazard_propagated", "hazard"),
}


class DigitalTwinEventHandler:
    """Handles incoming Kafka events and updates the twin state.

    Optionally broadcasts real-time updates to connected WebSocket
    clients via the WebSocketBroadcaster.

    All errors are caught and logged — failed events do not crash
    the consumer.

    Args:
        state_manager: The shared TwinStateManager singleton.
        broadcaster: Optional WebSocketBroadcaster for real-time push.
    """

    def __init__(
        self,
        state_manager: TwinStateManager,
        broadcaster: Optional["WebSocketBroadcaster"] = None,
        snapshot_service: Optional["SnapshotService"] = None,
    ) -> None:
        self._state = state_manager
        self._broadcaster = broadcaster
        self._snapshot_service = snapshot_service

        # Duplicate event detection
        self._processed_event_ids: Set[str] = set()
        self._max_event_cache_size: int = 10000

        # Metrics
        self._events_processed: int = 0
        self._events_failed: int = 0
        self._events_skipped: int = 0

    # ── Properties ──

    @property
    def events_processed(self) -> int:
        return self._events_processed

    @property
    def events_failed(self) -> int:
        return self._events_failed

    @property
    def events_skipped(self) -> int:
        return self._events_skipped

    @property
    def broadcaster(self) -> Optional["WebSocketBroadcaster"]:
        return self._broadcaster

    @broadcaster.setter
    def broadcaster(self, value: Optional["WebSocketBroadcaster"]) -> None:
        self._broadcaster = value

    # ── Main dispatch ──

    async def handle_event(
        self, topic: str, data: Dict[str, Any],
    ) -> None:
        """Process a single Kafka event.

        Routes to the appropriate handler based on topic.
        Deduplicates by event_id.
        After state mutation, broadcasts to WebSocket clients.
        """
        event_id = data.get("event_id", "")
        start = time.monotonic()

        # Deduplicate
        if event_id and event_id in self._processed_event_ids:
            self._events_skipped += 1
            logger.debug(
                "Skipping duplicate event: topic=%s event_id=%s",
                topic, event_id,
            )
            return

        topic_entry = _TOPIC_MAP.get(topic)
        if topic_entry is None:
            logger.warning(
                "Digital Twin received event for unhandled topic: %s",
                topic,
            )
            self._events_skipped += 1
            return

        handler_name, event_category = topic_entry

        try:
            handler_method = getattr(self, handler_name)
            zone_id, extra_context = handler_method(data)

            # Track dedup
            if event_id:
                self._processed_event_ids.add(event_id)
                if len(self._processed_event_ids) > self._max_event_cache_size:
                    # Evict oldest half
                    to_keep = list(self._processed_event_ids)[
                        self._max_event_cache_size // 2 :
                    ]
                    self._processed_event_ids = set(to_keep)

            self._events_processed += 1
            elapsed = (time.monotonic() - start) * 1000
            logger.debug(
                "Digital Twin processed event: topic=%s event_id=%s "
                "elapsed=%.1fms",
                topic, event_id, elapsed,
            )

            # ── Broadcast to WebSocket clients ──
            if self._broadcaster and self._broadcaster.connection_count > 0:
                await self._broadcast_updates(
                    event_category, zone_id, extra_context,
                )

            # ── Evaluate automatic snapshot trigger ──
            if self._snapshot_service:
                try:
                    await self._snapshot_service.evaluate_snapshot_trigger(
                        event_category=event_category,
                        zone_id=zone_id,
                    )
                except Exception:
                    logger.debug(
                        "Snapshot trigger evaluation failed",
                    )

        except Exception:
            self._events_failed += 1
            logger.exception(
                "Digital Twin failed to process event: "
                "topic=%s event_id=%s",
                topic, event_id,
            )

    async def _broadcast_updates(
        self,
        event_category: str,
        zone_id: str,
        extra_context: Dict[str, Any],
    ) -> None:
        """Broadcast typed updates + zone + facility updates.

        Non-blocking: errors are logged, never re-raised.
        """
        try:
            # 1. Type-specific broadcast
            if event_category == "sensor":
                sensor_id = extra_context.get("sensor_id", "")
                if sensor_id:
                    await self._broadcaster.broadcast_sensor_update(
                        zone_id, sensor_id,
                    )
            elif event_category == "risk":
                await self._broadcaster.broadcast_risk_update(zone_id)
            elif event_category == "hazard":
                await self._broadcaster.broadcast_hazard_update(zone_id)

            # 2. Zone-level update (always)
            await self._broadcaster.broadcast_zone_update(zone_id)

            # 3. Facility-level update (always)
            await self._broadcaster.broadcast_facility_update()

        except Exception:
            logger.debug(
                "WebSocket broadcast failed for zone=%s", zone_id,
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Sensor Intelligence handlers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _handle_sensor_anomaly(
        self, data: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", payload.get("location_zone_id", "UNKNOWN"))
        sensor_id = payload.get("sensor_id", "")
        self._state.update_sensor_anomaly(
            zone_id=zone_id,
            sensor_id=sensor_id,
            sensor_type=payload.get("sensor_type", payload.get("reading_type", "")),
            value=float(payload.get("value", payload.get("sensor_value", 0))),
            unit=payload.get("unit", payload.get("unit_of_measurement", "")),
            anomaly_score=float(
                payload.get(
                    "anomaly_score",
                    payload.get("isolation_forest_score", 0),
                )
            ),
        )
        return zone_id, {"sensor_id": sensor_id}

    def _handle_sensor_status(
        self, data: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", payload.get("location_zone_id", "UNKNOWN"))
        sensor_id = payload.get("sensor_id", "")
        self._state.update_sensor_status(
            zone_id=zone_id,
            sensor_id=sensor_id,
            status=payload.get("status", payload.get("new_status", "ACTIVE")),
        )
        return zone_id, {"sensor_id": sensor_id}

    def _handle_sensor_health(
        self, data: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", payload.get("location_zone_id", "UNKNOWN"))
        sensor_id = payload.get("sensor_id", "")
        self._state.update_sensor_health(
            zone_id=zone_id,
            sensor_id=sensor_id,
            health_score=float(payload.get("health_score", 100)),
        )
        return zone_id, {"sensor_id": sensor_id}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Risk Prediction handlers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _handle_risk_assessment(
        self, data: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", payload.get("location_zone_id", "UNKNOWN"))
        self._state.update_risk_assessment(
            zone_id=zone_id,
            risk_score=float(payload.get("risk_score", payload.get("accident_probability", 0)) or 0),
            risk_level=payload.get("risk_level", ""),
            accident_probability=float(payload.get("accident_probability", 0)),
            equipment_id=payload.get("equipment_id", ""),
        )
        return zone_id, {}

    def _handle_risk_score(
        self, data: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", "UNKNOWN")
        self._state.update_risk_score(
            zone_id=zone_id,
            risk_score=float(payload.get("risk_score", 0)),
            risk_level=payload.get("risk_level", ""),
        )
        return zone_id, {}

    def _handle_risk_threshold(
        self, data: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", "UNKNOWN")
        self._state.update_risk_threshold_exceeded(
            zone_id=zone_id,
            threshold_type=payload.get("threshold_type", ""),
            current_value=float(payload.get("current_value", 0)),
            threshold_value=float(payload.get("threshold_value", 0)),
        )
        return zone_id, {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Compound Risk Intelligence handlers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _handle_compound_risk(
        self, data: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", "UNKNOWN")
        self._state.update_compound_risk(
            zone_id=zone_id,
            compound_risk_score=float(payload.get("compound_risk_score", 0)),
            risk_level=payload.get("risk_level", ""),
            confidence_score=float(payload.get("confidence_score", 0)),
            contributing_factors=payload.get("contributing_factors"),
        )
        return zone_id, {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Hazard Propagation handlers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _handle_hazard_detected(
        self, data: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", payload.get("origin_zone", "UNKNOWN"))
        self._state.update_hazard_detected(
            zone_id=zone_id,
            hazard_id=payload.get("hazard_id", ""),
            hazard_type=payload.get("hazard_type", ""),
            severity=payload.get("severity", "HIGH"),
        )
        return zone_id, {}

    def _handle_hazard_propagated(
        self, data: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        payload = data.get("data", data)
        origin_zone = payload.get("origin_zone", "UNKNOWN")
        self._state.update_hazard_propagated(
            origin_zone=origin_zone,
            hazard_type=payload.get("hazard_type", ""),
            propagation_level=payload.get("propagation_level", "CONTAINED"),
            affected_zones=payload.get("affected_zones", []),
            propagation_id=payload.get("propagation_id", ""),
            severity=payload.get("severity", "HIGH"),
        )
        return origin_zone, {}
