"""Digital Twin event handler.

Routes incoming Kafka events to the appropriate TwinStateManager
update method. Follows the same pattern as:
  - HazardPropagationEventHandler
  - CompoundRiskEventHandler

Consumes (10 topics):
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

Publishes: NOTHING (Digital Twin is a pure consumer)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Set

from app.digital_twin.domain.enums import TwinUpdateType
from app.digital_twin.services.twin_state_manager import TwinStateManager
from app.shared.messaging.topics import KafkaTopics

logger = logging.getLogger(__name__)

# Map topic → handler method name
_TOPIC_MAP = {
    KafkaTopics.SENSOR_READING_ANOMALY: "_handle_sensor_anomaly",
    KafkaTopics.SENSOR_STATUS_CHANGED: "_handle_sensor_status",
    KafkaTopics.SENSOR_HEALTH_UPDATED: "_handle_sensor_health",
    KafkaTopics.RISK_ASSESSMENT_GENERATED: "_handle_risk_assessment",
    KafkaTopics.RISK_SCORE_UPDATED: "_handle_risk_score",
    KafkaTopics.RISK_THRESHOLD_EXCEEDED: "_handle_risk_threshold",
    KafkaTopics.COMPOUND_RISK_DETECTED: "_handle_compound_risk",
    KafkaTopics.HAZARD_DETECTED: "_handle_hazard_detected",
    KafkaTopics.HAZARD_PROPAGATED: "_handle_hazard_propagated",
}


class DigitalTwinEventHandler:
    """Handles incoming Kafka events and updates the twin state.

    All errors are caught and logged — failed events do not crash
    the consumer.

    Args:
        state_manager: The shared TwinStateManager singleton.
    """

    def __init__(self, state_manager: TwinStateManager) -> None:
        self._state = state_manager

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

    # ── Main dispatch ──

    async def handle_event(
        self, topic: str, data: Dict[str, Any],
    ) -> None:
        """Process a single Kafka event.

        Routes to the appropriate handler based on topic.
        Deduplicates by event_id.
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

        handler_name = _TOPIC_MAP.get(topic)
        if handler_name is None:
            logger.warning(
                "Digital Twin received event for unhandled topic: %s",
                topic,
            )
            self._events_skipped += 1
            return

        try:
            handler_method = getattr(self, handler_name)
            handler_method(data)

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

        except Exception:
            self._events_failed += 1
            logger.exception(
                "Digital Twin failed to process event: "
                "topic=%s event_id=%s",
                topic, event_id,
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Sensor Intelligence handlers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _handle_sensor_anomaly(self, data: Dict[str, Any]) -> None:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", payload.get("location_zone_id", "UNKNOWN"))
        self._state.update_sensor_anomaly(
            zone_id=zone_id,
            sensor_id=payload.get("sensor_id", ""),
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

    def _handle_sensor_status(self, data: Dict[str, Any]) -> None:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", payload.get("location_zone_id", "UNKNOWN"))
        self._state.update_sensor_status(
            zone_id=zone_id,
            sensor_id=payload.get("sensor_id", ""),
            status=payload.get("status", payload.get("new_status", "ACTIVE")),
        )

    def _handle_sensor_health(self, data: Dict[str, Any]) -> None:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", payload.get("location_zone_id", "UNKNOWN"))
        self._state.update_sensor_health(
            zone_id=zone_id,
            sensor_id=payload.get("sensor_id", ""),
            health_score=float(payload.get("health_score", 100)),
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Risk Prediction handlers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _handle_risk_assessment(self, data: Dict[str, Any]) -> None:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", payload.get("location_zone_id", "UNKNOWN"))
        self._state.update_risk_assessment(
            zone_id=zone_id,
            risk_score=float(payload.get("risk_score", payload.get("accident_probability", 0)) or 0),
            risk_level=payload.get("risk_level", ""),
            accident_probability=float(payload.get("accident_probability", 0)),
            equipment_id=payload.get("equipment_id", ""),
        )

    def _handle_risk_score(self, data: Dict[str, Any]) -> None:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", "UNKNOWN")
        self._state.update_risk_score(
            zone_id=zone_id,
            risk_score=float(payload.get("risk_score", 0)),
            risk_level=payload.get("risk_level", ""),
        )

    def _handle_risk_threshold(self, data: Dict[str, Any]) -> None:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", "UNKNOWN")
        self._state.update_risk_threshold_exceeded(
            zone_id=zone_id,
            threshold_type=payload.get("threshold_type", ""),
            current_value=float(payload.get("current_value", 0)),
            threshold_value=float(payload.get("threshold_value", 0)),
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Compound Risk Intelligence handlers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _handle_compound_risk(self, data: Dict[str, Any]) -> None:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", "UNKNOWN")
        self._state.update_compound_risk(
            zone_id=zone_id,
            compound_risk_score=float(payload.get("compound_risk_score", 0)),
            risk_level=payload.get("risk_level", ""),
            confidence_score=float(payload.get("confidence_score", 0)),
            contributing_factors=payload.get("contributing_factors"),
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Hazard Propagation handlers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _handle_hazard_detected(self, data: Dict[str, Any]) -> None:
        payload = data.get("data", data)
        zone_id = payload.get("zone_id", payload.get("origin_zone", "UNKNOWN"))
        self._state.update_hazard_detected(
            zone_id=zone_id,
            hazard_id=payload.get("hazard_id", ""),
            hazard_type=payload.get("hazard_type", ""),
            severity=payload.get("severity", "HIGH"),
        )

    def _handle_hazard_propagated(self, data: Dict[str, Any]) -> None:
        payload = data.get("data", data)
        self._state.update_hazard_propagated(
            origin_zone=payload.get("origin_zone", "UNKNOWN"),
            hazard_type=payload.get("hazard_type", ""),
            propagation_level=payload.get("propagation_level", "CONTAINED"),
            affected_zones=payload.get("affected_zones", []),
            propagation_id=payload.get("propagation_id", ""),
            severity=payload.get("severity", "HIGH"),
        )
