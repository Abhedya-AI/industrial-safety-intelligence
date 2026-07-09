"""Hazard Propagation event handler.

Processes incoming events from upstream modules and orchestrates
the full hazard propagation pipeline:

    Incoming Event → Validate → Load Graph → Execute Propagation
        → Score Zones → Score Equipment → Publish hazard.propagated

No business logic is implemented here — all computation is delegated
to the HazardPropagationEngine. This module is purely integration glue.

Consumes:
  - compound.risk.detected
  - hazard.detected

Publishes:
  - hazard.propagated
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

from app.hazard_propagation.domain.exceptions import (
    HazardPropagationError,
    PropagationSimulationError,
    ZoneNotFoundError,
)
from app.hazard_propagation.messaging.publisher import HazardPropagationPublisher
from app.hazard_propagation.repositories.graph_repository import GraphRepository
from app.hazard_propagation.services.propagation_engine import (
    HazardPropagationEngine,
    PropagationResult,
)
from app.shared.messaging.topics import KafkaTopics

logger = logging.getLogger(__name__)


class HazardPropagationEventHandler:
    """Handles incoming Kafka events and orchestrates hazard propagation.

    Event Processing Flow:
        1. Validate event structure (PS-1 v2.0 §5.3 format)
        2. Extract hazard_type, origin_zone, compound_risk_score
        3. Load graph topology via GraphRepository
        4. Execute propagation algorithm
        5. Publish hazard.propagated event

    All errors are caught and logged — failed events do not crash the consumer.

    Args:
        propagation_engine: The core propagation algorithm.
        publisher: The event publisher for hazard.propagated.
        graph_repo: The graph repository (for zone validation).
    """

    def __init__(
        self,
        propagation_engine: HazardPropagationEngine,
        publisher: HazardPropagationPublisher,
        graph_repo: GraphRepository,
    ) -> None:
        self._engine = propagation_engine
        self._publisher = publisher
        self._graph_repo = graph_repo

        # Duplicate event detection
        self._processed_event_ids: Set[str] = set()
        self._max_event_cache_size: int = 10000

        # Metrics
        self._events_processed: int = 0
        self._events_failed: int = 0
        self._events_skipped: int = 0
        self._propagations_executed: int = 0

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
    def propagations_executed(self) -> int:
        return self._propagations_executed

    # ── Topic routing ──

    async def handle_event(self, topic: str, data: Dict[str, Any]) -> bool:
        """Route an incoming event to the appropriate handler.

        Args:
            topic: Kafka topic the event came from.
            data: Deserialized event payload (full BaseEvent dict).

        Returns:
            True if the event was processed successfully, False otherwise.
        """
        trace_id = data.get("event_id", str(uuid.uuid4())[:8])
        start_time = time.monotonic()

        logger.info(
            "[%s] Processing event from '%s': event_type=%s",
            trace_id, topic, data.get("event_type", "unknown"),
        )

        try:
            # Step 1: Validate event structure
            event_data = self._validate_event(data)

            # Step 2: Check for duplicates
            event_id = data.get("event_id", "")
            if self._is_duplicate(event_id):
                self._events_skipped += 1
                logger.warning(
                    "[%s] Duplicate event skipped: event_id=%s",
                    trace_id, event_id,
                )
                return False

            # Step 3: Route to topic-specific handler
            if topic == KafkaTopics.COMPOUND_RISK_DETECTED:
                await self._handle_compound_risk(trace_id, event_data, data)
            elif topic == KafkaTopics.HAZARD_DETECTED:
                await self._handle_hazard_detected(trace_id, event_data, data)
            else:
                logger.warning(
                    "[%s] Unhandled topic: %s", trace_id, topic,
                )
                return False

            # Step 4: Mark as processed
            self._mark_processed(event_id)
            self._events_processed += 1

            elapsed = (time.monotonic() - start_time) * 1000
            logger.info(
                "[%s] Event processed successfully in %.1fms",
                trace_id, elapsed,
            )
            return True

        except HazardPropagationError as exc:
            self._events_failed += 1
            logger.error(
                "[%s] Domain error processing event: %s",
                trace_id, str(exc),
            )
            return False
        except Exception:
            self._events_failed += 1
            logger.exception(
                "[%s] Unexpected error processing event", trace_id,
            )
            return False

    # ── Validation ──

    @staticmethod
    def _validate_event(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the event has the required PS-1 v2.0 fields.

        Required fields: event_type, event_id, timestamp, data.

        Raises:
            HazardPropagationError: If required fields are missing.
        """
        missing = []
        for field_name in ("event_type", "event_id", "timestamp"):
            if field_name not in data:
                missing.append(field_name)

        if "data" not in data or not isinstance(data.get("data"), dict):
            missing.append("data")

        if missing:
            raise HazardPropagationError(
                f"Malformed event — missing required fields: "
                f"{', '.join(missing)}",
            )

        return data["data"]

    # ── Duplicate detection ──

    def _is_duplicate(self, event_id: str) -> bool:
        """Check if an event has already been processed."""
        if not event_id:
            return False
        return event_id in self._processed_event_ids

    def _mark_processed(self, event_id: str) -> None:
        """Mark an event as processed (with LRU eviction)."""
        if not event_id:
            return
        self._processed_event_ids.add(event_id)
        # LRU eviction when cache gets too large
        if len(self._processed_event_ids) > self._max_event_cache_size:
            # Remove oldest entries (set doesn't preserve order,
            # so we just clear half)
            to_remove = len(self._processed_event_ids) // 2
            for _ in range(to_remove):
                self._processed_event_ids.pop()

    # ── Per-topic handlers ──

    async def _handle_compound_risk(
        self,
        trace_id: str,
        event_data: Dict[str, Any],
        full_event: Dict[str, Any],
    ) -> None:
        """Handle a ``compound.risk.detected`` event.

        Extracts:
          - zone_id → origin_zone
          - compound_risk_score → propagation input
          - risk_level → used to determine hazard_type
          - equipment_id → optional context

        Triggers propagation when compound_risk_score indicates
        elevated risk (score ≥ 40).
        """
        zone_id = event_data.get("zone_id")
        if not zone_id:
            logger.warning(
                "[%s] compound.risk.detected missing zone_id", trace_id,
            )
            return

        compound_risk_score = float(
            event_data.get("compound_risk_score", 0.0),
        )
        risk_level = event_data.get("risk_level", "LOW")

        # Only trigger propagation for elevated risk
        if compound_risk_score < 40.0:
            logger.debug(
                "[%s] Compound risk score %.1f below propagation "
                "threshold (40.0) — skipping",
                trace_id, compound_risk_score,
            )
            self._events_skipped += 1
            return

        # Determine hazard type from context
        hazard_type = self._infer_hazard_type(event_data)

        logger.info(
            "[%s] Compound risk detected: zone=%s score=%.1f level=%s "
            "→ triggering propagation (type=%s)",
            trace_id, zone_id, compound_risk_score, risk_level, hazard_type,
        )

        await self._execute_propagation(
            trace_id=trace_id,
            hazard_type=hazard_type,
            origin_zone=zone_id,
            compound_risk_score=compound_risk_score,
            correlation_id=full_event.get("event_id"),
        )

    async def _handle_hazard_detected(
        self,
        trace_id: str,
        event_data: Dict[str, Any],
        full_event: Dict[str, Any],
    ) -> None:
        """Handle a ``hazard.detected`` event.

        Extracts:
          - hazard_type → direct from event
          - zone_id → origin_zone
          - severity → used as compound_risk_score proxy
        """
        zone_id = event_data.get("zone_id")
        if not zone_id:
            logger.warning(
                "[%s] hazard.detected missing zone_id", trace_id,
            )
            return

        hazard_type = event_data.get("hazard_type", "GAS_LEAK")
        severity = event_data.get("severity", "HIGH")

        # Map severity to a compound risk score
        severity_scores = {
            "LOW": 25.0,
            "MEDIUM": 50.0,
            "HIGH": 75.0,
            "CRITICAL": 95.0,
        }
        compound_risk_score = severity_scores.get(severity, 50.0)

        logger.info(
            "[%s] Hazard detected: zone=%s type=%s severity=%s "
            "→ triggering propagation",
            trace_id, zone_id, hazard_type, severity,
        )

        await self._execute_propagation(
            trace_id=trace_id,
            hazard_type=hazard_type,
            origin_zone=zone_id,
            compound_risk_score=compound_risk_score,
            correlation_id=full_event.get("event_id"),
        )

    # ── Core propagation pipeline ──

    async def _execute_propagation(
        self,
        trace_id: str,
        hazard_type: str,
        origin_zone: str,
        compound_risk_score: float,
        correlation_id: Optional[str] = None,
    ) -> Optional[PropagationResult]:
        """Execute the propagation algorithm and publish results.

        Steps:
            1. Validate origin zone exists in graph
            2. Execute propagation engine
            3. Publish hazard.propagated event
        """
        # Step 1: Validate zone exists
        zone = await self._graph_repo.get_zone(origin_zone)
        if zone is None:
            logger.warning(
                "[%s] Origin zone '%s' not found in graph — skipping",
                trace_id, origin_zone,
            )
            self._events_skipped += 1
            return None

        # Step 2: Execute propagation
        try:
            result = await self._engine.propagate(
                hazard_type=hazard_type,
                origin_zone=origin_zone,
                compound_risk_score=compound_risk_score,
            )
        except ZoneNotFoundError:
            logger.warning(
                "[%s] Zone '%s' not found during propagation",
                trace_id, origin_zone,
            )
            self._events_failed += 1
            return None
        except PropagationSimulationError as exc:
            logger.error(
                "[%s] Propagation simulation failed: %s",
                trace_id, str(exc),
            )
            self._events_failed += 1
            return None

        self._propagations_executed += 1

        logger.info(
            "[%s] Propagation complete: propagation_id=%s "
            "affected_zones=%d level=%s workers_at_risk=%d",
            trace_id, result.propagation_id,
            result.total_affected_zones, result.propagation_level.value,
            result.total_workers_at_risk,
        )

        # Step 3: Publish hazard.propagated
        try:
            self._publisher.publish_hazard_propagated(
                result=result,
                correlation_id=correlation_id,
            )
        except Exception:
            logger.exception(
                "[%s] Failed to publish hazard.propagated for "
                "propagation_id=%s",
                trace_id, result.propagation_id,
            )
            # Don't re-raise — propagation succeeded, publish failed

        return result

    # ── Utility ──

    @staticmethod
    def _infer_hazard_type(event_data: Dict[str, Any]) -> str:
        """Infer hazard type from compound risk event data.

        Uses contributing factors and sensor data to determine
        the most likely hazard type.
        """
        factors = event_data.get("contributing_factors", {})

        # Check for specific hazard indicators
        if factors.get("gas_risk", 0) > 0.5:
            return "GAS_LEAK"
        if factors.get("fire_risk", 0) > 0.5:
            return "FIRE"
        if factors.get("temperature_risk", 0) > 0.5:
            return "TEMPERATURE_ANOMALY"
        if factors.get("pressure_risk", 0) > 0.5:
            return "PRESSURE_ANOMALY"
        if factors.get("electrical_risk", 0) > 0.5:
            return "ELECTRICAL_FAULT"

        # Default based on risk level
        risk_level = event_data.get("risk_level", "HIGH")
        if risk_level == "CRITICAL":
            return "GAS_LEAK"

        return "GAS_LEAK"

    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        self._events_processed = 0
        self._events_failed = 0
        self._events_skipped = 0
        self._propagations_executed = 0
        self._processed_event_ids.clear()
