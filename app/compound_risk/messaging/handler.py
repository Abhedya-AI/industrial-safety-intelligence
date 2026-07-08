"""Compound Risk event handler.

Processes incoming events from upstream modules and orchestrates
the full compound risk pipeline:

    Incoming Event → Validate → Fetch Data → Rule Engine
        → Aggregation Engine → Explanation → Persist → Publish

No business logic is implemented here — all computation is delegated
to existing services. This module is purely integration glue.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.compound_risk.domain.exceptions import (
    CompoundRiskAnalysisFailedError,
    CompoundRiskError,
)
from app.compound_risk.messaging.publisher import CompoundRiskPublisher
from app.compound_risk.rules.rule_engine import CompoundRiskRuleEngine
from app.compound_risk.services.compound_risk_service import (
    CompoundRiskAggregationService,
    CompoundRiskInput,
)
from app.compound_risk.services.explainability_service import ExplainabilityService
from app.shared.messaging.topics import KafkaTopics

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event state accumulator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ZoneRiskState:
    """Accumulated risk state for a zone from incoming events.

    Collects signals from multiple event types before triggering
    compound risk analysis.
    """

    zone_id: str
    equipment_id: Optional[str] = None

    # From sensor.reading.anomaly
    isolation_forest_score: float = 0.0
    autoencoder_score: float = 0.0
    sensor_health_score: float = 100.0

    # From risk.assessment.generated / risk.score.updated
    accident_probability: float = 0.0
    risk_score: float = 0.0

    # Operational context
    active_alert_count: int = 0
    alert_severity_max: float = 0.0
    threshold_violation_count: int = 0

    # Sensor reading facts for rule engine
    sensor_facts: Dict[str, Any] = field(default_factory=dict)

    # Tracking
    last_updated: Optional[str] = None
    event_count: int = 0

    def to_compound_input(self) -> CompoundRiskInput:
        """Convert accumulated state to a CompoundRiskInput."""
        return CompoundRiskInput(
            isolation_forest_score=self.isolation_forest_score,
            autoencoder_score=self.autoencoder_score,
            accident_probability=self.accident_probability,
            risk_score=self.risk_score,
            sensor_health_score=self.sensor_health_score,
            active_alert_count=self.active_alert_count,
            alert_severity_max=self.alert_severity_max,
            threshold_violation_count=self.threshold_violation_count,
            equipment_id=self.equipment_id,
            zone_id=self.zone_id,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event handler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CompoundRiskEventHandler:
    """Handles incoming Kafka events and orchestrates compound risk analysis.

    Event Processing Flow:
        1. Validate event structure
        2. Extract and accumulate zone risk state
        3. Execute rule engine against sensor facts
        4. Execute aggregation engine
        5. Generate explanation
        6. Persist compound risk analysis
        7. Publish compound.risk.detected event

    All errors are caught and logged — failed events do not crash the consumer.

    Args:
        aggregation_service: The compound risk aggregation service.
        rule_engine: The configurable rule engine.
        explainability_service: The explanation generator.
        publisher: The event publisher.
    """

    def __init__(
        self,
        aggregation_service: CompoundRiskAggregationService,
        rule_engine: CompoundRiskRuleEngine,
        explainability_service: ExplainabilityService,
        publisher: CompoundRiskPublisher,
    ) -> None:
        self._aggregation = aggregation_service
        self._rule_engine = rule_engine
        self._explainability = explainability_service
        self._publisher = publisher

        # Per-zone accumulated state
        self._zone_states: Dict[str, ZoneRiskState] = {}

        # Metrics
        self._events_processed: int = 0
        self._events_failed: int = 0
        self._analyses_produced: int = 0

    @property
    def events_processed(self) -> int:
        return self._events_processed

    @property
    def events_failed(self) -> int:
        return self._events_failed

    @property
    def analyses_produced(self) -> int:
        return self._analyses_produced

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
            # Step 1: Validate
            event_data = self._validate_event(data)

            # Step 2: Route to handler
            if topic == KafkaTopics.SENSOR_READING_ANOMALY:
                await self._handle_anomaly(trace_id, event_data)
            elif topic == KafkaTopics.RISK_ASSESSMENT_GENERATED:
                await self._handle_risk_assessment(trace_id, event_data)
            elif topic == KafkaTopics.RISK_SCORE_UPDATED:
                await self._handle_risk_score_updated(trace_id, event_data)
            else:
                logger.warning("[%s] Unhandled topic: %s", trace_id, topic)
                return False

            self._events_processed += 1
            elapsed = (time.monotonic() - start_time) * 1000
            logger.info(
                "[%s] Event processed successfully in %.1fms",
                trace_id, elapsed,
            )
            return True

        except CompoundRiskError as exc:
            self._events_failed += 1
            logger.error(
                "[%s] Domain error processing event: %s",
                trace_id, exc.message,
            )
            return False
        except Exception:
            self._events_failed += 1
            logger.exception("[%s] Unexpected error processing event", trace_id)
            return False

    # ── Validation ──

    @staticmethod
    def _validate_event(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the event has the required PS-1 v2.0 fields.

        Required fields: event_type, event_id, timestamp, data.

        Raises:
            CompoundRiskAnalysisFailedError: If required fields are missing.
        """
        missing = []
        for field_name in ("event_type", "event_id", "timestamp"):
            if field_name not in data:
                missing.append(field_name)

        if "data" not in data or not isinstance(data.get("data"), dict):
            missing.append("data")

        if missing:
            raise CompoundRiskAnalysisFailedError(
                f"Malformed event — missing required fields: {', '.join(missing)}",
            )

        return data["data"]

    # ── Per-topic handlers ──

    async def _handle_anomaly(
        self, trace_id: str, event_data: Dict[str, Any],
    ) -> None:
        """Handle a ``sensor.reading.anomaly`` event.

        Extracts anomaly scores, updates zone state, and triggers
        compound risk analysis if sufficient data is available.
        """
        zone_id = event_data.get("zone_id", "UNKNOWN")
        sensor_id = event_data.get("sensor_id")
        equipment_id = event_data.get("equipment_id")

        state = self._get_or_create_state(zone_id, equipment_id)

        # Update anomaly scores
        state.isolation_forest_score = max(
            state.isolation_forest_score,
            float(event_data.get("isolation_forest_score", 0)),
        )
        state.autoencoder_score = max(
            state.autoencoder_score,
            float(event_data.get("autoencoder_score", 0)),
        )
        state.sensor_health_score = float(
            event_data.get("sensor_health_score", state.sensor_health_score),
        )

        # Update sensor facts for rule engine
        if "temperature_celsius" in event_data:
            state.sensor_facts["temperature_celsius"] = float(
                event_data["temperature_celsius"],
            )
        if "gas_level_ppm" in event_data:
            state.sensor_facts["gas_level_ppm"] = float(event_data["gas_level_ppm"])
        if "pressure_bar" in event_data:
            state.sensor_facts["pressure_bar"] = float(event_data["pressure_bar"])
        if "vibration_level" in event_data:
            state.sensor_facts["vibration_level"] = float(event_data["vibration_level"])

        state.sensor_facts["sensor_health_score"] = state.sensor_health_score
        state.threshold_violation_count = int(
            event_data.get("threshold_violation_count", state.threshold_violation_count),
        )
        state.active_alert_count = int(
            event_data.get("active_alert_count", state.active_alert_count),
        )
        state.alert_severity_max = float(
            event_data.get("alert_severity_max", state.alert_severity_max),
        )

        state.event_count += 1
        state.last_updated = datetime.now(timezone.utc).isoformat()

        logger.debug(
            "[%s] Updated zone state: zone=%s IF=%.2f AE=%.2f events=%d",
            trace_id, zone_id, state.isolation_forest_score,
            state.autoencoder_score, state.event_count,
        )

        # Trigger analysis
        await self._run_analysis(trace_id, state, event_data.get("event_id"))

    async def _handle_risk_assessment(
        self, trace_id: str, event_data: Dict[str, Any],
    ) -> None:
        """Handle a ``risk.assessment.generated`` event."""
        zone_id = event_data.get("zone_id", "UNKNOWN")
        equipment_id = event_data.get("equipment_id")

        state = self._get_or_create_state(zone_id, equipment_id)

        state.accident_probability = float(
            event_data.get("accident_probability", state.accident_probability),
        )
        state.risk_score = float(
            event_data.get("risk_score", state.risk_score),
        )

        state.event_count += 1
        state.last_updated = datetime.now(timezone.utc).isoformat()

        logger.debug(
            "[%s] Risk assessment received: zone=%s prob=%.2f score=%.1f",
            trace_id, zone_id, state.accident_probability, state.risk_score,
        )

        await self._run_analysis(trace_id, state, event_data.get("event_id"))

    async def _handle_risk_score_updated(
        self, trace_id: str, event_data: Dict[str, Any],
    ) -> None:
        """Handle a ``risk.score.updated`` event."""
        zone_id = event_data.get("zone_id", "UNKNOWN")
        equipment_id = event_data.get("equipment_id")

        state = self._get_or_create_state(zone_id, equipment_id)

        if "risk_score" in event_data:
            state.risk_score = float(event_data["risk_score"])
        if "accident_probability" in event_data:
            state.accident_probability = float(event_data["accident_probability"])
        if "risk_level" in event_data:
            state.sensor_facts["risk_level"] = event_data["risk_level"]

        state.event_count += 1
        state.last_updated = datetime.now(timezone.utc).isoformat()

        logger.debug(
            "[%s] Risk score updated: zone=%s score=%.1f",
            trace_id, zone_id, state.risk_score,
        )

        await self._run_analysis(trace_id, state, event_data.get("event_id"))

    # ── Core analysis pipeline ──

    async def _run_analysis(
        self,
        trace_id: str,
        state: ZoneRiskState,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Execute the full compound risk analysis pipeline.

        Steps:
            1. Build CompoundRiskInput from accumulated state
            2. Execute Rule Engine
            3. Execute Aggregation Engine
            4. Generate Explanation
            5. Persist result
            6. Publish compound.risk.detected
        """
        inp = state.to_compound_input()

        # Step 2: Rule engine
        rule_result = self._rule_engine.evaluate(state.sensor_facts)

        # Step 3: Aggregation
        result = self._aggregation.compute(inp)

        # Step 4: Explanation
        explanation = self._explainability.explain(inp, result, rule_result)

        # Step 5: Persist
        recommendation = "; ".join(explanation.recommendations) if explanation.recommendations else None
        model = await self._aggregation.compute_and_persist(
            inp, recommendation=recommendation,
        )

        # Step 6: Publish
        self._publisher.publish_compound_risk_detected(
            model=model,
            result=result,
            correlation_id=correlation_id,
        )

        self._analyses_produced += 1

        logger.info(
            "[%s] Compound risk analysis complete: id=%s zone=%s "
            "score=%.1f level=%s rules_triggered=%d",
            trace_id, model.id, state.zone_id,
            result.compound_risk_score, result.risk_level.value,
            len(rule_result.triggered_rules),
        )

    # ── State management ──

    def _get_or_create_state(
        self,
        zone_id: str,
        equipment_id: Optional[str] = None,
    ) -> ZoneRiskState:
        """Get or create the accumulated state for a zone."""
        if zone_id not in self._zone_states:
            self._zone_states[zone_id] = ZoneRiskState(
                zone_id=zone_id,
                equipment_id=equipment_id,
            )
        state = self._zone_states[zone_id]
        if equipment_id and not state.equipment_id:
            state.equipment_id = equipment_id
        return state

    def get_zone_state(self, zone_id: str) -> Optional[ZoneRiskState]:
        """Get the current accumulated state for a zone (for testing)."""
        return self._zone_states.get(zone_id)

    def reset_zone_state(self, zone_id: str) -> None:
        """Reset accumulated state for a zone."""
        self._zone_states.pop(zone_id, None)

    def reset_all(self) -> None:
        """Reset all accumulated state."""
        self._zone_states.clear()
        self._events_processed = 0
        self._events_failed = 0
        self._analyses_produced = 0
