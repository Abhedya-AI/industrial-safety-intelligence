"""Hazard Propagation event publisher.

Publishes ``hazard.propagated`` events via the shared Kafka
producer. No new producer/schema/topic is created — everything is
reused from ``app.shared.messaging``.

Follows the same pattern as CompoundRiskPublisher.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.hazard_propagation.services.propagation_engine import PropagationResult
from app.shared.messaging.events import BaseEvent
from app.shared.messaging.producer import KafkaEventProducer
from app.shared.messaging.topics import KafkaTopics

logger = logging.getLogger(__name__)

SOURCE_SYSTEM = "hazard_propagation_engine"


class HazardPropagationPublisher:
    """Publishes hazard propagation events to Kafka.

    Reuses the shared ``KafkaEventProducer`` — does NOT create its own.

    Usage:
        publisher = HazardPropagationPublisher(producer)
        publisher.publish_hazard_propagated(result)
    """

    def __init__(self, producer: KafkaEventProducer) -> None:
        self._producer = producer
        self._published_count: int = 0
        self._failed_count: int = 0

    @property
    def published_count(self) -> int:
        return self._published_count

    @property
    def failed_count(self) -> int:
        return self._failed_count

    def publish_hazard_propagated(
        self,
        result: PropagationResult,
        correlation_id: Optional[str] = None,
    ) -> Optional[BaseEvent]:
        """Publish a ``hazard.propagated`` event.

        Follows the PS-1 v2.0 standard event format (§5.3).

        Args:
            result: The propagation simulation result.
            correlation_id: Optional ID for event chain tracing.

        Returns:
            The published BaseEvent, or None on failure.
        """
        data = self._build_event_data(result)

        try:
            event = self._producer.publish(
                topic=KafkaTopics.HAZARD_PROPAGATED,
                data=data,
                source_system=SOURCE_SYSTEM,
                key=result.origin_zone,
                correlation_id=correlation_id,
            )
            self._published_count += 1
            logger.info(
                "Published hazard.propagated: propagation_id=%s origin=%s "
                "hazard_type=%s affected_zones=%d level=%s",
                result.propagation_id, result.origin_zone,
                result.hazard_type, result.total_affected_zones,
                result.propagation_level.value,
            )
            return event
        except Exception:
            self._failed_count += 1
            logger.exception(
                "Failed to publish hazard.propagated for propagation_id=%s",
                result.propagation_id,
            )
            return None

    @staticmethod
    def _build_event_data(result: PropagationResult) -> Dict[str, Any]:
        """Build the ``data`` payload for a hazard.propagated event."""
        return {
            "propagation_id": result.propagation_id,
            "hazard_type": result.hazard_type,
            "origin_zone": result.origin_zone,
            "propagation_level": result.propagation_level.value,
            "status": result.status.value,
            "affected_zones": result.affected_zone_ids,
            "total_affected_zones": result.total_affected_zones,
            "total_workers_at_risk": result.total_workers_at_risk,
            "impact_radius_meters": result.impact_radius_meters,
            "time_to_critical_minutes": result.time_to_critical_minutes,
            "impact_scores": result.impact_scores,
            "propagation_probabilities": result.propagation_probabilities,
            "affected_equipment": [
                {
                    "equipment_id": eq.equipment_id,
                    "equipment_type": eq.equipment_type,
                    "zone_id": eq.zone_id,
                    "impact_score": eq.impact_score,
                    "is_critical": eq.is_critical,
                }
                for eq in result.affected_equipment
            ],
            "propagation_paths": [
                {
                    "from_zone": p.from_zone,
                    "to_zone": p.to_zone,
                    "probability": p.probability,
                    "estimated_time_minutes": p.estimated_time_minutes,
                }
                for p in result.propagation_paths
            ],
            "recommended_action": result.recommended_action,
        }
