"""Compound Risk event publisher.

Publishes ``compound.risk.detected`` events via the shared Kafka
producer. No new producer/schema/topic is created — everything is
reused from ``app.shared.messaging``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.compound_risk.models.compound_risk_model import CompoundRiskModel
from app.compound_risk.services.compound_risk_service import CompoundRiskResult
from app.shared.messaging.events import BaseEvent
from app.shared.messaging.producer import KafkaEventProducer
from app.shared.messaging.topics import KafkaTopics

logger = logging.getLogger(__name__)

SOURCE_SYSTEM = "compound_risk_intelligence"


class CompoundRiskPublisher:
    """Publishes compound risk events to Kafka.

    Reuses the shared ``KafkaEventProducer`` — does NOT create its own.

    Usage:
        publisher = CompoundRiskPublisher(producer)
        publisher.publish_compound_risk_detected(model, result)
    """

    def __init__(self, producer: KafkaEventProducer) -> None:
        self._producer = producer

    def publish_compound_risk_detected(
        self,
        model: CompoundRiskModel,
        result: CompoundRiskResult,
        correlation_id: Optional[str] = None,
    ) -> Optional[BaseEvent]:
        """Publish a ``compound.risk.detected`` event.

        Follows the PS-1 v2.0 standard event format (§5.3).

        Args:
            model: The persisted CompoundRiskModel.
            result: The aggregation engine result.
            correlation_id: Optional ID for event chain tracing.

        Returns:
            The published BaseEvent, or None on failure.
        """
        data = self._build_event_data(model, result)

        try:
            event = self._producer.publish(
                topic=KafkaTopics.COMPOUND_RISK_DETECTED,
                data=data,
                source_system=SOURCE_SYSTEM,
                key=model.zone_id,
                correlation_id=correlation_id,
            )
            logger.info(
                "Published compound.risk.detected: id=%s zone=%s level=%s score=%.2f",
                model.id, model.zone_id, model.risk_level,
                model.compound_risk_score,
            )
            return event
        except Exception:
            logger.exception(
                "Failed to publish compound.risk.detected for id=%s",
                model.id,
            )
            return None

    @staticmethod
    def _build_event_data(
        model: CompoundRiskModel,
        result: CompoundRiskResult,
    ) -> Dict[str, Any]:
        """Build the ``data`` payload for a compound risk event."""
        return {
            "analysis_id": model.id,
            "equipment_id": model.equipment_id,
            "zone_id": model.zone_id,
            "anomaly_score": model.anomaly_score,
            "accident_probability": model.accident_probability,
            "risk_score": model.risk_score,
            "sensor_health_score": model.sensor_health_score,
            "compound_risk_score": result.compound_risk_score,
            "risk_level": result.risk_level.value,
            "confidence_score": result.confidence_score,
            "contributing_factors": result.contributing_factors,
            "component_scores": result.component_scores,
            "recommendation": model.recommendation,
            "created_at": (
                model.created_at.isoformat()
                if model.created_at else None
            ),
        }
