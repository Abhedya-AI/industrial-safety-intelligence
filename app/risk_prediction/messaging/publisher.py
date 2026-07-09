"""Risk Prediction event publisher.

Publishes domain events after successful prediction operations via the
shared Kafka producer. No new producer/schema/topic infrastructure is
created — everything is reused from ``app.shared.messaging``.

Events published:
  - risk.assessment.generated  — after a prediction is successfully generated
  - risk.score.updated         — when a new risk score is persisted
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.shared.messaging.events import BaseEvent
from app.shared.messaging.producer import KafkaEventProducer
from app.shared.messaging.topics import KafkaTopics

logger = logging.getLogger(__name__)

SOURCE_SYSTEM = "risk_prediction"


class RiskPredictionPublisher:
    """Publishes Risk Prediction domain events to Kafka.

    Reuses the shared ``KafkaEventProducer`` — does NOT create its own.
    All publishing failures are caught and logged — they must never
    crash the prediction operation that triggered them.

    Usage:
        publisher = RiskPredictionPublisher(producer)
        publisher.publish_assessment_generated(prediction)
    """

    def __init__(self, producer: KafkaEventProducer) -> None:
        self._producer = producer
        self._published_count: int = 0
        self._failed_count: int = 0

    # ── Properties ──

    @property
    def published_count(self) -> int:
        return self._published_count

    @property
    def failed_count(self) -> int:
        return self._failed_count

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # risk.assessment.generated
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def publish_assessment_generated(
        self,
        prediction_id: str,
        accident_probability: float,
        risk_score: int,
        risk_level: str,
        confidence_score: float,
        model_name: str,
        model_version: str,
        sensor_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        prediction_timestamp: Optional[str] = None,
    ) -> Optional[BaseEvent]:
        """Publish a ``risk.assessment.generated`` event.

        Called after a prediction is successfully computed and persisted.
        """
        data: Dict[str, Any] = {
            "prediction_id": prediction_id,
            "accident_probability": accident_probability,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "confidence_score": confidence_score,
            "model_name": model_name,
            "model_version": model_version,
        }
        if sensor_id:
            data["sensor_id"] = sensor_id
        if equipment_id:
            data["equipment_id"] = equipment_id
        if zone_id:
            data["zone_id"] = zone_id
        if prediction_timestamp:
            data["prediction_timestamp"] = prediction_timestamp

        return self._publish(
            KafkaTopics.RISK_ASSESSMENT_GENERATED,
            data,
            key=zone_id or sensor_id,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # risk.score.updated
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def publish_score_updated(
        self,
        prediction_id: str,
        risk_score: int,
        risk_level: str,
        accident_probability: float,
        confidence_score: float,
        sensor_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
        zone_id: Optional[str] = None,
    ) -> Optional[BaseEvent]:
        """Publish a ``risk.score.updated`` event.

        Called when a new risk score is persisted to the database.
        """
        data: Dict[str, Any] = {
            "prediction_id": prediction_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "accident_probability": accident_probability,
            "confidence_score": confidence_score,
        }
        if sensor_id:
            data["sensor_id"] = sensor_id
        if equipment_id:
            data["equipment_id"] = equipment_id
        if zone_id:
            data["zone_id"] = zone_id

        return self._publish(
            KafkaTopics.RISK_SCORE_UPDATED,
            data,
            key=zone_id or sensor_id,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Convenience: publish both after a prediction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def publish_prediction_events(self, prediction) -> None:
        """Publish both events for a completed prediction.

        Accepts a RiskPredictionModel and fires:
          1. risk.assessment.generated
          2. risk.score.updated

        This is the recommended API-layer hook — call this once after
        a successful prediction rather than calling each method
        individually.
        """
        ts = (
            prediction.prediction_timestamp.isoformat()
            if prediction.prediction_timestamp else None
        )

        self.publish_assessment_generated(
            prediction_id=prediction.id,
            accident_probability=prediction.accident_probability,
            risk_score=prediction.predicted_risk_score,
            risk_level=prediction.risk_level,
            confidence_score=prediction.confidence_score,
            model_name=prediction.model_name,
            model_version=prediction.model_version,
            sensor_id=prediction.sensor_id,
            equipment_id=prediction.equipment_id,
            zone_id=prediction.zone_id,
            prediction_timestamp=ts,
        )

        self.publish_score_updated(
            prediction_id=prediction.id,
            risk_score=prediction.predicted_risk_score,
            risk_level=prediction.risk_level,
            accident_probability=prediction.accident_probability,
            confidence_score=prediction.confidence_score,
            sensor_id=prediction.sensor_id,
            equipment_id=prediction.equipment_id,
            zone_id=prediction.zone_id,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _publish(
        self,
        topic: str,
        data: Dict[str, Any],
        key: Optional[str] = None,
    ) -> Optional[BaseEvent]:
        """Publish an event with error handling.

        Publishing failures are caught and logged — they must NEVER
        crash the caller.
        """
        try:
            event = self._producer.publish(
                topic=topic,
                data=data,
                source_system=SOURCE_SYSTEM,
                key=key,
            )
            self._published_count += 1
            logger.info(
                "Published %s: prediction=%s",
                topic, data.get("prediction_id", ""),
            )
            return event
        except Exception:
            self._failed_count += 1
            logger.exception("Failed to publish %s", topic)
            return None
