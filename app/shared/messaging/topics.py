"""Kafka topic constants — source of truth: PS1 SentinelAI Common Domain Names v2.0.

All topic names use snake_case with dots as defined in the team document (§3, §5.2).
Modules MUST use these constants instead of raw strings.
"""

from __future__ import annotations


class KafkaTopics:
    """Standardised Kafka topic names.

    Naming convention: snake_case with dots (§5.2 of PS-1 v2.0).
    All topics defined in §3 of the team document.
    """

    # ── Sensor Intelligence ──
    SENSOR_READING_CREATED = "sensor.reading.created"
    SENSOR_READING_ANOMALY = "sensor.reading.anomaly"
    SENSOR_STATUS_CHANGED = "sensor.status.changed"
    SENSOR_HEALTH_UPDATED = "sensor.health.updated"

    # ── Vision Intelligence ──
    VISION_EVENT_DETECTED = "vision.event.detected"
    VISION_WORKER_LOCATED = "vision.worker.located"

    # ── Risk Prediction ──
    RISK_ASSESSMENT_GENERATED = "risk.assessment.generated"
    RISK_SCORE_UPDATED = "risk.score.updated"
    RISK_THRESHOLD_EXCEEDED = "risk.threshold.exceeded"

    # ── Compound Risk Intelligence ──
    COMPOUND_RISK_DETECTED = "compound.risk.detected"

    # ── Forecast Intelligence ──
    FORECAST_GENERATED = "forecast.generated"
    FORECAST_UPDATED = "forecast.updated"

    # ── Permit Management ──
    PERMIT_CREATED = "permit.created"
    PERMIT_APPROVED = "permit.approved"
    PERMIT_ACTIVATED = "permit.activated"
    PERMIT_UPDATED = "permit.updated"
    PERMIT_EXPIRED = "permit.expired"
    PERMIT_REVOKED = "permit.revoked"

    # ── Maintenance ──
    MAINTENANCE_CREATED = "maintenance.created"
    MAINTENANCE_STARTED = "maintenance.started"
    MAINTENANCE_COMPLETED = "maintenance.completed"

    # ── Incident Management ──
    INCIDENT_CREATED = "incident.created"
    INCIDENT_UPDATED = "incident.updated"
    INCIDENT_RESOLVED = "incident.resolved"

    # ── Hazard & Emergency ──
    HAZARD_DETECTED = "hazard.detected"
    HAZARD_PROPAGATED = "hazard.propagated"
    EMERGENCY_TRIGGERED = "emergency.triggered"
    EVACUATION_INITIATED = "evacuation.initiated"
    ALL_CLEAR_SIGNAL = "all.clear.signal"

    # ── Alerts & Notifications ──
    ALERT_CREATED = "alert.created"
    ALERT_ACKNOWLEDGED = "alert.acknowledged"
    NOTIFICATION_SENT = "notification.sent"

    # ── AI Agents ──
    AGENT_DECISION_GENERATED = "agent.decision.generated"
    ROOT_CAUSE_ANALYSIS_COMPLETED = "root.cause.analysis.completed"

    @classmethod
    def all_topics(cls) -> list[str]:
        """Return all defined topic names."""
        return [
            v for k, v in vars(cls).items()
            if isinstance(v, str) and not k.startswith("_")
        ]

    @classmethod
    def sensor_topics(cls) -> list[str]:
        """Topics related to Sensor Intelligence."""
        return [
            cls.SENSOR_READING_CREATED,
            cls.SENSOR_READING_ANOMALY,
            cls.SENSOR_STATUS_CHANGED,
            cls.SENSOR_HEALTH_UPDATED,
        ]

    @classmethod
    def risk_topics(cls) -> list[str]:
        """Topics related to Risk Prediction."""
        return [
            cls.RISK_ASSESSMENT_GENERATED,
            cls.RISK_SCORE_UPDATED,
            cls.RISK_THRESHOLD_EXCEEDED,
            cls.COMPOUND_RISK_DETECTED,
        ]

    @classmethod
    def alert_topics(cls) -> list[str]:
        """Topics related to Alerts & Emergency."""
        return [
            cls.ALERT_CREATED,
            cls.ALERT_ACKNOWLEDGED,
            cls.EMERGENCY_TRIGGERED,
            cls.EVACUATION_INITIATED,
            cls.ALL_CLEAR_SIGNAL,
        ]
