"""Alert service — generates, manages, and resolves safety alerts.

Evaluates incoming sensor readings against configurable thresholds
and anomaly scores to generate operator-facing alerts. Handles:

  1. Threshold-based alerts (WARNING / CRITICAL / EMERGENCY per sensor type)
  2. Anomaly-score-based alerts (from IF/AE detectors)
  3. Duplicate prevention (one active alert per condition per sensor)
  4. Auto-resolution (clear alerts when values return to normal)

Consumed by:
  - Reading ingestion pipeline (post-ingestion hook)
  - Dashboard alerting endpoints

Architecture notes:
  - Uses domain entities (Alert) and the AlertRepository port
  - Publishes events via EventPublisher (NoOp or message broker)
  - Thresholds are configurable via AlertThresholdConfig
  - NO training logic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from app.sensor_intelligence.domain.entities.alert import Alert
from app.sensor_intelligence.domain.value_objects.alert_level import AlertLevel
from app.sensor_intelligence.repositories.alert_repository import AlertRepository
from app.sensor_intelligence.repositories.noop_publisher import EventPublisher

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configurable thresholds
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class SensorThreshold:
    """Configurable threshold for a sensor type."""

    warning: float
    critical: float
    emergency: float


@dataclass
class AlertThresholdConfig:
    """Application-level alert threshold configuration.

    All thresholds are configurable — no hardcoded values in the service.
    Defaults are reasonable industrial safety values.
    """

    # Anomaly score thresholds (from ML detectors, 0-1 scale)
    anomaly_warning: float = 0.6
    anomaly_critical: float = 0.8
    anomaly_emergency: float = 0.95

    # Sensor-type-specific value thresholds
    temperature: SensorThreshold = field(
        default_factory=lambda: SensorThreshold(warning=80.0, critical=120.0, emergency=200.0)
    )
    gas: SensorThreshold = field(
        default_factory=lambda: SensorThreshold(warning=50.0, critical=100.0, emergency=500.0)
    )
    pressure: SensorThreshold = field(
        default_factory=lambda: SensorThreshold(warning=8.0, critical=12.0, emergency=20.0)
    )
    humidity: SensorThreshold = field(
        default_factory=lambda: SensorThreshold(warning=80.0, critical=90.0, emergency=98.0)
    )
    vibration: SensorThreshold = field(
        default_factory=lambda: SensorThreshold(warning=5.0, critical=10.0, emergency=20.0)
    )

    # Number of simultaneous violations to trigger multi-violation alert
    multi_violation_threshold: int = 2

    def get_threshold_for_type(self, sensor_type: str) -> Optional[SensorThreshold]:
        """Get the threshold config for a sensor type."""
        mapping = {
            "TEMPERATURE": self.temperature,
            "GAS": self.gas,
            "PRESSURE": self.pressure,
            "HUMIDITY": self.humidity,
            "VIBRATION": self.vibration,
        }
        return mapping.get(sensor_type.upper())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Alert evaluation context
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ReadingContext:
    """Context object passed to alert evaluation.

    Contains all information needed to evaluate alert conditions.
    """

    sensor_id: UUID
    sensor_code: str  # Business sensor ID (e.g. "S001")
    sensor_type: str  # "TEMPERATURE", "GAS", etc.
    value: float
    anomaly_score: float = 0.0
    anomaly_status: str = "NORMAL"
    equipment_id: Optional[str] = None
    zone_id: Optional[str] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Alert service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AlertService:
    """Service for generating and managing safety alerts.

    Evaluates sensor readings against configurable thresholds and
    anomaly scores to produce actionable alerts for operators.
    """

    def __init__(
        self,
        alert_repo: AlertRepository,
        publisher: EventPublisher,
        config: Optional[AlertThresholdConfig] = None,
    ) -> None:
        self._repo = alert_repo
        self._publisher = publisher
        self._config = config or AlertThresholdConfig()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Core: evaluate reading and generate alerts
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def evaluate_reading(self, ctx: ReadingContext) -> list[Alert]:
        """Evaluate a reading and generate alerts if thresholds are breached.

        Steps:
          1. Check anomaly score thresholds
          2. Check sensor-type-specific value thresholds
          3. Check for multi-violation escalation
          4. Deduplicate against active alerts
          5. Persist new alerts
          6. Auto-resolve if reading is normal
          7. Publish alert events

        Args:
            ctx: ReadingContext with sensor info, value, and anomaly score.

        Returns:
            List of newly created Alert entities (may be empty).
        """
        violations = []

        # 1. Anomaly score check
        anomaly_alert = self._check_anomaly_score(ctx)
        if anomaly_alert:
            violations.append(anomaly_alert)

        # 2. Sensor-type value threshold checks
        value_alert = self._check_value_threshold(ctx)
        if value_alert:
            violations.append(value_alert)

        # 3. Multi-violation escalation
        if len(violations) >= self._config.multi_violation_threshold:
            violations.append(self._create_multi_violation_alert(ctx, violations))

        # 4. Deduplicate and persist
        created_alerts = []
        for alert in violations:
            existing = await self._repo.get_active_alert_for_sensor(
                ctx.sensor_id, alert.title
            )
            if existing is None:
                saved = await self._repo.save(alert)
                created_alerts.append(saved)
                logger.info(
                    "Alert created: sensor=%s type=%s level=%s",
                    ctx.sensor_code, alert.title, alert.level.value,
                )
            else:
                logger.debug(
                    "Duplicate alert suppressed: sensor=%s type=%s",
                    ctx.sensor_code, alert.title,
                )

        # 5. Auto-resolve if reading is normal and no violations
        if not violations:
            resolved_count = await self._auto_resolve(ctx)
            if resolved_count > 0:
                logger.info(
                    "Auto-resolved %d alerts for sensor %s (value=%.2f, anomaly_score=%.4f)",
                    resolved_count, ctx.sensor_code, ctx.value, ctx.anomaly_score,
                )

        # 6. Publish events
        for alert in created_alerts:
            await self._publish_alert(alert, ctx)

        return created_alerts

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Threshold checks (private)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _check_anomaly_score(self, ctx: ReadingContext) -> Optional[Alert]:
        """Check anomaly score against configured thresholds."""
        score = ctx.anomaly_score
        cfg = self._config

        if score >= cfg.anomaly_emergency:
            level = AlertLevel.EMERGENCY
        elif score >= cfg.anomaly_critical:
            level = AlertLevel.CRITICAL
        elif score >= cfg.anomaly_warning:
            level = AlertLevel.WARNING
        else:
            return None

        return Alert(
            sensor_id=ctx.sensor_id,
            level=level,
            title="ANOMALY_SCORE",
            message=(
                f"Anomaly detected on sensor {ctx.sensor_code} "
                f"({ctx.sensor_type}): score={score:.4f} [{level.value}]. "
                f"Zone: {ctx.zone_id or 'N/A'}, Equipment: {ctx.equipment_id or 'N/A'}."
            ),
        )

    def _check_value_threshold(self, ctx: ReadingContext) -> Optional[Alert]:
        """Check reading value against sensor-type-specific thresholds."""
        threshold = self._config.get_threshold_for_type(ctx.sensor_type)
        if threshold is None:
            return None

        value = ctx.value

        if value >= threshold.emergency:
            level = AlertLevel.EMERGENCY
        elif value >= threshold.critical:
            level = AlertLevel.CRITICAL
        elif value >= threshold.warning:
            level = AlertLevel.WARNING
        else:
            return None

        type_label_map = {
            "TEMPERATURE": "HIGH_TEMPERATURE",
            "GAS": "HIGH_GAS_CONCENTRATION",
            "PRESSURE": "HIGH_PRESSURE",
            "HUMIDITY": "HIGH_HUMIDITY",
            "VIBRATION": "HIGH_VIBRATION",
        }
        alert_type = type_label_map.get(ctx.sensor_type.upper(), f"HIGH_{ctx.sensor_type}")

        return Alert(
            sensor_id=ctx.sensor_id,
            level=level,
            title=alert_type,
            message=(
                f"{ctx.sensor_type} reading on sensor {ctx.sensor_code}: "
                f"value={value:.2f} [{level.value}] "
                f"(warning={threshold.warning}, critical={threshold.critical}, "
                f"emergency={threshold.emergency}). "
                f"Zone: {ctx.zone_id or 'N/A'}, Equipment: {ctx.equipment_id or 'N/A'}."
            ),
        )

    def _create_multi_violation_alert(
        self, ctx: ReadingContext, violations: list[Alert]
    ) -> Alert:
        """Create an escalated alert for simultaneous threshold violations."""
        violation_types = [v.title for v in violations]
        max_level = max(v.level for v in violations)

        # Escalate one level up from worst violation
        escalated = {
            AlertLevel.INFO: AlertLevel.WARNING,
            AlertLevel.WARNING: AlertLevel.CRITICAL,
            AlertLevel.CRITICAL: AlertLevel.EMERGENCY,
            AlertLevel.EMERGENCY: AlertLevel.EMERGENCY,
        }
        level = escalated.get(max_level, AlertLevel.EMERGENCY)

        return Alert(
            sensor_id=ctx.sensor_id,
            level=level,
            title="MULTI_VIOLATION",
            message=(
                f"Multiple simultaneous violations on sensor {ctx.sensor_code}: "
                f"{', '.join(violation_types)} [{level.value}]. "
                f"Zone: {ctx.zone_id or 'N/A'}, Equipment: {ctx.equipment_id or 'N/A'}."
            ),
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Auto-resolve
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _auto_resolve(self, ctx: ReadingContext) -> int:
        """Auto-resolve active alerts when readings return to normal.

        Returns the number of alerts resolved.
        """
        now = datetime.now(timezone.utc)
        return await self._repo.resolve_alerts_for_sensor(ctx.sensor_id, now)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Alert management
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def acknowledge_alert(
        self, alert_id: UUID, acknowledged_by: str
    ) -> Optional[Alert]:
        """Manually acknowledge an alert.

        Args:
            alert_id: UUID of the alert to acknowledge.
            acknowledged_by: Operator who acknowledged.

        Returns:
            Updated Alert entity, or None if not found.
        """
        alert = await self._repo.get_by_id(alert_id)
        if alert is None:
            return None

        now = datetime.now(timezone.utc)
        return await self._repo.acknowledge(alert_id, acknowledged_by, now)

    async def get_active_alerts(self) -> list[Alert]:
        """Get all unacknowledged alerts, ordered by recency."""
        return await self._repo.get_unacknowledged()

    async def get_alert_summary(self):
        """Get summary counts of alerts by level."""
        return await self._repo.get_summary()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Event publishing
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _publish_alert(self, alert: Alert, ctx: ReadingContext) -> None:
        """Publish an alert event to the configured message broker."""
        event = {
            "alert_id": str(alert.id),
            "sensor_id": ctx.sensor_code,
            "equipment_id": ctx.equipment_id,
            "zone_id": ctx.zone_id,
            "alert_type": alert.title,
            "severity": alert.level.value,
            "anomaly_score": ctx.anomaly_score,
            "message": alert.message,
            "created_at": alert.created_at.isoformat(),
            "acknowledged": alert.is_acknowledged,
            "resolved": False,
        }
        await self._publisher.publish("alerts", event)
