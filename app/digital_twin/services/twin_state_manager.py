"""Digital Twin state manager — the brain of the twin.

Maintains all zone states in memory and provides query methods for
the REST API layer. Updates are applied by the event handler via
typed update methods.

The TwinStateManager is a singleton, created once in DI and shared
between the event handler and the API endpoints.

Design decisions:
  - Zone-centric: every update targets a zone_id
  - In-memory only (Phase 1): no persistence
  - Thread-safe: all mutations go through methods (no direct dict access)
  - Graph-aware: loads facility topology from GraphRepository on init
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.digital_twin.domain.entities import (
    ActiveHazard,
    EquipmentState,
    FacilityState,
    SensorReading,
    ZoneState,
)
from app.digital_twin.domain.enums import HeatmapColor, RiskLevel
from app.digital_twin.domain.exceptions import ZoneNotFoundInTwinError
from app.hazard_propagation.repositories.graph_repository import GraphRepository

logger = logging.getLogger(__name__)


class TwinStateManager:
    """In-memory facility state manager.

    Lifecycle:
        1. Created in DI with a GraphRepository
        2. initialize() loads topology from the graph
        3. update_*() methods apply event-driven mutations
        4. get_*() methods serve the API layer

    Thread safety:
        A threading.Lock guards all mutations. Read methods acquire
        the same lock to prevent torn reads.
    """

    def __init__(self, graph_repo: GraphRepository) -> None:
        self._graph_repo = graph_repo
        self._zones: Dict[str, ZoneState] = {}
        self._initialized = False
        self._events_processed: int = 0
        self._lock = threading.Lock()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Initialization
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def initialize(self) -> None:
        """Load facility topology from the graph repository.

        Populates zone states with topology data (connections,
        equipment, sensors). Safe to call multiple times.
        """
        try:
            all_zones = await self._graph_repo.get_all_zones()

            with self._lock:
                for zone_node in all_zones:
                    zone_id = zone_node.zone_id
                    if zone_id not in self._zones:
                        self._zones[zone_id] = ZoneState(
                            zone_id=zone_id,
                            zone_name=zone_node.zone_name,
                            worker_capacity=zone_node.worker_capacity,
                            current_worker_count=zone_node.current_worker_count,
                            connected_zones=list(zone_node.connected_zones),
                        )

                    # Populate equipment from graph
                    zone_state = self._zones[zone_id]
                    for eq in zone_node.equipment:
                        eq_state = EquipmentState(
                            equipment_id=eq.equipment_id,
                            equipment_type=eq.equipment_type,
                            zone_id=zone_id,
                            operational_status=eq.operational_status,
                            health_score=eq.health_score,
                            sensor_count=eq.sensor_count,
                        )
                        zone_state.equipment.append(eq_state)

                    # Populate sensors from graph
                    for eq in zone_node.equipment:
                        for sensor in eq.sensors:
                            zone_state.latest_sensor_readings[
                                sensor.sensor_id
                            ] = SensorReading(
                                sensor_id=sensor.sensor_id,
                                sensor_type=sensor.sensor_type,
                                unit=sensor.unit_of_measurement,
                                status=sensor.sensor_status,
                            )
                    zone_state.sensor_count = len(
                        zone_state.latest_sensor_readings
                    )

                self._initialized = True

            logger.info(
                "Digital Twin initialized: %d zones loaded from graph",
                len(self._zones),
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize twin from graph: %s. "
                "Twin will populate zones on first event.",
                exc,
            )
            self._initialized = True

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _ensure_zone(self, zone_id: str) -> ZoneState:
        """Get or create a zone state. Must be called under lock."""
        if zone_id not in self._zones:
            self._zones[zone_id] = ZoneState(zone_id=zone_id)
            logger.debug("Auto-created zone state for: %s", zone_id)
        return self._zones[zone_id]

    def _risk_level_from_score(self, score: float) -> str:
        """Map a 0-100 score to a PS-1 §4.1 risk level."""
        if score >= 76:
            return RiskLevel.CRITICAL.value
        elif score >= 51:
            return RiskLevel.HIGH.value
        elif score >= 26:
            return RiskLevel.MEDIUM.value
        else:
            return RiskLevel.LOW.value

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Sensor Intelligence updates
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def update_sensor_anomaly(
        self,
        zone_id: str,
        sensor_id: str,
        sensor_type: str = "",
        value: float = 0.0,
        unit: str = "",
        anomaly_score: float = 0.0,
    ) -> None:
        """Process a sensor.reading.anomaly event."""
        with self._lock:
            zone = self._ensure_zone(zone_id)
            reading = zone.latest_sensor_readings.get(sensor_id)
            if reading is None:
                reading = SensorReading(sensor_id=sensor_id)
                zone.latest_sensor_readings[sensor_id] = reading

            reading.sensor_type = sensor_type or reading.sensor_type
            reading.value = value
            reading.unit = unit or reading.unit
            reading.anomaly_score = anomaly_score
            reading.is_anomalous = True
            reading.last_updated = datetime.now(timezone.utc).isoformat()

            zone.anomaly_count = sum(
                1
                for r in zone.latest_sensor_readings.values()
                if r.is_anomalous
            )
            zone.sensor_count = len(zone.latest_sensor_readings)
            zone.touch()
            self._events_processed += 1

    def update_sensor_status(
        self,
        zone_id: str,
        sensor_id: str,
        status: str,
    ) -> None:
        """Process a sensor.status.changed event."""
        with self._lock:
            zone = self._ensure_zone(zone_id)
            reading = zone.latest_sensor_readings.get(sensor_id)
            if reading is None:
                reading = SensorReading(sensor_id=sensor_id)
                zone.latest_sensor_readings[sensor_id] = reading

            reading.status = status
            reading.last_updated = datetime.now(timezone.utc).isoformat()
            zone.sensor_count = len(zone.latest_sensor_readings)
            zone.touch()
            self._events_processed += 1

    def update_sensor_health(
        self,
        zone_id: str,
        sensor_id: str,
        health_score: float,
    ) -> None:
        """Process a sensor.health.updated event."""
        with self._lock:
            zone = self._ensure_zone(zone_id)
            reading = zone.latest_sensor_readings.get(sensor_id)
            if reading is None:
                reading = SensorReading(sensor_id=sensor_id)
                zone.latest_sensor_readings[sensor_id] = reading

            reading.health_score = health_score
            reading.last_updated = datetime.now(timezone.utc).isoformat()

            # Recalculate zone-level sensor health (average)
            health_scores = [
                r.health_score
                for r in zone.latest_sensor_readings.values()
            ]
            zone.sensor_health = (
                sum(health_scores) / len(health_scores)
                if health_scores
                else 100.0
            )
            zone.sensor_count = len(zone.latest_sensor_readings)
            zone.touch()
            self._events_processed += 1

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Risk Prediction updates
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def update_risk_assessment(
        self,
        zone_id: str,
        risk_score: float,
        risk_level: str = "",
        accident_probability: float = 0.0,
        equipment_id: str = "",
    ) -> None:
        """Process a risk.assessment.generated event."""
        with self._lock:
            zone = self._ensure_zone(zone_id)
            zone.predicted_risk_score = max(
                zone.predicted_risk_score, risk_score,
            )
            zone.risk_level = (
                risk_level
                or self._risk_level_from_score(zone.predicted_risk_score)
            )
            zone.accident_probability = max(
                zone.accident_probability, accident_probability,
            )

            # Update equipment risk if equipment_id provided
            if equipment_id:
                for eq in zone.equipment:
                    if eq.equipment_id == equipment_id:
                        eq.risk_score = risk_score
                        eq.last_updated = (
                            datetime.now(timezone.utc).isoformat()
                        )
                        break

            zone.touch()
            self._events_processed += 1

    def update_risk_score(
        self,
        zone_id: str,
        risk_score: float,
        risk_level: str = "",
    ) -> None:
        """Process a risk.score.updated event."""
        with self._lock:
            zone = self._ensure_zone(zone_id)
            zone.predicted_risk_score = risk_score
            zone.risk_level = (
                risk_level or self._risk_level_from_score(risk_score)
            )
            zone.touch()
            self._events_processed += 1

    def update_risk_threshold_exceeded(
        self,
        zone_id: str,
        threshold_type: str,
        current_value: float,
        threshold_value: float = 0.0,
    ) -> None:
        """Process a risk.threshold.exceeded event."""
        with self._lock:
            zone = self._ensure_zone(zone_id)
            # A threshold breach implies elevated risk
            zone.predicted_risk_score = max(
                zone.predicted_risk_score, 76.0,
            )
            zone.risk_level = RiskLevel.CRITICAL.value
            zone.touch()
            self._events_processed += 1

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Compound Risk Intelligence updates
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def update_compound_risk(
        self,
        zone_id: str,
        compound_risk_score: float,
        risk_level: str = "",
        confidence_score: float = 0.0,
        contributing_factors: Optional[Dict[str, float]] = None,
    ) -> None:
        """Process a compound.risk.detected event."""
        with self._lock:
            zone = self._ensure_zone(zone_id)
            zone.compound_risk_score = compound_risk_score
            zone.compound_risk_level = (
                risk_level
                or self._risk_level_from_score(compound_risk_score)
            )
            zone.compound_risk_confidence = confidence_score
            if contributing_factors:
                zone.contributing_factors = dict(contributing_factors)

            # Estimate workers at risk based on worker count when risk is high
            if compound_risk_score >= 51:
                zone.workers_at_risk = zone.current_worker_count
            else:
                zone.workers_at_risk = 0

            zone.touch()
            self._events_processed += 1

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Hazard Propagation updates
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def update_hazard_detected(
        self,
        zone_id: str,
        hazard_id: str,
        hazard_type: str,
        severity: str = "HIGH",
    ) -> None:
        """Process a hazard.detected event."""
        with self._lock:
            zone = self._ensure_zone(zone_id)
            # Avoid duplicate hazards
            existing_ids = {h.hazard_id for h in zone.active_hazards}
            if hazard_id not in existing_ids:
                zone.active_hazards.append(
                    ActiveHazard(
                        hazard_id=hazard_id,
                        hazard_type=hazard_type,
                        severity=severity,
                        origin_zone=zone_id,
                    )
                )
            zone.workers_at_risk = zone.current_worker_count
            zone.touch()
            self._events_processed += 1

    def update_hazard_propagated(
        self,
        origin_zone: str,
        hazard_type: str,
        propagation_level: str,
        affected_zones: Optional[List[str]] = None,
        propagation_id: str = "",
        severity: str = "HIGH",
    ) -> None:
        """Process a hazard.propagated event."""
        affected = affected_zones or []
        with self._lock:
            # Update origin zone
            origin = self._ensure_zone(origin_zone)
            origin.affected_neighbors = list(affected)

            hazard_id = propagation_id or f"HAZ-{origin_zone}-{hazard_type}"
            existing_ids = {h.hazard_id for h in origin.active_hazards}
            if hazard_id not in existing_ids:
                origin.active_hazards.append(
                    ActiveHazard(
                        hazard_id=hazard_id,
                        hazard_type=hazard_type,
                        severity=severity,
                        origin_zone=origin_zone,
                        propagation_level=propagation_level,
                        affected_zones=list(affected),
                    )
                )
            origin.workers_at_risk = origin.current_worker_count
            origin.touch()

            # Mark affected zones
            for az_id in affected:
                if az_id == origin_zone:
                    continue
                az = self._ensure_zone(az_id)
                az_existing = {h.hazard_id for h in az.active_hazards}
                prop_hazard_id = f"{hazard_id}-{az_id}"
                if prop_hazard_id not in az_existing:
                    az.active_hazards.append(
                        ActiveHazard(
                            hazard_id=prop_hazard_id,
                            hazard_type=hazard_type,
                            severity=severity,
                            origin_zone=origin_zone,
                            propagation_level=propagation_level,
                            affected_zones=list(affected),
                        )
                    )
                az.workers_at_risk = az.current_worker_count
                az.touch()

            self._events_processed += 1

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Facility health calculation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _calculate_facility_health(self) -> float:
        """Calculate facility-wide health score (0-100).

        Weighted formula:
          facility_health =
            0.25 * avg(sensor_health)
          + 0.25 * avg(equipment_health)
          + 0.25 * (100 - avg(risk_score))
          + 0.25 * (100 - hazard_score)

        Must be called under lock.
        """
        if not self._zones:
            return 100.0

        zones = list(self._zones.values())

        # Average sensor health across all zones
        sensor_healths = [z.sensor_health for z in zones]
        avg_sensor = (
            sum(sensor_healths) / len(sensor_healths)
            if sensor_healths
            else 100.0
        )

        # Average equipment health
        all_eq_healths = []
        for z in zones:
            for eq in z.equipment:
                all_eq_healths.append(eq.health_score)
        avg_equipment = (
            sum(all_eq_healths) / len(all_eq_healths)
            if all_eq_healths
            else 100.0
        )

        # Average risk score (inverted: low risk = high health)
        risk_scores = [z.overall_risk_score for z in zones]
        avg_risk = (
            sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        )

        # Hazard score: proportion of zones with active hazards * 100
        zones_with_hazards = sum(
            1 for z in zones if z.active_hazard_count > 0
        )
        hazard_score = (
            (zones_with_hazards / len(zones)) * 100.0
            if zones
            else 0.0
        )

        facility_health = (
            0.25 * avg_sensor
            + 0.25 * avg_equipment
            + 0.25 * (100.0 - avg_risk)
            + 0.25 * (100.0 - hazard_score)
        )

        return max(0.0, min(100.0, round(facility_health, 2)))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Query methods (for API layer)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_facility_state(self) -> FacilityState:
        """Return facility-wide aggregated state."""
        with self._lock:
            zones = list(self._zones.values())
            risk_scores = [z.overall_risk_score for z in zones]

            return FacilityState(
                facility_health=self._calculate_facility_health(),
                total_zones=len(zones),
                active_hazards=sum(
                    z.active_hazard_count for z in zones
                ),
                critical_zones=sum(1 for z in zones if z.is_critical),
                workers_at_risk=sum(z.workers_at_risk for z in zones),
                total_workers=sum(
                    z.current_worker_count for z in zones
                ),
                total_equipment=sum(len(z.equipment) for z in zones),
                total_sensors=sum(z.sensor_count for z in zones),
                total_anomalies=sum(z.anomaly_count for z in zones),
                average_risk_score=(
                    round(sum(risk_scores) / len(risk_scores), 2)
                    if risk_scores
                    else 0.0
                ),
                max_risk_score=(
                    round(max(risk_scores), 2) if risk_scores else 0.0
                ),
                events_processed=self._events_processed,
                last_updated=datetime.now(timezone.utc).isoformat(),
                zone_ids=[z.zone_id for z in zones],
            )

    def get_all_zones(self) -> List[ZoneState]:
        """Return all zone states."""
        with self._lock:
            return list(self._zones.values())

    def get_zone(self, zone_id: str) -> ZoneState:
        """Return state for a single zone.

        Raises:
            ZoneNotFoundInTwinError: if zone_id is not tracked.
        """
        with self._lock:
            zone = self._zones.get(zone_id)
            if zone is None:
                raise ZoneNotFoundInTwinError(zone_id)
            return zone

    def get_heatmap(self) -> List[Dict[str, Any]]:
        """Return heatmap data for all zones."""
        with self._lock:
            return [
                {
                    "zone_id": z.zone_id,
                    "zone_name": z.zone_name,
                    "risk_score": round(z.overall_risk_score, 2),
                    "risk_level": z.risk_level,
                    "color": z.heatmap_color,
                }
                for z in self._zones.values()
            ]

    @property
    def zone_count(self) -> int:
        return len(self._zones)

    @property
    def events_processed(self) -> int:
        return self._events_processed

    @property
    def is_initialized(self) -> bool:
        return self._initialized
