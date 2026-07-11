"""Snapshot service for the Digital Twin.

Orchestrates snapshot creation, recovery, and retention management.

Responsibilities:
  - Capture current twin state → persist as snapshot
  - Recover latest snapshot → rebuild TwinStateManager
  - Manage snapshot retention (default: keep last 1000)
  - Automatic snapshot triggers (critical hazard, health change, etc.)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.digital_twin.domain.entities import (
    ActiveHazard,
    EquipmentState,
    FacilityState,
    SensorReading,
    ZoneState,
)
from app.digital_twin.models.facility_snapshot_model import (
    FacilitySnapshotModel,
)
from app.digital_twin.models.zone_state_model import ZoneStateModel
from app.digital_twin.repositories.twin_snapshot_repository import (
    TwinSnapshotRepository,
)
from app.digital_twin.services.twin_state_manager import TwinStateManager

logger = logging.getLogger(__name__)

# Default retention policy
DEFAULT_MAX_SNAPSHOTS = 1000

# Automatic snapshot trigger thresholds
HEALTH_CHANGE_THRESHOLD = 10.0  # Trigger if health changes by ≥10 points
COMPOUND_RISK_THRESHOLD = 75.0  # Trigger if compound risk score ≥75


class SnapshotService:
    """Orchestrates Digital Twin snapshot lifecycle.

    Args:
        state_manager: The shared TwinStateManager singleton.
        repository: Snapshot persistence repository.
        max_snapshots: Maximum snapshots to retain (default: 1000).
    """

    def __init__(
        self,
        state_manager: TwinStateManager,
        repository: TwinSnapshotRepository,
        max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
    ) -> None:
        self._state = state_manager
        self._repo = repository
        self._max_snapshots = max_snapshots

        # Track last snapshot health for change detection
        self._last_snapshot_health: Optional[float] = None
        self._snapshots_created: int = 0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Properties
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def snapshots_created(self) -> int:
        return self._snapshots_created

    @property
    def max_snapshots(self) -> int:
        return self._max_snapshots

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Snapshot Creation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def create_snapshot(
        self, trigger_reason: str = "manual",
    ) -> FacilitySnapshotModel:
        """Capture and persist the current twin state.

        Args:
            trigger_reason: Why this snapshot was created.
                Valid: manual | critical_hazard | health_change |
                       compound_risk_threshold | startup

        Returns:
            The persisted FacilitySnapshotModel.
        """
        facility = self._state.get_facility_state()
        zones = self._state.get_all_zones()
        snapshot_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Build full payload for recovery
        payload = self._build_snapshot_payload(facility, zones)

        snapshot = FacilitySnapshotModel(
            snapshot_id=snapshot_id,
            created_at=now,
            facility_health=facility.facility_health,
            total_zones=facility.total_zones,
            active_hazards=facility.active_hazards,
            critical_zones=facility.critical_zones,
            workers_at_risk=facility.workers_at_risk,
            events_processed=facility.events_processed,
            snapshot_payload=json.dumps(payload),
            trigger_reason=trigger_reason,
        )

        zone_models = [
            ZoneStateModel(
                snapshot_id=snapshot_id,
                zone_id=z.zone_id,
                risk_score=z.overall_risk_score,
                compound_risk_score=z.compound_risk_score,
                hazard_count=z.active_hazard_count,
                anomaly_count=z.anomaly_count,
                equipment_health=z.sensor_health,
                worker_count=z.current_worker_count,
                state_payload=json.dumps(
                    self._zone_to_dict(z),
                ),
                created_at=now,
            )
            for z in zones
        ]

        result = await self._repo.save_snapshot(snapshot, zone_models)
        self._snapshots_created += 1
        self._last_snapshot_health = facility.facility_health

        logger.info(
            "Snapshot created: id=%s trigger=%s health=%.1f zones=%d",
            snapshot_id, trigger_reason,
            facility.facility_health, len(zones),
        )

        # Enforce retention
        await self._enforce_retention()

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Recovery
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def recover_latest_snapshot(self) -> bool:
        """Recover the twin state from the latest snapshot.

        Returns True if recovery was successful, False if no
        snapshot exists or recovery failed.
        """
        try:
            snapshot = await self._repo.get_latest_snapshot()
            if snapshot is None:
                logger.info(
                    "No snapshots found — twin starts with empty state",
                )
                return False

            payload = json.loads(snapshot.snapshot_payload)
            zones_data = payload.get("zones", [])

            restored_count = 0
            for zone_data in zones_data:
                zone_state = self._dict_to_zone(zone_data)
                self._state.restore_zone(zone_state)
                restored_count += 1

            self._last_snapshot_health = snapshot.facility_health

            logger.info(
                "Twin recovered from snapshot: id=%s health=%.1f "
                "zones=%d age=%s",
                snapshot.snapshot_id,
                snapshot.facility_health,
                restored_count,
                datetime.now(timezone.utc) - snapshot.created_at,
            )
            return True

        except Exception:
            logger.exception(
                "Failed to recover twin from snapshot — "
                "starting with empty state",
            )
            return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Queries
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def list_snapshots(
        self, offset: int = 0, limit: int = 50,
    ) -> List[FacilitySnapshotModel]:
        """List snapshots in reverse chronological order."""
        return await self._repo.list_snapshots(
            offset=offset, limit=limit,
        )

    async def get_snapshot(
        self, snapshot_id: str,
    ) -> Optional[FacilitySnapshotModel]:
        """Get a snapshot by its ID."""
        return await self._repo.get_snapshot(snapshot_id)

    async def get_snapshot_with_zones(
        self, snapshot_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a snapshot with all its zone states."""
        snapshot = await self._repo.get_snapshot(snapshot_id)
        if snapshot is None:
            return None

        zone_states = await self._repo.get_zone_states_for_snapshot(
            snapshot_id,
        )
        return {
            "snapshot": snapshot,
            "zone_states": zone_states,
        }

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot by its ID."""
        deleted = await self._repo.delete_snapshot(snapshot_id)
        if deleted:
            logger.info("Snapshot deleted: id=%s", snapshot_id)
        return deleted

    async def count_snapshots(self) -> int:
        """Count total snapshots."""
        return await self._repo.count_snapshots()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Automatic Trigger Evaluation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def evaluate_snapshot_trigger(
        self,
        event_category: str,
        zone_id: str,
    ) -> Optional[FacilitySnapshotModel]:
        """Evaluate whether the current state warrants a snapshot.

        Called after each event is processed. Returns the snapshot
        if one was created, None otherwise.

        Triggers:
          - Critical hazard detected
          - Facility health changed by ≥ HEALTH_CHANGE_THRESHOLD
          - Compound risk score exceeded COMPOUND_RISK_THRESHOLD
        """
        try:
            # 1. Critical hazard
            if event_category == "hazard":
                try:
                    zone = self._state.get_zone(zone_id)
                    if zone.is_critical:
                        return await self.create_snapshot(
                            trigger_reason="critical_hazard",
                        )
                except Exception:
                    pass

            # 2. Compound risk threshold
            if event_category == "risk":
                try:
                    zone = self._state.get_zone(zone_id)
                    if zone.compound_risk_score >= COMPOUND_RISK_THRESHOLD:
                        return await self.create_snapshot(
                            trigger_reason="compound_risk_threshold",
                        )
                except Exception:
                    pass

            # 3. Health change
            facility = self._state.get_facility_state()
            if self._last_snapshot_health is not None:
                delta = abs(
                    facility.facility_health - self._last_snapshot_health
                )
                if delta >= HEALTH_CHANGE_THRESHOLD:
                    return await self.create_snapshot(
                        trigger_reason="health_change",
                    )

        except Exception:
            logger.debug("Snapshot trigger evaluation failed")

        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Retention
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _enforce_retention(self) -> None:
        """Delete oldest snapshots if count exceeds max_snapshots."""
        try:
            count = await self._repo.count_snapshots()
            if count > self._max_snapshots:
                deleted = await self._repo.delete_oldest_snapshots(
                    keep_count=self._max_snapshots,
                )
                if deleted > 0:
                    logger.info(
                        "Retention: deleted %d old snapshots "
                        "(max=%d, had=%d)",
                        deleted, self._max_snapshots, count,
                    )
        except Exception:
            logger.debug("Retention enforcement failed")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Serialization helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _zone_to_dict(zone: ZoneState) -> Dict[str, Any]:
        """Serialize a ZoneState to a JSON-safe dict."""
        return {
            "zone_id": zone.zone_id,
            "zone_name": zone.zone_name,
            "sensor_health": zone.sensor_health,
            "anomaly_count": zone.anomaly_count,
            "sensor_count": zone.sensor_count,
            "latest_sensor_readings": {
                sid: {
                    "sensor_id": r.sensor_id,
                    "sensor_type": r.sensor_type,
                    "value": r.value,
                    "unit": r.unit,
                    "anomaly_score": r.anomaly_score,
                    "is_anomalous": r.is_anomalous,
                    "health_score": r.health_score,
                    "status": r.status,
                    "last_updated": r.last_updated,
                }
                for sid, r in zone.latest_sensor_readings.items()
            },
            "predicted_risk_score": zone.predicted_risk_score,
            "risk_level": zone.risk_level,
            "accident_probability": zone.accident_probability,
            "compound_risk_score": zone.compound_risk_score,
            "compound_risk_level": zone.compound_risk_level,
            "compound_risk_confidence": zone.compound_risk_confidence,
            "contributing_factors": zone.contributing_factors,
            "active_hazards": [
                {
                    "hazard_id": h.hazard_id,
                    "hazard_type": h.hazard_type,
                    "severity": h.severity,
                    "origin_zone": h.origin_zone,
                    "propagation_level": h.propagation_level,
                    "affected_zones": h.affected_zones,
                    "detected_at": h.detected_at,
                }
                for h in zone.active_hazards
            ],
            "affected_neighbors": zone.affected_neighbors,
            "workers_at_risk": zone.workers_at_risk,
            "worker_capacity": zone.worker_capacity,
            "current_worker_count": zone.current_worker_count,
            "equipment": [
                {
                    "equipment_id": eq.equipment_id,
                    "equipment_type": eq.equipment_type,
                    "zone_id": eq.zone_id,
                    "operational_status": eq.operational_status,
                    "health_score": eq.health_score,
                    "risk_score": eq.risk_score,
                    "sensor_count": eq.sensor_count,
                    "last_updated": eq.last_updated,
                }
                for eq in zone.equipment
            ],
            "connected_zones": zone.connected_zones,
            "last_updated": zone.last_updated,
            "event_count": zone.event_count,
        }

    @staticmethod
    def _dict_to_zone(data: Dict[str, Any]) -> ZoneState:
        """Deserialize a dict back to a ZoneState entity."""
        zone = ZoneState(
            zone_id=data["zone_id"],
            zone_name=data.get("zone_name", ""),
            sensor_health=data.get("sensor_health", 100.0),
            anomaly_count=data.get("anomaly_count", 0),
            sensor_count=data.get("sensor_count", 0),
            predicted_risk_score=data.get("predicted_risk_score", 0.0),
            risk_level=data.get("risk_level", "LOW"),
            accident_probability=data.get("accident_probability", 0.0),
            compound_risk_score=data.get("compound_risk_score", 0.0),
            compound_risk_level=data.get("compound_risk_level", "LOW"),
            compound_risk_confidence=data.get(
                "compound_risk_confidence", 0.0,
            ),
            contributing_factors=data.get("contributing_factors", {}),
            affected_neighbors=data.get("affected_neighbors", []),
            workers_at_risk=data.get("workers_at_risk", 0),
            worker_capacity=data.get("worker_capacity", 0),
            current_worker_count=data.get("current_worker_count", 0),
            connected_zones=data.get("connected_zones", []),
            last_updated=data.get("last_updated", ""),
            event_count=data.get("event_count", 0),
        )

        # Restore sensor readings
        for sid, r_data in data.get(
            "latest_sensor_readings", {},
        ).items():
            zone.latest_sensor_readings[sid] = SensorReading(
                sensor_id=r_data.get("sensor_id", sid),
                sensor_type=r_data.get("sensor_type", ""),
                value=r_data.get("value", 0.0),
                unit=r_data.get("unit", ""),
                anomaly_score=r_data.get("anomaly_score", 0.0),
                is_anomalous=r_data.get("is_anomalous", False),
                health_score=r_data.get("health_score", 100.0),
                status=r_data.get("status", "ACTIVE"),
                last_updated=r_data.get("last_updated", ""),
            )

        # Restore equipment
        for eq_data in data.get("equipment", []):
            zone.equipment.append(EquipmentState(
                equipment_id=eq_data.get("equipment_id", ""),
                equipment_type=eq_data.get("equipment_type", ""),
                zone_id=eq_data.get("zone_id", ""),
                operational_status=eq_data.get(
                    "operational_status", "ACTIVE",
                ),
                health_score=eq_data.get("health_score", 100.0),
                risk_score=eq_data.get("risk_score", 0.0),
                sensor_count=eq_data.get("sensor_count", 0),
                last_updated=eq_data.get("last_updated", ""),
            ))

        # Restore active hazards
        for h_data in data.get("active_hazards", []):
            zone.active_hazards.append(ActiveHazard(
                hazard_id=h_data.get("hazard_id", ""),
                hazard_type=h_data.get("hazard_type", ""),
                severity=h_data.get("severity", "HIGH"),
                origin_zone=h_data.get("origin_zone", ""),
                propagation_level=h_data.get(
                    "propagation_level", "CONTAINED",
                ),
                affected_zones=h_data.get("affected_zones", []),
                detected_at=h_data.get("detected_at", ""),
            ))

        return zone

    @staticmethod
    def _build_snapshot_payload(
        facility: FacilityState,
        zones: List[ZoneState],
    ) -> Dict[str, Any]:
        """Build the full JSON payload for a snapshot."""
        return {
            "facility": {
                "facility_health": facility.facility_health,
                "total_zones": facility.total_zones,
                "active_hazards": facility.active_hazards,
                "critical_zones": facility.critical_zones,
                "workers_at_risk": facility.workers_at_risk,
                "total_workers": facility.total_workers,
                "total_equipment": facility.total_equipment,
                "total_sensors": facility.total_sensors,
                "total_anomalies": facility.total_anomalies,
                "average_risk_score": facility.average_risk_score,
                "max_risk_score": facility.max_risk_score,
                "events_processed": facility.events_processed,
                "last_updated": facility.last_updated,
            },
            "zones": [
                SnapshotService._zone_to_dict(z)
                for z in zones
            ],
        }
