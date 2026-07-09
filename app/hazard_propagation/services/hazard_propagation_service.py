"""Hazard Propagation Service — main façade.

Coordinates all Hazard Propagation sub-components:
  - GraphRepository (graph topology retrieval)
  - HazardPropagationEngine (BFS propagation algorithm)
  - HazardPropagationPublisher (Kafka event publishing)

This is the single entry-point used by the API layer and event handlers.
All business logic is delegated to specialised components; this service
is responsible for orchestration, error handling, and logging.

Design:
  - Follows the existing DI pattern (constructor injection)
  - All dependencies are injected — no hardcoded infrastructure
  - Stateless per request (state lives in GraphRepository and Kafka)
  - Exceptions follow the domain exception hierarchy

Checklist items addressed:
  - Impact Radius Calculation
  - Affected Worker Detection
  - Hazard Spread Simulation
  - Critical Time Window Estimation
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.hazard_propagation.domain.exceptions import (
    HazardPropagationError,
    InvalidHazardError,
    PropagationSimulationError,
    ZoneNotFoundError,
)
from app.hazard_propagation.domain.models import (
    Hazard,
    HazardPropagation,
    PropagationPath,
    ZoneRiskState,
)
from app.hazard_propagation.domain.value_objects import (
    HazardType,
    PropagationLevel,
    PropagationStatus,
    RiskLevel,
)
from app.hazard_propagation.graph.entities import HazardNode, ZoneNode
from app.hazard_propagation.messaging.publisher import HazardPropagationPublisher
from app.hazard_propagation.repositories.graph_repository import GraphRepository
from app.hazard_propagation.services.config import PropagationConfig
from app.hazard_propagation.services.propagation_engine import (
    HazardPropagationEngine,
    PropagationResult,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Propagation analysis result (full output)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class PropagationAnalysisResult:
    """Complete output of a hazard propagation analysis.

    Aggregates engine output + persisted hazard + summary.
    """

    # Engine output
    propagation_result: PropagationResult

    # Persisted domain model
    hazard: Optional[Hazard] = None
    propagation: Optional[HazardPropagation] = None

    # Summary
    summary: str = ""
    explanation: str = ""
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full analysis result."""
        r = self.propagation_result
        return {
            "propagation_id": r.propagation_id,
            "hazard_type": r.hazard_type,
            "origin_zone": r.origin_zone,
            "propagation_level": r.propagation_level.value,
            "status": r.status.value,
            "affected_zones": r.affected_zone_ids,
            "total_affected_zones": r.total_affected_zones,
            "total_workers_at_risk": r.total_workers_at_risk,
            "impact_radius_meters": r.impact_radius_meters,
            "time_to_critical_minutes": r.time_to_critical_minutes,
            "impact_scores": r.impact_scores,
            "propagation_probabilities": r.propagation_probabilities,
            "affected_equipment": [
                {
                    "equipment_id": eq.equipment_id,
                    "equipment_type": eq.equipment_type,
                    "zone_id": eq.zone_id,
                    "impact_score": eq.impact_score,
                    "is_critical": eq.is_critical,
                }
                for eq in r.affected_equipment
            ],
            "propagation_paths": [
                {
                    "from_zone": p.from_zone,
                    "to_zone": p.to_zone,
                    "probability": p.probability,
                    "estimated_time_minutes": p.estimated_time_minutes,
                }
                for p in r.propagation_paths
            ],
            "recommended_action": r.recommended_action,
            "summary": self.summary,
            "explanation": self.explanation,
            "recommendations": self.recommendations,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hazard Propagation Service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class HazardPropagationService:
    """Main façade for Hazard Propagation Intelligence.

    Coordinates all sub-components to produce a complete propagation
    analysis from hazard events.

    Dependencies (all constructor-injected):
        graph_repo: Graph topology data access.
        engine: BFS propagation algorithm.
        publisher: Kafka event publisher (optional).
        config: Propagation algorithm configuration.

    Usage:
        service = HazardPropagationService(repo, engine, publisher)
        result = await service.propagate_hazard(
            hazard_type="GAS_LEAK",
            origin_zone="ZONE_A",
            compound_risk_score=80.0,
        )
    """

    def __init__(
        self,
        graph_repo: GraphRepository,
        engine: Optional[HazardPropagationEngine] = None,
        publisher: Optional[HazardPropagationPublisher] = None,
        config: Optional[PropagationConfig] = None,
    ) -> None:
        self._graph_repo = graph_repo
        self._config = config or PropagationConfig()
        self._engine = engine or HazardPropagationEngine(
            graph_repo, self._config,
        )
        self._publisher = publisher

        # Metrics
        self._total_propagations: int = 0
        self._failed_propagations: int = 0
        self._total_zones_affected: int = 0
        self._total_workers_at_risk: int = 0

        logger.info(
            "HazardPropagationService initialised "
            "(decay=%.2f, max_depth=%d, threshold=%.2f, publisher=%s)",
            self._config.propagation_decay_factor,
            self._config.max_depth,
            self._config.minimum_propagation_threshold,
            "enabled" if publisher else "disabled",
        )

    # ── Properties ──

    @property
    def total_propagations(self) -> int:
        return self._total_propagations

    @property
    def failed_propagations(self) -> int:
        return self._failed_propagations

    @property
    def total_zones_affected(self) -> int:
        return self._total_zones_affected

    @property
    def total_workers_at_risk(self) -> int:
        return self._total_workers_at_risk

    @property
    def config(self) -> PropagationConfig:
        return self._config

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Core: full propagation analysis
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def propagate_hazard(
        self,
        hazard_type: str,
        origin_zone: str,
        compound_risk_score: float = 50.0,
        hazard_id: Optional[str] = None,
        max_depth: Optional[int] = None,
        correlation_id: Optional[str] = None,
    ) -> PropagationAnalysisResult:
        """Execute a full hazard propagation analysis.

        Pipeline:
            1. Validate inputs
            2. Verify origin zone exists in graph
            3. Execute propagation engine
            4. Persist hazard to graph
            5. Generate summary and explanation
            6. Publish hazard.propagated event
            7. Return complete analysis result

        Args:
            hazard_type: PS-1 HazardType enum value.
            origin_zone: Zone ID where hazard originated.
            compound_risk_score: Input risk score (0–100).
            hazard_id: Optional hazard ID (auto-generated if None).
            max_depth: Override max propagation depth.
            correlation_id: Optional ID for event chain tracing.

        Returns:
            PropagationAnalysisResult with all outputs.

        Raises:
            InvalidHazardError: If hazard_type is invalid.
            ZoneNotFoundError: If origin_zone doesn't exist.
            PropagationSimulationError: If the algorithm fails.
        """
        start_time = time.monotonic()
        hid = hazard_id or str(uuid.uuid4())

        logger.info(
            "Starting propagation: hazard_id=%s type=%s origin=%s score=%.1f",
            hid, hazard_type, origin_zone, compound_risk_score,
        )

        try:
            # Step 1: Validate
            self._validate_hazard_type(hazard_type)
            self._validate_risk_score(compound_risk_score)

            # Step 2: Verify origin zone
            zone = await self._graph_repo.get_zone(origin_zone)
            if zone is None:
                raise ZoneNotFoundError(origin_zone)

            # Step 3: Execute propagation
            result = await self._engine.propagate(
                hazard_type=hazard_type,
                origin_zone=origin_zone,
                compound_risk_score=compound_risk_score,
                hazard_id=hid,
                max_depth=max_depth,
            )

            # Step 4: Persist hazard node to graph
            hazard = await self._persist_hazard(
                hid, hazard_type, result,
            )

            # Step 5: Generate summary and explanation
            summary = self._generate_summary(result)
            explanation = self._generate_explanation(result)
            recommendations = self._generate_recommendations(result)

            # Step 6: Publish event
            self._publish_event(result, correlation_id)

            # Build final result
            elapsed = (time.monotonic() - start_time) * 1000
            self._total_propagations += 1
            self._total_zones_affected += result.total_affected_zones
            self._total_workers_at_risk += result.total_workers_at_risk

            analysis = PropagationAnalysisResult(
                propagation_result=result,
                hazard=None,  # Domain model not persisted to RDBMS
                summary=summary,
                explanation=explanation,
                recommendations=recommendations,
                processing_time_ms=elapsed,
            )

            logger.info(
                "Propagation complete: hazard_id=%s zones=%d level=%s "
                "workers=%d time=%.1fms",
                hid, result.total_affected_zones,
                result.propagation_level.value,
                result.total_workers_at_risk, elapsed,
            )

            return analysis

        except HazardPropagationError:
            self._failed_propagations += 1
            raise
        except Exception as exc:
            self._failed_propagations += 1
            logger.exception("Propagation failed unexpectedly: %s", exc)
            raise PropagationSimulationError(str(exc)) from exc

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Core: compute only (no persist / publish)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def simulate(
        self,
        hazard_type: str,
        origin_zone: str,
        compound_risk_score: float = 50.0,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Simulate propagation without persisting or publishing.

        Useful for what-if / preview scenarios.

        Returns:
            Dictionary with propagation results.
        """
        self._validate_hazard_type(hazard_type)

        zone = await self._graph_repo.get_zone(origin_zone)
        if zone is None:
            raise ZoneNotFoundError(origin_zone)

        result = await self._engine.propagate(
            hazard_type=hazard_type,
            origin_zone=origin_zone,
            compound_risk_score=compound_risk_score,
            max_depth=max_depth,
        )

        return {
            "propagation_id": result.propagation_id,
            "hazard_type": result.hazard_type,
            "origin_zone": result.origin_zone,
            "propagation_level": result.propagation_level.value,
            "affected_zones": result.affected_zone_ids,
            "total_affected_zones": result.total_affected_zones,
            "total_workers_at_risk": result.total_workers_at_risk,
            "impact_radius_meters": result.impact_radius_meters,
            "time_to_critical_minutes": result.time_to_critical_minutes,
            "impact_scores": result.impact_scores,
            "propagation_probabilities": result.propagation_probabilities,
            "recommended_action": result.recommended_action,
            "summary": self._generate_summary(result),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Queries
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_graph_stats(self) -> Dict[str, int]:
        """Return graph statistics (zones, equipment, sensors, edges)."""
        return await self._graph_repo.get_graph_stats()

    async def get_zone_neighbors(
        self, zone_id: str, max_hops: int = 2,
    ) -> Dict[str, int]:
        """Get all zones reachable from zone_id within max_hops."""
        zone = await self._graph_repo.get_zone(zone_id)
        if zone is None:
            raise ZoneNotFoundError(zone_id)
        return await self._graph_repo.get_neighbors(zone_id, max_hops)

    async def get_zone_risk_assessment(
        self, zone_id: str,
    ) -> Dict[str, Any]:
        """Get current risk assessment for a zone."""
        zone = await self._graph_repo.get_zone(zone_id)
        if zone is None:
            raise ZoneNotFoundError(zone_id)

        equipment = await self._graph_repo.get_equipment_in_zone(zone_id)
        sensors = await self._graph_repo.get_sensors_in_zone(zone_id)
        neighbors = await self._graph_repo.get_neighbors(zone_id, max_hops=1)

        return {
            "zone_id": zone.zone_id,
            "zone_name": zone.zone_name,
            "risk_level_baseline": zone.risk_level_baseline,
            "current_risk_score": zone.current_risk_score,
            "worker_count": zone.current_worker_count,
            "worker_capacity": zone.worker_capacity,
            "equipment_count": len(equipment),
            "sensor_count": len(sensors),
            "connected_zones": len(neighbors) - 1,  # Exclude self
            "has_active_hazards": zone.has_active_hazards,
            "is_restricted": zone.is_restricted,
        }

    async def get_hazard_paths(
        self, origin_zone: str, max_depth: int = 3,
    ) -> List[List[str]]:
        """Get all possible propagation paths from a zone."""
        zone = await self._graph_repo.get_zone(origin_zone)
        if zone is None:
            raise ZoneNotFoundError(origin_zone)
        return await self._graph_repo.get_hazard_paths(origin_zone, max_depth)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal: validation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _validate_hazard_type(hazard_type: str) -> None:
        """Validate hazard type against PS-1 §4.6 enum."""
        valid_types = {ht.value for ht in HazardType}
        if hazard_type.upper() not in valid_types:
            raise InvalidHazardError(
                f"Invalid hazard type: '{hazard_type}'. "
                f"Valid types: {', '.join(sorted(valid_types))}",
            )

    @staticmethod
    def _validate_risk_score(score: float) -> None:
        """Validate compound risk score is in range 0–100."""
        if not 0.0 <= score <= 100.0:
            raise InvalidHazardError(
                f"compound_risk_score must be 0.0–100.0, got {score}",
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal: persistence
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _persist_hazard(
        self,
        hazard_id: str,
        hazard_type: str,
        result: PropagationResult,
    ) -> Optional[HazardNode]:
        """Persist hazard node with AFFECTS relationships to the graph."""
        try:
            hazard_node = HazardNode(
                hazard_id=hazard_id,
                hazard_type=hazard_type,
                severity=result.propagation_level.value,
                affected_zones=result.affected_zone_ids,
            )
            await self._graph_repo.create_hazard(hazard_node)
            logger.debug(
                "Persisted hazard %s → %d zones",
                hazard_id, len(result.affected_zone_ids),
            )
            return hazard_node
        except Exception:
            # Persistence failure should not crash the propagation
            logger.exception(
                "Failed to persist hazard %s to graph", hazard_id,
            )
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal: Kafka publishing
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _publish_event(
        self,
        result: PropagationResult,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Publish hazard.propagated if publisher is available."""
        if self._publisher is None:
            logger.debug(
                "No publisher configured — skipping event publish",
            )
            return

        try:
            self._publisher.publish_hazard_propagated(
                result=result,
                correlation_id=correlation_id,
            )
        except Exception:
            # Publishing failures must not crash the analysis pipeline
            logger.exception(
                "Failed to publish hazard.propagated for "
                "propagation_id=%s",
                result.propagation_id,
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal: summaries and explanations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _generate_summary(result: PropagationResult) -> str:
        """Generate a concise human-readable summary."""
        level = result.propagation_level.value
        n_zones = result.total_affected_zones
        n_workers = result.total_workers_at_risk
        radius = result.impact_radius_meters
        ttc = result.time_to_critical_minutes

        return (
            f"{result.hazard_type} propagation from {result.origin_zone}: "
            f"{level} — {n_zones} zone(s) affected, "
            f"{n_workers} worker(s) at risk, "
            f"impact radius {radius:.0f}m, "
            f"time to critical {ttc:.1f} min."
        )

    @staticmethod
    def _generate_explanation(result: PropagationResult) -> str:
        """Generate a detailed explanation of the propagation."""
        lines = [
            f"Hazard Type: {result.hazard_type}",
            f"Origin Zone: {result.origin_zone}",
            f"Propagation Level: {result.propagation_level.value}",
            f"Status: {result.status.value}",
            "",
            "Zone Impact Assessment:",
        ]

        for zone in sorted(
            result.affected_zones,
            key=lambda z: z.risk_score,
            reverse=True,
        ):
            lines.append(
                f"  • {zone.zone_id}: risk_score={zone.risk_score:.1f} "
                f"({zone.risk_level.value}), "
                f"probability={zone.propagation_probability:.2%}, "
                f"workers={zone.worker_count}"
            )

        if result.affected_equipment:
            lines.append("")
            lines.append("Equipment Impact Assessment:")
            for eq in sorted(
                result.affected_equipment,
                key=lambda e: e.impact_score,
                reverse=True,
            ):
                critical_tag = " [CRITICAL]" if eq.is_critical else ""
                lines.append(
                    f"  • {eq.equipment_id} ({eq.equipment_type}): "
                    f"impact={eq.impact_score:.1f}{critical_tag}"
                )

        if result.propagation_paths:
            lines.append("")
            lines.append("Propagation Paths:")
            for path in result.propagation_paths:
                lines.append(
                    f"  {path.from_zone} → {path.to_zone} "
                    f"(prob={path.probability:.2%}, "
                    f"ETA={path.estimated_time_minutes:.0f} min)"
                )

        return "\n".join(lines)

    @staticmethod
    def _generate_recommendations(
        result: PropagationResult,
    ) -> List[str]:
        """Generate actionable recommendations based on propagation level."""
        recommendations = []
        level = result.propagation_level

        if level == PropagationLevel.EMERGENCY:
            recommendations.extend([
                "IMMEDIATE: Initiate facility-wide evacuation protocol.",
                f"Evacuate all affected zones: "
                f"{', '.join(result.affected_zone_ids)}.",
                f"Alert {result.total_workers_at_risk} workers at risk.",
                "Deploy all available emergency response teams.",
                "Notify local emergency services.",
            ])
        elif level == PropagationLevel.CRITICAL:
            recommendations.extend([
                f"URGENT: Evacuate affected zones: "
                f"{', '.join(result.affected_zone_ids)}.",
                "Deploy hazard containment teams immediately.",
                "Restrict access to adjacent zones.",
                "Prepare facility-wide evacuation as contingency.",
            ])
        elif level == PropagationLevel.SPREADING:
            recommendations.extend([
                f"WARNING: Hazard spreading from {result.origin_zone}.",
                "Restrict access to adjacent zones.",
                "Deploy monitoring equipment to track spread.",
                "Alert workers in potentially affected zones.",
                "Prepare containment resources.",
            ])
        else:  # CONTAINED
            recommendations.extend([
                f"MONITOR: Hazard contained in {result.origin_zone}.",
                "Continue monitoring for changes.",
                "Ensure containment measures are in place.",
                "Review sensor readings for anomalies.",
            ])

        # Equipment-specific recommendations
        critical_equipment = [
            eq for eq in result.affected_equipment if eq.is_critical
        ]
        if critical_equipment:
            eq_ids = ", ".join(eq.equipment_id for eq in critical_equipment)
            recommendations.append(
                f"EQUIPMENT: Critical impact on: {eq_ids}. "
                f"Initiate emergency shutdown procedures.",
            )

        return recommendations
