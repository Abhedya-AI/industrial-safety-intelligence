"""Hazard Propagation Engine — core propagation algorithm.

Implements BFS-based hazard spread simulation through the facility
graph. Works with ANY GraphRepository implementation (InMemory, Neo4j)
without code changes.

Algorithm:
  1. Start at the origin zone with probability 1.0
  2. BFS outward through CONNECTED_TO edges
  3. At each hop, multiply probability by decay_factor
  4. Skip zones below minimum_propagation_threshold
  5. Score each zone based on equipment health + worker count + baseline risk
  6. Track propagation paths, affected equipment, and impact scores
  7. Classify overall propagation level (CONTAINED → EMERGENCY)

Architecture:
  - GraphRepository is the ONLY data access layer (no direct Neo4j)
  - Configuration via PropagationConfig dataclass
  - Hazard-type-specific decay overrides
  - Pure algorithm — no Kafka, no API, no persistence

Checklist items addressed:
  - Impact Radius Calculation
  - Affected Worker Detection
  - Hazard Spread Simulation
  - Critical Time Window Estimation
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from app.hazard_propagation.domain.exceptions import (
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
from app.hazard_propagation.graph.entities import EquipmentNode, ZoneNode
from app.hazard_propagation.repositories.graph_repository import GraphRepository
from app.hazard_propagation.services.config import (
    HAZARD_DECAY_OVERRIDES,
    RISK_LEVEL_SCORES,
    PropagationConfig,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result container for equipment impact
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class EquipmentImpact:
    """Impact assessment for a single piece of equipment."""

    equipment_id: str
    equipment_type: str
    zone_id: str
    health_score: float
    operational_status: str
    impact_score: float = 0.0
    is_critical: bool = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result container for the full propagation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class PropagationResult:
    """Complete result of a propagation simulation.

    This is the service-layer output. It can be mapped to the
    HazardPropagation domain model or API response schemas.
    """

    propagation_id: str = field(
        default_factory=lambda: str(uuid.uuid4()),
    )
    hazard_type: str = ""
    origin_zone: str = ""
    status: PropagationStatus = PropagationStatus.COMPLETED
    propagation_level: PropagationLevel = PropagationLevel.CONTAINED

    # Affected zones with full risk state
    affected_zones: List[ZoneRiskState] = field(default_factory=list)

    # All propagation paths
    propagation_paths: List[PropagationPath] = field(default_factory=list)

    # Equipment impacts
    affected_equipment: List[EquipmentImpact] = field(default_factory=list)

    # Affected worker IDs (aggregated from zone worker counts)
    affected_workers: List[str] = field(default_factory=list)

    # Impact scores per zone
    impact_scores: Dict[str, float] = field(default_factory=dict)

    # Propagation probabilities per zone
    propagation_probabilities: Dict[str, float] = field(default_factory=dict)

    # Aggregate metrics
    impact_radius_meters: float = 0.0
    time_to_critical_minutes: float = 0.0
    total_workers_at_risk: int = 0
    recommended_action: str = ""

    @property
    def total_affected_zones(self) -> int:
        return sum(1 for z in self.affected_zones if z.is_affected)

    @property
    def affected_zone_ids(self) -> List[str]:
        return [z.zone_id for z in self.affected_zones if z.is_affected]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Propagation Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class HazardPropagationEngine:
    """Core hazard propagation algorithm.

    Uses BFS through the GraphRepository to simulate hazard spread.
    Works identically with InMemoryGraphRepository and Neo4jGraphRepository.

    Usage:
        engine = HazardPropagationEngine(repo)
        result = await engine.propagate(
            hazard_type="GAS_LEAK",
            origin_zone="ZONE_A",
            compound_risk_score=75.0,
        )
    """

    def __init__(
        self,
        graph_repo: GraphRepository,
        config: Optional[PropagationConfig] = None,
    ) -> None:
        self._repo = graph_repo
        self._config = config or PropagationConfig()

    @property
    def config(self) -> PropagationConfig:
        return self._config

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Main propagation entry point
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def propagate(
        self,
        hazard_type: str,
        origin_zone: str,
        compound_risk_score: float = 50.0,
        hazard_id: Optional[str] = None,
        max_depth: Optional[int] = None,
    ) -> PropagationResult:
        """Run a hazard propagation simulation.

        Args:
            hazard_type:         PS-1 HazardType enum value.
            origin_zone:         Zone ID where hazard originated.
            compound_risk_score: Input risk score (0–100) from upstream.
            hazard_id:           Optional hazard ID (auto-generated if None).
            max_depth:           Override max propagation depth.

        Returns:
            PropagationResult with affected zones, equipment, paths,
            impact scores, and propagation probabilities.

        Raises:
            ZoneNotFoundError: If origin_zone doesn't exist.
            PropagationSimulationError: If the simulation fails.
        """
        depth = max_depth or self._config.max_depth
        hid = hazard_id or str(uuid.uuid4())

        # Validate origin zone exists
        origin = await self._repo.get_zone(origin_zone)
        if origin is None:
            raise ZoneNotFoundError(origin_zone)

        try:
            result = PropagationResult(
                propagation_id=hid,
                hazard_type=hazard_type,
                origin_zone=origin_zone,
            )

            # Step 1: BFS propagation
            zone_probabilities = await self._bfs_propagate(
                origin_zone, hazard_type, depth,
            )

            # Step 2: Score each zone
            for zone_id, (probability, hop) in zone_probabilities.items():
                zone_state = await self._score_zone(
                    zone_id=zone_id,
                    probability=probability,
                    hop=hop,
                    hazard_type=hazard_type,
                    compound_risk_score=compound_risk_score,
                    is_origin=(zone_id == origin_zone),
                )
                result.affected_zones.append(zone_state)
                result.impact_scores[zone_id] = zone_state.risk_score
                result.propagation_probabilities[zone_id] = probability

            # Step 3: Score equipment
            result.affected_equipment = await self._score_equipment(
                zone_probabilities, compound_risk_score,
            )

            # Step 4: Build propagation paths
            result.propagation_paths = await self._build_paths(
                origin_zone, zone_probabilities,
            )

            # Step 5: Compute aggregate metrics
            result.total_workers_at_risk = sum(
                z.worker_count for z in result.affected_zones if z.is_affected
            )
            result.impact_radius_meters = self._compute_impact_radius(
                zone_probabilities,
            )
            result.time_to_critical_minutes = self._estimate_time_to_critical(
                zone_probabilities, compound_risk_score,
            )

            # Step 6: Classify propagation level
            result.propagation_level = self._classify_level(result)

            # Step 7: Generate recommendation
            result.recommended_action = self._generate_recommendation(result)

            result.status = PropagationStatus.COMPLETED
            logger.info(
                "Propagation %s completed: %d zones affected, level=%s",
                hid, result.total_affected_zones, result.propagation_level,
            )
            return result

        except ZoneNotFoundError:
            raise
        except Exception as exc:
            logger.exception("Propagation simulation failed: %s", exc)
            raise PropagationSimulationError(str(exc)) from exc

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 1: BFS Propagation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _bfs_propagate(
        self,
        origin: str,
        hazard_type: str,
        max_depth: int,
    ) -> Dict[str, Tuple[float, int]]:
        """BFS with weighted decay.

        Returns: {zone_id: (probability, hop_distance)}

        At each hop the probability decays by the decay factor.
        Hazard-specific decay overrides are applied when available.
        Zones below minimum_propagation_threshold are excluded.
        """
        decay = HAZARD_DECAY_OVERRIDES.get(
            hazard_type, self._config.propagation_decay_factor,
        )
        threshold = self._config.minimum_propagation_threshold

        # Result: zone_id → (probability, hop)
        visited: Dict[str, Tuple[float, int]] = {origin: (1.0, 0)}

        # BFS queue: (zone_id, current_probability, hop_count)
        queue: deque[Tuple[str, float, int]] = deque()
        queue.append((origin, 1.0, 0))

        while queue:
            current_zone, current_prob, hop = queue.popleft()

            if hop >= max_depth:
                continue

            # Get neighbors from GraphRepository
            neighbors = await self._repo.get_connected_zones(current_zone)
            next_prob = current_prob * decay

            if next_prob < threshold:
                continue

            for neighbor in neighbors:
                nid = neighbor.zone_id
                if nid not in visited or visited[nid][0] < next_prob:
                    visited[nid] = (next_prob, hop + 1)
                    queue.append((nid, next_prob, hop + 1))

        return visited

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 2: Zone Impact Scoring
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _score_zone(
        self,
        zone_id: str,
        probability: float,
        hop: int,
        hazard_type: str,
        compound_risk_score: float,
        is_origin: bool,
    ) -> ZoneRiskState:
        """Compute impact score for a single zone.

        Impact score = compound_risk_score × probability
                     × (baseline_weight × baseline_risk
                        + worker_weight × normalised_worker_count
                        + equipment_weight × equipment_factor)

        The result is clamped to 0–100.
        """
        zone = await self._repo.get_zone(zone_id)
        if zone is None:
            return ZoneRiskState(
                zone_id=zone_id, is_affected=True,
                propagation_probability=probability,
            )

        # Baseline risk from zone's risk_level_baseline
        baseline_score = RISK_LEVEL_SCORES.get(
            zone.risk_level_baseline, 10.0,
        )

        # Normalise worker count (cap at 50 for scoring)
        worker_factor = min(zone.current_worker_count / 50.0, 1.0)

        # Equipment factor: proportion of non-operational equipment
        equipment = await self._repo.get_equipment_in_zone(zone_id)
        if equipment:
            non_operational = sum(
                1 for eq in equipment if not eq.is_operational
            )
            equipment_factor = (non_operational / len(equipment))
        else:
            equipment_factor = 0.0

        # Weighted impact score
        weighted = (
            self._config.baseline_risk_weight * (baseline_score / 100.0)
            + self._config.worker_risk_weight * worker_factor
            + self._config.equipment_risk_weight * equipment_factor
        )

        # Final score: compound_risk × probability × weighted factors
        raw_score = compound_risk_score * probability * (0.5 + weighted)
        impact_score = min(max(raw_score, 0.0), 100.0)

        # Classify risk level from score
        risk_level = self._score_to_risk_level(impact_score)

        return ZoneRiskState(
            zone_id=zone_id,
            zone_name=zone.zone_name,
            risk_level=risk_level,
            risk_score=round(impact_score, 2),
            is_origin=is_origin,
            is_affected=True,
            arrival_time_minutes=hop * self._config.time_per_hop_minutes,
            propagation_probability=round(probability, 4),
            worker_count=zone.current_worker_count,
            equipment_count=zone.equipment_count,
            active_hazards=[hazard_type] if is_origin else [],
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 3: Equipment Impact Scoring
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _score_equipment(
        self,
        zone_probabilities: Dict[str, Tuple[float, int]],
        compound_risk_score: float,
    ) -> List[EquipmentImpact]:
        """Score all equipment in affected zones.

        Equipment impact = zone_probability × (1 - health_score/100)
                         × compound_risk_score / 100
        """
        impacts: List[EquipmentImpact] = []

        for zone_id, (prob, _) in zone_probabilities.items():
            equipment = await self._repo.get_equipment_in_zone(zone_id)
            for eq in equipment:
                vulnerability = 1.0 - (eq.health_score / 100.0)
                impact = prob * vulnerability * (compound_risk_score / 100.0)
                impact_score = min(impact * 100.0, 100.0)

                impacts.append(EquipmentImpact(
                    equipment_id=eq.equipment_id,
                    equipment_type=eq.equipment_type,
                    zone_id=zone_id,
                    health_score=eq.health_score,
                    operational_status=eq.operational_status,
                    impact_score=round(impact_score, 2),
                    is_critical=impact_score > 70.0,
                ))

        return impacts

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 4: Propagation Paths
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _build_paths(
        self,
        origin: str,
        zone_probabilities: Dict[str, Tuple[float, int]],
    ) -> List[PropagationPath]:
        """Build PropagationPath objects for each edge in the BFS tree."""
        paths: List[PropagationPath] = []
        visited_edges: Set[Tuple[str, str]] = set()

        for zone_id, (prob, hop) in zone_probabilities.items():
            if hop == 0:
                continue  # Skip origin

            # Find which connected zones could be the parent
            neighbors = await self._repo.get_connected_zones(zone_id)
            for neighbor in neighbors:
                nid = neighbor.zone_id
                if nid in zone_probabilities:
                    parent_prob, parent_hop = zone_probabilities[nid]
                    if parent_hop == hop - 1:
                        edge = (nid, zone_id)
                        if edge not in visited_edges:
                            visited_edges.add(edge)
                            paths.append(PropagationPath(
                                from_zone=nid,
                                to_zone=zone_id,
                                probability=round(prob, 4),
                                estimated_time_minutes=(
                                    hop * self._config.time_per_hop_minutes
                                ),
                                path_type="CONNECTED_TO",
                            ))

        return paths

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 5: Aggregate Metrics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _compute_impact_radius(
        self,
        zone_probabilities: Dict[str, Tuple[float, int]],
    ) -> float:
        """Estimate impact radius based on maximum hop distance."""
        if not zone_probabilities:
            return 0.0
        max_hop = max(hop for _, hop in zone_probabilities.values())
        return self._config.base_impact_radius_meters * (1 + max_hop)

    def _estimate_time_to_critical(
        self,
        zone_probabilities: Dict[str, Tuple[float, int]],
        compound_risk_score: float,
    ) -> float:
        """Estimate time until the hazard reaches critical threshold.

        If compound_risk_score is already critical (>80), time is 0.
        Otherwise, estimate based on propagation speed and risk growth.
        """
        if compound_risk_score >= 80.0:
            return 0.0
        if not zone_probabilities:
            return 0.0

        max_hop = max(hop for _, hop in zone_probabilities.values())
        # Scale time based on gap between current risk and critical
        risk_gap_factor = (80.0 - compound_risk_score) / 80.0
        return max_hop * self._config.time_per_hop_minutes * risk_gap_factor

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 6: Classification
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _classify_level(self, result: PropagationResult) -> PropagationLevel:
        """Classify overall propagation severity.

        CONTAINED:  1 zone affected
        SPREADING:  2–3 zones affected
        CRITICAL:   4+ zones OR any zone with risk_score > 80
        EMERGENCY:  5+ zones AND workers at risk > 10
        """
        n_affected = result.total_affected_zones
        max_score = max(
            (z.risk_score for z in result.affected_zones if z.is_affected),
            default=0.0,
        )

        if n_affected >= 5 and result.total_workers_at_risk > 10:
            return PropagationLevel.EMERGENCY
        if n_affected >= 4 or max_score > 80.0:
            return PropagationLevel.CRITICAL
        if n_affected >= 2:
            return PropagationLevel.SPREADING
        return PropagationLevel.CONTAINED

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 7: Recommendations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _generate_recommendation(
        self, result: PropagationResult,
    ) -> str:
        """Generate a human-readable recommended action."""
        level = result.propagation_level

        if level == PropagationLevel.EMERGENCY:
            zones = ", ".join(result.affected_zone_ids)
            return (
                f"EMERGENCY: Initiate facility-wide evacuation. "
                f"Affected zones: {zones}. "
                f"{result.total_workers_at_risk} workers at risk."
            )

        if level == PropagationLevel.CRITICAL:
            zones = ", ".join(result.affected_zone_ids)
            return (
                f"CRITICAL: Evacuate affected zones ({zones}). "
                f"Deploy hazard containment teams. "
                f"Restrict access to adjacent zones."
            )

        if level == PropagationLevel.SPREADING:
            return (
                f"WARNING: Hazard spreading from {result.origin_zone}. "
                f"Restrict access to adjacent zones and deploy monitoring."
            )

        return (
            f"CONTAINED: Hazard contained in {result.origin_zone}. "
            f"Monitor conditions and prepare containment if needed."
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Utilities
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _score_to_risk_level(score: float) -> RiskLevel:
        """Map a 0–100 score to a RiskLevel."""
        if score >= 80.0:
            return RiskLevel.CRITICAL
        if score >= 50.0:
            return RiskLevel.HIGH
        if score >= 25.0:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
