"""Compound Risk Aggregation Engine.

Computes a compound risk score by combining signals from multiple upstream
modules through a configurable weighted aggregation mechanism.

Inputs:
  - Isolation Forest anomaly score
  - Autoencoder anomaly score
  - Accident probability from Risk Prediction
  - Sensor health score
  - Active alert count / severity
  - Threshold violation count

Outputs:
  - compound_risk_score (0–100)
  - risk_level (LOW | MEDIUM | HIGH | CRITICAL)
  - confidence_score (0.0–1.0)
  - contributing_factors breakdown

Architecture:
  - All business logic lives in this service layer
  - Weights are configurable via CompoundRiskWeights dataclass
  - Risk level thresholds are configurable via RiskLevelThresholds
  - No hardcoded values — everything flows through configuration
  - Thread-safe (no mutable shared state beyond config)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.compound_risk.domain.value_objects import RiskLevel
from app.compound_risk.models.compound_risk_model import CompoundRiskModel
from app.compound_risk.repositories.compound_risk_repository import (
    CompoundRiskRepository,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration data classes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class CompoundRiskWeights:
    """Configurable weights for each risk component.

    All weights should sum to 1.0 for a properly normalised score.
    The engine auto-normalises if they don't, but logs a warning.

    Default weights derived from architecture document Feature 1:
      gas_weight(0.40), maintenance_weight(0.25),
      worker_weight(0.20), permit_weight(0.15)
    Adapted for our aggregation inputs.
    """

    risk_prediction_weight: float = 0.30
    isolation_forest_weight: float = 0.20
    autoencoder_weight: float = 0.15
    sensor_health_weight: float = 0.15
    alert_weight: float = 0.10
    threshold_violation_weight: float = 0.10

    @property
    def total(self) -> float:
        return (
            self.risk_prediction_weight
            + self.isolation_forest_weight
            + self.autoencoder_weight
            + self.sensor_health_weight
            + self.alert_weight
            + self.threshold_violation_weight
        )

    def normalised(self) -> CompoundRiskWeights:
        """Return a copy with weights normalised to sum to 1.0."""
        t = self.total
        if t == 0:
            return CompoundRiskWeights()
        return CompoundRiskWeights(
            risk_prediction_weight=self.risk_prediction_weight / t,
            isolation_forest_weight=self.isolation_forest_weight / t,
            autoencoder_weight=self.autoencoder_weight / t,
            sensor_health_weight=self.sensor_health_weight / t,
            alert_weight=self.alert_weight / t,
            threshold_violation_weight=self.threshold_violation_weight / t,
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class RiskLevelThresholds:
    """Configurable thresholds for risk level classification.

    Compound risk score (0–100) is mapped to:
      score < low_max      → LOW
      score < medium_max   → MEDIUM
      score < high_max     → HIGH
      score >= high_max    → CRITICAL
    """

    low_max: float = 25.0
    medium_max: float = 50.0
    high_max: float = 75.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input data class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class CompoundRiskInput:
    """All inputs required for a compound risk calculation.

    Scores are 0.0–1.0 unless noted otherwise.
    """

    # ── Upstream model scores ──
    isolation_forest_score: float = 0.0    # 0.0–1.0
    autoencoder_score: float = 0.0         # 0.0–1.0
    accident_probability: float = 0.0      # 0.0–1.0
    risk_score: float = 0.0               # 0–100 (from risk prediction)

    # ── Sensor health ──
    sensor_health_score: float = 100.0     # 0–100 (higher = healthier)

    # ── Operational context ──
    active_alert_count: int = 0
    alert_severity_max: float = 0.0        # 0.0–1.0
    threshold_violation_count: int = 0

    # ── Metadata ──
    equipment_id: Optional[str] = None
    zone_id: Optional[str] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Aggregation result
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class CompoundRiskResult:
    """Output of the compound risk aggregation engine."""

    compound_risk_score: float    # 0–100
    risk_level: RiskLevel
    confidence_score: float       # 0.0–1.0
    contributing_factors: List[Dict[str, Any]]
    component_scores: Dict[str, float]  # Raw per-component scores (0–100 scale)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Compound Risk Aggregation Service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CompoundRiskAggregationService:
    """Service that computes compound risk from multiple upstream signals.

    All business logic is contained here. No hardcoded values — weights
    and thresholds are fully configurable.
    """

    def __init__(
        self,
        repo: CompoundRiskRepository,
        weights: Optional[CompoundRiskWeights] = None,
        thresholds: Optional[RiskLevelThresholds] = None,
    ) -> None:
        self._repo = repo

        raw_weights = weights or CompoundRiskWeights()
        # Auto-normalise if weights don't sum to 1.0
        if abs(raw_weights.total - 1.0) > 0.001:
            logger.warning(
                "Compound risk weights sum to %.4f (expected 1.0) — auto-normalising",
                raw_weights.total,
            )
            self._weights = raw_weights.normalised()
        else:
            self._weights = raw_weights

        self._thresholds = thresholds or RiskLevelThresholds()

    @property
    def weights(self) -> CompoundRiskWeights:
        return self._weights

    @property
    def thresholds(self) -> RiskLevelThresholds:
        return self._thresholds

    # ── Core aggregation ──

    def compute(self, inp: CompoundRiskInput) -> CompoundRiskResult:
        """Compute the compound risk score from upstream signals.

        Steps:
          1. Normalise each input to a 0–100 risk scale
          2. Apply configurable weights
          3. Sum to compound_risk_score (0–100)
          4. Classify risk_level via thresholds
          5. Calculate confidence from input completeness and agreement
          6. Build contributing_factors breakdown

        Args:
            inp: All upstream signals and metadata.

        Returns:
            CompoundRiskResult with score, level, confidence, and factors.
        """
        # 1. Normalise each component to 0–100 risk scale
        components = self._normalise_components(inp)

        # 2. Weighted aggregation
        compound_score = (
            components["risk_prediction"] * self._weights.risk_prediction_weight
            + components["isolation_forest"] * self._weights.isolation_forest_weight
            + components["autoencoder"] * self._weights.autoencoder_weight
            + components["sensor_health"] * self._weights.sensor_health_weight
            + components["alert"] * self._weights.alert_weight
            + components["threshold_violation"] * self._weights.threshold_violation_weight
        )

        # Clamp to [0, 100]
        compound_score = max(0.0, min(100.0, compound_score))

        # 3. Classify
        risk_level = self._classify(compound_score)

        # 4. Confidence
        confidence = self._calculate_confidence(inp, components)

        # 5. Contributing factors
        factors = self._build_contributing_factors(components)

        return CompoundRiskResult(
            compound_risk_score=round(compound_score, 2),
            risk_level=risk_level,
            confidence_score=round(confidence, 4),
            contributing_factors=factors,
            component_scores={k: round(v, 2) for k, v in components.items()},
        )

    async def compute_and_persist(
        self,
        inp: CompoundRiskInput,
        recommendation: Optional[str] = None,
    ) -> CompoundRiskModel:
        """Compute compound risk and persist the result.

        Args:
            inp: Upstream signals.
            recommendation: Optional text recommendation.

        Returns:
            Persisted CompoundRiskModel.
        """
        result = self.compute(inp)

        model = CompoundRiskModel(
            id=str(uuid.uuid4()),
            equipment_id=inp.equipment_id,
            zone_id=inp.zone_id,
            anomaly_score=max(inp.isolation_forest_score, inp.autoencoder_score),
            accident_probability=inp.accident_probability,
            risk_score=inp.risk_score,
            sensor_health_score=inp.sensor_health_score,
            compound_risk_score=result.compound_risk_score / 100.0,  # Store as 0–1
            risk_level=result.risk_level.value,
            confidence_score=result.confidence_score,
            contributing_factors=json.dumps(result.contributing_factors),
            recommendation=recommendation,
            created_at=datetime.now(timezone.utc),
        )

        persisted = await self._repo.create(model)
        logger.info(
            "Compound risk computed: zone=%s score=%.1f level=%s",
            inp.zone_id, result.compound_risk_score, result.risk_level.value,
        )
        return persisted

    # ── Query helpers ──

    async def get_by_id(self, analysis_id: str) -> Optional[CompoundRiskModel]:
        return await self._repo.get_by_id(analysis_id)

    async def get_latest(
        self,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
    ) -> Optional[CompoundRiskModel]:
        return await self._repo.get_latest(
            zone_id=zone_id, equipment_id=equipment_id,
        )

    async def get_history(
        self,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[CompoundRiskModel]:
        return await self._repo.get_history(
            zone_id=zone_id, equipment_id=equipment_id,
            risk_level=risk_level, offset=offset, limit=limit,
        )

    async def count(
        self,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> int:
        return await self._repo.count(
            zone_id=zone_id, equipment_id=equipment_id,
            risk_level=risk_level,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal: normalisation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _normalise_components(inp: CompoundRiskInput) -> Dict[str, float]:
        """Normalise each input to a 0–100 risk scale.

        Higher value = higher risk across all components.
        """
        # Risk prediction: accident_probability (0–1) → 0–100
        risk_pred = inp.accident_probability * 100.0

        # Isolation Forest score: 0–1 → 0–100
        iso_forest = inp.isolation_forest_score * 100.0

        # Autoencoder score: 0–1 → 0–100
        autoenc = inp.autoencoder_score * 100.0

        # Sensor health: 0–100 where 100=healthy → invert (100=unhealthy risk)
        sensor_health_risk = 100.0 - inp.sensor_health_score

        # Alerts: count + severity → 0–100
        # Scale: 0 alerts = 0, 1+ alerts scaled by max severity
        if inp.active_alert_count > 0:
            count_factor = min(inp.active_alert_count / 10.0, 1.0)
            severity_factor = max(inp.alert_severity_max, 0.3)
            alert_risk = count_factor * severity_factor * 100.0
        else:
            alert_risk = 0.0

        # Threshold violations: count → 0–100
        # Scale: 0 = 0, 5+ = 100
        violation_risk = min(inp.threshold_violation_count / 5.0, 1.0) * 100.0

        return {
            "risk_prediction": max(0.0, min(100.0, risk_pred)),
            "isolation_forest": max(0.0, min(100.0, iso_forest)),
            "autoencoder": max(0.0, min(100.0, autoenc)),
            "sensor_health": max(0.0, min(100.0, sensor_health_risk)),
            "alert": max(0.0, min(100.0, alert_risk)),
            "threshold_violation": max(0.0, min(100.0, violation_risk)),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal: classification
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _classify(self, score: float) -> RiskLevel:
        """Classify compound risk score into RiskLevel using thresholds."""
        if score < self._thresholds.low_max:
            return RiskLevel.LOW
        elif score < self._thresholds.medium_max:
            return RiskLevel.MEDIUM
        elif score < self._thresholds.high_max:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal: confidence
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _calculate_confidence(
        inp: CompoundRiskInput,
        components: Dict[str, float],
    ) -> float:
        """Calculate confidence score based on input completeness and agreement.

        Factors:
          1. Data completeness: how many inputs are non-default
          2. Inter-model agreement: how consistent anomaly signals are
          3. Sensor health: degraded sensors lower confidence
        """
        # 1. Completeness (each non-default input adds confidence)
        data_points = 0
        if inp.accident_probability > 0:
            data_points += 1
        if inp.isolation_forest_score > 0:
            data_points += 1
        if inp.autoencoder_score > 0:
            data_points += 1
        if inp.sensor_health_score < 100:
            data_points += 1
        if inp.active_alert_count > 0:
            data_points += 1
        if inp.threshold_violation_count > 0:
            data_points += 1

        completeness = min(data_points / 4.0, 1.0)  # 4+ inputs = full confidence

        # 2. Inter-model agreement
        model_scores = [
            components["risk_prediction"],
            components["isolation_forest"],
            components["autoencoder"],
        ]
        non_zero = [s for s in model_scores if s > 0]
        if len(non_zero) >= 2:
            mean = sum(non_zero) / len(non_zero)
            variance = sum((s - mean) ** 2 for s in non_zero) / len(non_zero)
            # Lower variance → higher agreement → higher confidence
            agreement = max(0.0, 1.0 - (variance / 2500.0))
        else:
            agreement = 0.5  # Single model → moderate confidence

        # 3. Sensor health factor
        health_factor = inp.sensor_health_score / 100.0

        # Weighted combination
        confidence = (
            completeness * 0.4
            + agreement * 0.4
            + health_factor * 0.2
        )

        return max(0.1, min(1.0, confidence))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal: contributing factors
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_contributing_factors(
        self, components: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Build sorted contributing factors with weights and contributions.

        Returns top factors ordered by weighted contribution (highest first).
        """
        weight_map = {
            "risk_prediction": self._weights.risk_prediction_weight,
            "isolation_forest": self._weights.isolation_forest_weight,
            "autoencoder": self._weights.autoencoder_weight,
            "sensor_health": self._weights.sensor_health_weight,
            "alert": self._weights.alert_weight,
            "threshold_violation": self._weights.threshold_violation_weight,
        }

        display_names = {
            "risk_prediction": "Accident Probability",
            "isolation_forest": "Isolation Forest Anomaly",
            "autoencoder": "Autoencoder Anomaly",
            "sensor_health": "Sensor Health Degradation",
            "alert": "Active Alerts",
            "threshold_violation": "Threshold Violations",
        }

        factors = []
        for name, score in components.items():
            weight = weight_map[name]
            contribution = score * weight  # Weighted contribution to total
            factors.append({
                "factor": display_names.get(name, name),
                "weight": round(weight, 4),
                "current_value": str(round(score, 2)),
                "contribution": round(contribution / 100.0, 4),
            })

        # Sort by contribution descending
        factors.sort(key=lambda f: float(f["current_value"]) * f["weight"], reverse=True)
        return factors
