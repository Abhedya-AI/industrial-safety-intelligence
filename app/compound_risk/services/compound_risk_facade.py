"""Compound Risk Intelligence Service — main façade.

Coordinates all Compound Risk sub-components:
  - Aggregation Engine (weighted score computation)
  - Rule Engine (configurable condition evaluation)
  - Explainability Service (human-readable explanations)
  - Repository (persistence)
  - Kafka Publisher (event publishing)

This is the single entry-point used by the API layer and event handlers.
All business logic is delegated to specialised components; this service
is responsible for orchestration, error handling, and logging.

Design:
  - Follows the existing DI pattern (constructor injection)
  - All dependencies are injected — no hardcoded infrastructure
  - Stateless per request (state lives in DB and Kafka)
  - Exceptions follow the domain exception hierarchy
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.compound_risk.domain.exceptions import (
    CompoundRiskAnalysisFailedError,
    CompoundRiskError,
    InsufficientScenarioDataError,
)
from app.compound_risk.domain.value_objects import RiskLevel
from app.compound_risk.messaging.publisher import CompoundRiskPublisher
from app.compound_risk.models.compound_risk_model import CompoundRiskModel
from app.compound_risk.repositories.compound_risk_repository import (
    CompoundRiskRepository,
)
from app.compound_risk.rules.rule_engine import (
    CompoundRiskRuleEngine,
    RuleEngineResult,
)
from app.compound_risk.services.compound_risk_service import (
    CompoundRiskAggregationService,
    CompoundRiskInput,
    CompoundRiskResult,
)
from app.compound_risk.services.explainability_service import (
    CompoundRiskExplanation,
    ExplainabilityService,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis result (full output)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class CompoundRiskAnalysisResult:
    """Complete output of a compound risk analysis.

    Aggregates outputs from every sub-component into a single response.
    """

    # Persisted model
    model: CompoundRiskModel

    # Aggregation engine output
    compound_risk_score: float
    risk_level: RiskLevel
    confidence_score: float
    contributing_factors: List[Dict[str, Any]]
    component_scores: Dict[str, float]

    # Rule engine output
    rule_result: RuleEngineResult

    # Explainability output
    explanation: CompoundRiskExplanation

    # Metadata
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full analysis result."""
        return {
            "analysis_id": self.model.id,
            "equipment_id": self.model.equipment_id,
            "zone_id": self.model.zone_id,
            "compound_risk_score": round(self.compound_risk_score, 2),
            "risk_level": self.risk_level.value,
            "confidence_score": round(self.confidence_score, 4),
            "contributing_factors": self.contributing_factors,
            "component_scores": self.component_scores,
            "triggered_rules": [
                r.to_dict() for r in self.explanation.triggered_rules
            ],
            "recommendation": self.model.recommendation,
            "explanation": self.explanation.summary,
            "key_drivers": self.explanation.key_drivers,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "created_at": (
                self.model.created_at.isoformat()
                if self.model.created_at else None
            ),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Compound Risk Service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CompoundRiskService:
    """Main façade for Compound Risk Intelligence.

    Coordinates all sub-components to produce a complete compound risk
    analysis from upstream inputs.

    Dependencies (all constructor-injected):
        aggregation_service: Weighted score computation + persistence.
        rule_engine: Configurable rule evaluation.
        explainability_service: Human-readable explanation generator.
        publisher: Kafka event publisher.

    Usage:
        service = CompoundRiskService(aggregation, rules, explain, publisher)
        result = await service.analyze(input, sensor_facts)
    """

    def __init__(
        self,
        aggregation_service: CompoundRiskAggregationService,
        rule_engine: CompoundRiskRuleEngine,
        explainability_service: ExplainabilityService,
        publisher: Optional[CompoundRiskPublisher] = None,
    ) -> None:
        self._aggregation = aggregation_service
        self._rule_engine = rule_engine
        self._explainability = explainability_service
        self._publisher = publisher

        # Metrics
        self._total_analyses: int = 0
        self._failed_analyses: int = 0

        logger.info(
            "CompoundRiskService initialised (rules=%d, publisher=%s)",
            rule_engine.rule_count,
            "enabled" if publisher else "disabled",
        )

    # ── Properties ──

    @property
    def total_analyses(self) -> int:
        return self._total_analyses

    @property
    def failed_analyses(self) -> int:
        return self._failed_analyses

    @property
    def rule_count(self) -> int:
        return self._rule_engine.rule_count

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Core: full analysis
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def analyze(
        self,
        inp: CompoundRiskInput,
        sensor_facts: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> CompoundRiskAnalysisResult:
        """Execute a full compound risk analysis.

        Pipeline:
            1. Validate inputs
            2. Run Rule Engine against sensor facts
            3. Run Aggregation Engine (weighted score computation)
            4. Generate Explanation
            5. Persist result to database
            6. Publish compound.risk.detected event
            7. Return complete analysis result

        Args:
            inp: Upstream signals (anomaly scores, accident probability,
                 sensor health, alerts, threshold violations).
            sensor_facts: Raw sensor measurements for rule evaluation
                (e.g. temperature_celsius, gas_level_ppm).
            correlation_id: Optional ID for event chain tracing.

        Returns:
            CompoundRiskAnalysisResult with all outputs.

        Raises:
            InsufficientScenarioDataError: If inputs are empty/invalid.
            CompoundRiskAnalysisFailedError: If computation fails.
        """
        start_time = time.monotonic()

        try:
            # Step 1: Validate
            self._validate_input(inp)

            # Step 2: Rule Engine
            facts = sensor_facts or {}
            rule_result = self._rule_engine.evaluate(facts)
            logger.debug(
                "Rule engine: %d triggered, total_impact=%.2f",
                len(rule_result.triggered_rules),
                rule_result.total_impact,
            )

            # Step 3: Aggregation Engine
            agg_result = self._aggregation.compute(inp)
            logger.debug(
                "Aggregation: score=%.1f level=%s confidence=%.3f",
                agg_result.compound_risk_score,
                agg_result.risk_level.value,
                agg_result.confidence_score,
            )

            # Step 4: Explainability
            explanation = self._explainability.explain(
                inp, agg_result, rule_result,
            )

            # Step 5: Persist
            recommendation = self._build_recommendation(explanation, rule_result)
            model = await self._aggregation.compute_and_persist(
                inp, recommendation=recommendation,
            )

            # Step 6: Publish
            self._publish_event(model, agg_result, correlation_id)

            # Build result
            elapsed = (time.monotonic() - start_time) * 1000
            self._total_analyses += 1

            result = CompoundRiskAnalysisResult(
                model=model,
                compound_risk_score=agg_result.compound_risk_score,
                risk_level=agg_result.risk_level,
                confidence_score=agg_result.confidence_score,
                contributing_factors=agg_result.contributing_factors,
                component_scores=agg_result.component_scores,
                rule_result=rule_result,
                explanation=explanation,
                processing_time_ms=elapsed,
            )

            logger.info(
                "Compound risk analysis complete: "
                "id=%s zone=%s score=%.1f level=%s rules=%d time=%.1fms",
                model.id, model.zone_id,
                agg_result.compound_risk_score,
                agg_result.risk_level.value,
                len(rule_result.triggered_rules),
                elapsed,
            )

            return result

        except CompoundRiskError:
            self._failed_analyses += 1
            raise
        except Exception as exc:
            self._failed_analyses += 1
            logger.exception("Compound risk analysis failed unexpectedly")
            raise CompoundRiskAnalysisFailedError(str(exc)) from exc

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Core: compute only (no persist / publish)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def compute(
        self,
        inp: CompoundRiskInput,
        sensor_facts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute compound risk without persisting or publishing.

        Useful for dry-run / preview endpoints.

        Returns:
            Dictionary with score, level, confidence, rules, explanation.
        """
        self._validate_input(inp)
        facts = sensor_facts or {}

        rule_result = self._rule_engine.evaluate(facts)
        agg_result = self._aggregation.compute(inp)
        explanation = self._explainability.explain(inp, agg_result, rule_result)

        return {
            "compound_risk_score": agg_result.compound_risk_score,
            "risk_level": agg_result.risk_level.value,
            "confidence_score": agg_result.confidence_score,
            "contributing_factors": agg_result.contributing_factors,
            "component_scores": agg_result.component_scores,
            "triggered_rules": len(rule_result.triggered_rules),
            "total_rule_impact": rule_result.total_impact,
            "explanation": explanation.summary,
            "key_drivers": explanation.key_drivers,
            "recommendations": explanation.recommendations,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Queries (delegated to aggregation service)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_by_id(self, analysis_id: str) -> Optional[CompoundRiskModel]:
        """Retrieve a compound risk analysis by ID."""
        return await self._aggregation.get_by_id(analysis_id)

    async def get_latest(
        self,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
    ) -> Optional[CompoundRiskModel]:
        """Get the most recent analysis for a zone/equipment."""
        return await self._aggregation.get_latest(
            zone_id=zone_id, equipment_id=equipment_id,
        )

    async def get_history(
        self,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> List[CompoundRiskModel]:
        """Get analysis history with optional filters."""
        return await self._aggregation.get_history(
            zone_id=zone_id, equipment_id=equipment_id,
            risk_level=risk_level, offset=offset, limit=limit,
        )

    async def count(
        self,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> int:
        """Count analyses matching filters."""
        return await self._aggregation.count(
            zone_id=zone_id, equipment_id=equipment_id,
            risk_level=risk_level,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _validate_input(inp: CompoundRiskInput) -> None:
        """Validate that the input has at least one meaningful signal.

        Raises:
            InsufficientScenarioDataError: If all inputs are default/empty.
        """
        has_anomaly = (
            inp.isolation_forest_score > 0 or inp.autoencoder_score > 0
        )
        has_risk = inp.accident_probability > 0 or inp.risk_score > 0
        has_health = inp.sensor_health_score < 100.0
        has_alerts = inp.active_alert_count > 0
        has_violations = inp.threshold_violation_count > 0

        if not any([has_anomaly, has_risk, has_health, has_alerts, has_violations]):
            raise InsufficientScenarioDataError(
                missing=["anomaly_scores", "risk_prediction", "sensor_health",
                         "alerts", "threshold_violations"],
            )

    @staticmethod
    def _build_recommendation(
        explanation: CompoundRiskExplanation,
        rule_result: RuleEngineResult,
    ) -> Optional[str]:
        """Build a recommendation string from explanation and rules."""
        parts = []

        # Explanation recommendations (deduplicated)
        for rec in explanation.recommendations:
            if rec not in parts:
                parts.append(rec)

        if not parts:
            return None

        return "; ".join(parts)

    def _publish_event(
        self,
        model: CompoundRiskModel,
        result: CompoundRiskResult,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Publish compound.risk.detected if publisher is available."""
        if self._publisher is None:
            logger.debug("No publisher configured — skipping event publish")
            return

        try:
            self._publisher.publish_compound_risk_detected(
                model=model,
                result=result,
                correlation_id=correlation_id,
            )
        except Exception:
            # Publishing failures must not crash the analysis pipeline
            logger.exception(
                "Failed to publish compound.risk.detected for id=%s",
                model.id,
            )
