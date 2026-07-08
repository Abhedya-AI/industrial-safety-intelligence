"""Explainability Service for Compound Risk Intelligence.

Generates human-readable, structured explanations for every compound risk
result. Takes inputs from the Aggregation Engine and Rule Engine and
produces explanations that describe *why* a particular risk level was
assigned.

Explanation components:
  - Anomaly score contribution (IF + AE)
  - Accident probability contribution
  - Sensor health contribution
  - Active alerts contribution
  - Triggered rules and their impact
  - Overall narrative summary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from app.compound_risk.domain.value_objects import RiskLevel
from app.compound_risk.rules.rule_engine import RuleEngineResult, RuleResult
from app.compound_risk.services.compound_risk_service import (
    CompoundRiskInput,
    CompoundRiskResult,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class ExplainabilityThresholds:
    """Configurable thresholds for triggering explanation clauses.

    All values are normalised 0–1 unless otherwise noted.
    """

    anomaly_notable: float = 0.3
    anomaly_high: float = 0.7
    accident_prob_notable: float = 0.3
    accident_prob_high: float = 0.7
    sensor_health_degraded: float = 70.0   # 0–100 scale
    sensor_health_poor: float = 40.0       # 0–100 scale
    alert_count_notable: int = 1
    alert_count_high: int = 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Explanation data classes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ContributionLevel(str, Enum):
    """How much a factor contributed to the risk score."""

    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class FactorExplanation:
    """Structured explanation for a single contributing factor."""

    factor_name: str
    raw_value: float
    contribution_level: ContributionLevel
    explanation: str
    impact_score: float = 0.0    # 0–100 (weighted contribution)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "raw_value": self.raw_value,
            "contribution_level": self.contribution_level.value,
            "explanation": self.explanation,
            "impact_score": round(self.impact_score, 2),
        }


@dataclass
class TriggeredRuleExplanation:
    """Structured explanation for a triggered rule."""

    rule_name: str
    severity: str
    risk_impact: float
    explanation: str
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "severity": self.severity,
            "risk_impact": round(self.risk_impact, 4),
            "explanation": self.explanation,
            "recommendation": self.recommendation,
        }


@dataclass
class CompoundRiskExplanation:
    """Full structured explanation for a compound risk assessment."""

    risk_level: RiskLevel
    compound_risk_score: float
    confidence_score: float
    summary: str
    factor_explanations: List[FactorExplanation] = field(default_factory=list)
    triggered_rules: List[TriggeredRuleExplanation] = field(default_factory=list)
    key_drivers: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level.value,
            "compound_risk_score": round(self.compound_risk_score, 2),
            "confidence_score": round(self.confidence_score, 4),
            "summary": self.summary,
            "factor_explanations": [f.to_dict() for f in self.factor_explanations],
            "triggered_rules": [r.to_dict() for r in self.triggered_rules],
            "key_drivers": self.key_drivers,
            "recommendations": self.recommendations,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Explainability Service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ExplainabilityService:
    """Generates structured, human-readable explanations for compound risk.

    Stateless service — all configuration is injected at init-time.
    """

    def __init__(
        self,
        thresholds: Optional[ExplainabilityThresholds] = None,
    ) -> None:
        self._thresholds = thresholds or ExplainabilityThresholds()

    @property
    def thresholds(self) -> ExplainabilityThresholds:
        return self._thresholds

    def explain(
        self,
        inp: CompoundRiskInput,
        result: CompoundRiskResult,
        rule_result: Optional[RuleEngineResult] = None,
    ) -> CompoundRiskExplanation:
        """Generate a full structured explanation.

        Args:
            inp: The original input signals.
            result: The aggregation engine output.
            rule_result: Optional rule engine evaluation result.

        Returns:
            CompoundRiskExplanation with summary, factor breakdowns,
            triggered rules, key drivers, and recommendations.
        """
        factor_explanations = self._explain_factors(inp, result)
        triggered_rules = self._explain_rules(rule_result)
        key_drivers = self._identify_key_drivers(factor_explanations, triggered_rules)
        recommendations = self._collect_recommendations(triggered_rules, result)
        summary = self._build_summary(result, key_drivers, triggered_rules)

        return CompoundRiskExplanation(
            risk_level=result.risk_level,
            compound_risk_score=result.compound_risk_score,
            confidence_score=result.confidence_score,
            summary=summary,
            factor_explanations=factor_explanations,
            triggered_rules=triggered_rules,
            key_drivers=key_drivers,
            recommendations=recommendations,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Factor explanations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _explain_factors(
        self,
        inp: CompoundRiskInput,
        result: CompoundRiskResult,
    ) -> List[FactorExplanation]:
        """Generate explanations for each contributing factor."""
        factors = []

        # 1. Anomaly score (combine IF + AE)
        factors.append(self._explain_anomaly(inp, result))

        # 2. Accident probability
        factors.append(self._explain_accident_probability(inp, result))

        # 3. Sensor health
        factors.append(self._explain_sensor_health(inp, result))

        # 4. Active alerts
        factors.append(self._explain_alerts(inp, result))

        # 5. Threshold violations
        factors.append(self._explain_threshold_violations(inp, result))

        return factors

    def _explain_anomaly(
        self, inp: CompoundRiskInput, result: CompoundRiskResult,
    ) -> FactorExplanation:
        """Explain anomaly score contribution (IF + AE combined)."""
        max_score = max(inp.isolation_forest_score, inp.autoencoder_score)

        if max_score >= self._thresholds.anomaly_high:
            level = ContributionLevel.CRITICAL
            text = (
                f"Anomaly detection identified highly abnormal behavior "
                f"(IF: {inp.isolation_forest_score:.2f}, "
                f"AE: {inp.autoencoder_score:.2f}). "
                f"This is a strong indicator of unusual operating conditions."
            )
        elif max_score >= self._thresholds.anomaly_notable:
            level = ContributionLevel.MODERATE
            text = (
                f"Anomaly detection flagged moderate abnormalities "
                f"(IF: {inp.isolation_forest_score:.2f}, "
                f"AE: {inp.autoencoder_score:.2f}). "
                f"Conditions deviate from normal patterns."
            )
        elif max_score > 0:
            level = ContributionLevel.LOW
            text = (
                f"Minor anomaly signals detected "
                f"(IF: {inp.isolation_forest_score:.2f}, "
                f"AE: {inp.autoencoder_score:.2f})."
            )
        else:
            level = ContributionLevel.NONE
            text = "No anomalies detected by either model."

        impact = result.component_scores.get("isolation_forest", 0) + \
                 result.component_scores.get("autoencoder", 0)

        return FactorExplanation(
            factor_name="Anomaly Detection",
            raw_value=max_score,
            contribution_level=level,
            explanation=text,
            impact_score=impact / 2,
        )

    def _explain_accident_probability(
        self, inp: CompoundRiskInput, result: CompoundRiskResult,
    ) -> FactorExplanation:
        """Explain accident probability contribution."""
        prob = inp.accident_probability

        if prob >= self._thresholds.accident_prob_high:
            level = ContributionLevel.CRITICAL
            text = (
                f"Accident probability is critically elevated at "
                f"{prob:.0%}. The risk prediction model indicates "
                f"a high likelihood of an incident."
            )
        elif prob >= self._thresholds.accident_prob_notable:
            level = ContributionLevel.MODERATE
            text = (
                f"Accident probability is elevated at {prob:.0%}. "
                f"Conditions show increased risk compared to normal operations."
            )
        elif prob > 0:
            level = ContributionLevel.LOW
            text = f"Accident probability is low at {prob:.0%}."
        else:
            level = ContributionLevel.NONE
            text = "No accident risk detected by the prediction model."

        return FactorExplanation(
            factor_name="Accident Probability",
            raw_value=prob,
            contribution_level=level,
            explanation=text,
            impact_score=result.component_scores.get("risk_prediction", 0),
        )

    def _explain_sensor_health(
        self, inp: CompoundRiskInput, result: CompoundRiskResult,
    ) -> FactorExplanation:
        """Explain sensor health contribution."""
        health = inp.sensor_health_score

        if health < self._thresholds.sensor_health_poor:
            level = ContributionLevel.HIGH
            text = (
                f"Sensor health is poor ({health:.0f}/100). "
                f"Readings may be unreliable, increasing uncertainty "
                f"and overall risk."
            )
        elif health < self._thresholds.sensor_health_degraded:
            level = ContributionLevel.MODERATE
            text = (
                f"Sensor health is degraded ({health:.0f}/100). "
                f"Some readings may have reduced accuracy."
            )
        else:
            level = ContributionLevel.NONE
            text = f"Sensor health is good ({health:.0f}/100)."

        return FactorExplanation(
            factor_name="Sensor Health",
            raw_value=health,
            contribution_level=level,
            explanation=text,
            impact_score=result.component_scores.get("sensor_health", 0),
        )

    def _explain_alerts(
        self, inp: CompoundRiskInput, result: CompoundRiskResult,
    ) -> FactorExplanation:
        """Explain active alerts contribution."""
        count = inp.active_alert_count
        severity = inp.alert_severity_max

        if count >= self._thresholds.alert_count_high:
            level = ContributionLevel.HIGH
            text = (
                f"{count} active alert(s) with maximum severity "
                f"{severity:.2f}. Multiple concurrent alerts indicate "
                f"compounding hazardous conditions."
            )
        elif count >= self._thresholds.alert_count_notable:
            level = ContributionLevel.MODERATE
            text = (
                f"{count} active alert(s) detected with maximum severity "
                f"{severity:.2f}."
            )
        else:
            level = ContributionLevel.NONE
            text = "No active alerts."

        return FactorExplanation(
            factor_name="Active Alerts",
            raw_value=float(count),
            contribution_level=level,
            explanation=text,
            impact_score=result.component_scores.get("alert", 0),
        )

    def _explain_threshold_violations(
        self, inp: CompoundRiskInput, result: CompoundRiskResult,
    ) -> FactorExplanation:
        """Explain threshold violation contribution."""
        count = inp.threshold_violation_count

        if count >= 3:
            level = ContributionLevel.HIGH
            text = (
                f"{count} threshold violation(s) detected. "
                f"Multiple simultaneous violations indicate "
                f"widespread unsafe conditions."
            )
        elif count >= 1:
            level = ContributionLevel.MODERATE
            text = f"{count} threshold violation(s) detected."
        else:
            level = ContributionLevel.NONE
            text = "No threshold violations."

        return FactorExplanation(
            factor_name="Threshold Violations",
            raw_value=float(count),
            contribution_level=level,
            explanation=text,
            impact_score=result.component_scores.get("threshold_violation", 0),
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Rule explanations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _explain_rules(
        rule_result: Optional[RuleEngineResult],
    ) -> List[TriggeredRuleExplanation]:
        """Convert triggered rules to explanation objects."""
        if rule_result is None:
            return []

        explanations = []
        for rr in rule_result.triggered_rules:
            explanations.append(TriggeredRuleExplanation(
                rule_name=rr.rule_name,
                severity=rr.severity,
                risk_impact=rr.risk_impact,
                explanation=rr.explanation,
                recommendation=rr.recommendation,
            ))

        return explanations

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Key drivers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _identify_key_drivers(
        factors: List[FactorExplanation],
        rules: List[TriggeredRuleExplanation],
    ) -> List[str]:
        """Identify the top contributing factors as human-readable strings."""
        drivers = []

        # Factors with MODERATE or higher contribution
        significant = [
            f for f in factors
            if f.contribution_level not in (ContributionLevel.NONE, ContributionLevel.LOW)
        ]
        # Sort by impact score descending
        significant.sort(key=lambda f: f.impact_score, reverse=True)

        for f in significant:
            drivers.append(f"{f.factor_name} ({f.contribution_level.value})")

        # Critical/High severity rules
        for r in rules:
            if r.severity in ("CRITICAL", "HIGH"):
                drivers.append(f"Rule: {r.rule_name} ({r.severity})")

        return drivers

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Recommendations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _collect_recommendations(
        rules: List[TriggeredRuleExplanation],
        result: CompoundRiskResult,
    ) -> List[str]:
        """Collect unique recommendations from triggered rules and risk level."""
        recs = []
        seen = set()

        # Rule-based recommendations
        for r in rules:
            if r.recommendation and r.recommendation not in seen:
                recs.append(r.recommendation)
                seen.add(r.recommendation)

        # Risk-level-based generic recommendations
        if result.risk_level == RiskLevel.CRITICAL:
            generic = "Initiate emergency response protocol. Consider zone evacuation."
            if generic not in seen:
                recs.append(generic)
        elif result.risk_level == RiskLevel.HIGH:
            generic = "Increase monitoring frequency. Alert safety supervisor."
            if generic not in seen:
                recs.append(generic)

        return recs

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Summary narrative
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _build_summary(
        result: CompoundRiskResult,
        key_drivers: List[str],
        triggered_rules: List[TriggeredRuleExplanation],
    ) -> str:
        """Build a natural-language summary of the compound risk assessment."""
        level = result.risk_level.value
        score = result.compound_risk_score

        if not key_drivers and not triggered_rules:
            return (
                f"Risk level is {level} with a compound score of {score:.1f}/100. "
                f"All monitored parameters are within normal operating ranges."
            )

        # Build "because" clause
        because_parts = []
        for driver in key_drivers[:3]:  # Top 3 drivers
            because_parts.append(driver.lower())

        if triggered_rules:
            rule_count = len(triggered_rules)
            because_parts.append(
                f"{rule_count} compound risk rule{'s' if rule_count > 1 else ''} triggered"
            )

        because = ", ".join(because_parts[:-1])
        if len(because_parts) > 1:
            because += f", and {because_parts[-1]}"
        elif because_parts:
            because = because_parts[0]

        return (
            f"Risk level is {level} (score: {score:.1f}/100) "
            f"because {because}."
        )
