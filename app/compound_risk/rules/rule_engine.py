"""Compound Risk Rule Engine.

A configurable, declarative rule engine that evaluates safety conditions
against a fact dictionary and returns triggered rules with explanations
and risk impact.

Supports logical operators:
  - AND: All conditions must be true
  - OR:  At least one condition must be true
  - NOT: Inverts a condition

Rules are pure data — defined as ``RuleDefinition`` dataclasses — making
them fully configurable at init-time, via config files, or environment
variables. No thresholds are hardcoded.

Architecture:
  - ``Condition``: A single predicate (field, operator, threshold)
  - ``RuleDefinition``: A named collection of conditions with a logical operator
  - ``RuleResult``: The outcome of evaluating one rule
  - ``RuleEngineResult``: All triggered rules + aggregate impact
  - ``CompoundRiskRuleEngine``: Stateless evaluator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Comparison operators
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ComparisonOp(str, Enum):
    """Supported comparison operators for conditions."""

    GT = ">"           # Greater than
    GTE = ">="         # Greater than or equal
    LT = "<"           # Less than
    LTE = "<="         # Less than or equal
    EQ = "=="          # Equal
    NEQ = "!="         # Not equal
    IN = "in"          # Value in a set
    NOT_IN = "not_in"  # Value not in a set


class LogicalOp(str, Enum):
    """Logical operator for combining conditions within a rule."""

    AND = "AND"
    OR = "OR"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Condition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class Condition:
    """A single predicate evaluated against a fact dictionary.

    Examples:
        Condition("temperature_celsius", ComparisonOp.GT, 60)
        Condition("gas_level_ppm", ComparisonOp.GTE, 100)
        Condition("maintenance_active", ComparisonOp.EQ, True)
        Condition("permit_type", ComparisonOp.IN, ["HOT_WORK", "CONFINED_SPACE"])

    Set ``negate=True`` to apply NOT logic.
    """

    field: str
    operator: ComparisonOp
    threshold: Any
    negate: bool = False  # NOT support
    description: Optional[str] = None  # Human-readable label

    def evaluate(self, facts: Dict[str, Any]) -> bool:
        """Evaluate this condition against the given facts.

        Returns False (not True) if the field is missing from facts,
        since we can't evaluate without data.
        """
        if self.field not in facts:
            result = False
        else:
            value = facts[self.field]
            result = self._compare(value)

        return (not result) if self.negate else result

    def _compare(self, value: Any) -> bool:
        """Apply the comparison operator."""
        try:
            if self.operator == ComparisonOp.GT:
                return value > self.threshold
            elif self.operator == ComparisonOp.GTE:
                return value >= self.threshold
            elif self.operator == ComparisonOp.LT:
                return value < self.threshold
            elif self.operator == ComparisonOp.LTE:
                return value <= self.threshold
            elif self.operator == ComparisonOp.EQ:
                return value == self.threshold
            elif self.operator == ComparisonOp.NEQ:
                return value != self.threshold
            elif self.operator == ComparisonOp.IN:
                return value in self.threshold
            elif self.operator == ComparisonOp.NOT_IN:
                return value not in self.threshold
            else:
                logger.warning("Unknown operator: %s", self.operator)
                return False
        except TypeError:
            logger.warning(
                "Type error comparing %s (%s) with %s",
                self.field, type(value).__name__, self.threshold,
            )
            return False

    def explain(self, facts: Dict[str, Any]) -> str:
        """Generate a human-readable explanation of this condition."""
        desc = self.description or self.field
        value = facts.get(self.field, "N/A")
        prefix = "NOT " if self.negate else ""
        return f"{prefix}{desc} {self.operator.value} {self.threshold} (actual: {value})"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule Definition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class RuleDefinition:
    """A named rule composed of conditions joined by a logical operator.

    Attributes:
        name: Unique rule identifier.
        conditions: List of conditions to evaluate.
        logical_op: How to combine conditions (AND/OR).
        risk_impact: How much this rule adds to compound risk (0.0–1.0).
        severity: Severity label for the triggered rule.
        description: Human-readable rule description.
        recommendation: Suggested action if the rule triggers.
    """

    name: str
    conditions: List[Condition]
    logical_op: LogicalOp = LogicalOp.AND
    risk_impact: float = 0.1
    severity: str = "MEDIUM"
    description: str = ""
    recommendation: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule Result
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class RuleResult:
    """Outcome of evaluating a single rule."""

    rule_name: str
    triggered: bool
    risk_impact: float
    severity: str
    explanation: str
    recommendation: str
    condition_results: List[Dict[str, Any]]


@dataclass
class RuleEngineResult:
    """Aggregate outcome of evaluating all rules."""

    triggered_rules: List[RuleResult]
    total_impact: float
    explanation: str
    all_results: List[RuleResult]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CompoundRiskRuleEngine:
    """Stateless, configurable rule engine for compound risk evaluation.

    All rules are injected at init-time — no thresholds are hardcoded.
    The engine evaluates each rule against a fact dictionary and returns
    triggered rules with explanations and cumulative risk impact.
    """

    def __init__(self, rules: Optional[List[RuleDefinition]] = None) -> None:
        self._rules: List[RuleDefinition] = rules or []

    @property
    def rules(self) -> List[RuleDefinition]:
        return list(self._rules)

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def add_rule(self, rule: RuleDefinition) -> None:
        """Add a rule to the engine."""
        self._rules.append(rule)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name. Returns True if found and removed."""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != rule_name]
        return len(self._rules) < before

    def evaluate(self, facts: Dict[str, Any]) -> RuleEngineResult:
        """Evaluate all rules against the provided facts.

        Args:
            facts: Dictionary of current sensor/operational values.
                   Keys should match Condition.field names.

        Returns:
            RuleEngineResult with triggered rules, impact, and explanation.
        """
        all_results: List[RuleResult] = []
        triggered: List[RuleResult] = []

        for rule in self._rules:
            result = self._evaluate_rule(rule, facts)
            all_results.append(result)
            if result.triggered:
                triggered.append(result)

        total_impact = sum(r.risk_impact for r in triggered)
        total_impact = min(total_impact, 1.0)  # Cap at 1.0

        explanation = self._build_explanation(triggered)

        return RuleEngineResult(
            triggered_rules=triggered,
            total_impact=round(total_impact, 4),
            explanation=explanation,
            all_results=all_results,
        )

    def _evaluate_rule(
        self, rule: RuleDefinition, facts: Dict[str, Any],
    ) -> RuleResult:
        """Evaluate a single rule against facts."""
        condition_results = []
        condition_outcomes = []

        for cond in rule.conditions:
            result = cond.evaluate(facts)
            condition_outcomes.append(result)
            condition_results.append({
                "field": cond.field,
                "operator": cond.operator.value,
                "threshold": cond.threshold,
                "actual_value": facts.get(cond.field, None),
                "passed": result,
                "negated": cond.negate,
                "explanation": cond.explain(facts),
            })

        # Apply logical operator
        if rule.logical_op == LogicalOp.AND:
            triggered = all(condition_outcomes) if condition_outcomes else False
        elif rule.logical_op == LogicalOp.OR:
            triggered = any(condition_outcomes) if condition_outcomes else False
        else:
            triggered = False

        # Build explanation
        if triggered:
            cond_explanations = [
                cr["explanation"] for cr in condition_results if cr["passed"]
            ]
            explanation = (
                f"Rule '{rule.name}' triggered: "
                f"{rule.description or 'No description'}. "
                f"Conditions met: {'; '.join(cond_explanations)}"
            )
        else:
            explanation = f"Rule '{rule.name}' not triggered."

        return RuleResult(
            rule_name=rule.name,
            triggered=triggered,
            risk_impact=rule.risk_impact if triggered else 0.0,
            severity=rule.severity if triggered else "NONE",
            explanation=explanation,
            recommendation=rule.recommendation if triggered else "",
            condition_results=condition_results,
        )

    @staticmethod
    def _build_explanation(triggered: List[RuleResult]) -> str:
        """Build aggregate explanation from triggered rules."""
        if not triggered:
            return "No compound risk rules triggered."

        count = len(triggered)
        names = [r.rule_name for r in triggered]
        severities = [r.severity for r in triggered]
        max_severity = "CRITICAL" if "CRITICAL" in severities else (
            "HIGH" if "HIGH" in severities else (
                "MEDIUM" if "MEDIUM" in severities else "LOW"
            )
        )

        return (
            f"{count} rule(s) triggered: {', '.join(names)}. "
            f"Maximum severity: {max_severity}."
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Default rule factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_default_rules(
    temperature_threshold: float = 60.0,
    gas_threshold: float = 100.0,
    pressure_threshold: float = 5.0,
    equipment_health_threshold: float = 0.3,
    vibration_threshold: float = 10.0,
    worker_density_threshold: int = 15,
) -> List[RuleDefinition]:
    """Create the default compound risk rule set.

    All thresholds are parameters — nothing is hardcoded.
    Callers can override any threshold or create entirely custom rules.

    Returns:
        List of RuleDefinition objects ready for CompoundRiskRuleEngine.
    """
    return [
        # Rule 1: High temp + high gas → explosion / fire risk
        RuleDefinition(
            name="high_temp_and_gas",
            conditions=[
                Condition(
                    "temperature_celsius", ComparisonOp.GT, temperature_threshold,
                    description="Temperature",
                ),
                Condition(
                    "gas_level_ppm", ComparisonOp.GTE, gas_threshold,
                    description="Gas Level",
                ),
            ],
            logical_op=LogicalOp.AND,
            risk_impact=0.35,
            severity="CRITICAL",
            description="High temperature combined with elevated gas levels",
            recommendation="Stop hot work activities, increase ventilation, consider evacuation",
        ),

        # Rule 2: Pressure anomaly + poor equipment health
        RuleDefinition(
            name="pressure_and_equipment_health",
            conditions=[
                Condition(
                    "pressure_bar", ComparisonOp.GT, pressure_threshold,
                    description="Pressure",
                ),
                Condition(
                    "equipment_health", ComparisonOp.LT, equipment_health_threshold,
                    description="Equipment Health",
                ),
            ],
            logical_op=LogicalOp.AND,
            risk_impact=0.30,
            severity="HIGH",
            description="High pressure with degraded equipment health",
            recommendation="Reduce pressure, schedule emergency maintenance",
        ),

        # Rule 3: Gas + active maintenance → ignition risk
        RuleDefinition(
            name="gas_and_maintenance",
            conditions=[
                Condition(
                    "gas_level_ppm", ComparisonOp.GTE, gas_threshold,
                    description="Gas Level",
                ),
                Condition(
                    "maintenance_active", ComparisonOp.EQ, True,
                    description="Maintenance Active",
                ),
            ],
            logical_op=LogicalOp.AND,
            risk_impact=0.30,
            severity="CRITICAL",
            description="Elevated gas during active maintenance",
            recommendation="Halt maintenance activities, ventilate area",
        ),

        # Rule 4: Gas + hot work permit → highest risk
        RuleDefinition(
            name="gas_and_hot_work",
            conditions=[
                Condition(
                    "gas_level_ppm", ComparisonOp.GTE, gas_threshold,
                    description="Gas Level",
                ),
                Condition(
                    "permit_type", ComparisonOp.EQ, "HOT_WORK",
                    description="Permit Type",
                ),
                Condition(
                    "permit_active", ComparisonOp.EQ, True,
                    description="Permit Active",
                ),
            ],
            logical_op=LogicalOp.AND,
            risk_impact=0.40,
            severity="CRITICAL",
            description="Gas level exceeds threshold while hot work permit is active",
            recommendation="Immediately revoke hot work permit, evacuate zone",
        ),

        # Rule 5: High temp OR high gas (either alone is concerning)
        RuleDefinition(
            name="temp_or_gas_elevated",
            conditions=[
                Condition(
                    "temperature_celsius", ComparisonOp.GT, temperature_threshold,
                    description="Temperature",
                ),
                Condition(
                    "gas_level_ppm", ComparisonOp.GTE, gas_threshold,
                    description="Gas Level",
                ),
            ],
            logical_op=LogicalOp.OR,
            risk_impact=0.15,
            severity="MEDIUM",
            description="Temperature or gas levels elevated beyond safe thresholds",
            recommendation="Monitor closely, prepare contingency measures",
        ),

        # Rule 6: High vibration + poor equipment health
        RuleDefinition(
            name="vibration_and_equipment_health",
            conditions=[
                Condition(
                    "vibration_level", ComparisonOp.GT, vibration_threshold,
                    description="Vibration Level",
                ),
                Condition(
                    "equipment_health", ComparisonOp.LT, equipment_health_threshold,
                    description="Equipment Health",
                ),
            ],
            logical_op=LogicalOp.AND,
            risk_impact=0.25,
            severity="HIGH",
            description="High vibration on equipment with poor health score",
            recommendation="Stop equipment, perform immediate inspection",
        ),

        # Rule 7: Worker density too high in a risky zone
        RuleDefinition(
            name="high_worker_density",
            conditions=[
                Condition(
                    "worker_count", ComparisonOp.GT, worker_density_threshold,
                    description="Worker Count",
                ),
            ],
            logical_op=LogicalOp.AND,
            risk_impact=0.10,
            severity="MEDIUM",
            description="Worker density exceeds safe threshold for the zone",
            recommendation="Reduce personnel in zone, stagger shifts",
        ),

        # Rule 8: Night shift + maintenance (reduced supervision)
        RuleDefinition(
            name="night_shift_maintenance",
            conditions=[
                Condition(
                    "shift_type", ComparisonOp.EQ, "NIGHT",
                    description="Shift Type",
                ),
                Condition(
                    "maintenance_active", ComparisonOp.EQ, True,
                    description="Maintenance Active",
                ),
            ],
            logical_op=LogicalOp.AND,
            risk_impact=0.15,
            severity="MEDIUM",
            description="Maintenance during night shift with reduced supervision",
            recommendation="Ensure supervisor present, increase monitoring frequency",
        ),

        # Rule 9: Sensor NOT healthy (negated condition)
        RuleDefinition(
            name="sensor_health_degraded",
            conditions=[
                Condition(
                    "sensor_health_score", ComparisonOp.GTE, 50.0,
                    negate=True,
                    description="Sensor Health >= 50",
                ),
            ],
            logical_op=LogicalOp.AND,
            risk_impact=0.10,
            severity="LOW",
            description="Sensor health below acceptable threshold (readings may be unreliable)",
            recommendation="Schedule sensor calibration or replacement",
        ),
    ]
