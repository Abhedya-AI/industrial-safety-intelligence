"""Unit tests for the Compound Risk Explainability Service.

Tests cover:
  - Factor explanations (anomaly, accident probability, sensor health, alerts, violations)
  - Contribution level classification
  - Triggered rule explanations
  - Key driver identification
  - Recommendations collection
  - Summary narrative generation
  - Serialization (to_dict)
  - Custom thresholds
  - Edge cases (no inputs, no rules, all inputs)
"""

from __future__ import annotations

import pytest

from app.compound_risk.domain.value_objects import RiskLevel
from app.compound_risk.rules.rule_engine import (
    ComparisonOp,
    CompoundRiskRuleEngine,
    Condition,
    LogicalOp,
    RuleDefinition,
    RuleEngineResult,
    RuleResult,
)
from app.compound_risk.services.compound_risk_service import (
    CompoundRiskInput,
    CompoundRiskResult,
)
from app.compound_risk.services.explainability_service import (
    CompoundRiskExplanation,
    ContributionLevel,
    ExplainabilityService,
    ExplainabilityThresholds,
    FactorExplanation,
    TriggeredRuleExplanation,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_result(
    score: float = 50.0,
    level: RiskLevel = RiskLevel.MEDIUM,
    confidence: float = 0.85,
    **component_overrides,
) -> CompoundRiskResult:
    """Build a CompoundRiskResult with defaults."""
    components = {
        "risk_prediction": 50.0,
        "isolation_forest": 30.0,
        "autoencoder": 20.0,
        "sensor_health": 10.0,
        "alert": 5.0,
        "threshold_violation": 0.0,
    }
    components.update(component_overrides)
    return CompoundRiskResult(
        compound_risk_score=score,
        risk_level=level,
        confidence_score=confidence,
        contributing_factors=[],
        component_scores=components,
    )


def _make_input(**overrides) -> CompoundRiskInput:
    defaults = {
        "isolation_forest_score": 0.3,
        "autoencoder_score": 0.2,
        "accident_probability": 0.5,
        "sensor_health_score": 80.0,
        "active_alert_count": 0,
        "alert_severity_max": 0.0,
        "threshold_violation_count": 0,
    }
    defaults.update(overrides)
    return CompoundRiskInput(**defaults)


def _make_rule_result(triggered_names=None) -> RuleEngineResult:
    """Build a RuleEngineResult with given triggered rule names."""
    triggered = []
    if triggered_names:
        for name, sev, impact, rec in triggered_names:
            triggered.append(RuleResult(
                rule_name=name,
                triggered=True,
                risk_impact=impact,
                severity=sev,
                explanation=f"Rule '{name}' triggered",
                recommendation=rec,
                condition_results=[],
            ))
    return RuleEngineResult(
        triggered_rules=triggered,
        total_impact=sum(r.risk_impact for r in triggered),
        explanation="Test",
        all_results=triggered,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def service() -> ExplainabilityService:
    return ExplainabilityService()


@pytest.fixture
def custom_service() -> ExplainabilityService:
    return ExplainabilityService(
        thresholds=ExplainabilityThresholds(
            anomaly_notable=0.2,
            anomaly_high=0.5,
            accident_prob_notable=0.2,
            accident_prob_high=0.6,
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Anomaly explanation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAnomalyExplanation:
    def test_no_anomaly(self, service):
        inp = _make_input(isolation_forest_score=0.0, autoencoder_score=0.0)
        result = _make_result()
        expl = service.explain(inp, result)
        anomaly = expl.factor_explanations[0]
        assert anomaly.factor_name == "Anomaly Detection"
        assert anomaly.contribution_level == ContributionLevel.NONE
        assert "No anomalies" in anomaly.explanation

    def test_low_anomaly(self, service):
        inp = _make_input(isolation_forest_score=0.15, autoencoder_score=0.1)
        result = _make_result()
        expl = service.explain(inp, result)
        anomaly = expl.factor_explanations[0]
        assert anomaly.contribution_level == ContributionLevel.LOW
        assert "Minor" in anomaly.explanation

    def test_moderate_anomaly(self, service):
        inp = _make_input(isolation_forest_score=0.5, autoencoder_score=0.4)
        result = _make_result()
        expl = service.explain(inp, result)
        anomaly = expl.factor_explanations[0]
        assert anomaly.contribution_level == ContributionLevel.MODERATE
        assert "moderate" in anomaly.explanation.lower()

    def test_critical_anomaly(self, service):
        inp = _make_input(isolation_forest_score=0.85, autoencoder_score=0.9)
        result = _make_result()
        expl = service.explain(inp, result)
        anomaly = expl.factor_explanations[0]
        assert anomaly.contribution_level == ContributionLevel.CRITICAL
        assert "highly abnormal" in anomaly.explanation.lower()

    def test_includes_both_scores(self, service):
        inp = _make_input(isolation_forest_score=0.5, autoencoder_score=0.3)
        result = _make_result()
        expl = service.explain(inp, result)
        text = expl.factor_explanations[0].explanation
        assert "IF: 0.50" in text
        assert "AE: 0.30" in text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Accident probability explanation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAccidentProbExplanation:
    def test_no_risk(self, service):
        inp = _make_input(accident_probability=0.0)
        result = _make_result()
        expl = service.explain(inp, result)
        acc = expl.factor_explanations[1]
        assert acc.factor_name == "Accident Probability"
        assert acc.contribution_level == ContributionLevel.NONE

    def test_low_risk(self, service):
        inp = _make_input(accident_probability=0.15)
        result = _make_result()
        expl = service.explain(inp, result)
        acc = expl.factor_explanations[1]
        assert acc.contribution_level == ContributionLevel.LOW

    def test_moderate_risk(self, service):
        inp = _make_input(accident_probability=0.5)
        result = _make_result()
        expl = service.explain(inp, result)
        acc = expl.factor_explanations[1]
        assert acc.contribution_level == ContributionLevel.MODERATE
        assert "elevated" in acc.explanation.lower()

    def test_critical_risk(self, service):
        inp = _make_input(accident_probability=0.85)
        result = _make_result()
        expl = service.explain(inp, result)
        acc = expl.factor_explanations[1]
        assert acc.contribution_level == ContributionLevel.CRITICAL
        assert "85%" in acc.explanation

    def test_has_impact_score(self, service):
        inp = _make_input(accident_probability=0.5)
        result = _make_result(risk_prediction=50.0)
        expl = service.explain(inp, result)
        assert expl.factor_explanations[1].impact_score == 50.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Sensor health explanation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSensorHealthExplanation:
    def test_good_health(self, service):
        inp = _make_input(sensor_health_score=90.0)
        result = _make_result()
        expl = service.explain(inp, result)
        sh = expl.factor_explanations[2]
        assert sh.factor_name == "Sensor Health"
        assert sh.contribution_level == ContributionLevel.NONE
        assert "good" in sh.explanation.lower()

    def test_degraded_health(self, service):
        inp = _make_input(sensor_health_score=55.0)
        result = _make_result()
        expl = service.explain(inp, result)
        sh = expl.factor_explanations[2]
        assert sh.contribution_level == ContributionLevel.MODERATE
        assert "degraded" in sh.explanation.lower()

    def test_poor_health(self, service):
        inp = _make_input(sensor_health_score=25.0)
        result = _make_result()
        expl = service.explain(inp, result)
        sh = expl.factor_explanations[2]
        assert sh.contribution_level == ContributionLevel.HIGH
        assert "poor" in sh.explanation.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Alerts explanation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAlertsExplanation:
    def test_no_alerts(self, service):
        inp = _make_input(active_alert_count=0)
        result = _make_result()
        expl = service.explain(inp, result)
        al = expl.factor_explanations[3]
        assert al.factor_name == "Active Alerts"
        assert al.contribution_level == ContributionLevel.NONE

    def test_some_alerts(self, service):
        inp = _make_input(active_alert_count=2, alert_severity_max=0.6)
        result = _make_result()
        expl = service.explain(inp, result)
        al = expl.factor_explanations[3]
        assert al.contribution_level == ContributionLevel.MODERATE
        assert "2 active alert" in al.explanation

    def test_many_alerts(self, service):
        inp = _make_input(active_alert_count=5, alert_severity_max=0.9)
        result = _make_result()
        expl = service.explain(inp, result)
        al = expl.factor_explanations[3]
        assert al.contribution_level == ContributionLevel.HIGH
        assert "compounding" in al.explanation.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Threshold violations explanation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestThresholdViolationsExplanation:
    def test_no_violations(self, service):
        inp = _make_input(threshold_violation_count=0)
        result = _make_result()
        expl = service.explain(inp, result)
        tv = expl.factor_explanations[4]
        assert tv.factor_name == "Threshold Violations"
        assert tv.contribution_level == ContributionLevel.NONE

    def test_some_violations(self, service):
        inp = _make_input(threshold_violation_count=2)
        result = _make_result()
        expl = service.explain(inp, result)
        tv = expl.factor_explanations[4]
        assert tv.contribution_level == ContributionLevel.MODERATE

    def test_many_violations(self, service):
        inp = _make_input(threshold_violation_count=5)
        result = _make_result()
        expl = service.explain(inp, result)
        tv = expl.factor_explanations[4]
        assert tv.contribution_level == ContributionLevel.HIGH
        assert "widespread" in tv.explanation.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Triggered rules explanation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTriggeredRules:
    def test_no_rule_result(self, service):
        inp = _make_input()
        result = _make_result()
        expl = service.explain(inp, result, rule_result=None)
        assert expl.triggered_rules == []

    def test_no_triggered_rules(self, service):
        inp = _make_input()
        result = _make_result()
        rule_result = _make_rule_result(triggered_names=[])
        expl = service.explain(inp, result, rule_result)
        assert expl.triggered_rules == []

    def test_triggered_rules_populated(self, service):
        inp = _make_input()
        result = _make_result()
        rule_result = _make_rule_result([
            ("gas_high", "CRITICAL", 0.35, "Stop hot work"),
            ("temp_high", "HIGH", 0.20, "Increase ventilation"),
        ])
        expl = service.explain(inp, result, rule_result)
        assert len(expl.triggered_rules) == 2
        assert expl.triggered_rules[0].rule_name == "gas_high"
        assert expl.triggered_rules[0].severity == "CRITICAL"
        assert expl.triggered_rules[0].recommendation == "Stop hot work"

    def test_triggered_rule_to_dict(self, service):
        inp = _make_input()
        result = _make_result()
        rule_result = _make_rule_result([
            ("test_rule", "HIGH", 0.25, "Take action"),
        ])
        expl = service.explain(inp, result, rule_result)
        d = expl.triggered_rules[0].to_dict()
        assert d["rule_name"] == "test_rule"
        assert d["severity"] == "HIGH"
        assert d["risk_impact"] == 0.25


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Key drivers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestKeyDrivers:
    def test_no_drivers_when_safe(self, service):
        inp = _make_input(
            isolation_forest_score=0.0, autoencoder_score=0.0,
            accident_probability=0.0, sensor_health_score=95.0,
        )
        result = _make_result()
        expl = service.explain(inp, result)
        assert len(expl.key_drivers) == 0

    def test_includes_significant_factors(self, service):
        inp = _make_input(
            accident_probability=0.85,
            isolation_forest_score=0.8,
        )
        result = _make_result()
        expl = service.explain(inp, result)
        driver_text = " ".join(expl.key_drivers)
        assert "Accident Probability" in driver_text
        assert "Anomaly Detection" in driver_text

    def test_includes_critical_rules(self, service):
        inp = _make_input()
        result = _make_result()
        rule_result = _make_rule_result([
            ("gas_critical", "CRITICAL", 0.4, "Evacuate"),
        ])
        expl = service.explain(inp, result, rule_result)
        driver_text = " ".join(expl.key_drivers)
        assert "gas_critical" in driver_text
        assert "CRITICAL" in driver_text

    def test_does_not_include_low_rules(self, service):
        inp = _make_input()
        result = _make_result()
        rule_result = _make_rule_result([
            ("minor_rule", "LOW", 0.05, "Monitor"),
        ])
        expl = service.explain(inp, result, rule_result)
        rule_drivers = [d for d in expl.key_drivers if "Rule:" in d]
        assert len(rule_drivers) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Recommendations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRecommendations:
    def test_no_recommendations_safe(self, service):
        inp = _make_input()
        result = _make_result(level=RiskLevel.LOW)
        expl = service.explain(inp, result)
        assert len(expl.recommendations) == 0

    def test_rule_recommendations(self, service):
        inp = _make_input()
        result = _make_result(level=RiskLevel.LOW)
        rule_result = _make_rule_result([
            ("r1", "HIGH", 0.2, "Increase ventilation"),
        ])
        expl = service.explain(inp, result, rule_result)
        assert "Increase ventilation" in expl.recommendations

    def test_critical_adds_emergency(self, service):
        inp = _make_input()
        result = _make_result(level=RiskLevel.CRITICAL)
        expl = service.explain(inp, result)
        recs_text = " ".join(expl.recommendations)
        assert "emergency" in recs_text.lower()

    def test_high_adds_monitoring(self, service):
        inp = _make_input()
        result = _make_result(level=RiskLevel.HIGH)
        expl = service.explain(inp, result)
        recs_text = " ".join(expl.recommendations)
        assert "monitoring" in recs_text.lower()

    def test_no_duplicate_recommendations(self, service):
        inp = _make_input()
        result = _make_result(level=RiskLevel.LOW)
        rule_result = _make_rule_result([
            ("r1", "HIGH", 0.2, "Same action"),
            ("r2", "HIGH", 0.3, "Same action"),
        ])
        expl = service.explain(inp, result, rule_result)
        assert expl.recommendations.count("Same action") == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Summary narrative
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSummary:
    def test_safe_summary(self, service):
        inp = _make_input(
            isolation_forest_score=0.0, autoencoder_score=0.0,
            accident_probability=0.0, sensor_health_score=95.0,
        )
        result = _make_result(score=10.0, level=RiskLevel.LOW)
        expl = service.explain(inp, result)
        assert "LOW" in expl.summary
        assert "normal" in expl.summary.lower()

    def test_high_risk_summary_has_because(self, service):
        inp = _make_input(accident_probability=0.85, isolation_forest_score=0.8)
        result = _make_result(score=72.0, level=RiskLevel.HIGH)
        expl = service.explain(inp, result)
        assert "HIGH" in expl.summary
        assert "because" in expl.summary

    def test_summary_includes_score(self, service):
        inp = _make_input(accident_probability=0.5)
        result = _make_result(score=45.0, level=RiskLevel.MEDIUM)
        expl = service.explain(inp, result)
        assert "45.0" in expl.summary

    def test_summary_with_rules(self, service):
        inp = _make_input(accident_probability=0.5)
        result = _make_result(score=65.0, level=RiskLevel.HIGH)
        rule_result = _make_rule_result([
            ("gas_high", "CRITICAL", 0.35, "Stop work"),
        ])
        expl = service.explain(inp, result, rule_result)
        assert "rule" in expl.summary.lower()

    def test_example_from_requirements(self, service):
        """Verify the example from the user's requirements is achievable."""
        inp = _make_input(
            accident_probability=0.87,
            isolation_forest_score=0.75,
            autoencoder_score=0.6,
        )
        result = _make_result(score=72.0, level=RiskLevel.HIGH)
        expl = service.explain(inp, result)
        # Summary should mention HIGH, and key drivers
        assert "HIGH" in expl.summary
        assert len(expl.key_drivers) >= 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Serialization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSerialization:
    def test_factor_to_dict(self):
        f = FactorExplanation(
            factor_name="Test",
            raw_value=0.5,
            contribution_level=ContributionLevel.MODERATE,
            explanation="Test explanation",
            impact_score=25.0,
        )
        d = f.to_dict()
        assert d["factor_name"] == "Test"
        assert d["contribution_level"] == "MODERATE"
        assert d["impact_score"] == 25.0

    def test_full_explanation_to_dict(self, service):
        inp = _make_input(accident_probability=0.8)
        result = _make_result(score=60.0, level=RiskLevel.HIGH)
        rule_result = _make_rule_result([
            ("test", "HIGH", 0.2, "Do something"),
        ])
        expl = service.explain(inp, result, rule_result)
        d = expl.to_dict()
        assert d["risk_level"] == "HIGH"
        assert d["compound_risk_score"] == 60.0
        assert len(d["factor_explanations"]) == 5
        assert len(d["triggered_rules"]) == 1
        assert isinstance(d["key_drivers"], list)
        assert isinstance(d["recommendations"], list)

    def test_to_dict_roundtrip_types(self, service):
        inp = _make_input()
        result = _make_result()
        expl = service.explain(inp, result)
        d = expl.to_dict()
        assert isinstance(d["risk_level"], str)
        assert isinstance(d["compound_risk_score"], float)
        assert isinstance(d["summary"], str)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. Custom thresholds
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCustomThresholds:
    def test_lower_anomaly_threshold(self, custom_service):
        """Custom: anomaly_notable=0.2 (default 0.3)."""
        inp = _make_input(isolation_forest_score=0.25)
        result = _make_result()
        expl = custom_service.explain(inp, result)
        anomaly = expl.factor_explanations[0]
        assert anomaly.contribution_level == ContributionLevel.MODERATE

    def test_lower_anomaly_high_threshold(self, custom_service):
        """Custom: anomaly_high=0.5 (default 0.7)."""
        inp = _make_input(isolation_forest_score=0.55)
        result = _make_result()
        expl = custom_service.explain(inp, result)
        anomaly = expl.factor_explanations[0]
        assert anomaly.contribution_level == ContributionLevel.CRITICAL

    def test_thresholds_property(self, service):
        assert isinstance(service.thresholds, ExplainabilityThresholds)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. Edge cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEdgeCases:
    def test_all_zero_inputs(self, service):
        inp = CompoundRiskInput()
        result = _make_result(score=0.0, level=RiskLevel.LOW)
        expl = service.explain(inp, result)
        assert expl.risk_level == RiskLevel.LOW
        assert len(expl.factor_explanations) == 5
        assert "normal" in expl.summary.lower()

    def test_all_max_inputs(self, service):
        inp = _make_input(
            isolation_forest_score=1.0,
            autoencoder_score=1.0,
            accident_probability=1.0,
            sensor_health_score=0.0,
            active_alert_count=10,
            alert_severity_max=1.0,
            threshold_violation_count=10,
        )
        result = _make_result(score=100.0, level=RiskLevel.CRITICAL)
        expl = service.explain(inp, result)
        assert expl.risk_level == RiskLevel.CRITICAL
        assert len(expl.key_drivers) > 0
        assert len(expl.recommendations) > 0

    def test_explanation_is_correct_type(self, service):
        inp = _make_input()
        result = _make_result()
        expl = service.explain(inp, result)
        assert isinstance(expl, CompoundRiskExplanation)

    def test_five_factors_always(self, service):
        """Always exactly 5 factor explanations."""
        inp = _make_input()
        result = _make_result()
        expl = service.explain(inp, result)
        assert len(expl.factor_explanations) == 5
        names = [f.factor_name for f in expl.factor_explanations]
        assert "Anomaly Detection" in names
        assert "Accident Probability" in names
        assert "Sensor Health" in names
        assert "Active Alerts" in names
        assert "Threshold Violations" in names


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. Integration with real Rule Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestWithRealRuleEngine:
    def test_explain_with_real_rules(self, service):
        """Use the actual rule engine for a realistic scenario."""
        from app.compound_risk.rules.rule_engine import create_default_rules

        rules = create_default_rules()
        engine = CompoundRiskRuleEngine(rules)

        facts = {
            "temperature_celsius": 70,
            "gas_level_ppm": 120,
            "maintenance_active": True,
            "permit_type": "HOT_WORK",
            "permit_active": True,
            "shift_type": "NIGHT",
            "sensor_health_score": 30,
        }
        rule_result = engine.evaluate(facts)

        inp = _make_input(
            isolation_forest_score=0.8,
            autoencoder_score=0.6,
            accident_probability=0.85,
            sensor_health_score=30.0,
            active_alert_count=3,
            alert_severity_max=0.9,
            threshold_violation_count=4,
        )
        result = _make_result(score=82.0, level=RiskLevel.CRITICAL)

        expl = service.explain(inp, result, rule_result)

        assert expl.risk_level == RiskLevel.CRITICAL
        assert len(expl.triggered_rules) > 0
        assert len(expl.key_drivers) > 0
        assert len(expl.recommendations) > 0
        assert "CRITICAL" in expl.summary

        # Serializable
        d = expl.to_dict()
        assert d["risk_level"] == "CRITICAL"
        assert len(d["triggered_rules"]) > 0
