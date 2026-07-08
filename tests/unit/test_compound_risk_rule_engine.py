"""Unit tests for the Compound Risk Rule Engine.

Tests cover:
  - Condition evaluation (all 8 comparison operators)
  - NOT (negate) logic
  - AND / OR logical operators
  - Rule evaluation (single and multi-condition)
  - Full engine evaluation
  - Default rules factory
  - Edge cases (missing fields, type errors, empty rules)
  - Explanation and recommendation output
  - Rule management (add/remove)
"""

from __future__ import annotations

import pytest

from app.compound_risk.rules.rule_engine import (
    ComparisonOp,
    CompoundRiskRuleEngine,
    Condition,
    LogicalOp,
    RuleDefinition,
    RuleEngineResult,
    RuleResult,
    create_default_rules,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Condition evaluation — all operators
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConditionOperators:
    def test_gt_true(self):
        c = Condition("temp", ComparisonOp.GT, 60)
        assert c.evaluate({"temp": 65}) is True

    def test_gt_false(self):
        c = Condition("temp", ComparisonOp.GT, 60)
        assert c.evaluate({"temp": 60}) is False

    def test_gte_true_equal(self):
        c = Condition("temp", ComparisonOp.GTE, 60)
        assert c.evaluate({"temp": 60}) is True

    def test_gte_true_above(self):
        c = Condition("temp", ComparisonOp.GTE, 60)
        assert c.evaluate({"temp": 61}) is True

    def test_gte_false(self):
        c = Condition("temp", ComparisonOp.GTE, 60)
        assert c.evaluate({"temp": 59}) is False

    def test_lt_true(self):
        c = Condition("health", ComparisonOp.LT, 0.3)
        assert c.evaluate({"health": 0.2}) is True

    def test_lt_false(self):
        c = Condition("health", ComparisonOp.LT, 0.3)
        assert c.evaluate({"health": 0.3}) is False

    def test_lte_true(self):
        c = Condition("health", ComparisonOp.LTE, 0.3)
        assert c.evaluate({"health": 0.3}) is True

    def test_eq_true(self):
        c = Condition("active", ComparisonOp.EQ, True)
        assert c.evaluate({"active": True}) is True

    def test_eq_false(self):
        c = Condition("active", ComparisonOp.EQ, True)
        assert c.evaluate({"active": False}) is False

    def test_eq_string(self):
        c = Condition("permit_type", ComparisonOp.EQ, "HOT_WORK")
        assert c.evaluate({"permit_type": "HOT_WORK"}) is True
        assert c.evaluate({"permit_type": "CONFINED_SPACE"}) is False

    def test_neq_true(self):
        c = Condition("status", ComparisonOp.NEQ, "OFFLINE")
        assert c.evaluate({"status": "ACTIVE"}) is True

    def test_neq_false(self):
        c = Condition("status", ComparisonOp.NEQ, "OFFLINE")
        assert c.evaluate({"status": "OFFLINE"}) is False

    def test_in_true(self):
        c = Condition("permit_type", ComparisonOp.IN, ["HOT_WORK", "CONFINED_SPACE"])
        assert c.evaluate({"permit_type": "HOT_WORK"}) is True

    def test_in_false(self):
        c = Condition("permit_type", ComparisonOp.IN, ["HOT_WORK", "CONFINED_SPACE"])
        assert c.evaluate({"permit_type": "ELECTRICAL"}) is False

    def test_not_in_true(self):
        c = Condition("shift", ComparisonOp.NOT_IN, ["NIGHT"])
        assert c.evaluate({"shift": "MORNING"}) is True

    def test_not_in_false(self):
        c = Condition("shift", ComparisonOp.NOT_IN, ["NIGHT"])
        assert c.evaluate({"shift": "NIGHT"}) is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. NOT (negate) logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNegate:
    def test_negate_turns_true_to_false(self):
        c = Condition("temp", ComparisonOp.GT, 60, negate=True)
        assert c.evaluate({"temp": 65}) is False

    def test_negate_turns_false_to_true(self):
        c = Condition("temp", ComparisonOp.GT, 60, negate=True)
        assert c.evaluate({"temp": 55}) is True

    def test_negate_missing_field(self):
        """Missing field → False, negated → True."""
        c = Condition("temp", ComparisonOp.GT, 60, negate=True)
        assert c.evaluate({}) is True

    def test_negate_with_eq(self):
        c = Condition("active", ComparisonOp.EQ, True, negate=True)
        assert c.evaluate({"active": True}) is False
        assert c.evaluate({"active": False}) is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Missing fields and type errors
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEdgeCases:
    def test_missing_field_returns_false(self):
        c = Condition("missing_field", ComparisonOp.GT, 10)
        assert c.evaluate({"other_field": 20}) is False

    def test_type_mismatch_returns_false(self):
        c = Condition("temp", ComparisonOp.GT, 60)
        assert c.evaluate({"temp": "not_a_number"}) is False

    def test_empty_facts(self):
        c = Condition("temp", ComparisonOp.GT, 60)
        assert c.evaluate({}) is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Condition explain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConditionExplain:
    def test_explain_with_description(self):
        c = Condition("temp", ComparisonOp.GT, 60, description="Temperature")
        text = c.explain({"temp": 75})
        assert "Temperature" in text
        assert "> 60" in text
        assert "75" in text

    def test_explain_without_description(self):
        c = Condition("temp", ComparisonOp.GT, 60)
        text = c.explain({"temp": 75})
        assert "temp" in text

    def test_explain_missing_field(self):
        c = Condition("temp", ComparisonOp.GT, 60)
        text = c.explain({})
        assert "N/A" in text

    def test_explain_negate(self):
        c = Condition("temp", ComparisonOp.GT, 60, negate=True)
        text = c.explain({"temp": 55})
        assert "NOT" in text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. AND rules
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestANDRules:
    def test_and_all_true(self):
        rule = RuleDefinition(
            name="test_and",
            conditions=[
                Condition("temp", ComparisonOp.GT, 60),
                Condition("gas", ComparisonOp.GTE, 100),
            ],
            logical_op=LogicalOp.AND,
            risk_impact=0.3,
            severity="CRITICAL",
        )
        engine = CompoundRiskRuleEngine([rule])
        result = engine.evaluate({"temp": 70, "gas": 120})
        assert len(result.triggered_rules) == 1
        assert result.triggered_rules[0].triggered is True

    def test_and_one_false(self):
        rule = RuleDefinition(
            name="test_and",
            conditions=[
                Condition("temp", ComparisonOp.GT, 60),
                Condition("gas", ComparisonOp.GTE, 100),
            ],
            logical_op=LogicalOp.AND,
        )
        engine = CompoundRiskRuleEngine([rule])
        result = engine.evaluate({"temp": 70, "gas": 50})
        assert len(result.triggered_rules) == 0

    def test_and_all_false(self):
        rule = RuleDefinition(
            name="test_and",
            conditions=[
                Condition("temp", ComparisonOp.GT, 60),
                Condition("gas", ComparisonOp.GTE, 100),
            ],
            logical_op=LogicalOp.AND,
        )
        engine = CompoundRiskRuleEngine([rule])
        result = engine.evaluate({"temp": 50, "gas": 50})
        assert len(result.triggered_rules) == 0

    def test_and_empty_conditions(self):
        rule = RuleDefinition(name="empty", conditions=[], logical_op=LogicalOp.AND)
        engine = CompoundRiskRuleEngine([rule])
        result = engine.evaluate({"temp": 70})
        assert len(result.triggered_rules) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. OR rules
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestORRules:
    def test_or_one_true(self):
        rule = RuleDefinition(
            name="test_or",
            conditions=[
                Condition("temp", ComparisonOp.GT, 60),
                Condition("gas", ComparisonOp.GTE, 100),
            ],
            logical_op=LogicalOp.OR,
            risk_impact=0.15,
        )
        engine = CompoundRiskRuleEngine([rule])
        result = engine.evaluate({"temp": 70, "gas": 50})
        assert len(result.triggered_rules) == 1

    def test_or_all_true(self):
        rule = RuleDefinition(
            name="test_or",
            conditions=[
                Condition("temp", ComparisonOp.GT, 60),
                Condition("gas", ComparisonOp.GTE, 100),
            ],
            logical_op=LogicalOp.OR,
        )
        engine = CompoundRiskRuleEngine([rule])
        result = engine.evaluate({"temp": 70, "gas": 120})
        assert len(result.triggered_rules) == 1

    def test_or_all_false(self):
        rule = RuleDefinition(
            name="test_or",
            conditions=[
                Condition("temp", ComparisonOp.GT, 60),
                Condition("gas", ComparisonOp.GTE, 100),
            ],
            logical_op=LogicalOp.OR,
        )
        engine = CompoundRiskRuleEngine([rule])
        result = engine.evaluate({"temp": 50, "gas": 50})
        assert len(result.triggered_rules) == 0

    def test_or_empty_conditions(self):
        rule = RuleDefinition(name="empty", conditions=[], logical_op=LogicalOp.OR)
        engine = CompoundRiskRuleEngine([rule])
        result = engine.evaluate({"temp": 70})
        assert len(result.triggered_rules) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Multi-rule evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMultiRuleEvaluation:
    def test_multiple_rules_some_triggered(self):
        rules = [
            RuleDefinition(
                name="temp_high",
                conditions=[Condition("temp", ComparisonOp.GT, 60)],
                risk_impact=0.2,
                severity="HIGH",
            ),
            RuleDefinition(
                name="gas_high",
                conditions=[Condition("gas", ComparisonOp.GTE, 100)],
                risk_impact=0.3,
                severity="CRITICAL",
            ),
            RuleDefinition(
                name="pressure_high",
                conditions=[Condition("pressure", ComparisonOp.GT, 5)],
                risk_impact=0.1,
            ),
        ]
        engine = CompoundRiskRuleEngine(rules)
        result = engine.evaluate({"temp": 70, "gas": 120, "pressure": 3})
        assert len(result.triggered_rules) == 2
        assert result.total_impact == pytest.approx(0.5)
        assert len(result.all_results) == 3

    def test_no_rules_triggered(self):
        rules = [
            RuleDefinition(
                name="temp_high",
                conditions=[Condition("temp", ComparisonOp.GT, 60)],
            ),
        ]
        engine = CompoundRiskRuleEngine(rules)
        result = engine.evaluate({"temp": 50})
        assert len(result.triggered_rules) == 0
        assert result.total_impact == 0.0

    def test_all_rules_triggered(self):
        rules = [
            RuleDefinition(
                name="r1",
                conditions=[Condition("a", ComparisonOp.GT, 0)],
                risk_impact=0.3,
            ),
            RuleDefinition(
                name="r2",
                conditions=[Condition("b", ComparisonOp.GT, 0)],
                risk_impact=0.4,
            ),
        ]
        engine = CompoundRiskRuleEngine(rules)
        result = engine.evaluate({"a": 1, "b": 1})
        assert len(result.triggered_rules) == 2
        assert result.total_impact == pytest.approx(0.7)

    def test_total_impact_capped_at_one(self):
        rules = [
            RuleDefinition(
                name=f"r{i}",
                conditions=[Condition("a", ComparisonOp.GT, 0)],
                risk_impact=0.5,
            )
            for i in range(5)
        ]
        engine = CompoundRiskRuleEngine(rules)
        result = engine.evaluate({"a": 1})
        assert result.total_impact <= 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. RuleResult structure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRuleResult:
    def test_triggered_has_impact(self):
        rule = RuleDefinition(
            name="test",
            conditions=[Condition("temp", ComparisonOp.GT, 60)],
            risk_impact=0.25,
            severity="HIGH",
            recommendation="Reduce temperature",
        )
        engine = CompoundRiskRuleEngine([rule])
        result = engine.evaluate({"temp": 70})
        tr = result.triggered_rules[0]
        assert tr.risk_impact == 0.25
        assert tr.severity == "HIGH"
        assert tr.recommendation == "Reduce temperature"

    def test_not_triggered_has_zero_impact(self):
        rule = RuleDefinition(
            name="test",
            conditions=[Condition("temp", ComparisonOp.GT, 60)],
            risk_impact=0.25,
        )
        engine = CompoundRiskRuleEngine([rule])
        result = engine.evaluate({"temp": 50})
        ar = result.all_results[0]
        assert ar.triggered is False
        assert ar.risk_impact == 0.0
        assert ar.severity == "NONE"
        assert ar.recommendation == ""

    def test_condition_results_populated(self):
        rule = RuleDefinition(
            name="test",
            conditions=[
                Condition("temp", ComparisonOp.GT, 60),
                Condition("gas", ComparisonOp.GTE, 100),
            ],
            logical_op=LogicalOp.AND,
        )
        engine = CompoundRiskRuleEngine([rule])
        result = engine.evaluate({"temp": 70, "gas": 120})
        tr = result.triggered_rules[0]
        assert len(tr.condition_results) == 2
        assert tr.condition_results[0]["field"] == "temp"
        assert tr.condition_results[0]["passed"] is True
        assert tr.condition_results[0]["actual_value"] == 70


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Explanation output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestExplanation:
    def test_no_triggers_explanation(self):
        engine = CompoundRiskRuleEngine([
            RuleDefinition(
                name="test",
                conditions=[Condition("temp", ComparisonOp.GT, 60)],
            ),
        ])
        result = engine.evaluate({"temp": 50})
        assert "No compound risk rules triggered" in result.explanation

    def test_triggered_explanation_has_rule_name(self):
        engine = CompoundRiskRuleEngine([
            RuleDefinition(
                name="high_temp",
                conditions=[Condition("temp", ComparisonOp.GT, 60)],
                severity="HIGH",
            ),
        ])
        result = engine.evaluate({"temp": 70})
        assert "high_temp" in result.explanation
        assert "HIGH" in result.explanation

    def test_triggered_rule_explanation_has_conditions(self):
        engine = CompoundRiskRuleEngine([
            RuleDefinition(
                name="combined",
                conditions=[Condition("temp", ComparisonOp.GT, 60, description="Temp")],
                description="Test rule",
            ),
        ])
        result = engine.evaluate({"temp": 70})
        rule_expl = result.triggered_rules[0].explanation
        assert "triggered" in rule_expl
        assert "Test rule" in rule_expl

    def test_multiple_triggered_explanation(self):
        engine = CompoundRiskRuleEngine([
            RuleDefinition(
                name="r1",
                conditions=[Condition("a", ComparisonOp.GT, 0)],
                severity="HIGH",
            ),
            RuleDefinition(
                name="r2",
                conditions=[Condition("b", ComparisonOp.GT, 0)],
                severity="CRITICAL",
            ),
        ])
        result = engine.evaluate({"a": 1, "b": 1})
        assert "r1" in result.explanation
        assert "r2" in result.explanation
        assert "CRITICAL" in result.explanation  # Max severity


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Rule management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRuleManagement:
    def test_add_rule(self):
        engine = CompoundRiskRuleEngine()
        assert engine.rule_count == 0
        engine.add_rule(
            RuleDefinition(name="new", conditions=[Condition("a", ComparisonOp.GT, 0)]),
        )
        assert engine.rule_count == 1

    def test_remove_rule(self):
        engine = CompoundRiskRuleEngine([
            RuleDefinition(name="keep", conditions=[Condition("a", ComparisonOp.GT, 0)]),
            RuleDefinition(name="drop", conditions=[Condition("b", ComparisonOp.GT, 0)]),
        ])
        removed = engine.remove_rule("drop")
        assert removed is True
        assert engine.rule_count == 1
        assert engine.rules[0].name == "keep"

    def test_remove_nonexistent(self):
        engine = CompoundRiskRuleEngine([
            RuleDefinition(name="keep", conditions=[Condition("a", ComparisonOp.GT, 0)]),
        ])
        removed = engine.remove_rule("nonexistent")
        assert removed is False
        assert engine.rule_count == 1

    def test_rules_property_returns_copy(self):
        engine = CompoundRiskRuleEngine([
            RuleDefinition(name="r1", conditions=[Condition("a", ComparisonOp.GT, 0)]),
        ])
        rules = engine.rules
        rules.clear()
        assert engine.rule_count == 1  # Original unaffected

    def test_empty_engine(self):
        engine = CompoundRiskRuleEngine()
        result = engine.evaluate({"temp": 100})
        assert len(result.triggered_rules) == 0
        assert result.total_impact == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. Default rules factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDefaultRules:
    def test_creates_nine_rules(self):
        rules = create_default_rules()
        assert len(rules) == 9

    def test_custom_thresholds(self):
        rules = create_default_rules(temperature_threshold=80, gas_threshold=200)
        engine = CompoundRiskRuleEngine(rules)
        # Below custom thresholds → no high_temp_and_gas
        result = engine.evaluate({"temperature_celsius": 70, "gas_level_ppm": 150})
        triggered_names = {r.rule_name for r in result.triggered_rules}
        assert "high_temp_and_gas" not in triggered_names

    def test_default_thresholds_trigger(self):
        rules = create_default_rules()
        engine = CompoundRiskRuleEngine(rules)
        # Above default thresholds
        result = engine.evaluate({
            "temperature_celsius": 70,
            "gas_level_ppm": 120,
        })
        triggered_names = {r.rule_name for r in result.triggered_rules}
        assert "high_temp_and_gas" in triggered_names
        assert "temp_or_gas_elevated" in triggered_names

    def test_gas_and_hot_work_rule(self):
        rules = create_default_rules()
        engine = CompoundRiskRuleEngine(rules)
        result = engine.evaluate({
            "gas_level_ppm": 120,
            "permit_type": "HOT_WORK",
            "permit_active": True,
        })
        triggered_names = {r.rule_name for r in result.triggered_rules}
        assert "gas_and_hot_work" in triggered_names

    def test_night_shift_maintenance(self):
        rules = create_default_rules()
        engine = CompoundRiskRuleEngine(rules)
        result = engine.evaluate({
            "shift_type": "NIGHT",
            "maintenance_active": True,
        })
        triggered_names = {r.rule_name for r in result.triggered_rules}
        assert "night_shift_maintenance" in triggered_names

    def test_sensor_health_degraded_negate(self):
        """Uses NOT: sensor health < 50 → triggered."""
        rules = create_default_rules()
        engine = CompoundRiskRuleEngine(rules)
        result = engine.evaluate({"sensor_health_score": 30})
        triggered_names = {r.rule_name for r in result.triggered_rules}
        assert "sensor_health_degraded" in triggered_names

    def test_sensor_health_good_not_triggered(self):
        """Sensor health >= 50 → NOT triggered."""
        rules = create_default_rules()
        engine = CompoundRiskRuleEngine(rules)
        result = engine.evaluate({"sensor_health_score": 80})
        triggered_names = {r.rule_name for r in result.triggered_rules}
        assert "sensor_health_degraded" not in triggered_names

    def test_all_rules_have_required_fields(self):
        rules = create_default_rules()
        for r in rules:
            assert r.name
            assert len(r.conditions) > 0
            assert r.risk_impact > 0
            assert r.severity
            assert r.description
            assert r.recommendation


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. Complex scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestComplexScenarios:
    def test_cascading_risk(self):
        """Multiple compound conditions trigger simultaneously."""
        rules = create_default_rules()
        engine = CompoundRiskRuleEngine(rules)
        result = engine.evaluate({
            "temperature_celsius": 70,
            "gas_level_ppm": 120,
            "pressure_bar": 6,
            "equipment_health": 0.2,
            "maintenance_active": True,
            "permit_type": "HOT_WORK",
            "permit_active": True,
            "shift_type": "NIGHT",
            "worker_count": 20,
            "sensor_health_score": 30,
        })
        assert len(result.triggered_rules) >= 5
        assert result.total_impact > 0.5

    def test_safe_scenario(self):
        """Normal operating conditions → few or no triggers."""
        rules = create_default_rules()
        engine = CompoundRiskRuleEngine(rules)
        result = engine.evaluate({
            "temperature_celsius": 25,
            "gas_level_ppm": 10,
            "pressure_bar": 2,
            "equipment_health": 0.95,
            "maintenance_active": False,
            "shift_type": "MORNING",
            "worker_count": 5,
            "sensor_health_score": 95,
        })
        assert len(result.triggered_rules) == 0
        assert result.total_impact == 0.0
