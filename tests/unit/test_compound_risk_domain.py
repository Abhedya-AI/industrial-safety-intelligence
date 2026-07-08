"""Unit tests for the Compound Risk Intelligence domain layer.

Tests cover:
  - Value objects (enums matching PS-1 v2.0 exactly)
  - ORM model (CompoundRiskModel)
  - Pydantic schemas (request, response, nested types)
  - Domain exceptions
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.compound_risk.domain.exceptions import (
    CompoundRiskAnalysisFailedError,
    CompoundRiskError,
    CompoundRiskModelNotLoadedError,
    InsufficientScenarioDataError,
    InvalidRiskComponentError,
    ZoneNotFoundError,
)
from app.compound_risk.domain.value_objects import (
    CompoundRiskStatus,
    HazardType,
    PermitType,
    RiskLevel,
    ShiftType,
)
from app.compound_risk.models.compound_risk_model import CompoundRiskModel
from app.compound_risk.schemas import (
    CompoundRiskHistoryResponse,
    CompoundRiskRequest,
    CompoundRiskResponse,
    ContributingFactor,
    DangerousCombination,
    HistoricalContext,
    RecommendedAction,
    ScenarioInput,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Value objects (enums) — PS-1 v2.0 exact values
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRiskLevel:
    """Must match EXACTLY: LOW | MEDIUM | HIGH | CRITICAL (§4.1)."""

    def test_values(self):
        assert RiskLevel.LOW == "LOW"
        assert RiskLevel.MEDIUM == "MEDIUM"
        assert RiskLevel.HIGH == "HIGH"
        assert RiskLevel.CRITICAL == "CRITICAL"

    def test_four_levels(self):
        assert len(RiskLevel) == 4

    def test_string_enum(self):
        assert isinstance(RiskLevel.LOW, str)

    def test_uppercase(self):
        """Per §6.1 Rule #1: CRITICAL not Critical."""
        for level in RiskLevel:
            assert level.value == level.value.upper()


class TestCompoundRiskStatus:
    def test_values(self):
        assert CompoundRiskStatus.PENDING == "PENDING"
        assert CompoundRiskStatus.COMPLETED == "COMPLETED"
        assert CompoundRiskStatus.FAILED == "FAILED"

    def test_count(self):
        assert len(CompoundRiskStatus) == 3


class TestHazardType:
    """Must match EXACTLY per PS-1 v2.0 §4.6."""

    def test_all_nine_types(self):
        assert len(HazardType) == 9
        expected = {
            "GAS_LEAK", "FIRE", "SMOKE", "CHEMICAL_SPILL", "PPE_VIOLATION",
            "FALL_DETECTED", "ELECTRICAL_FAULT", "TEMPERATURE_ANOMALY",
            "PRESSURE_ANOMALY",
        }
        assert {h.value for h in HazardType} == expected

    def test_uppercase(self):
        for h in HazardType:
            assert h.value == h.value.upper()


class TestShiftType:
    """Must match: MORNING | AFTERNOON | NIGHT."""

    def test_values(self):
        assert ShiftType.MORNING == "MORNING"
        assert ShiftType.AFTERNOON == "AFTERNOON"
        assert ShiftType.NIGHT == "NIGHT"
        assert len(ShiftType) == 3


class TestPermitType:
    def test_hot_work(self):
        assert PermitType.HOT_WORK == "HOT_WORK"

    def test_all_types(self):
        assert len(PermitType) == 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. ORM model — exact field list
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCompoundRiskModel:
    def test_create_model_all_fields(self):
        m = CompoundRiskModel(
            id=str(uuid.uuid4()),
            equipment_id="EQ001",
            zone_id="ZONE_A",
            anomaly_score=0.85,
            accident_probability=0.65,
            risk_score=72.0,
            sensor_health_score=45.0,
            compound_risk_score=0.89,
            risk_level="CRITICAL",
            confidence_score=0.92,
            contributing_factors=json.dumps([{"factor": "Gas", "weight": 0.4,
                                              "current_value": "120 ppm", "contribution": 0.65}]),
            recommendation="Stop hot work activities",
            created_at=datetime.now(timezone.utc),
        )
        assert m.risk_level == "CRITICAL"
        assert m.compound_risk_score == 0.89
        assert m.risk_score == 72.0
        assert m.equipment_id == "EQ001"
        assert m.zone_id == "ZONE_A"

    def test_id_format(self):
        """ID must be UUID (per PS-1 conventions)."""
        uid = str(uuid.uuid4())
        m = CompoundRiskModel(
            id=uid,
            anomaly_score=0.5, accident_probability=0.5,
            risk_score=50.0, sensor_health_score=50.0,
            compound_risk_score=0.5, risk_level="MEDIUM",
            confidence_score=0.8,
        )
        assert m.id == uid
        uuid.UUID(m.id)  # Should not raise

    def test_tablename(self):
        """Per §5: lowercase table names."""
        assert CompoundRiskModel.__tablename__ == "compound_risk_analyses"

    def test_repr(self):
        m = CompoundRiskModel(
            id="test-id", zone_id="ZONE_B",
            anomaly_score=0.5, accident_probability=0.5,
            risk_score=60.0, sensor_health_score=50.0,
            compound_risk_score=0.72, risk_level="HIGH",
            confidence_score=0.8,
        )
        r = repr(m)
        assert "ZONE_B" in r
        assert "HIGH" in r
        assert "0.72" in r

    def test_optional_fields_nullable(self):
        m = CompoundRiskModel(
            anomaly_score=0.5, accident_probability=0.5,
            risk_score=50.0, sensor_health_score=50.0,
            compound_risk_score=0.5, risk_level="MEDIUM",
            confidence_score=0.8,
        )
        assert m.equipment_id is None
        assert m.zone_id is None
        assert m.contributing_factors is None
        assert m.recommendation is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Schema: ScenarioInput
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestScenarioInput:
    def test_from_api_spec(self):
        """Matches POST /risk/compound-analysis request body."""
        s = ScenarioInput(
            gas_level_ppm=120,
            temperature_celsius=45,
            pressure_bar=2.5,
            maintenance_active=True,
            worker_count=12,
            permit_type="HOT_WORK",
            permit_active=True,
            shift_type="NIGHT",
            equipment_health=0.75,
        )
        assert s.gas_level_ppm == 120
        assert s.shift_type == "NIGHT"
        assert s.permit_active is True

    def test_defaults(self):
        s = ScenarioInput()
        assert s.gas_level_ppm == 0.0
        assert s.worker_count == 0
        assert s.equipment_health == 1.0
        assert s.anomaly_score == 0.0
        assert s.risk_score == 0.0
        assert s.sensor_health_score == 100.0

    def test_rejects_negative_gas(self):
        with pytest.raises(ValidationError):
            ScenarioInput(gas_level_ppm=-1)

    def test_rejects_invalid_humidity(self):
        with pytest.raises(ValidationError):
            ScenarioInput(humidity_percent=101)

    def test_enrichment_fields(self):
        s = ScenarioInput(
            anomaly_score=0.85,
            accident_probability=0.65,
            risk_score=72.0,
            sensor_health_score=45.0,
        )
        assert s.anomaly_score == 0.85
        assert s.risk_score == 72.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Schema: CompoundRiskRequest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCompoundRiskRequest:
    def test_valid_request(self):
        r = CompoundRiskRequest(
            zone_id="ZONE_A",
            scenario=ScenarioInput(gas_level_ppm=120, temperature_celsius=45),
        )
        assert r.zone_id == "ZONE_A"
        assert r.scenario.gas_level_ppm == 120

    def test_zone_id_required(self):
        with pytest.raises(ValidationError):
            CompoundRiskRequest(scenario=ScenarioInput())

    def test_optional_fields(self):
        r = CompoundRiskRequest(
            zone_id="ZONE_B",
            scenario=ScenarioInput(),
            equipment_id="EQ001",
            include_historical=True,
        )
        assert r.equipment_id == "EQ001"
        assert r.include_historical is True

    def test_zone_id_nonempty(self):
        with pytest.raises(ValidationError):
            CompoundRiskRequest(zone_id="", scenario=ScenarioInput())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Schema: Nested response types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestContributingFactor:
    def test_from_api_spec(self):
        cf = ContributingFactor(
            factor="Gas Level",
            weight=0.40,
            current_value="120 ppm",
            contribution=0.65,
        )
        assert cf.factor == "Gas Level"
        assert cf.weight == 0.40

    def test_rejects_weight_above_one(self):
        with pytest.raises(ValidationError):
            ContributingFactor(
                factor="X", weight=1.5, current_value="5",
                contribution=0.1,
            )


class TestDangerousCombination:
    def test_from_api_spec(self):
        dc = DangerousCombination(
            condition_1="High Gas Level (120 ppm)",
            condition_2="Hot Work Permit Active",
            condition_3="Maintenance Activity",
            risk_score=0.95,
            severity=RiskLevel.CRITICAL,
            historical_incidents=["INC-2024-005", "INC-2024-012"],
            probability_of_incident=0.78,
        )
        assert dc.severity == RiskLevel.CRITICAL
        assert len(dc.historical_incidents) == 2


class TestRecommendedAction:
    def test_from_api_spec(self):
        ra = RecommendedAction(
            priority=1,
            action="Stop hot work activities",
            rationale="Gas level + hot work permit = explosion risk",
            estimated_effect="Reduces risk by 45%",
        )
        assert ra.priority == 1

    def test_rejects_zero_priority(self):
        with pytest.raises(ValidationError):
            RecommendedAction(
                priority=0, action="X", rationale="Y", estimated_effect="Z",
            )


class TestHistoricalContext:
    def test_from_api_spec(self):
        hc = HistoricalContext(
            similar_incidents_count=5,
            most_severe_incident="INC-2024-005",
            most_severe_outcome="3 workers hospitalized, plant evacuated",
            pattern="Occurs when maintenance + permits overlap + shift changes",
        )
        assert hc.similar_incidents_count == 5
        assert hc.most_severe_incident == "INC-2024-005"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Schema: CompoundRiskResponse
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCompoundRiskResponse:
    def test_from_orm(self):
        m = CompoundRiskModel(
            id=str(uuid.uuid4()),
            equipment_id="EQ001",
            zone_id="ZONE_A",
            anomaly_score=0.85,
            accident_probability=0.65,
            risk_score=72.0,
            sensor_health_score=45.0,
            compound_risk_score=0.89,
            risk_level="CRITICAL",
            confidence_score=0.92,
            contributing_factors=json.dumps([
                {"factor": "Gas", "weight": 0.4,
                 "current_value": "120 ppm", "contribution": 0.65},
            ]),
            recommendation="Evacuate zone",
            created_at=datetime.now(timezone.utc),
        )
        resp = CompoundRiskResponse.model_validate(m)
        assert resp.risk_level == RiskLevel.CRITICAL
        assert resp.compound_risk_score == 0.89
        assert resp.risk_score == 72.0
        assert resp.contributing_factors is not None
        assert resp.contributing_factors[0].factor == "Gas"

    def test_json_deserialisation(self):
        """Contributing factors stored as JSON string in DB are parsed."""
        m = CompoundRiskModel(
            id="test",
            anomaly_score=0.5, accident_probability=0.5,
            risk_score=50.0, sensor_health_score=50.0,
            compound_risk_score=0.5, risk_level="MEDIUM",
            confidence_score=0.8,
            contributing_factors='[{"factor":"X","weight":0.1,"current_value":"5","contribution":0.1}]',
            created_at=datetime.now(timezone.utc),
        )
        resp = CompoundRiskResponse.model_validate(m)
        assert isinstance(resp.contributing_factors, list)
        assert resp.contributing_factors[0].factor == "X"

    def test_null_contributing_factors(self):
        m = CompoundRiskModel(
            id="test2",
            anomaly_score=0.5, accident_probability=0.5,
            risk_score=50.0, sensor_health_score=50.0,
            compound_risk_score=0.5, risk_level="LOW",
            confidence_score=0.8,
            created_at=datetime.now(timezone.utc),
        )
        resp = CompoundRiskResponse.model_validate(m)
        assert resp.contributing_factors is None

    def test_all_risk_levels(self):
        for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            m = CompoundRiskModel(
                id=str(uuid.uuid4()),
                anomaly_score=0.5, accident_probability=0.5,
                risk_score=50.0, sensor_health_score=50.0,
                compound_risk_score=0.5, risk_level=level,
                confidence_score=0.8,
                created_at=datetime.now(timezone.utc),
            )
            resp = CompoundRiskResponse.model_validate(m)
            assert resp.risk_level.value == level


class TestCompoundRiskHistoryResponse:
    def test_empty(self):
        h = CompoundRiskHistoryResponse()
        assert h.success is True
        assert h.predictions == []
        assert h.total == 0

    def test_with_pagination(self):
        h = CompoundRiskHistoryResponse(
            predictions=[], total=42, offset=10, limit=20,
        )
        assert h.total == 42
        assert h.offset == 10
        assert h.limit == 20


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Domain exceptions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDomainExceptions:
    def test_base_error(self):
        e = CompoundRiskError()
        assert "compound risk" in str(e).lower()

    def test_analysis_failed(self):
        e = CompoundRiskAnalysisFailedError("Timeout")
        assert "Timeout" in str(e)
        assert e.error_code == "COMPOUND_RISK_ANALYSIS_FAILED"

    def test_insufficient_data(self):
        e = InsufficientScenarioDataError(["gas_level_ppm", "worker_count"])
        assert "gas_level_ppm" in str(e)
        assert e.missing_fields == ["gas_level_ppm", "worker_count"]

    def test_invalid_component(self):
        e = InvalidRiskComponentError("gas_risk", 1.5)
        assert "gas_risk" in str(e)
        assert e.value == 1.5

    def test_zone_not_found(self):
        e = ZoneNotFoundError("ZONE_X")
        assert "ZONE_X" in str(e)
        assert e.zone_id == "ZONE_X"

    def test_model_not_loaded(self):
        e = CompoundRiskModelNotLoadedError("Rules engine unavailable")
        assert "Rules engine" in str(e)

    def test_inheritance(self):
        e = CompoundRiskAnalysisFailedError("test")
        assert isinstance(e, CompoundRiskError)

    def test_insufficient_data_empty(self):
        e = InsufficientScenarioDataError()
        assert e.missing_fields == []
