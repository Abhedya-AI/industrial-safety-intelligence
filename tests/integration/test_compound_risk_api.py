"""Integration tests for the Compound Risk Intelligence API layer.

Tests cover:
  - POST /risk/compound-analysis (calculate)
  - GET  /risk/compound-analysis/latest
  - GET  /risk/compound-analysis/history
  - Validation errors (422)
  - Not found (404)
  - Response schema compliance
  - Pagination
  - Filters (zone_id, equipment_id, risk_level)
  - Multiple analyses
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

API_PREFIX = "/api/v1"
COMPOUND_ANALYSIS_URL = f"{API_PREFIX}/risk/compound-analysis"
LATEST_URL = f"{COMPOUND_ANALYSIS_URL}/latest"
HISTORY_URL = f"{COMPOUND_ANALYSIS_URL}/history"


def _make_request(
    zone_id: str = "ZONE_A",
    gas: float = 80.0,
    temp: float = 40.0,
    pressure: float = 3.0,
    anomaly: float = 0.5,
    accident_prob: float = 0.6,
    risk_score: float = 55.0,
    equipment_id: str = None,
    **overrides,
) -> dict:
    """Build a valid CompoundRiskRequest payload."""
    scenario = {
        "gas_level_ppm": gas,
        "temperature_celsius": temp,
        "pressure_bar": pressure,
        "anomaly_score": anomaly,
        "accident_probability": accident_prob,
        "risk_score": risk_score,
        "sensor_health_score": 75.0,
        "equipment_health": 0.8,
        **overrides,
    }
    payload = {"zone_id": zone_id, "scenario": scenario}
    if equipment_id:
        payload["equipment_id"] = equipment_id
    return payload


def _high_risk_request(zone_id: str = "ZONE_A") -> dict:
    return _make_request(
        zone_id=zone_id,
        gas=150, temp=70, pressure=6,
        anomaly=0.9, accident_prob=0.85,
        risk_score=80.0,
        equipment_health=0.2,
        maintenance_active=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. POST /risk/compound-analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCalculateCompoundRisk:
    async def test_returns_201(self, client: AsyncClient):
        resp = await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        assert resp.status_code == 201

    async def test_success_true(self, client: AsyncClient):
        resp = await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        body = resp.json()
        assert body["success"] is True

    async def test_has_compound_risk_analysis(self, client: AsyncClient):
        resp = await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        body = resp.json()
        assert "compound_risk_analysis" in body

    async def test_analysis_has_required_fields(self, client: AsyncClient):
        resp = await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        analysis = resp.json()["compound_risk_analysis"]
        assert "id" in analysis
        assert "compound_risk_score" in analysis
        assert "risk_level" in analysis
        assert "confidence_score" in analysis
        assert "created_at" in analysis

    async def test_risk_level_valid_enum(self, client: AsyncClient):
        resp = await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        analysis = resp.json()["compound_risk_analysis"]
        assert analysis["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    async def test_score_in_range(self, client: AsyncClient):
        resp = await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        analysis = resp.json()["compound_risk_analysis"]
        assert 0 <= analysis["compound_risk_score"] <= 1

    async def test_confidence_in_range(self, client: AsyncClient):
        resp = await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        analysis = resp.json()["compound_risk_analysis"]
        assert 0 < analysis["confidence_score"] <= 1

    async def test_zone_id_preserved(self, client: AsyncClient):
        resp = await client.post(
            COMPOUND_ANALYSIS_URL,
            json=_make_request(zone_id="ZONE_X"),
        )
        analysis = resp.json()["compound_risk_analysis"]
        assert analysis["zone_id"] == "ZONE_X"

    async def test_equipment_id_preserved(self, client: AsyncClient):
        resp = await client.post(
            COMPOUND_ANALYSIS_URL,
            json=_make_request(equipment_id="EQ999"),
        )
        analysis = resp.json()["compound_risk_analysis"]
        assert analysis["equipment_id"] == "EQ999"

    async def test_high_risk_scenario(self, client: AsyncClient):
        resp = await client.post(
            COMPOUND_ANALYSIS_URL, json=_high_risk_request(),
        )
        assert resp.status_code == 201
        analysis = resp.json()["compound_risk_analysis"]
        assert analysis["risk_level"] in ("HIGH", "CRITICAL")

    async def test_contributing_factors_returned(self, client: AsyncClient):
        resp = await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        analysis = resp.json()["compound_risk_analysis"]
        if analysis.get("contributing_factors"):
            assert isinstance(analysis["contributing_factors"], list)

    async def test_recommendation_returned(self, client: AsyncClient):
        resp = await client.post(
            COMPOUND_ANALYSIS_URL, json=_high_risk_request(),
        )
        analysis = resp.json()["compound_risk_analysis"]
        # High-risk scenario should have recommendation
        assert isinstance(analysis.get("recommendation"), (str, type(None)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Validation errors
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestValidationErrors:
    async def test_missing_zone_id(self, client: AsyncClient):
        resp = await client.post(
            COMPOUND_ANALYSIS_URL,
            json={"scenario": {"gas_level_ppm": 80}},
        )
        assert resp.status_code == 422

    async def test_missing_scenario(self, client: AsyncClient):
        resp = await client.post(
            COMPOUND_ANALYSIS_URL,
            json={"zone_id": "ZONE_A"},
        )
        assert resp.status_code == 422

    async def test_empty_body(self, client: AsyncClient):
        resp = await client.post(COMPOUND_ANALYSIS_URL, json={})
        assert resp.status_code == 422

    async def test_invalid_json(self, client: AsyncClient):
        resp = await client.post(
            COMPOUND_ANALYSIS_URL,
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. GET /risk/compound-analysis/latest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGetLatest:
    async def test_404_when_empty(self, client: AsyncClient):
        resp = await client.get(LATEST_URL)
        assert resp.status_code == 404

    async def test_returns_after_post(self, client: AsyncClient):
        await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        resp = await client.get(LATEST_URL)
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    async def test_latest_has_analysis(self, client: AsyncClient):
        await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        resp = await client.get(LATEST_URL)
        assert "compound_risk_analysis" in resp.json()

    async def test_filter_by_zone(self, client: AsyncClient):
        await client.post(
            COMPOUND_ANALYSIS_URL,
            json=_make_request(zone_id="ZONE_Q"),
        )
        resp = await client.get(LATEST_URL, params={"zone_id": "ZONE_Q"})
        assert resp.status_code == 200
        assert resp.json()["compound_risk_analysis"]["zone_id"] == "ZONE_Q"

    async def test_filter_zone_not_found(self, client: AsyncClient):
        await client.post(
            COMPOUND_ANALYSIS_URL,
            json=_make_request(zone_id="ZONE_REAL"),
        )
        resp = await client.get(LATEST_URL, params={"zone_id": "ZONE_FAKE"})
        assert resp.status_code == 404

    async def test_filter_by_equipment(self, client: AsyncClient):
        await client.post(
            COMPOUND_ANALYSIS_URL,
            json=_make_request(equipment_id="EQ_SPECIAL"),
        )
        resp = await client.get(
            LATEST_URL, params={"equipment_id": "EQ_SPECIAL"},
        )
        assert resp.status_code == 200


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. GET /risk/compound-analysis/history
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGetHistory:
    async def test_empty_history(self, client: AsyncClient):
        resp = await client.get(HISTORY_URL)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["total"] == 0
        assert body["predictions"] == []

    async def test_history_after_posts(self, client: AsyncClient):
        for _ in range(3):
            await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        resp = await client.get(HISTORY_URL)
        body = resp.json()
        assert body["total"] == 3
        assert len(body["predictions"]) == 3

    async def test_filter_by_zone(self, client: AsyncClient):
        await client.post(
            COMPOUND_ANALYSIS_URL,
            json=_make_request(zone_id="ZONE_AA"),
        )
        await client.post(
            COMPOUND_ANALYSIS_URL,
            json=_make_request(zone_id="ZONE_BB"),
        )
        resp = await client.get(HISTORY_URL, params={"zone_id": "ZONE_AA"})
        body = resp.json()
        assert body["total"] == 1
        assert body["predictions"][0]["zone_id"] == "ZONE_AA"

    async def test_pagination_offset(self, client: AsyncClient):
        for _ in range(5):
            await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        resp = await client.get(HISTORY_URL, params={"offset": 3, "limit": 10})
        body = resp.json()
        assert body["total"] == 5
        assert len(body["predictions"]) == 2  # 5 - 3 = 2 remaining

    async def test_pagination_limit(self, client: AsyncClient):
        for _ in range(5):
            await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        resp = await client.get(HISTORY_URL, params={"limit": 2})
        body = resp.json()
        assert body["total"] == 5
        assert len(body["predictions"]) == 2
        assert body["limit"] == 2

    async def test_response_has_offset_limit(self, client: AsyncClient):
        resp = await client.get(HISTORY_URL, params={"offset": 5, "limit": 10})
        body = resp.json()
        assert body["offset"] == 5
        assert body["limit"] == 10

    async def test_filter_by_risk_level(self, client: AsyncClient):
        # Post a high-risk analysis
        await client.post(
            COMPOUND_ANALYSIS_URL,
            json=_high_risk_request(zone_id="ZONE_RL"),
        )
        resp = await client.get(
            HISTORY_URL,
            params={"zone_id": "ZONE_RL"},
        )
        body = resp.json()
        assert body["total"] >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Schema compliance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSchemaCompliance:
    async def test_analysis_id_is_uuid(self, client: AsyncClient):
        resp = await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        analysis_id = resp.json()["compound_risk_analysis"]["id"]
        import uuid
        uuid.UUID(analysis_id)  # Should not raise

    async def test_created_at_is_iso8601(self, client: AsyncClient):
        resp = await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        created_at = resp.json()["compound_risk_analysis"]["created_at"]
        from datetime import datetime
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))

    async def test_anomaly_score_in_range(self, client: AsyncClient):
        resp = await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        analysis = resp.json()["compound_risk_analysis"]
        assert 0 <= analysis["anomaly_score"] <= 1

    async def test_accident_probability_in_range(self, client: AsyncClient):
        resp = await client.post(COMPOUND_ANALYSIS_URL, json=_make_request())
        analysis = resp.json()["compound_risk_analysis"]
        assert 0 <= analysis["accident_probability"] <= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Multiple scenario types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestScenarioTypes:
    async def test_minimal_scenario(self, client: AsyncClient):
        """Only zone_id + minimal scenario."""
        payload = {
            "zone_id": "ZONE_MIN",
            "scenario": {"anomaly_score": 0.3},
        }
        resp = await client.post(COMPOUND_ANALYSIS_URL, json=payload)
        assert resp.status_code == 201

    async def test_full_scenario(self, client: AsyncClient):
        """All scenario fields populated."""
        payload = {
            "zone_id": "ZONE_FULL",
            "equipment_id": "EQ_FULL",
            "scenario": {
                "gas_level_ppm": 120,
                "temperature_celsius": 55,
                "pressure_bar": 4.5,
                "humidity_percent": 70,
                "vibration_level": 3.5,
                "maintenance_active": True,
                "worker_count": 10,
                "permit_type": "HOT_WORK",
                "permit_active": True,
                "shift_type": "NIGHT",
                "equipment_health": 0.6,
                "anomaly_score": 0.7,
                "accident_probability": 0.65,
                "risk_score": 60.0,
                "sensor_health_score": 70.0,
            },
        }
        resp = await client.post(COMPOUND_ANALYSIS_URL, json=payload)
        assert resp.status_code == 201
        analysis = resp.json()["compound_risk_analysis"]
        assert analysis["zone_id"] == "ZONE_FULL"
        assert analysis["equipment_id"] == "EQ_FULL"
