"""Integration tests for the Risk Prediction API endpoints.

Uses the shared ``client`` fixture (in-memory SQLite + real FastAPI app)
with the XGBoost model mocked via dependency override.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.risk_prediction.models.risk_prediction_model import RiskPredictionModel
from app.risk_prediction.repositories.sqlalchemy_risk_prediction_repo import (
    SQLAlchemyRiskPredictionRepository,
)
from app.risk_prediction.services.risk_prediction_service import (
    RiskPredictionService,
    _ModelHolder,
)


# ── Fixtures ──


def _make_mock_holder(proba: float = 0.65) -> _ModelHolder:
    """Create a pre-loaded model holder with mocked model/scaler."""
    holder = _ModelHolder()
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1.0 - proba, proba]])
    model.predict.return_value = np.array([1 if proba >= 0.5 else 0])
    model.feature_importances_ = np.array([0.3, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05, 0.1])

    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, 8))

    holder.model = model
    holder.scaler = scaler
    holder.column_order = [f"feature_{i}" for i in range(8)]
    holder.feature_importances = {
        "Temp": 0.30, "Service_Days": 0.12, "Pressure": 0.08,
        "Age": 0.03, "Gas": 0.02, "Humidity": 0.01,
    }
    holder._loaded = True
    return holder


@pytest_asyncio.fixture
async def risk_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client with the risk model mocked."""
    from app.main import app
    from app.shared.database.connection import get_async_session

    mock_holder = _make_mock_holder(proba=0.65)

    # Override the session
    async def override_get_session():
        yield db_session

    # Override the risk prediction service to inject mock model holder
    def override_risk_service(session=None):
        repo = SQLAlchemyRiskPredictionRepository(db_session)
        return RiskPredictionService(repo, model_holder=mock_holder)

    from app.core.dependencies import get_risk_prediction_service

    app.dependency_overrides[get_async_session] = override_get_session
    app.dependency_overrides[get_risk_prediction_service] = override_risk_service

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def seeded_client(
    db_session: AsyncSession,
) -> AsyncGenerator[tuple[AsyncClient, list[str]], None]:
    """Client with 5 predictions pre-seeded in the DB."""
    from app.main import app
    from app.shared.database.connection import get_async_session

    mock_holder = _make_mock_holder(proba=0.65)

    async def override_get_session():
        yield db_session

    def override_risk_service(session=None):
        repo = SQLAlchemyRiskPredictionRepository(db_session)
        return RiskPredictionService(repo, model_holder=mock_holder)

    from app.core.dependencies import get_risk_prediction_service

    app.dependency_overrides[get_async_session] = override_get_session
    app.dependency_overrides[get_risk_prediction_service] = override_risk_service

    # Seed predictions
    repo = SQLAlchemyRiskPredictionRepository(db_session)
    ids = []
    for i in range(5):
        pred = RiskPredictionModel(
            id=str(uuid.uuid4()),
            sensor_id="S001" if i < 3 else "S002",
            zone_id="ZONE_A" if i < 3 else "ZONE_B",
            equipment_id="EQ-001",
            prediction_timestamp=datetime(2026, 7, 1, 10 + i, 0, tzinfo=timezone.utc),
            accident_probability=0.1 * (i + 1),
            predicted_risk_score=10 * (i + 1),
            risk_level=["LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH"][i],
            confidence_score=0.9,
            model_name="xgboost_risk_prediction",
            model_version="1.0.0",
            status="COMPLETED",
        )
        await repo.create_prediction(pred)
        ids.append(pred.id)
    await db_session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac, ids

    app.dependency_overrides.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. POST /risk/predict
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPredictRisk:
    async def test_basic_prediction(self, risk_client: AsyncClient):
        resp = await risk_client.post(
            "/api/v1/risk/predict",
            json={
                "sensor_id": "S001",
                "zone_id": "ZONE_A",
                "features": {
                    "Temp": 85.0,
                    "Pressure": 22.0,
                    "Humidity": 55.0,
                    "Gas": 110.0,
                    "Vibration": 3.5,
                    "Speed": 120.0,
                },
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["success"] is True
        pred = data["prediction"]
        assert pred["sensor_id"] == "S001"
        assert pred["zone_id"] == "ZONE_A"
        assert pred["risk_level"] == "HIGH"
        assert 0 <= pred["accident_probability"] <= 1
        assert 0 <= pred["predicted_risk_score"] <= 100
        assert 0 <= pred["confidence_score"] <= 1
        assert pred["model_name"] == "xgboost_risk_prediction"
        assert pred["status"] == "COMPLETED"

    async def test_prediction_with_explanation(self, risk_client: AsyncClient):
        resp = await risk_client.post(
            "/api/v1/risk/predict",
            json={
                "features": {"Temp": 85.0, "Pressure": 22.0, "Humidity": 55.0,
                             "Gas": 110.0, "Vibration": 3.5, "Speed": 120.0},
                "include_explanation": True,
            },
        )
        assert resp.status_code == 201
        pred = resp.json()["prediction"]
        assert pred["explanation"] is not None
        assert len(pred["explanation"]) > 0

    async def test_prediction_with_breakdown(self, risk_client: AsyncClient):
        resp = await risk_client.post(
            "/api/v1/risk/predict",
            json={
                "features": {"Temp": 85.0, "Pressure": 22.0, "Humidity": 55.0,
                             "Gas": 110.0, "Vibration": 3.5, "Speed": 120.0},
                "include_breakdown": True,
            },
        )
        assert resp.status_code == 201
        pred = resp.json()["prediction"]
        assert pred["top_contributing_factors"] is not None

    async def test_prediction_persisted(self, risk_client: AsyncClient):
        resp = await risk_client.post(
            "/api/v1/risk/predict",
            json={
                "sensor_id": "S999",
                "features": {"Temp": 85.0, "Pressure": 22.0, "Humidity": 55.0,
                             "Gas": 110.0, "Vibration": 3.5, "Speed": 120.0},
            },
        )
        assert resp.status_code == 201
        pred_id = resp.json()["prediction"]["id"]

        # Verify it appears in history
        history = await risk_client.get(
            "/api/v1/risk/predictions", params={"sensor_id": "S999"},
        )
        assert history.status_code == 200
        assert history.json()["total"] >= 1

    async def test_empty_features_get_defaults(self, risk_client: AsyncClient):
        resp = await risk_client.post(
            "/api/v1/risk/predict",
            json={"features": {"Temp": 50.0, "Pressure": 15.0, "Humidity": 40.0,
                               "Gas": 20.0, "Vibration": 1.0, "Speed": 80.0}},
        )
        assert resp.status_code == 201

    async def test_prediction_id_is_uuid(self, risk_client: AsyncClient):
        resp = await risk_client.post(
            "/api/v1/risk/predict",
            json={"features": {"Temp": 50.0, "Pressure": 15.0, "Humidity": 40.0,
                               "Gas": 20.0, "Vibration": 1.0, "Speed": 80.0}},
        )
        pred_id = resp.json()["prediction"]["id"]
        uuid.UUID(pred_id)  # Validates it's a valid UUID


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. GET /risk/predictions (history)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPredictionHistory:
    async def test_empty_history(self, risk_client: AsyncClient):
        resp = await risk_client.get("/api/v1/risk/predictions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["predictions"] == []
        assert data["total"] == 0

    async def test_seeded_history(self, seeded_client):
        client, ids = seeded_client
        resp = await client.get("/api/v1/risk/predictions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["predictions"]) == 5

    async def test_filter_by_sensor(self, seeded_client):
        client, _ = seeded_client
        resp = await client.get(
            "/api/v1/risk/predictions", params={"sensor_id": "S001"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert all(p["sensor_id"] == "S001" for p in data["predictions"])

    async def test_filter_by_zone(self, seeded_client):
        client, _ = seeded_client
        resp = await client.get(
            "/api/v1/risk/predictions", params={"zone_id": "ZONE_B"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    async def test_filter_by_risk_level(self, seeded_client):
        client, _ = seeded_client
        resp = await client.get(
            "/api/v1/risk/predictions", params={"risk_level": "HIGH"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert all(p["risk_level"] == "HIGH" for p in data["predictions"])

    async def test_pagination(self, seeded_client):
        client, _ = seeded_client
        resp = await client.get(
            "/api/v1/risk/predictions", params={"offset": 0, "limit": 2},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 2
        assert data["total"] == 5
        assert data["offset"] == 0
        assert data["limit"] == 2

    async def test_pagination_offset(self, seeded_client):
        client, _ = seeded_client
        page1 = await client.get(
            "/api/v1/risk/predictions", params={"offset": 0, "limit": 3},
        )
        page2 = await client.get(
            "/api/v1/risk/predictions", params={"offset": 3, "limit": 3},
        )
        ids1 = {p["id"] for p in page1.json()["predictions"]}
        ids2 = {p["id"] for p in page2.json()["predictions"]}
        assert ids1.isdisjoint(ids2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. GET /risk/predictions/latest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLatestPrediction:
    async def test_no_predictions_404(self, risk_client: AsyncClient):
        resp = await risk_client.get("/api/v1/risk/predictions/latest")
        assert resp.status_code == 404

    async def test_returns_latest(self, seeded_client):
        client, _ = seeded_client
        resp = await client.get("/api/v1/risk/predictions/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        pred = data["prediction"]
        # The latest by timestamp is the 5th one (14:00)
        assert pred["predicted_risk_score"] == 50

    async def test_latest_by_sensor(self, seeded_client):
        client, _ = seeded_client
        resp = await client.get(
            "/api/v1/risk/predictions/latest", params={"sensor_id": "S001"},
        )
        assert resp.status_code == 200
        pred = resp.json()["prediction"]
        assert pred["sensor_id"] == "S001"
        # Latest S001 is the 3rd one (12:00, score=30)
        assert pred["predicted_risk_score"] == 30

    async def test_latest_by_zone(self, seeded_client):
        client, _ = seeded_client
        resp = await client.get(
            "/api/v1/risk/predictions/latest", params={"zone_id": "ZONE_B"},
        )
        assert resp.status_code == 200
        pred = resp.json()["prediction"]
        assert pred["zone_id"] == "ZONE_B"

    async def test_latest_nonexistent_sensor_404(self, seeded_client):
        client, _ = seeded_client
        resp = await client.get(
            "/api/v1/risk/predictions/latest", params={"sensor_id": "SXXX"},
        )
        assert resp.status_code == 404


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. End-to-end flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEndToEndFlow:
    async def test_predict_then_retrieve(self, risk_client: AsyncClient):
        """Full flow: predict → get latest → get history."""
        # 1. Predict
        predict_resp = await risk_client.post(
            "/api/v1/risk/predict",
            json={
                "sensor_id": "S001",
                "zone_id": "ZONE_A",
                "features": {"Temp": 85.0, "Pressure": 22.0, "Humidity": 55.0,
                             "Gas": 110.0, "Vibration": 3.5, "Speed": 120.0},
            },
        )
        assert predict_resp.status_code == 201
        pred_id = predict_resp.json()["prediction"]["id"]

        # 2. Get latest
        latest_resp = await risk_client.get(
            "/api/v1/risk/predictions/latest", params={"sensor_id": "S001"},
        )
        assert latest_resp.status_code == 200
        assert latest_resp.json()["prediction"]["id"] == pred_id

        # 3. Get history
        history_resp = await risk_client.get(
            "/api/v1/risk/predictions", params={"sensor_id": "S001"},
        )
        assert history_resp.status_code == 200
        assert history_resp.json()["total"] >= 1
