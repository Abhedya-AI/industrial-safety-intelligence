"""End-to-end integration tests for the complete Risk Prediction pipeline.

Verifies the full data flow:

  Sensor Reading → Statistics → Anomaly Detection → Feature Builder →
  Risk Prediction Service → Database Persistence → API Response

Scenarios tested:
  1.  Normal conditions — low-risk prediction through entire pipeline
  2.  High-risk scenarios — elevated readings produce HIGH/CRITICAL risk
  3.  Missing data — graceful handling of absent features
  4.  Invalid data — NaN/Inf rejection, schema validation
  5.  Model loading failures — 503 graceful degradation
  6.  Prediction failures — scaler/inference errors
  7.  Feature builder → service integration — structured input path
  8.  Monitoring integration — stats tracking through pipeline
  9.  Pagination and filtering — query pipeline via API
  10. End-to-end API flow — predict → retrieve → history → latest
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Dict, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.risk_prediction.domain.exceptions import (
    InsufficientFeaturesError,
    RiskModelNotLoadedError,
    RiskPredictionFailedError,
)
from app.risk_prediction.domain.value_objects import PredictionStatus, RiskLevel
from app.risk_prediction.models.risk_prediction_model import RiskPredictionModel
from app.risk_prediction.preprocessing.feature_builder import (
    RiskFeatureInput,
    build_risk_feature_vector,
    build_risk_features_from_dict,
    build_risk_features_from_input,
    validate_features,
    SENSOR_FEATURES,
)
from app.risk_prediction.repositories.sqlalchemy_risk_prediction_repo import (
    SQLAlchemyRiskPredictionRepository,
)
from app.risk_prediction.services.monitoring_service import (
    RiskPredictionMonitoringService,
)
from app.risk_prediction.services.risk_prediction_service import (
    MODEL_NAME,
    MODEL_VERSION,
    RiskPredictionService,
    _ModelHolder,
    classify_risk_level,
    probability_to_risk_score,
    calculate_confidence,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mock factories
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_mock_model(proba: float = 0.65):
    """Create a mock XGBoost model returning a fixed probability."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1.0 - proba, proba]])
    model.predict.return_value = np.array([1 if proba >= 0.5 else 0])
    return model


def _make_mock_scaler():
    """Create a mock scaler that returns zeros (no transform)."""
    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, 8))
    return scaler


def _make_holder(proba: float = 0.65) -> _ModelHolder:
    """Create a pre-loaded _ModelHolder with mocks."""
    holder = _ModelHolder()
    holder.model = _make_mock_model(proba)
    holder.scaler = _make_mock_scaler()
    holder.column_order = [f"feature_{i}" for i in range(8)]
    holder.feature_importances = {
        "Temp": 0.30, "Service_Days": 0.12, "Pressure": 0.08,
        "Age": 0.03, "Gas": 0.02, "Humidity": 0.01,
    }
    holder._loaded = True
    return holder


def _make_broken_holder() -> _ModelHolder:
    """Model holder that fails on load."""
    holder = _ModelHolder()
    return holder  # _loaded = False, no model


def _make_failing_scaler_holder() -> _ModelHolder:
    """Model holder whose scaler raises ValueError."""
    holder = _make_holder()
    holder.scaler.transform.side_effect = ValueError("Feature dimension mismatch")
    return holder


def _make_failing_model_holder() -> _ModelHolder:
    """Model holder whose model.predict_proba raises RuntimeError."""
    holder = _make_holder()
    holder.model.predict_proba.side_effect = RuntimeError("XGBoost inference crash")
    return holder


# ── Feature helpers ──


def _normal_features() -> Dict[str, float]:
    """Normal operating conditions — low risk expected."""
    return {
        "Temp": 45.0, "Pressure": 12.0, "Humidity": 50.0,
        "Gas": 20.0, "Vibration": 1.0, "Speed": 80.0,
        "Sparks": 0.0, "Workers": 5.0, "Age": 35.0,
        "Service_Days": 200.0,
    }


def _high_risk_features() -> Dict[str, float]:
    """Extreme conditions — high risk expected."""
    return {
        "Temp": 350.0, "Pressure": 95.0, "Humidity": 95.0,
        "Gas": 500.0, "Vibration": 25.0, "Speed": 500.0,
        "Sparks": 10.0, "Workers": 25.0, "Age": 60.0,
        "Service_Days": 3000.0,
        "anomaly_score_if": 0.95, "anomaly_score_ae": 0.90,
        "sensor_health_score": 15.0,
    }


def _partial_features() -> Dict[str, float]:
    """Only core sensor features — everything else uses defaults."""
    return {"Temp": 70.0, "Pressure": 18.0, "Humidity": 55.0,
            "Gas": 60.0, "Vibration": 2.0, "Speed": 100.0}


# ── Fixtures ──


@pytest_asyncio.fixture
async def risk_repo(db_session: AsyncSession) -> SQLAlchemyRiskPredictionRepository:
    return SQLAlchemyRiskPredictionRepository(db_session)


@pytest_asyncio.fixture
async def service_normal(
    db_session: AsyncSession,
) -> RiskPredictionService:
    """Service with model returning P(accident)=0.15 → LOW risk."""
    repo = SQLAlchemyRiskPredictionRepository(db_session)
    return RiskPredictionService(repo, model_holder=_make_holder(proba=0.15))


@pytest_asyncio.fixture
async def service_high(
    db_session: AsyncSession,
) -> RiskPredictionService:
    """Service with model returning P(accident)=0.82 → CRITICAL risk."""
    repo = SQLAlchemyRiskPredictionRepository(db_session)
    return RiskPredictionService(repo, model_holder=_make_holder(proba=0.82))


@pytest_asyncio.fixture
async def service_broken(
    db_session: AsyncSession,
) -> RiskPredictionService:
    """Service with no model loaded."""
    repo = SQLAlchemyRiskPredictionRepository(db_session)
    return RiskPredictionService(
        repo, model_holder=_make_broken_holder(),
        model_path="/nonexistent/model.pkl",
    )


@pytest_asyncio.fixture
async def api_client_low(
    db_session: AsyncSession,
) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client with model returning LOW risk."""
    yield await _build_api_client(db_session, proba=0.15)


@pytest_asyncio.fixture
async def api_client_critical(
    db_session: AsyncSession,
) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client with model returning CRITICAL risk."""
    yield await _build_api_client(db_session, proba=0.90)


@pytest_asyncio.fixture
async def api_client_broken(
    db_session: AsyncSession,
) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client with model that fails to load."""
    from app.main import app
    from app.shared.database.connection import get_async_session
    from app.core.dependencies import get_risk_prediction_service

    async def override_session():
        yield db_session

    def override_service(session=None):
        repo = SQLAlchemyRiskPredictionRepository(db_session)
        return RiskPredictionService(
            repo, model_holder=_make_broken_holder(),
            model_path="/nonexistent/model.pkl",
        )

    app.dependency_overrides[get_async_session] = override_session
    app.dependency_overrides[get_risk_prediction_service] = override_service

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


async def _build_api_client(
    db_session: AsyncSession, proba: float,
) -> AsyncClient:
    from app.main import app
    from app.shared.database.connection import get_async_session
    from app.core.dependencies import get_risk_prediction_service

    holder = _make_holder(proba)

    async def override_session():
        yield db_session

    def override_service(session=None):
        repo = SQLAlchemyRiskPredictionRepository(db_session)
        return RiskPredictionService(repo, model_holder=holder)

    app.dependency_overrides[get_async_session] = override_session
    app.dependency_overrides[get_risk_prediction_service] = override_service

    return AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Normal conditions — full pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNormalConditionsPipeline:
    """Normal readings → feature builder → model → LOW/MEDIUM risk → DB → response."""

    async def test_feature_build_produces_valid_dict(self):
        features = build_risk_features_from_dict(_normal_features())
        issues = validate_features(features)
        assert issues == [], f"Validation issues: {issues}"

    async def test_feature_input_to_dict_roundtrip(self):
        inp = RiskFeatureInput(
            temperature=45.0, pressure=12.0, humidity=50.0,
            gas=20.0, vibration=1.0, speed=80.0,
        )
        features = build_risk_features_from_input(inp)
        issues = validate_features(features)
        assert issues == []

    async def test_normal_prediction_low_risk(self, service_normal):
        result = await service_normal.predict_from_features(
            _normal_features(), sensor_id="S001", zone_id="ZONE_A",
        )
        assert result.risk_level == RiskLevel.LOW.value
        assert result.predicted_risk_score < 25
        assert result.confidence_score > 0
        assert result.status == PredictionStatus.COMPLETED.value

    async def test_prediction_persisted_to_db(self, service_normal, risk_repo):
        result = await service_normal.predict_from_features(
            _normal_features(), sensor_id="S001",
        )
        stored = await risk_repo.get_prediction(result.id)
        assert stored is not None
        assert stored.id == result.id
        assert stored.accident_probability == result.accident_probability

    async def test_latest_returns_most_recent(self, service_normal, risk_repo):
        # Make two predictions
        await service_normal.predict_from_features(
            _normal_features(), sensor_id="S001",
        )
        second = await service_normal.predict_from_features(
            _normal_features(), sensor_id="S001",
        )
        latest = await risk_repo.get_latest_prediction(sensor_id="S001")
        assert latest is not None
        assert latest.id == second.id

    async def test_history_populates(self, service_normal, risk_repo):
        for _ in range(3):
            await service_normal.predict_from_features(
                _normal_features(), sensor_id="S001",
            )
        history = await risk_repo.get_prediction_history(sensor_id="S001")
        assert len(history) == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. High-risk scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHighRiskScenarios:
    """Extreme readings → model produces HIGH/CRITICAL risk."""

    async def test_critical_risk_prediction(self, service_high):
        result = await service_high.predict_from_features(
            _high_risk_features(), sensor_id="S-DANGER",
        )
        assert result.risk_level == RiskLevel.CRITICAL.value
        assert result.predicted_risk_score >= 75
        assert result.accident_probability >= 0.75

    async def test_critical_with_explanation(self, service_high):
        result = await service_high.predict_from_features(
            _high_risk_features(),
            include_explanation=True,
        )
        assert result.explanation is not None
        assert "immediate action" in result.explanation.lower()

    async def test_critical_with_breakdown(self, service_high):
        result = await service_high.predict_from_features(
            _high_risk_features(),
            include_breakdown=True,
        )
        assert result.top_contributing_factors is not None
        factors = json.loads(result.top_contributing_factors)
        assert len(factors) > 0
        # Top factor should be Temp (highest importance in our mock)
        assert factors[0]["factor"] == "Temp"

    async def test_high_risk_with_anomaly_enrichment(self, service_high):
        """Features with high anomaly scores pass through cleanly."""
        features = _high_risk_features()
        assert features["anomaly_score_if"] == 0.95
        result = await service_high.predict_from_features(features)
        assert result.risk_level == RiskLevel.CRITICAL.value

    async def test_multiple_high_risk_sensors(self, service_high, risk_repo):
        """Multiple sensors each get independent predictions."""
        ids = []
        for sid in ["S-A", "S-B", "S-C"]:
            r = await service_high.predict_from_features(
                _high_risk_features(), sensor_id=sid,
            )
            ids.append(r.id)
        assert len(set(ids)) == 3  # All unique prediction IDs

        for sid in ["S-A", "S-B", "S-C"]:
            latest = await risk_repo.get_latest_prediction(sensor_id=sid)
            assert latest is not None
            assert latest.sensor_id == sid


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Missing data handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMissingDataHandling:
    """Pipeline handles missing features gracefully via defaults."""

    async def test_partial_features_filled_with_defaults(self):
        features = build_risk_features_from_dict(_partial_features())
        # Defaults should have been applied
        assert "Workers" in features
        assert "anomaly_score_if" in features
        assert "sensor_health_score" in features

    async def test_partial_features_pass_validation(self):
        features = build_risk_features_from_dict(_partial_features())
        issues = validate_features(features)
        assert issues == []

    async def test_partial_features_produce_prediction(self, service_normal):
        result = await service_normal.predict_from_features(
            _partial_features(),
        )
        assert result.risk_level in [r.value for r in RiskLevel]
        assert result.status == PredictionStatus.COMPLETED.value

    async def test_empty_dict_fills_all_defaults(self):
        features = build_risk_features_from_dict({})
        # All sensor features should have been set to 0.0
        for col in SENSOR_FEATURES:
            assert col in features

    async def test_empty_dict_still_predicts(self, service_normal):
        """Even an empty feature dict → defaults → prediction."""
        result = await service_normal.predict_from_features({})
        assert result.status == PredictionStatus.COMPLETED.value

    async def test_risk_feature_input_defaults(self):
        """Default RiskFeatureInput produces valid features."""
        inp = RiskFeatureInput()  # All defaults
        features = build_risk_features_from_input(inp)
        issues = validate_features(features)
        assert issues == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Invalid data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInvalidDataHandling:
    """NaN, Inf, and invalid values are caught."""

    async def test_nan_feature_detected_by_validation(self):
        features = _normal_features()
        features["Temp"] = float("nan")
        issues = validate_features(features)
        assert "Temp" in issues

    async def test_inf_feature_detected_by_validation(self):
        features = _normal_features()
        features["Pressure"] = float("inf")
        issues = validate_features(features)
        assert "Pressure" in issues

    async def test_nan_feature_raises_insufficient(self, service_normal):
        features = {k: float("nan") for k in SENSOR_FEATURES}
        with pytest.raises(InsufficientFeaturesError):
            await service_normal.predict_from_features(features)

    async def test_mixed_valid_nan_raises(self, service_normal):
        features = _normal_features()
        features["Temp"] = float("nan")
        with pytest.raises(InsufficientFeaturesError):
            await service_normal.predict_from_features(features)

    async def test_feature_vector_handles_missing_columns(self):
        """build_risk_feature_vector defaults missing columns to 0.0."""
        features = {"A": 1.0, "B": 2.0}
        vector = build_risk_feature_vector(features, ["A", "B", "C", "D"])
        assert vector.shape == (4,)
        assert vector[2] == 0.0  # C missing → 0.0
        assert vector[3] == 0.0  # D missing → 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Model loading failures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestModelLoadingFailures:
    """Model not available → service raises → API returns 503."""

    async def test_service_raises_model_not_loaded(self, service_broken):
        with pytest.raises(RiskModelNotLoadedError):
            await service_broken.predict_from_features(_normal_features())

    async def test_api_returns_503(self, api_client_broken):
        resp = await api_client_broken.post(
            "/api/v1/risk/predict",
            json={"features": _normal_features()},
        )
        assert resp.status_code == 503

    async def test_503_has_detail(self, api_client_broken):
        resp = await api_client_broken.post(
            "/api/v1/risk/predict",
            json={"features": _normal_features()},
        )
        assert "not loaded" in resp.json()["detail"].lower()

    async def test_history_still_works_when_model_broken(self, api_client_broken):
        """Read endpoints should work even if model is unavailable."""
        resp = await api_client_broken.get("/api/v1/risk/predictions")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Prediction failures (scaler/inference errors)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPredictionFailures:
    """Scaler or model inference errors → RiskPredictionFailedError → 500."""

    async def test_scaler_failure(self, db_session):
        repo = SQLAlchemyRiskPredictionRepository(db_session)
        service = RiskPredictionService(
            repo, model_holder=_make_failing_scaler_holder(),
        )
        with pytest.raises(RiskPredictionFailedError, match="scaling"):
            await service.predict_from_features(_normal_features())

    async def test_model_inference_failure(self, db_session):
        repo = SQLAlchemyRiskPredictionRepository(db_session)
        service = RiskPredictionService(
            repo, model_holder=_make_failing_model_holder(),
        )
        with pytest.raises(RiskPredictionFailedError, match="inference"):
            await service.predict_from_features(_normal_features())

    async def test_scaler_failure_via_api(self, db_session):
        from app.main import app
        from app.shared.database.connection import get_async_session
        from app.core.dependencies import get_risk_prediction_service

        async def override_session():
            yield db_session

        def override_service(session=None):
            repo = SQLAlchemyRiskPredictionRepository(db_session)
            return RiskPredictionService(
                repo, model_holder=_make_failing_scaler_holder(),
            )

        app.dependency_overrides[get_async_session] = override_session
        app.dependency_overrides[get_risk_prediction_service] = override_service

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver",
        ) as ac:
            resp = await ac.post(
                "/api/v1/risk/predict",
                json={"features": _normal_features()},
            )
        app.dependency_overrides.clear()
        assert resp.status_code == 500
        assert "scaling" in resp.json()["detail"].lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Feature builder → service integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFeatureBuilderServiceIntegration:
    """Structured RiskFeatureInput → predict_from_input → DB."""

    async def test_predict_from_input(self, service_normal, risk_repo):
        inp = RiskFeatureInput(
            temperature=45.0, pressure=12.0, humidity=50.0,
            gas=20.0, vibration=1.0, speed=80.0,
            workers=5, training=True, experience="Senior",
            zone_id="ZONE_B",
        )
        result = await service_normal.predict_from_input(
            inp, sensor_id="S001",
        )
        assert result.zone_id == "ZONE_B"
        assert result.sensor_id == "S001"
        stored = await risk_repo.get_prediction(result.id)
        assert stored is not None

    async def test_predict_from_input_high_risk(self, service_high):
        inp = RiskFeatureInput(
            temperature=350.0, pressure=95.0, humidity=95.0,
            gas=500.0, vibration=25.0, speed=500.0,
            anomaly_score_if=0.95, anomaly_score_ae=0.90,
            sensor_health_score=15.0,
        )
        result = await service_high.predict_from_input(inp)
        assert result.risk_level == RiskLevel.CRITICAL.value

    async def test_all_context_fields_propagated(self, service_normal):
        inp = RiskFeatureInput()
        result = await service_normal.predict_from_input(
            inp,
            sensor_id="S-42",
            equipment_id="EQ-007",
            zone_id="ZONE_C",
        )
        assert result.sensor_id == "S-42"
        assert result.equipment_id == "EQ-007"
        assert result.zone_id == "ZONE_C"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Monitoring integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMonitoringIntegration:
    """Monitoring service tracks predictions from the pipeline."""

    async def test_monitor_records_predictions(self):
        monitor = RiskPredictionMonitoringService()
        monitor.record_prediction(
            probability=0.15, risk_level=RiskLevel.LOW,
            confidence=0.85, latency_ms=8.5,
            features=_normal_features(), sensor_id="S001",
        )
        monitor.record_prediction(
            probability=0.82, risk_level=RiskLevel.CRITICAL,
            confidence=0.95, latency_ms=12.0,
            features=_high_risk_features(), sensor_id="S002",
        )
        assert monitor.prediction_count == 2
        dist = monitor.get_distribution()
        assert dist["counts"]["low"] == 1
        assert dist["counts"]["critical"] == 1

    async def test_monitor_drift_detection(self):
        monitor = RiskPredictionMonitoringService()
        monitor.set_drift_baseline(
            means={"Temp": 50.0, "Pressure": 15.0},
            stds={"Temp": 5.0, "Pressure": 2.0},
        )
        # Record shifted data
        for _ in range(20):
            monitor.record_prediction(
                probability=0.5, risk_level=RiskLevel.HIGH,
                confidence=0.8, latency_ms=10.0,
                features={"Temp": 100.0, "Pressure": 15.5},
            )
        report = monitor.get_drift_report()
        assert report["has_significant_drift"] is True
        assert report["top_drift"][0]["feature"] == "Temp"

    async def test_monitor_summary(self):
        monitor = RiskPredictionMonitoringService()
        for prob, level in [(0.1, RiskLevel.LOW), (0.8, RiskLevel.CRITICAL)]:
            monitor.record_prediction(
                probability=prob, risk_level=level,
                confidence=0.9, latency_ms=10.0,
            )
        summary = monitor.get_summary()
        assert summary["prediction_count"] == 2
        assert summary["avg_confidence"] == pytest.approx(0.9)
        assert summary["model_version"] == "1.0.0"

    async def test_monitor_integrates_with_si_framework(self):
        from app.sensor_intelligence.services.model_monitoring_service import (
            ModelMonitoringService,
        )
        mm = ModelMonitoringService()
        monitor = RiskPredictionMonitoringService(model_monitoring=mm)
        monitor.record_prediction(
            probability=0.7, risk_level=RiskLevel.HIGH,
            confidence=0.85, latency_ms=12.0,
            sensor_id="S001",
        )
        stats = mm.get_inference_stats("xgboost_risk_prediction")
        assert stats.prediction_count == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. API pagination and filtering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAPIPaginationAndFiltering:
    """Verify query pipeline: filters, pagination, ordering."""

    async def _seed_predictions(
        self, db_session: AsyncSession, count: int = 10,
    ) -> list[str]:
        repo = SQLAlchemyRiskPredictionRepository(db_session)
        ids = []
        levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        for i in range(count):
            pred = RiskPredictionModel(
                id=str(uuid.uuid4()),
                sensor_id=f"S00{i % 3 + 1}",
                zone_id=f"ZONE_{'A' if i < 5 else 'B'}",
                equipment_id="EQ-001",
                prediction_timestamp=datetime(
                    2026, 7, 1, 10 + i, 0, tzinfo=timezone.utc,
                ),
                accident_probability=0.1 * (i + 1),
                predicted_risk_score=10 * (i + 1),
                risk_level=levels[i % 4],
                confidence_score=0.9,
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                status="COMPLETED",
            )
            await repo.create_prediction(pred)
            ids.append(pred.id)
        await db_session.commit()
        return ids

    async def test_pagination_limit(self, db_session):
        from app.main import app
        from app.shared.database.connection import get_async_session
        from app.core.dependencies import get_risk_prediction_service

        await self._seed_predictions(db_session, count=10)

        async def override_session():
            yield db_session

        def override_service(session=None):
            repo = SQLAlchemyRiskPredictionRepository(db_session)
            return RiskPredictionService(repo, model_holder=_make_holder())

        app.dependency_overrides[get_async_session] = override_session
        app.dependency_overrides[get_risk_prediction_service] = override_service

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver",
        ) as ac:
            resp = await ac.get(
                "/api/v1/risk/predictions",
                params={"limit": 3, "offset": 0},
            )

        app.dependency_overrides.clear()
        data = resp.json()
        assert resp.status_code == 200
        assert len(data["predictions"]) == 3
        assert data["total"] == 10

    async def test_filter_by_risk_level(self, db_session):
        from app.main import app
        from app.shared.database.connection import get_async_session
        from app.core.dependencies import get_risk_prediction_service

        await self._seed_predictions(db_session, count=8)

        async def override_session():
            yield db_session

        def override_service(session=None):
            repo = SQLAlchemyRiskPredictionRepository(db_session)
            return RiskPredictionService(repo, model_holder=_make_holder())

        app.dependency_overrides[get_async_session] = override_session
        app.dependency_overrides[get_risk_prediction_service] = override_service

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver",
        ) as ac:
            resp = await ac.get(
                "/api/v1/risk/predictions",
                params={"risk_level": "HIGH"},
            )

        app.dependency_overrides.clear()
        data = resp.json()
        assert resp.status_code == 200
        assert all(p["risk_level"] == "HIGH" for p in data["predictions"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. End-to-end API flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEndToEndAPIFlow:
    """Full API lifecycle: predict → retrieve → history → latest."""

    async def test_full_lifecycle_low_risk(self, api_client_low):
        ac = api_client_low

        # 1. Predict — normal conditions
        resp = await ac.post(
            "/api/v1/risk/predict",
            json={
                "sensor_id": "S001", "zone_id": "ZONE_A",
                "features": _normal_features(),
            },
        )
        assert resp.status_code == 201
        pred = resp.json()["prediction"]
        assert pred["risk_level"] == "LOW"
        pred_id = pred["id"]

        # 2. Get latest
        resp = await ac.get(
            "/api/v1/risk/predictions/latest",
            params={"sensor_id": "S001"},
        )
        assert resp.status_code == 200
        assert resp.json()["prediction"]["id"] == pred_id

        # 3. History
        resp = await ac.get(
            "/api/v1/risk/predictions",
            params={"sensor_id": "S001"},
        )
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    async def test_full_lifecycle_critical_risk(self, api_client_critical):
        ac = api_client_critical

        # Predict — extreme conditions
        resp = await ac.post(
            "/api/v1/risk/predict",
            json={
                "sensor_id": "S-DANGER", "zone_id": "ZONE_B",
                "features": _high_risk_features(),
                "include_explanation": True,
                "include_breakdown": True,
            },
        )
        assert resp.status_code == 201
        pred = resp.json()["prediction"]
        assert pred["risk_level"] == "CRITICAL"
        assert pred["predicted_risk_score"] >= 75
        assert pred["explanation"] is not None
        assert pred["top_contributing_factors"] is not None
        assert "immediate action" in pred["explanation"].lower()

    async def test_multiple_predictions_then_query(self, api_client_low):
        ac = api_client_low

        # Make predictions for different sensors
        for sid in ["S-A", "S-B", "S-C"]:
            await ac.post(
                "/api/v1/risk/predict",
                json={"sensor_id": sid, "features": _normal_features()},
            )

        # Query all
        resp = await ac.get("/api/v1/risk/predictions")
        assert resp.json()["total"] == 3

        # Query by sensor
        resp = await ac.get(
            "/api/v1/risk/predictions",
            params={"sensor_id": "S-B"},
        )
        assert resp.json()["total"] == 1
        assert resp.json()["predictions"][0]["sensor_id"] == "S-B"

    async def test_latest_returns_404_before_any_predictions(self, api_client_low):
        ac = api_client_low
        resp = await ac.get(
            "/api/v1/risk/predictions/latest",
            params={"sensor_id": "NON_EXISTENT"},
        )
        assert resp.status_code == 404

    async def test_predict_with_minimal_features(self, api_client_low):
        ac = api_client_low
        resp = await ac.post(
            "/api/v1/risk/predict",
            json={"features": _partial_features()},
        )
        assert resp.status_code == 201
        assert resp.json()["success"] is True
