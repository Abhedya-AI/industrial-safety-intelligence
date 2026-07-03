"""Integration tests for the Risk Prediction repository layer.

Uses the shared ``db_session`` fixture (in-memory async SQLite) so every
test gets a fresh, isolated database.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.risk_prediction.models.risk_prediction_model import RiskPredictionModel
from app.risk_prediction.repositories.sqlalchemy_risk_prediction_repo import (
    SQLAlchemyRiskPredictionRepository,
)


# ── Fixtures ──


@pytest_asyncio.fixture
async def repo(db_session: AsyncSession) -> SQLAlchemyRiskPredictionRepository:
    """Provide a fresh repository instance backed by the test session."""
    return SQLAlchemyRiskPredictionRepository(db_session)


def _make_prediction(
    *,
    sensor_id: str = "S001",
    zone_id: str = "ZONE_A",
    equipment_id: str = "EQ-001",
    risk_level: str = "HIGH",
    accident_probability: float = 0.65,
    predicted_risk_score: int = 72,
    confidence_score: float = 0.88,
    model_name: str = "xgboost_ensemble",
    model_version: str = "1.0.0",
    prediction_timestamp: datetime | None = None,
    status: str = "COMPLETED",
) -> RiskPredictionModel:
    """Factory helper for creating test prediction models."""
    return RiskPredictionModel(
        id=str(uuid.uuid4()),
        sensor_id=sensor_id,
        zone_id=zone_id,
        equipment_id=equipment_id,
        prediction_timestamp=prediction_timestamp or datetime.now(timezone.utc),
        accident_probability=accident_probability,
        predicted_risk_score=predicted_risk_score,
        risk_level=risk_level,
        confidence_score=confidence_score,
        model_name=model_name,
        model_version=model_version,
        status=status,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. create_prediction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCreatePrediction:
    async def test_creates_and_returns(self, repo):
        pred = _make_prediction()
        result = await repo.create_prediction(pred)
        assert result.id == pred.id
        assert result.risk_level == "HIGH"
        assert result.predicted_risk_score == 72
        assert result.created_at is not None

    async def test_server_defaults_populated(self, repo):
        pred = _make_prediction()
        result = await repo.create_prediction(pred)
        # created_at has server_default=func.now()
        assert result.created_at is not None

    async def test_all_fields_persisted(self, repo):
        pred = _make_prediction(
            sensor_id="S999",
            zone_id="ZONE_X",
            equipment_id="EQ-X",
            risk_level="CRITICAL",
            accident_probability=0.92,
            predicted_risk_score=95,
            confidence_score=0.99,
            model_name="autoencoder_risk",
            model_version="3.0.0",
            status="COMPLETED",
        )
        result = await repo.create_prediction(pred)
        assert result.sensor_id == "S999"
        assert result.zone_id == "ZONE_X"
        assert result.equipment_id == "EQ-X"
        assert result.risk_level == "CRITICAL"
        assert result.accident_probability == 0.92
        assert result.predicted_risk_score == 95
        assert result.confidence_score == 0.99
        assert result.model_name == "autoencoder_risk"
        assert result.model_version == "3.0.0"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. get_prediction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGetPrediction:
    async def test_returns_by_id(self, repo):
        pred = await repo.create_prediction(_make_prediction())
        fetched = await repo.get_prediction(pred.id)
        assert fetched is not None
        assert fetched.id == pred.id
        assert fetched.risk_level == pred.risk_level

    async def test_returns_none_for_unknown_id(self, repo):
        result = await repo.get_prediction("nonexistent-id")
        assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. get_latest_prediction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGetLatestPrediction:
    async def test_returns_most_recent(self, repo):
        t1 = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
        await repo.create_prediction(
            _make_prediction(sensor_id="S001", prediction_timestamp=t1, risk_level="LOW"),
        )
        await repo.create_prediction(
            _make_prediction(sensor_id="S001", prediction_timestamp=t2, risk_level="HIGH"),
        )
        latest = await repo.get_latest_prediction(sensor_id="S001")
        assert latest is not None
        assert latest.risk_level == "HIGH"
        # SQLite strips tz info; compare by value
        assert latest.prediction_timestamp.replace(tzinfo=timezone.utc) == t2

    async def test_filters_by_sensor(self, repo):
        t = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
        await repo.create_prediction(
            _make_prediction(sensor_id="S001", prediction_timestamp=t),
        )
        await repo.create_prediction(
            _make_prediction(sensor_id="S002", prediction_timestamp=t),
        )
        latest = await repo.get_latest_prediction(sensor_id="S002")
        assert latest is not None
        assert latest.sensor_id == "S002"

    async def test_filters_by_zone(self, repo):
        t = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
        await repo.create_prediction(
            _make_prediction(zone_id="ZONE_A", prediction_timestamp=t),
        )
        await repo.create_prediction(
            _make_prediction(zone_id="ZONE_B", prediction_timestamp=t),
        )
        latest = await repo.get_latest_prediction(zone_id="ZONE_B")
        assert latest is not None
        assert latest.zone_id == "ZONE_B"

    async def test_returns_none_if_empty(self, repo):
        result = await repo.get_latest_prediction(sensor_id="S999")
        assert result is None

    async def test_no_filter_returns_global_latest(self, repo):
        t1 = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
        await repo.create_prediction(
            _make_prediction(sensor_id="S001", prediction_timestamp=t1),
        )
        await repo.create_prediction(
            _make_prediction(sensor_id="S002", prediction_timestamp=t2),
        )
        latest = await repo.get_latest_prediction()
        assert latest is not None
        assert latest.sensor_id == "S002"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. get_prediction_history
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGetPredictionHistory:
    async def _seed_history(self, repo, count: int = 5, sensor_id: str = "S001"):
        """Create N predictions spread across time."""
        base = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
        preds = []
        for i in range(count):
            pred = await repo.create_prediction(
                _make_prediction(
                    sensor_id=sensor_id,
                    prediction_timestamp=base + timedelta(hours=i),
                    predicted_risk_score=10 * (i + 1),
                    risk_level=["LOW", "MEDIUM", "HIGH", "CRITICAL", "HIGH"][i % 5],
                ),
            )
            preds.append(pred)
        return preds

    async def test_returns_all_unfiltered(self, repo):
        await self._seed_history(repo, 5)
        history = await repo.get_prediction_history()
        assert len(history) == 5

    async def test_ordered_newest_first(self, repo):
        await self._seed_history(repo, 3)
        history = await repo.get_prediction_history()
        timestamps = [p.prediction_timestamp for p in history]
        assert timestamps == sorted(timestamps, reverse=True)

    async def test_filter_by_sensor_id(self, repo):
        await self._seed_history(repo, 3, sensor_id="S001")
        await self._seed_history(repo, 2, sensor_id="S002")
        history = await repo.get_prediction_history(sensor_id="S001")
        assert len(history) == 3
        assert all(p.sensor_id == "S001" for p in history)

    async def test_filter_by_zone_id(self, repo):
        await repo.create_prediction(
            _make_prediction(zone_id="ZONE_A"),
        )
        await repo.create_prediction(
            _make_prediction(zone_id="ZONE_B"),
        )
        history = await repo.get_prediction_history(zone_id="ZONE_A")
        assert len(history) == 1
        assert history[0].zone_id == "ZONE_A"

    async def test_filter_by_risk_level(self, repo):
        await self._seed_history(repo, 5)
        history = await repo.get_prediction_history(risk_level="CRITICAL")
        assert all(p.risk_level == "CRITICAL" for p in history)

    async def test_filter_by_time_range(self, repo):
        preds = await self._seed_history(repo, 5)
        start = preds[1].prediction_timestamp
        end = preds[3].prediction_timestamp
        history = await repo.get_prediction_history(
            start_time=start, end_time=end,
        )
        for p in history:
            assert start <= p.prediction_timestamp <= end

    async def test_pagination_offset_limit(self, repo):
        await self._seed_history(repo, 10)
        page1 = await repo.get_prediction_history(offset=0, limit=3)
        page2 = await repo.get_prediction_history(offset=3, limit=3)
        assert len(page1) == 3
        assert len(page2) == 3
        # No overlap
        ids1 = {p.id for p in page1}
        ids2 = {p.id for p in page2}
        assert ids1.isdisjoint(ids2)

    async def test_empty_history(self, repo):
        history = await repo.get_prediction_history(sensor_id="S999")
        assert history == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. count_predictions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCountPredictions:
    async def test_counts_all(self, repo):
        for _ in range(4):
            await repo.create_prediction(_make_prediction())
        assert await repo.count_predictions() == 4

    async def test_counts_filtered(self, repo):
        await repo.create_prediction(_make_prediction(sensor_id="S001"))
        await repo.create_prediction(_make_prediction(sensor_id="S001"))
        await repo.create_prediction(_make_prediction(sensor_id="S002"))
        assert await repo.count_predictions(sensor_id="S001") == 2

    async def test_counts_zero_when_empty(self, repo):
        assert await repo.count_predictions() == 0

    async def test_counts_by_risk_level(self, repo):
        await repo.create_prediction(_make_prediction(risk_level="LOW"))
        await repo.create_prediction(_make_prediction(risk_level="LOW"))
        await repo.create_prediction(_make_prediction(risk_level="CRITICAL"))
        assert await repo.count_predictions(risk_level="LOW") == 2
        assert await repo.count_predictions(risk_level="CRITICAL") == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. delete_prediction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeletePrediction:
    async def test_deletes_existing(self, repo):
        pred = await repo.create_prediction(_make_prediction())
        deleted = await repo.delete_prediction(pred.id)
        assert deleted is True
        assert await repo.get_prediction(pred.id) is None

    async def test_returns_false_for_nonexistent(self, repo):
        deleted = await repo.delete_prediction("nonexistent-id")
        assert deleted is False

    async def test_does_not_affect_others(self, repo):
        pred1 = await repo.create_prediction(_make_prediction())
        pred2 = await repo.create_prediction(_make_prediction())
        await repo.delete_prediction(pred1.id)
        assert await repo.get_prediction(pred2.id) is not None
        assert await repo.count_predictions() == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. delete_prediction_history
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeletePredictionHistory:
    async def test_delete_by_sensor(self, repo):
        await repo.create_prediction(_make_prediction(sensor_id="S001"))
        await repo.create_prediction(_make_prediction(sensor_id="S001"))
        await repo.create_prediction(_make_prediction(sensor_id="S002"))
        deleted = await repo.delete_prediction_history(sensor_id="S001")
        assert deleted == 2
        assert await repo.count_predictions() == 1

    async def test_delete_by_zone(self, repo):
        await repo.create_prediction(_make_prediction(zone_id="ZONE_A"))
        await repo.create_prediction(_make_prediction(zone_id="ZONE_B"))
        deleted = await repo.delete_prediction_history(zone_id="ZONE_A")
        assert deleted == 1
        assert await repo.count_predictions() == 1

    async def test_delete_before_timestamp(self, repo):
        t1 = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
        t3 = datetime(2026, 7, 1, 18, 0, tzinfo=timezone.utc)
        await repo.create_prediction(_make_prediction(prediction_timestamp=t1))
        await repo.create_prediction(_make_prediction(prediction_timestamp=t2))
        await repo.create_prediction(_make_prediction(prediction_timestamp=t3))
        cutoff = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
        deleted = await repo.delete_prediction_history(before=cutoff)
        assert deleted == 2
        remaining = await repo.get_prediction_history()
        assert len(remaining) == 1
        # SQLite strips tz info; compare by value
        assert remaining[0].prediction_timestamp.replace(tzinfo=timezone.utc) == t3

    async def test_rejects_no_filters(self, repo):
        with pytest.raises(ValueError, match="At least one filter"):
            await repo.delete_prediction_history()

    async def test_combined_filters(self, repo):
        t1 = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
        await repo.create_prediction(
            _make_prediction(sensor_id="S001", prediction_timestamp=t1),
        )
        await repo.create_prediction(
            _make_prediction(sensor_id="S001", prediction_timestamp=t2),
        )
        await repo.create_prediction(
            _make_prediction(sensor_id="S002", prediction_timestamp=t1),
        )
        # Delete S001 predictions before noon
        cutoff = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
        deleted = await repo.delete_prediction_history(
            sensor_id="S001", before=cutoff,
        )
        assert deleted == 1
        assert await repo.count_predictions() == 2
