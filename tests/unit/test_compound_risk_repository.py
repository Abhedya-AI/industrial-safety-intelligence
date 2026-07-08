"""Repository unit tests for the Compound Risk Intelligence module.

Uses the shared ``db_session`` fixture from conftest.py (in-memory SQLite).
Tests the SQLAlchemy repository implementation for all CRUD operations.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.compound_risk.models.compound_risk_model import CompoundRiskModel
from app.compound_risk.repositories.compound_risk_repository import (
    CompoundRiskRepository,
)
from app.compound_risk.repositories.sqlalchemy_compound_risk_repo import (
    SQLAlchemyCompoundRiskRepository,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_analysis(**overrides) -> CompoundRiskModel:
    """Create a CompoundRiskModel with sensible defaults."""
    defaults = {
        "id": str(uuid.uuid4()),
        "equipment_id": "EQ001",
        "zone_id": "ZONE_A",
        "anomaly_score": 0.5,
        "accident_probability": 0.3,
        "risk_score": 45.0,
        "sensor_health_score": 80.0,
        "compound_risk_score": 0.42,
        "risk_level": "MEDIUM",
        "confidence_score": 0.85,
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return CompoundRiskModel(**defaults)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest_asyncio.fixture
async def repo(db_session: AsyncSession) -> SQLAlchemyCompoundRiskRepository:
    return SQLAlchemyCompoundRiskRepository(db_session)


@pytest_asyncio.fixture
async def seeded_repo(
    db_session: AsyncSession,
) -> tuple[SQLAlchemyCompoundRiskRepository, list[str]]:
    """Repository pre-seeded with 5 analyses for query tests."""
    repo = SQLAlchemyCompoundRiskRepository(db_session)
    ids = []
    levels = ["LOW", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    zones = ["ZONE_A", "ZONE_A", "ZONE_B", "ZONE_A", "ZONE_B"]
    equips = ["EQ001", "EQ001", "EQ002", "EQ001", "EQ002"]
    for i in range(5):
        a = _make_analysis(
            id=str(uuid.uuid4()),
            zone_id=zones[i],
            equipment_id=equips[i],
            risk_level=levels[i],
            compound_risk_score=0.1 * (i + 1),
            created_at=datetime(2026, 7, 1, 10 + i, 0, tzinfo=timezone.utc),
        )
        await repo.create(a)
        ids.append(a.id)
    await db_session.commit()
    return repo, ids


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Interface contract
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRepositoryInterface:
    def test_is_abstract(self):
        assert issubclass(SQLAlchemyCompoundRiskRepository, CompoundRiskRepository)

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            CompoundRiskRepository()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. create()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCreate:
    async def test_create_and_persist(self, repo):
        analysis = _make_analysis()
        result = await repo.create(analysis)
        assert result.id == analysis.id
        assert result.compound_risk_score == analysis.compound_risk_score

    async def test_create_populates_id(self, repo):
        analysis = _make_analysis()
        result = await repo.create(analysis)
        assert result.id is not None
        uuid.UUID(result.id)  # Should not raise

    async def test_create_sets_created_at(self, repo):
        analysis = _make_analysis()
        result = await repo.create(analysis)
        assert result.created_at is not None

    async def test_create_with_contributing_factors(self, repo):
        import json
        factors = json.dumps([
            {"factor": "Gas", "weight": 0.4,
             "current_value": "120 ppm", "contribution": 0.65},
        ])
        analysis = _make_analysis(contributing_factors=factors)
        result = await repo.create(analysis)
        assert result.contributing_factors is not None
        parsed = json.loads(result.contributing_factors)
        assert parsed[0]["factor"] == "Gas"

    async def test_create_with_recommendation(self, repo):
        analysis = _make_analysis(recommendation="Evacuate immediately")
        result = await repo.create(analysis)
        assert result.recommendation == "Evacuate immediately"

    async def test_create_optional_fields_null(self, repo):
        analysis = _make_analysis(
            equipment_id=None, zone_id=None,
            contributing_factors=None, recommendation=None,
        )
        result = await repo.create(analysis)
        assert result.equipment_id is None
        assert result.zone_id is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. get_by_id()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGetById:
    async def test_found(self, repo):
        analysis = _make_analysis()
        await repo.create(analysis)
        result = await repo.get_by_id(analysis.id)
        assert result is not None
        assert result.id == analysis.id

    async def test_not_found(self, repo):
        result = await repo.get_by_id("nonexistent-uuid")
        assert result is None

    async def test_returns_all_fields(self, repo):
        analysis = _make_analysis(
            recommendation="Test recommendation",
            contributing_factors='[{"factor":"X","weight":0.1,"current_value":"5","contribution":0.1}]',
        )
        await repo.create(analysis)
        result = await repo.get_by_id(analysis.id)
        assert result.recommendation == "Test recommendation"
        assert result.contributing_factors is not None
        assert result.anomaly_score == analysis.anomaly_score
        assert result.risk_score == analysis.risk_score
        assert result.compound_risk_score == analysis.compound_risk_score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. get_latest()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGetLatest:
    async def test_empty_returns_none(self, repo):
        result = await repo.get_latest()
        assert result is None

    async def test_returns_most_recent(self, seeded_repo):
        repo, ids = seeded_repo
        result = await repo.get_latest()
        assert result is not None
        # ids[4] is the newest (10+4 = 14:00)
        assert result.id == ids[4]

    async def test_filter_by_zone(self, seeded_repo):
        repo, ids = seeded_repo
        result = await repo.get_latest(zone_id="ZONE_A")
        assert result is not None
        # ZONE_A: ids[0] (10:00), ids[1] (11:00), ids[3] (13:00) → newest = ids[3]
        assert result.id == ids[3]

    async def test_filter_by_equipment(self, seeded_repo):
        repo, ids = seeded_repo
        result = await repo.get_latest(equipment_id="EQ002")
        assert result is not None
        # EQ002: ids[2] (12:00), ids[4] (14:00) → newest = ids[4]
        assert result.id == ids[4]

    async def test_filter_by_zone_and_equipment(self, seeded_repo):
        repo, ids = seeded_repo
        result = await repo.get_latest(zone_id="ZONE_B", equipment_id="EQ002")
        assert result is not None
        # ZONE_B + EQ002: ids[2] (12:00), ids[4] (14:00) → newest = ids[4]
        assert result.id == ids[4]

    async def test_no_match_returns_none(self, seeded_repo):
        repo, _ = seeded_repo
        result = await repo.get_latest(zone_id="ZONE_X")
        assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. get_history()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGetHistory:
    async def test_empty(self, repo):
        result = await repo.get_history()
        assert result == []

    async def test_returns_all(self, seeded_repo):
        repo, ids = seeded_repo
        result = await repo.get_history()
        assert len(result) == 5

    async def test_newest_first(self, seeded_repo):
        repo, ids = seeded_repo
        result = await repo.get_history()
        assert result[0].id == ids[4]
        assert result[4].id == ids[0]

    async def test_filter_by_zone(self, seeded_repo):
        repo, _ = seeded_repo
        result = await repo.get_history(zone_id="ZONE_A")
        assert len(result) == 3
        assert all(r.zone_id == "ZONE_A" for r in result)

    async def test_filter_by_equipment(self, seeded_repo):
        repo, _ = seeded_repo
        result = await repo.get_history(equipment_id="EQ002")
        assert len(result) == 2
        assert all(r.equipment_id == "EQ002" for r in result)

    async def test_filter_by_risk_level(self, seeded_repo):
        repo, _ = seeded_repo
        result = await repo.get_history(risk_level="LOW")
        assert len(result) == 2
        assert all(r.risk_level == "LOW" for r in result)

    async def test_filter_by_time_range(self, seeded_repo):
        repo, _ = seeded_repo
        start = datetime(2026, 7, 1, 11, 0, tzinfo=timezone.utc)
        end = datetime(2026, 7, 1, 13, 0, tzinfo=timezone.utc)
        result = await repo.get_history(start_time=start, end_time=end)
        assert len(result) == 3  # 11:00, 12:00, 13:00

    async def test_pagination_limit(self, seeded_repo):
        repo, ids = seeded_repo
        result = await repo.get_history(limit=2)
        assert len(result) == 2
        assert result[0].id == ids[4]
        assert result[1].id == ids[3]

    async def test_pagination_offset(self, seeded_repo):
        repo, ids = seeded_repo
        result = await repo.get_history(offset=3, limit=10)
        assert len(result) == 2
        assert result[0].id == ids[1]
        assert result[1].id == ids[0]

    async def test_combined_filters(self, seeded_repo):
        repo, _ = seeded_repo
        result = await repo.get_history(
            zone_id="ZONE_A", risk_level="LOW",
        )
        assert len(result) == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. count()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCount:
    async def test_empty(self, repo):
        assert await repo.count() == 0

    async def test_total(self, seeded_repo):
        repo, _ = seeded_repo
        assert await repo.count() == 5

    async def test_filter_by_zone(self, seeded_repo):
        repo, _ = seeded_repo
        assert await repo.count(zone_id="ZONE_A") == 3
        assert await repo.count(zone_id="ZONE_B") == 2

    async def test_filter_by_risk_level(self, seeded_repo):
        repo, _ = seeded_repo
        assert await repo.count(risk_level="CRITICAL") == 1
        assert await repo.count(risk_level="LOW") == 2

    async def test_filter_by_equipment(self, seeded_repo):
        repo, _ = seeded_repo
        assert await repo.count(equipment_id="EQ001") == 3

    async def test_combined_filters(self, seeded_repo):
        repo, _ = seeded_repo
        assert await repo.count(zone_id="ZONE_A", risk_level="HIGH") == 1

    async def test_no_match(self, seeded_repo):
        repo, _ = seeded_repo
        assert await repo.count(zone_id="ZONE_X") == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. delete()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDelete:
    async def test_delete_existing(self, repo):
        analysis = _make_analysis()
        await repo.create(analysis)
        deleted = await repo.delete(analysis.id)
        assert deleted is True
        assert await repo.get_by_id(analysis.id) is None

    async def test_delete_nonexistent(self, repo):
        deleted = await repo.delete("nonexistent-uuid")
        assert deleted is False

    async def test_delete_does_not_affect_others(self, seeded_repo):
        repo, ids = seeded_repo
        await repo.delete(ids[0])
        assert await repo.count() == 4
        # Others still exist
        assert await repo.get_by_id(ids[1]) is not None
        assert await repo.get_by_id(ids[4]) is not None

    async def test_delete_then_recreate(self, repo):
        analysis = _make_analysis()
        await repo.create(analysis)
        await repo.delete(analysis.id)
        # Can recreate with same ID
        analysis2 = _make_analysis(id=analysis.id)
        result = await repo.create(analysis2)
        assert result.id == analysis.id
