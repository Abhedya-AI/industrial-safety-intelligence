"""Comprehensive unit tests for the Baseline Learning Service.

Covers:
  1. BaselineResult computation (learn_baseline)
     - Descriptive statistics
     - Normal operating range
     - Rolling averages
     - Seasonal hourly patterns
     - Trend analysis
  2. BaselineRepository (CRUD)
  3. BaselineLearningService integration
     - update_baseline (create + update)
     - get_baseline
     - is_within_normal_range
     - update_all_baselines
     - Insufficient data handling
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.sensor_intelligence.models.reading_model import ReadingModel
from app.sensor_intelligence.models.sensor_model import SensorModel
from app.sensor_intelligence.repositories.sqlalchemy_reading_repo import (
    SQLAlchemyReadingRepository,
)
from app.sensor_intelligence.repositories.sqlalchemy_sensor_repo import (
    SQLAlchemySensorRepository,
)
from app.sensor_intelligence.schemas.reading_schemas import ReadingCreateRequest
from app.sensor_intelligence.services.baseline_service import (
    BaselineLearningService,
    BaselineRepository,
    BaselineResult,
)
from app.sensor_intelligence.services.reading_service import ReadingService


# ── Helpers ──


def _make_sensor(**overrides) -> SensorModel:
    defaults = {
        "sensor_id": f"S-{uuid.uuid4().hex[:6]}",
        "sensor_name": "Test Sensor",
        "sensor_type": "GAS",
        "status": "NORMAL",
        "location_zone": "ZONE_A",
        "unit": "ppm",
        "min_value": 0.0,
        "max_value": 10000.0,
    }
    defaults.update(overrides)
    return SensorModel(**defaults)


async def _ingest_readings(
    reading_service: ReadingService,
    sensor_code: str,
    values: list[float],
    base_time: datetime | None = None,
    interval_minutes: int = 60,
):
    """Ingest a series of readings for a sensor."""
    base = base_time or (datetime.now(timezone.utc) - timedelta(hours=len(values)))
    for i, v in enumerate(values):
        await reading_service.ingest_reading(
            ReadingCreateRequest(
                sensor_id=sensor_code,
                value=v,
                timestamp=(base + timedelta(minutes=i * interval_minutes)).isoformat(),
                confidence=95.0,
            )
        )


# ── Fixtures ──


@pytest_asyncio.fixture
async def sensor_repo(db_session: AsyncSession) -> SQLAlchemySensorRepository:
    return SQLAlchemySensorRepository(db_session)


@pytest_asyncio.fixture
async def reading_repo(db_session: AsyncSession) -> SQLAlchemyReadingRepository:
    return SQLAlchemyReadingRepository(db_session)


@pytest_asyncio.fixture
async def baseline_repo(db_session: AsyncSession) -> BaselineRepository:
    return BaselineRepository(db_session)


@pytest_asyncio.fixture
async def reading_service(
    reading_repo: SQLAlchemyReadingRepository,
    sensor_repo: SQLAlchemySensorRepository,
) -> ReadingService:
    return ReadingService(reading_repo, sensor_repo)


@pytest_asyncio.fixture
async def baseline_service(
    baseline_repo: BaselineRepository,
    sensor_repo: SQLAlchemySensorRepository,
    reading_repo: SQLAlchemyReadingRepository,
) -> BaselineLearningService:
    return BaselineLearningService(
        baseline_repo, sensor_repo, reading_repo,
        min_samples=5,  # Lower threshold for tests
    )


@pytest_asyncio.fixture
async def sensor_with_readings(
    sensor_repo: SQLAlchemySensorRepository,
    reading_service: ReadingService,
) -> SensorModel:
    """Create a sensor with 20 normal readings."""
    sensor = await sensor_repo.create_sensor(
        _make_sensor(sensor_id="S001")
    )
    values = [
        42.0, 43.1, 41.5, 44.2, 42.8,
        43.5, 41.9, 44.0, 42.3, 43.7,
        42.1, 43.3, 41.8, 44.1, 42.5,
        43.6, 41.7, 44.3, 42.9, 43.0,
    ]
    await _ingest_readings(reading_service, "S001", values)
    return sensor


@pytest_asyncio.fixture
async def sensor_few_readings(
    sensor_repo: SQLAlchemySensorRepository,
    reading_service: ReadingService,
) -> SensorModel:
    """Create a sensor with too few readings for baseline."""
    sensor = await sensor_repo.create_sensor(
        _make_sensor(sensor_id="S-FEW")
    )
    await _ingest_readings(reading_service, "S-FEW", [10.0, 20.0, 30.0])
    return sensor


@pytest_asyncio.fixture
async def sensor_no_readings(
    sensor_repo: SQLAlchemySensorRepository,
) -> SensorModel:
    """Create a sensor with no readings."""
    return await sensor_repo.create_sensor(
        _make_sensor(sensor_id="S-EMPTY")
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. learn_baseline — computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_learn_baseline_descriptive_stats(
    baseline_service: BaselineLearningService,
    sensor_with_readings: SensorModel,
):
    result = await baseline_service.learn_baseline(sensor_with_readings.id)
    assert result is not None
    assert result.sample_count == 20
    assert 41.0 < result.mean < 44.0
    assert 41.0 < result.median < 44.0
    assert result.std_dev > 0
    assert result.variance > 0
    assert result.min_value <= result.mean <= result.max_value


async def test_learn_baseline_normal_range(
    baseline_service: BaselineLearningService,
    sensor_with_readings: SensorModel,
):
    result = await baseline_service.learn_baseline(sensor_with_readings.id)
    assert result is not None
    # Normal range = mean ± 2·σ
    assert result.normal_range_low < result.mean
    assert result.normal_range_high > result.mean
    assert result.normal_range_low == round(
        result.mean - result.sigma_multiplier * result.std_dev, 6
    )


async def test_learn_baseline_rolling_averages(
    baseline_service: BaselineLearningService,
    sensor_with_readings: SensorModel,
):
    result = await baseline_service.learn_baseline(sensor_with_readings.id)
    assert result is not None
    assert len(result.rolling_avg_5) > 0  # 20 readings → 16 entries
    assert len(result.rolling_avg_10) > 0  # 20 readings → 11 entries


async def test_learn_baseline_hourly_pattern(
    baseline_service: BaselineLearningService,
    sensor_with_readings: SensorModel,
):
    result = await baseline_service.learn_baseline(sensor_with_readings.id)
    assert result is not None
    assert isinstance(result.hourly_pattern, dict)
    # Each hour should have a float mean
    for hour, avg in result.hourly_pattern.items():
        assert 0 <= hour <= 23
        assert isinstance(avg, float)


async def test_learn_baseline_trend(
    baseline_service: BaselineLearningService,
    sensor_with_readings: SensorModel,
):
    result = await baseline_service.learn_baseline(sensor_with_readings.id)
    assert result is not None
    assert result.trend_direction in ("increasing", "decreasing", "stable")
    assert result.trend_slope is not None


async def test_learn_baseline_insufficient_data(
    baseline_service: BaselineLearningService,
    sensor_few_readings: SensorModel,
):
    """Should return None when below min_samples threshold."""
    result = await baseline_service.learn_baseline(sensor_few_readings.id)
    assert result is None


async def test_learn_baseline_no_readings(
    baseline_service: BaselineLearningService,
    sensor_no_readings: SensorModel,
):
    result = await baseline_service.learn_baseline(sensor_no_readings.id)
    assert result is None


async def test_learn_baseline_custom_sigma(
    baseline_service: BaselineLearningService,
    sensor_with_readings: SensorModel,
):
    result_2 = await baseline_service.learn_baseline(
        sensor_with_readings.id, sigma_multiplier=2.0
    )
    result_3 = await baseline_service.learn_baseline(
        sensor_with_readings.id, sigma_multiplier=3.0
    )
    assert result_2 is not None and result_3 is not None
    # Wider sigma → wider range
    range_width_2 = result_2.normal_range_high - result_2.normal_range_low
    range_width_3 = result_3.normal_range_high - result_3.normal_range_low
    assert range_width_3 > range_width_2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. BaselineRepository CRUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_repo_save_and_get(
    baseline_repo: BaselineRepository,
    sensor_with_readings: SensorModel,
):
    from app.sensor_intelligence.models.sensor_baseline_model import SensorBaselineModel
    model = SensorBaselineModel(
        id=str(uuid.uuid4()),
        sensor_id=sensor_with_readings.id,
        mean=42.5,
        median=42.3,
        std_dev=0.8,
        variance=0.64,
        min_value=41.5,
        max_value=44.3,
        normal_range_low=40.9,
        normal_range_high=44.1,
        sample_count=20,
    )
    saved = await baseline_repo.save(model)
    fetched = await baseline_repo.get_by_sensor_id(sensor_with_readings.id)
    assert fetched is not None
    assert fetched.mean == 42.5
    assert fetched.id == saved.id


async def test_repo_list_all(
    baseline_service: BaselineLearningService,
    baseline_repo: BaselineRepository,
    sensor_with_readings: SensorModel,
):
    await baseline_service.update_baseline(sensor_with_readings.id)
    items = await baseline_repo.list_all()
    assert len(items) >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. update_baseline (persist)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_update_baseline_creates_record(
    baseline_service: BaselineLearningService,
    baseline_repo: BaselineRepository,
    sensor_with_readings: SensorModel,
):
    model = await baseline_service.update_baseline(sensor_with_readings.id)
    assert model is not None
    assert model.sensor_id == sensor_with_readings.id
    assert model.mean > 0

    # Verify persisted
    fetched = await baseline_repo.get_by_sensor_id(sensor_with_readings.id)
    assert fetched is not None
    assert fetched.mean == model.mean


async def test_update_baseline_updates_existing(
    baseline_service: BaselineLearningService,
    sensor_with_readings: SensorModel,
):
    m1 = await baseline_service.update_baseline(sensor_with_readings.id)
    m2 = await baseline_service.update_baseline(sensor_with_readings.id)
    assert m1 is not None and m2 is not None
    assert m1.id == m2.id  # Same record


async def test_update_baseline_insufficient_data_returns_none(
    baseline_service: BaselineLearningService,
    sensor_few_readings: SensorModel,
):
    model = await baseline_service.update_baseline(sensor_few_readings.id)
    assert model is None


async def test_update_baseline_persists_rolling_avg(
    baseline_service: BaselineLearningService,
    sensor_with_readings: SensorModel,
):
    model = await baseline_service.update_baseline(sensor_with_readings.id)
    assert model is not None
    assert model.rolling_avg_5 is not None
    parsed = json.loads(model.rolling_avg_5)
    assert isinstance(parsed, list)
    assert len(parsed) > 0


async def test_update_baseline_persists_hourly_pattern(
    baseline_service: BaselineLearningService,
    sensor_with_readings: SensorModel,
):
    model = await baseline_service.update_baseline(sensor_with_readings.id)
    assert model is not None
    assert model.hourly_pattern is not None
    parsed = json.loads(model.hourly_pattern)
    assert isinstance(parsed, dict)


async def test_update_baseline_persists_trend(
    baseline_service: BaselineLearningService,
    sensor_with_readings: SensorModel,
):
    model = await baseline_service.update_baseline(sensor_with_readings.id)
    assert model is not None
    assert model.trend_direction in ("increasing", "decreasing", "stable")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. get_baseline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_get_baseline_by_business_id(
    baseline_service: BaselineLearningService,
    sensor_with_readings: SensorModel,
):
    await baseline_service.update_baseline(sensor_with_readings.id)
    result = await baseline_service.get_baseline("S001")
    assert result is not None
    assert result.sample_count == 20


async def test_get_baseline_nonexistent(
    baseline_service: BaselineLearningService,
):
    result = await baseline_service.get_baseline("NONEXISTENT")
    assert result is None


async def test_get_baseline_by_pk(
    baseline_service: BaselineLearningService,
    sensor_with_readings: SensorModel,
):
    await baseline_service.update_baseline(sensor_with_readings.id)
    result = await baseline_service.get_baseline_by_pk(sensor_with_readings.id)
    assert result is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. is_within_normal_range
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_within_normal_range(
    baseline_service: BaselineLearningService,
    sensor_with_readings: SensorModel,
):
    await baseline_service.update_baseline(sensor_with_readings.id)
    # Mean ≈ 42.8, should be within range
    result = await baseline_service.is_within_normal_range("S001", 42.5)
    assert result is True


async def test_outside_normal_range(
    baseline_service: BaselineLearningService,
    sensor_with_readings: SensorModel,
):
    await baseline_service.update_baseline(sensor_with_readings.id)
    # Far outside any 2σ range for values ~42
    result = await baseline_service.is_within_normal_range("S001", 100.0)
    assert result is False


async def test_normal_range_no_baseline(
    baseline_service: BaselineLearningService,
):
    result = await baseline_service.is_within_normal_range("NONEXISTENT", 42.0)
    assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. update_all_baselines
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def test_update_all_baselines(
    baseline_service: BaselineLearningService,
    sensor_with_readings: SensorModel,
    sensor_no_readings: SensorModel,
):
    results = await baseline_service.update_all_baselines()
    # Only sensor_with_readings should have enough data
    assert len(results) == 1
    assert results[0].sensor_id == sensor_with_readings.id


async def test_update_all_baselines_multiple_sensors(
    baseline_service: BaselineLearningService,
    reading_service: ReadingService,
    sensor_repo: SQLAlchemySensorRepository,
):
    """Multiple sensors with enough data should all get baselines."""
    s1 = await sensor_repo.create_sensor(_make_sensor(sensor_id="S-A"))
    s2 = await sensor_repo.create_sensor(_make_sensor(sensor_id="S-B"))

    values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    await _ingest_readings(reading_service, "S-A", values)
    await _ingest_readings(reading_service, "S-B", [v * 2 for v in values])

    results = await baseline_service.update_all_baselines()
    assert len(results) == 2
    pks = {r.sensor_id for r in results}
    assert s1.id in pks
    assert s2.id in pks
