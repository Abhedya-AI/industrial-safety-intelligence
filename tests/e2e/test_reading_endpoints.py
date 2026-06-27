"""Comprehensive API integration tests for the Sensor Reading endpoints.

Covers:
  - Successful reading ingestion (single and batch)
  - Reading retrieval (latest, history, stats)
  - Invalid sensor ID → 404
  - Invalid request body → 422
  - Business rule violations (OFFLINE sensor, out-of-range, future timestamp,
    duplicate reading) → 409 / 422
  - Not Found responses
  - Error format compliance
  - OpenAPI/Swagger doc visibility
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient

API_PREFIX = "/api/v1"


# ── Fixtures ──


@pytest.fixture
def gas_sensor():
    return {
        "sensor_id": "S001",
        "sensor_name": "Zone A Gas Detector",
        "sensor_type": "GAS",
        "location_zone": "ZONE_A",
        "unit": "ppm",
        "min_value": 0.0,
        "max_value": 1000.0,
    }


@pytest.fixture
def offline_sensor():
    return {
        "sensor_id": "S-OFFLINE",
        "sensor_name": "Offline Sensor",
        "sensor_type": "TEMPERATURE",
        "location_zone": "ZONE_B",
        "unit": "°C",
        "min_value": -40.0,
        "max_value": 200.0,
        "status": "OFFLINE",
    }


def _reading(sensor_id: str = "S001", **overrides) -> dict:
    """Build a valid reading payload."""
    defaults = {
        "sensor_id": sensor_id,
        "value": 42.0,
        "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
        "confidence": 95.0,
    }
    defaults.update(overrides)
    return defaults


async def _register_sensor(client: AsyncClient, sensor: dict):
    """Helper to pre-register a sensor."""
    resp = await client.post(f"{API_PREFIX}/sensors", json=sensor)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _assert_error_format(body: dict):
    """Assert error response follows the spec format."""
    assert body["success"] is False
    assert "error" in body
    assert "message" in body
    assert "request_id" in body
    assert "timestamp" in body


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST /readings/ingest — Single ingestion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_ingest_single_reading_success(client: AsyncClient, gas_sensor):
    await _register_sensor(client, gas_sensor)

    resp = await client.post(f"{API_PREFIX}/readings/ingest", json=_reading())
    assert resp.status_code == 201

    body = resp.json()
    assert body["success"] is True
    assert body["reading"]["value"] == 42.0
    assert body["reading"]["sensor_id"] is not None
    assert body["reading"]["confidence"] == 95.0
    assert "id" in body["reading"]
    assert "received_at" in body["reading"]


@pytest.mark.asyncio
async def test_ingest_reading_nonexistent_sensor_returns_404(client: AsyncClient):
    resp = await client.post(
        f"{API_PREFIX}/readings/ingest",
        json=_reading(sensor_id="NONEXISTENT"),
    )
    assert resp.status_code == 404
    _assert_error_format(resp.json())
    assert resp.json()["error"] == "RESOURCE_NOT_FOUND"


@pytest.mark.asyncio
async def test_ingest_reading_offline_sensor_returns_409(
    client: AsyncClient, gas_sensor, offline_sensor
):
    await _register_sensor(client, gas_sensor)
    # Create sensor as NORMAL first, then update to OFFLINE
    offline_data = {k: v for k, v in offline_sensor.items() if k != "status"}
    await _register_sensor(client, offline_data)
    await client.put(
        f"{API_PREFIX}/sensors/{offline_sensor['sensor_id']}",
        json={"status": "OFFLINE"},
    )

    resp = await client.post(
        f"{API_PREFIX}/readings/ingest",
        json=_reading(sensor_id="S-OFFLINE"),
    )
    assert resp.status_code == 409
    body = resp.json()
    _assert_error_format(body)
    assert "OFFLINE" in body["message"]


@pytest.mark.asyncio
async def test_ingest_reading_value_out_of_range_returns_422(
    client: AsyncClient, gas_sensor
):
    await _register_sensor(client, gas_sensor)

    # Above max (1000)
    resp = await client.post(
        f"{API_PREFIX}/readings/ingest",
        json=_reading(value=9999.0),
    )
    assert resp.status_code == 422
    _assert_error_format(resp.json())
    assert "exceeds" in resp.json()["message"]


@pytest.mark.asyncio
async def test_ingest_reading_below_min_returns_422(
    client: AsyncClient, gas_sensor
):
    await _register_sensor(client, gas_sensor)

    resp = await client.post(
        f"{API_PREFIX}/readings/ingest",
        json=_reading(value=-10.0),
    )
    assert resp.status_code == 422
    _assert_error_format(resp.json())
    assert "below" in resp.json()["message"]


@pytest.mark.asyncio
async def test_ingest_reading_future_timestamp_returns_422(
    client: AsyncClient, gas_sensor
):
    await _register_sensor(client, gas_sensor)

    future = datetime.now(timezone.utc) + timedelta(hours=2)
    resp = await client.post(
        f"{API_PREFIX}/readings/ingest",
        json=_reading(timestamp=future.isoformat()),
    )
    assert resp.status_code == 422
    _assert_error_format(resp.json())
    assert "future" in resp.json()["message"]


@pytest.mark.asyncio
async def test_ingest_reading_duplicate_returns_409(
    client: AsyncClient, gas_sensor
):
    await _register_sensor(client, gas_sensor)

    ts = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    resp1 = await client.post(
        f"{API_PREFIX}/readings/ingest",
        json=_reading(value=10.0, timestamp=ts),
    )
    assert resp1.status_code == 201

    resp2 = await client.post(
        f"{API_PREFIX}/readings/ingest",
        json=_reading(value=20.0, timestamp=ts),
    )
    assert resp2.status_code == 409
    body = resp2.json()
    _assert_error_format(body)
    assert "Duplicate" in body["message"]


@pytest.mark.asyncio
async def test_ingest_reading_invalid_body_returns_422(client: AsyncClient):
    """Missing required fields."""
    resp = await client.post(
        f"{API_PREFIX}/readings/ingest",
        json={"value": 10.0},  # missing sensor_id and timestamp
    )
    assert resp.status_code == 422
    body = resp.json()
    _assert_error_format(body)
    assert body["error"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_ingest_reading_with_metadata(client: AsyncClient, gas_sensor):
    await _register_sensor(client, gas_sensor)

    resp = await client.post(
        f"{API_PREFIX}/readings/ingest",
        json=_reading(metadata={"equipment_id": "EQ-001", "zone": "A"}),
    )
    assert resp.status_code == 201
    assert resp.json()["reading"]["value"] == 42.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST /readings/ingest/batch — Batch ingestion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_ingest_batch_success(client: AsyncClient, gas_sensor):
    await _register_sensor(client, gas_sensor)

    base = datetime.now(timezone.utc) - timedelta(hours=10)
    readings = [
        _reading(value=float(i * 10), timestamp=(base + timedelta(hours=i)).isoformat())
        for i in range(3)
    ]
    resp = await client.post(
        f"{API_PREFIX}/readings/ingest/batch",
        json={"readings": readings},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["ingested"] == 3
    assert len(body["readings"]) == 3


@pytest.mark.asyncio
async def test_ingest_batch_rejects_all_on_one_invalid(
    client: AsyncClient, gas_sensor
):
    await _register_sensor(client, gas_sensor)

    base = datetime.now(timezone.utc) - timedelta(hours=10)
    readings = [
        _reading(value=10.0, timestamp=(base + timedelta(hours=1)).isoformat()),
        _reading(value=99999.0, timestamp=(base + timedelta(hours=2)).isoformat()),
    ]
    resp = await client.post(
        f"{API_PREFIX}/readings/ingest/batch",
        json={"readings": readings},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ingest_batch_empty_returns_422(client: AsyncClient):
    resp = await client.post(
        f"{API_PREFIX}/readings/ingest/batch",
        json={"readings": []},
    )
    assert resp.status_code == 422


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /readings/latest/{sensor_id} — Latest reading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_get_latest_reading_success(client: AsyncClient, gas_sensor):
    await _register_sensor(client, gas_sensor)

    t1 = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
    t2 = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()

    await client.post(
        f"{API_PREFIX}/readings/ingest", json=_reading(value=10.0, timestamp=t1)
    )
    await client.post(
        f"{API_PREFIX}/readings/ingest", json=_reading(value=99.0, timestamp=t2)
    )

    resp = await client.get(f"{API_PREFIX}/readings/latest/S001")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["reading"]["value"] == 99.0


@pytest.mark.asyncio
async def test_get_latest_reading_not_found(client: AsyncClient, gas_sensor):
    await _register_sensor(client, gas_sensor)

    resp = await client.get(f"{API_PREFIX}/readings/latest/S001")
    assert resp.status_code == 404
    _assert_error_format(resp.json())


@pytest.mark.asyncio
async def test_get_latest_reading_nonexistent_sensor(client: AsyncClient):
    resp = await client.get(f"{API_PREFIX}/readings/latest/NONEXISTENT")
    assert resp.status_code == 404
    _assert_error_format(resp.json())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /readings/{sensor_id} — Historical readings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_get_sensor_readings_history(client: AsyncClient, gas_sensor):
    await _register_sensor(client, gas_sensor)

    base = datetime.now(timezone.utc) - timedelta(hours=10)
    for i in range(5):
        await client.post(
            f"{API_PREFIX}/readings/ingest",
            json=_reading(
                value=float(i * 10),
                timestamp=(base + timedelta(hours=i)).isoformat(),
            ),
        )

    resp = await client.get(
        f"{API_PREFIX}/readings/S001",
        params={
            "start": (base - timedelta(minutes=1)).isoformat(),
            "end": (base + timedelta(hours=10)).isoformat(),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["sensor_id"] == "S001"
    assert body["count"] == 5
    assert len(body["readings"]) == 5


@pytest.mark.asyncio
async def test_get_sensor_readings_empty_range(client: AsyncClient, gas_sensor):
    await _register_sensor(client, gas_sensor)

    resp = await client.get(
        f"{API_PREFIX}/readings/S001",
        params={
            "start": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            "end": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["readings"] == []


@pytest.mark.asyncio
async def test_get_sensor_readings_nonexistent_sensor(client: AsyncClient):
    resp = await client.get(f"{API_PREFIX}/readings/NONEXISTENT")
    assert resp.status_code == 404
    _assert_error_format(resp.json())


@pytest.mark.asyncio
async def test_get_sensor_readings_defaults_to_24h(client: AsyncClient, gas_sensor):
    """When no start/end given, defaults to last 24h."""
    await _register_sensor(client, gas_sensor)

    # Insert a reading within last 24h
    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    await client.post(
        f"{API_PREFIX}/readings/ingest",
        json=_reading(value=55.0, timestamp=recent),
    )

    resp = await client.get(f"{API_PREFIX}/readings/S001")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /readings/{sensor_id}/stats — Statistics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_get_reading_stats_success(client: AsyncClient, gas_sensor):
    await _register_sensor(client, gas_sensor)

    base = datetime.now(timezone.utc) - timedelta(hours=10)
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    for i, v in enumerate(values):
        await client.post(
            f"{API_PREFIX}/readings/ingest",
            json=_reading(value=v, timestamp=(base + timedelta(hours=i)).isoformat()),
        )

    resp = await client.get(
        f"{API_PREFIX}/readings/S001/stats",
        params={
            "start": (base - timedelta(minutes=1)).isoformat(),
            "end": (base + timedelta(hours=10)).isoformat(),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    stats = body["statistics"]
    assert stats["count"] == 5
    assert stats["min_value"] == 10.0
    assert stats["max_value"] == 50.0
    assert stats["mean"] == 30.0


@pytest.mark.asyncio
async def test_get_reading_stats_empty(client: AsyncClient, gas_sensor):
    await _register_sensor(client, gas_sensor)

    resp = await client.get(
        f"{API_PREFIX}/readings/S001/stats",
        params={
            "start": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            "end": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["statistics"] is None


@pytest.mark.asyncio
async def test_get_reading_stats_nonexistent_sensor(client: AsyncClient):
    resp = await client.get(f"{API_PREFIX}/readings/NONEXISTENT/stats")
    assert resp.status_code == 404
    _assert_error_format(resp.json())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenAPI/Swagger visibility
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_readings_endpoints_appear_in_openapi_docs(client: AsyncClient):
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]

    expected_paths = [
        f"{API_PREFIX}/readings/ingest",
        f"{API_PREFIX}/readings/ingest/batch",
    ]
    for path in expected_paths:
        assert path in paths, f"Missing endpoint in OpenAPI: {path}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sensor Registry still works (regression check)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_sensor_registry_still_works(client: AsyncClient, gas_sensor):
    """Ensure adding reading endpoints did not break sensor CRUD."""
    resp = await client.post(f"{API_PREFIX}/sensors", json=gas_sensor)
    assert resp.status_code == 201

    resp = await client.get(f"{API_PREFIX}/sensors/S001")
    assert resp.status_code == 200
    assert resp.json()["sensor_id"] == "S001"

    resp = await client.delete(f"{API_PREFIX}/sensors/S001")
    assert resp.status_code == 204
