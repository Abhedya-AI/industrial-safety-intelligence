"""Comprehensive API integration tests for the Sensor Registry.

Verifies the API layer against the finalized API specification:
  - Response schema compliance (every field present per spec)
  - Error response format (success, error, message, request_id, timestamp)
  - Correct HTTP status codes
  - Business rule enforcement through the API
  - Query parameter filtering
  - Pagination
  - Spec endpoints 6 (GET /sensors/current) and 7 (GET /sensors/{sensor_id}/history)
  - OpenAPI/Swagger doc visibility
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


# ── Fixtures ──


@pytest.fixture
def gas_sensor():
    return {
        "sensor_id": "S001",
        "sensor_name": "Zone A Gas Detector",
        "sensor_type": "GAS",
        "location_zone": "ZONE_A",
        "equipment_id": "EQ001",
        "manufacturer": "Dräger",
        "model": "POLYTRON 8700",
        "unit": "ppm",
        "min_value": 0.0,
        "max_value": 10000.0,
        "accuracy_rating": 0.99,
        "installation_date": "2024-01-15",
        "last_calibration": "2025-06-01",
        "next_calibration_due": "2025-09-01",
    }


@pytest.fixture
def temp_sensor():
    return {
        "sensor_id": "S002",
        "sensor_name": "Zone B Temperature Probe",
        "sensor_type": "TEMPERATURE",
        "location_zone": "ZONE_B",
        "equipment_id": "EQ002",
        "manufacturer": "Honeywell",
        "model": "STT850",
        "unit": "°C",
        "min_value": -40.0,
        "max_value": 200.0,
        "accuracy_rating": 0.97,
        "installation_date": "2024-03-10",
        "last_calibration": "2025-05-20",
        "next_calibration_due": "2025-08-20",
    }


@pytest.fixture
def pressure_sensor():
    return {
        "sensor_id": "S003",
        "sensor_name": "Zone A Pressure Gauge",
        "sensor_type": "PRESSURE",
        "location_zone": "ZONE_A",
        "equipment_id": "EQ003",
        "manufacturer": "Emerson",
        "model": "3051S",
        "unit": "bar",
        "min_value": 0.0,
        "max_value": 100.0,
    }


@pytest.fixture
def minimal_sensor():
    """Sensor with only required fields."""
    return {
        "sensor_id": "S-MIN",
        "sensor_name": "Minimal Sensor",
        "sensor_type": "HUMIDITY",
        "unit": "%RH",
    }


async def _register(client: AsyncClient, payload: dict) -> dict:
    """Helper: create sensor and return response body."""
    resp = await client.post("/api/v1/sensors", json=payload)
    assert resp.status_code == 201
    return resp.json()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST /sensors — Create
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCreateSensor:

    async def test_create_returns_201_with_all_fields(
        self, client: AsyncClient, gas_sensor
    ):
        resp = await client.post("/api/v1/sensors", json=gas_sensor)
        assert resp.status_code == 201
        body = resp.json()

        # Every SensorResponse field must be present
        assert body["sensor_id"] == "S001"
        assert body["sensor_name"] == "Zone A Gas Detector"
        assert body["sensor_type"] == "GAS"
        assert body["status"] == "NORMAL"  # default
        assert body["location_zone"] == "ZONE_A"
        assert body["equipment_id"] == "EQ001"
        assert body["manufacturer"] == "Dräger"
        assert body["model"] == "POLYTRON 8700"
        assert body["unit"] == "ppm"
        assert body["min_value"] == 0.0
        assert body["max_value"] == 10000.0
        assert body["accuracy_rating"] == 0.99
        assert body["installation_date"] == "2024-01-15"
        assert body["last_calibration"] == "2025-06-01"
        assert body["next_calibration_due"] == "2025-09-01"
        assert "id" in body
        assert "created_at" in body
        assert "updated_at" in body

    async def test_create_with_minimal_fields(
        self, client: AsyncClient, minimal_sensor
    ):
        resp = await client.post("/api/v1/sensors", json=minimal_sensor)
        assert resp.status_code == 201
        body = resp.json()
        assert body["sensor_id"] == "S-MIN"
        assert body["status"] == "NORMAL"
        # Optional fields should be null
        assert body["location_zone"] is None
        assert body["equipment_id"] is None
        assert body["manufacturer"] is None
        assert body["model"] is None
        assert body["min_value"] is None
        assert body["max_value"] is None
        assert body["accuracy_rating"] is None

    async def test_create_duplicate_returns_409_spec_error_format(
        self, client: AsyncClient, gas_sensor
    ):
        await _register(client, gas_sensor)
        resp = await client.post("/api/v1/sensors", json=gas_sensor)
        assert resp.status_code == 409
        body = resp.json()
        # Spec error format fields
        assert body["success"] is False
        assert body["error"] == "DUPLICATE_RESOURCE"
        assert "message" in body
        assert "request_id" in body
        assert "timestamp" in body

    async def test_create_missing_required_fields_returns_422(
        self, client: AsyncClient
    ):
        resp = await client.post("/api/v1/sensors", json={})
        assert resp.status_code == 422
        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "VALIDATION_ERROR"
        assert "details" in body
        assert "validation_errors" in body["details"]

    async def test_create_invalid_sensor_type_returns_422(
        self, client: AsyncClient
    ):
        resp = await client.post("/api/v1/sensors", json={
            "sensor_id": "BAD",
            "sensor_name": "Bad Type",
            "sensor_type": "INVALID_TYPE",
            "unit": "x",
        })
        assert resp.status_code == 422

    async def test_create_min_gte_max_returns_422(self, client: AsyncClient):
        resp = await client.post("/api/v1/sensors", json={
            "sensor_id": "MM01",
            "sensor_name": "Bad Range",
            "sensor_type": "GAS",
            "unit": "ppm",
            "min_value": 100.0,
            "max_value": 50.0,
        })
        assert resp.status_code == 422
        body = resp.json()
        assert body["success"] is False
        assert "min_value" in body["message"]

    async def test_create_calibration_before_install_returns_422(
        self, client: AsyncClient
    ):
        resp = await client.post("/api/v1/sensors", json={
            "sensor_id": "CAL01",
            "sensor_name": "Bad Calibration",
            "sensor_type": "GAS",
            "unit": "ppm",
            "installation_date": "2025-06-01",
            "last_calibration": "2024-01-01",
        })
        assert resp.status_code == 422
        body = resp.json()
        assert body["success"] is False
        assert "calibration" in body["message"].lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /sensors/{sensor_id} — Read
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGetSensor:

    async def test_get_returns_full_response(
        self, client: AsyncClient, gas_sensor
    ):
        await _register(client, gas_sensor)
        resp = await client.get("/api/v1/sensors/S001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["sensor_id"] == "S001"
        assert body["sensor_type"] == "GAS"
        assert body["status"] == "NORMAL"
        assert "created_at" in body

    async def test_get_not_found_returns_404_spec_format(
        self, client: AsyncClient
    ):
        resp = await client.get("/api/v1/sensors/NONEXISTENT")
        assert resp.status_code == 404
        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "RESOURCE_NOT_FOUND"
        assert "request_id" in body
        assert "timestamp" in body


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /sensors — List
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestListSensors:

    async def test_list_empty(self, client: AsyncClient):
        resp = await client.get("/api/v1/sensors")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["total"] == 0
        assert body["items"] == []
        assert body["offset"] == 0
        assert body["limit"] == 50

    async def test_list_returns_multiple(
        self, client: AsyncClient, gas_sensor, temp_sensor
    ):
        await _register(client, gas_sensor)
        await _register(client, temp_sensor)
        resp = await client.get("/api/v1/sensors")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert len(body["items"]) == 2

    async def test_list_filter_by_sensor_type(
        self, client: AsyncClient, gas_sensor, temp_sensor, pressure_sensor
    ):
        await _register(client, gas_sensor)
        await _register(client, temp_sensor)
        await _register(client, pressure_sensor)

        resp = await client.get("/api/v1/sensors?sensor_type=GAS")
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["sensor_id"] == "S001"

        resp = await client.get("/api/v1/sensors?sensor_type=TEMPERATURE")
        assert resp.json()["total"] == 1

        resp = await client.get("/api/v1/sensors?sensor_type=VIBRATION")
        assert resp.json()["total"] == 0

    async def test_list_filter_by_status(
        self, client: AsyncClient, gas_sensor, temp_sensor
    ):
        await _register(client, gas_sensor)
        await _register(client, temp_sensor)
        # Update one to WARNING
        await client.put("/api/v1/sensors/S002", json={"status": "WARNING"})

        resp = await client.get("/api/v1/sensors?status=WARNING")
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["sensor_id"] == "S002"

        resp = await client.get("/api/v1/sensors?status=NORMAL")
        assert resp.json()["total"] == 1

    async def test_list_filter_by_zone(
        self, client: AsyncClient, gas_sensor, temp_sensor
    ):
        await _register(client, gas_sensor)   # ZONE_A
        await _register(client, temp_sensor)  # ZONE_B

        resp = await client.get("/api/v1/sensors?zone_id=ZONE_A")
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["location_zone"] == "ZONE_A"

    async def test_list_pagination(
        self, client: AsyncClient, gas_sensor, temp_sensor, pressure_sensor
    ):
        await _register(client, gas_sensor)
        await _register(client, temp_sensor)
        await _register(client, pressure_sensor)

        # Page 1
        resp = await client.get("/api/v1/sensors?offset=0&limit=2")
        body = resp.json()
        assert body["total"] == 3
        assert len(body["items"]) == 2
        assert body["offset"] == 0
        assert body["limit"] == 2

        # Page 2
        resp = await client.get("/api/v1/sensors?offset=2&limit=2")
        body = resp.json()
        assert body["total"] == 3
        assert len(body["items"]) == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PUT /sensors/{sensor_id} — Update
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestUpdateSensor:

    async def test_update_partial_fields(
        self, client: AsyncClient, gas_sensor
    ):
        await _register(client, gas_sensor)
        resp = await client.put("/api/v1/sensors/S001", json={
            "sensor_name": "Renamed Detector",
            "status": "WARNING",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["sensor_name"] == "Renamed Detector"
        assert body["status"] == "WARNING"
        # Untouched fields preserved
        assert body["manufacturer"] == "Dräger"
        assert body["sensor_type"] == "GAS"

    async def test_update_not_found_returns_404(self, client: AsyncClient):
        resp = await client.put(
            "/api/v1/sensors/NONEXISTENT",
            json={"sensor_name": "X"},
        )
        assert resp.status_code == 404

    async def test_update_min_gte_max_returns_422(
        self, client: AsyncClient, gas_sensor
    ):
        await _register(client, gas_sensor)
        resp = await client.put("/api/v1/sensors/S001", json={
            "min_value": 99999.0,  # exceeds existing max
        })
        assert resp.status_code == 422
        body = resp.json()
        assert body["success"] is False

    async def test_update_calibration_before_install_returns_422(
        self, client: AsyncClient, gas_sensor
    ):
        await _register(client, gas_sensor)
        resp = await client.put("/api/v1/sensors/S001", json={
            "last_calibration": "2020-01-01",
        })
        assert resp.status_code == 422

    async def test_update_reflects_in_subsequent_get(
        self, client: AsyncClient, gas_sensor
    ):
        await _register(client, gas_sensor)
        await client.put("/api/v1/sensors/S001", json={
            "location_zone": "ZONE_C",
        })
        resp = await client.get("/api/v1/sensors/S001")
        assert resp.json()["location_zone"] == "ZONE_C"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DELETE /sensors/{sensor_id}
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeleteSensor:

    async def test_delete_returns_204_no_body(
        self, client: AsyncClient, gas_sensor
    ):
        await _register(client, gas_sensor)
        resp = await client.delete("/api/v1/sensors/S001")
        assert resp.status_code == 204
        assert resp.content == b""

    async def test_delete_makes_sensor_unfetchable(
        self, client: AsyncClient, gas_sensor
    ):
        await _register(client, gas_sensor)
        await client.delete("/api/v1/sensors/S001")
        resp = await client.get("/api/v1/sensors/S001")
        assert resp.status_code == 404

    async def test_delete_not_found_returns_404(self, client: AsyncClient):
        resp = await client.delete("/api/v1/sensors/NONEXISTENT")
        assert resp.status_code == 404

    async def test_delete_does_not_affect_other_sensors(
        self, client: AsyncClient, gas_sensor, temp_sensor
    ):
        await _register(client, gas_sensor)
        await _register(client, temp_sensor)
        await client.delete("/api/v1/sensors/S001")

        resp = await client.get("/api/v1/sensors")
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["sensor_id"] == "S002"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Spec Endpoint 6: GET /sensors/current
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGetCurrentSensors:

    async def test_response_schema_matches_spec(
        self, client: AsyncClient, gas_sensor, temp_sensor
    ):
        await _register(client, gas_sensor)
        await _register(client, temp_sensor)

        resp = await client.get("/api/v1/sensors/current")
        assert resp.status_code == 200
        body = resp.json()

        # Top-level fields per spec
        assert body["success"] is True
        assert "timestamp" in body
        assert isinstance(body["sensors"], list)
        assert isinstance(body["summary"], dict)

        # Summary fields per spec
        summary = body["summary"]
        assert "total_sensors" in summary
        assert "sensors_normal" in summary
        assert "sensors_warning" in summary
        assert "sensors_critical" in summary
        assert "sensors_offline" in summary
        assert "anomalies_detected" in summary
        assert summary["total_sensors"] == 2

        # Sensor item fields per spec
        sensor_item = body["sensors"][0]
        assert "sensor_id" in sensor_item
        assert "sensor_type" in sensor_item
        assert "location_zone" in sensor_item
        assert "status" in sensor_item
        assert "anomaly_detected" in sensor_item

    async def test_filter_by_sensor_type(
        self, client: AsyncClient, gas_sensor, temp_sensor
    ):
        await _register(client, gas_sensor)
        await _register(client, temp_sensor)

        resp = await client.get("/api/v1/sensors/current?sensor_type=GAS")
        body = resp.json()
        assert len(body["sensors"]) == 1
        assert body["sensors"][0]["sensor_type"] == "GAS"

    async def test_filter_by_zone_id(
        self, client: AsyncClient, gas_sensor, temp_sensor
    ):
        await _register(client, gas_sensor)   # ZONE_A
        await _register(client, temp_sensor)  # ZONE_B

        resp = await client.get("/api/v1/sensors/current?zone_id=ZONE_A")
        body = resp.json()
        assert len(body["sensors"]) == 1
        assert body["sensors"][0]["sensor_id"] == "S001"

    async def test_filter_by_status(
        self, client: AsyncClient, gas_sensor, temp_sensor
    ):
        await _register(client, gas_sensor)
        await _register(client, temp_sensor)
        await client.put("/api/v1/sensors/S002", json={"status": "CRITICAL"})

        resp = await client.get("/api/v1/sensors/current?status=CRITICAL")
        body = resp.json()
        assert len(body["sensors"]) == 1
        assert body["sensors"][0]["sensor_id"] == "S002"

    async def test_summary_counts_match_status_distribution(
        self, client: AsyncClient, gas_sensor, temp_sensor, pressure_sensor
    ):
        await _register(client, gas_sensor)
        await _register(client, temp_sensor)
        await _register(client, pressure_sensor)
        # S001 = NORMAL, S002 → WARNING, S003 → CRITICAL
        await client.put("/api/v1/sensors/S002", json={"status": "WARNING"})
        await client.put("/api/v1/sensors/S003", json={"status": "CRITICAL"})

        resp = await client.get("/api/v1/sensors/current")
        summary = resp.json()["summary"]
        assert summary["total_sensors"] == 3
        assert summary["sensors_normal"] == 1
        assert summary["sensors_warning"] == 1
        assert summary["sensors_critical"] == 1
        assert summary["sensors_offline"] == 0

    async def test_empty_fleet(self, client: AsyncClient):
        resp = await client.get("/api/v1/sensors/current")
        assert resp.status_code == 200
        body = resp.json()
        assert body["sensors"] == []
        assert body["summary"]["total_sensors"] == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Spec Endpoint 7: GET /sensors/{sensor_id}/history
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGetSensorHistory:

    async def test_response_schema_matches_spec(
        self, client: AsyncClient, gas_sensor
    ):
        await _register(client, gas_sensor)
        resp = await client.get("/api/v1/sensors/S001/history")
        assert resp.status_code == 200
        body = resp.json()

        assert body["success"] is True

        # Sensor detail block per spec
        sensor = body["sensor"]
        assert sensor["sensor_id"] == "S001"
        assert sensor["sensor_type"] == "GAS"
        assert sensor["location_zone"] == "ZONE_A"
        assert sensor["equipment_id"] == "EQ001"
        assert sensor["manufacturer"] == "Dräger"
        assert sensor["model"] == "POLYTRON 8700"
        assert sensor["installation_date"] == "2024-01-15"
        assert sensor["last_calibration"] == "2025-06-01"
        assert sensor["next_calibration_due"] == "2025-09-01"
        assert sensor["accuracy_rating"] == 0.99

        # Data arrays (currently stubbed but must be present)
        assert "readings" in body
        assert "statistics" in body
        assert "anomalies_detected" in body
        assert "forecast" in body

    async def test_not_found_returns_404(self, client: AsyncClient):
        resp = await client.get("/api/v1/sensors/NONEXISTENT/history")
        assert resp.status_code == 404
        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "RESOURCE_NOT_FOUND"

    async def test_query_params_accepted(
        self, client: AsyncClient, gas_sensor
    ):
        await _register(client, gas_sensor)
        resp = await client.get(
            "/api/v1/sensors/S001/history?time_range=1h&granularity=5m"
        )
        assert resp.status_code == 200

    async def test_optional_sensor_fields_nullable(
        self, client: AsyncClient, minimal_sensor
    ):
        """Sensor with no optional fields should still return valid history."""
        await _register(client, minimal_sensor)
        resp = await client.get("/api/v1/sensors/S-MIN/history")
        assert resp.status_code == 200
        sensor = resp.json()["sensor"]
        assert sensor["equipment_id"] is None
        assert sensor["manufacturer"] is None
        assert sensor["model"] is None
        assert sensor["installation_date"] is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenAPI / Swagger Documentation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSwaggerDocs:

    async def test_openapi_json_available(self, client: AsyncClient):
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        spec = resp.json()
        paths = spec["paths"]

        # All sensor endpoints visible in OpenAPI
        assert "/api/v1/sensors" in paths
        assert "/api/v1/sensors/current" in paths
        assert "/api/v1/sensors/{sensor_id}" in paths
        assert "/api/v1/sensors/{sensor_id}/history" in paths

    async def test_swagger_ui_available(self, client: AsyncClient):
        resp = await client.get("/docs")
        assert resp.status_code == 200

    async def test_redoc_available(self, client: AsyncClient):
        resp = await client.get("/redoc")
        assert resp.status_code == 200
