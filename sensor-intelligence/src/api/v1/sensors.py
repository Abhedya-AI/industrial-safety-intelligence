"""Sensor endpoints — stub router.

Business logic will be wired in when use cases are implemented.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/sensors", tags=["Sensors"])


# TODO: Wire up use cases
# POST   /sensors             → RegisterSensor use case
# GET    /sensors              → List sensors (filterable)
# GET    /sensors/{sensor_id}  → Get sensor details
# PATCH  /sensors/{sensor_id}  → Update sensor metadata
# GET    /sensors/{sensor_id}/health → Get health score
