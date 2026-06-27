"""Reading endpoints — stub router.

Business logic will be wired in when use cases are implemented.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/readings", tags=["Readings"])


# TODO: Wire up use cases
# POST   /readings              → IngestReading use case
# POST   /readings/batch        → Batch ingest readings
# GET    /readings               → Query readings (filterable)
# GET    /readings/{sensor_id}/latest → Latest reading
# GET    /readings/{sensor_id}/stats  → Aggregated stats
