"""Anomaly endpoints — stub router.

Business logic will be wired in when use cases are implemented.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/anomalies", tags=["Anomalies"])


# TODO: Wire up use cases
# GET    /anomalies              → List anomalies (filterable)
# GET    /anomalies/{anomaly_id} → Get anomaly details
# PATCH  /anomalies/{anomaly_id}/resolve → Resolve anomaly
