"""Alert endpoints — stub router.

Business logic will be wired in when use cases are implemented.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/alerts", tags=["Alerts"])


# TODO: Wire up use cases
# GET    /alerts                      → List alerts (filterable)
# GET    /alerts/{alert_id}           → Get alert details
# PATCH  /alerts/{alert_id}/acknowledge → Acknowledge alert
# GET    /alerts/summary              → Alert counts by level
