"""Threshold endpoints — stub router.

Business logic will be wired in when use cases are implemented.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/thresholds", tags=["Thresholds"])


# TODO: Wire up use cases
# POST   /thresholds              → Create threshold config
# GET    /thresholds              → List thresholds (filterable)
# PUT    /thresholds/{threshold_id} → Update threshold
# DELETE /thresholds/{threshold_id} → Deactivate threshold
