"""Value objects for the Hazard Propagation Engine.

Enumerations aligned with PS-1 SentinelAI Common Domain Names &
Shared Conventions v2.0.

HazardType values from §4.6 (exact match required per §6.1 Rule #1).
PropagationStatus and PropagationLevel are engine-specific enums
derived from the architecture document's hazard spread simulation.
"""

from __future__ import annotations

from enum import Enum


class HazardType(str, Enum):
    """Hazard types as defined in PS-1 v2.0, §4.6.

    Must match EXACTLY as shown in the domain conventions document.
    """

    GAS_LEAK = "GAS_LEAK"
    FIRE = "FIRE"
    SMOKE = "SMOKE"
    CHEMICAL_SPILL = "CHEMICAL_SPILL"
    PPE_VIOLATION = "PPE_VIOLATION"
    FALL_DETECTED = "FALL_DETECTED"
    ELECTRICAL_FAULT = "ELECTRICAL_FAULT"
    TEMPERATURE_ANOMALY = "TEMPERATURE_ANOMALY"
    PRESSURE_ANOMALY = "PRESSURE_ANOMALY"


class PropagationStatus(str, Enum):
    """Status of a hazard propagation simulation.

    PENDING    — simulation queued but not started
    RUNNING    — simulation in progress
    COMPLETED  — propagation analysis finished
    FAILED     — simulation encountered an error
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PropagationLevel(str, Enum):
    """Severity classification for propagation impact.

    Derived from RiskLevel (§4.1) applied to hazard spread context:
      CONTAINED  — hazard confined to origin zone
      SPREADING  — hazard reached adjacent zones
      CRITICAL   — hazard affecting multiple zones / critical infrastructure
      EMERGENCY  — facility-wide propagation requiring immediate evacuation
    """

    CONTAINED = "CONTAINED"
    SPREADING = "SPREADING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class RiskLevel(str, Enum):
    """Risk level classification — PS-1 v2.0, §4.1.

    Must match EXACTLY: LOW | MEDIUM | HIGH | CRITICAL
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
