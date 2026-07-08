"""Value objects for the Compound Risk Intelligence module.

Enumerations aligned with the PS-1 SentinelAI Common Domain Names
& Shared Conventions v2.0 document.

Uses EXACT enum values as specified in Section 4.1:
  RiskLevel: LOW | MEDIUM | HIGH | CRITICAL

All values are UPPERCASE as per Section 6.1 Critical Rule #1:
  "Use EXACT enum values — no variations (CRITICAL not Critical)"
"""

from __future__ import annotations

from enum import Enum


class RiskLevel(str, Enum):
    """Risk level classification.

    Must match EXACTLY: LOW | MEDIUM | HIGH | CRITICAL
    (per PS-1 Common Domain Names v2.0, Section 4.1)
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CompoundRiskStatus(str, Enum):
    """Status of a compound risk analysis computation."""

    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class HazardType(str, Enum):
    """Hazard types as defined in PS-1 Common Domain Names v2.0, Section 4.6.

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


class ShiftType(str, Enum):
    """Shift types as defined in PS-1 Common Domain Names v2.0.

    Worker.shift: MORNING | AFTERNOON | NIGHT
    """

    MORNING = "MORNING"
    AFTERNOON = "AFTERNOON"
    NIGHT = "NIGHT"


class PermitType(str, Enum):
    """Permit types referenced in compound risk scenarios."""

    HOT_WORK = "HOT_WORK"
    CONFINED_SPACE = "CONFINED_SPACE"
    ELECTRICAL = "ELECTRICAL"
    EXCAVATION = "EXCAVATION"
    GENERAL = "GENERAL"
