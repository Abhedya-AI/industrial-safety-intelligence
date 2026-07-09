"""Propagation engine configuration.

All tuneable parameters for the hazard propagation algorithm.
Follows the same frozen-dataclass pattern used by CompoundRiskWeights.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PropagationConfig:
    """Configuration for the hazard propagation algorithm.

    Attributes:
        propagation_decay_factor: Multiplicative decay per hop.
            At each hop the risk is multiplied by this factor.
            Range: 0.0–1.0. Default 0.6 (40% loss per hop).
        max_depth: Maximum number of zone hops to simulate.
            Range: 1–10. Default 3.
        minimum_propagation_threshold: Minimum propagation probability
            below which a zone is no longer considered affected.
            Range: 0.0–1.0. Default 0.1 (10%).
        base_impact_radius_meters: Base radius (meters) for the origin
            zone. Decays with each hop. Default 50.0.
        time_per_hop_minutes: Estimated time for hazard to traverse
            one connection. Default 5.0.
        equipment_risk_weight: Weight applied to equipment health
            score when computing zone impact. Default 0.3.
        worker_risk_weight: Weight applied to worker count when
            computing zone impact. Default 0.5.
        baseline_risk_weight: Weight applied to zone baseline risk
            level when computing zone impact. Default 0.2.
    """

    propagation_decay_factor: float = 0.6
    max_depth: int = 3
    minimum_propagation_threshold: float = 0.1
    base_impact_radius_meters: float = 50.0
    time_per_hop_minutes: float = 5.0
    equipment_risk_weight: float = 0.3
    worker_risk_weight: float = 0.5
    baseline_risk_weight: float = 0.2

    def __post_init__(self):
        if not 0.0 <= self.propagation_decay_factor <= 1.0:
            raise ValueError(
                f"propagation_decay_factor must be 0.0–1.0, "
                f"got {self.propagation_decay_factor}"
            )
        if not 1 <= self.max_depth <= 10:
            raise ValueError(
                f"max_depth must be 1–10, got {self.max_depth}"
            )
        if not 0.0 <= self.minimum_propagation_threshold <= 1.0:
            raise ValueError(
                f"minimum_propagation_threshold must be 0.0–1.0, "
                f"got {self.minimum_propagation_threshold}"
            )


# ── Hazard-type specific decay overrides ──

HAZARD_DECAY_OVERRIDES: dict[str, float] = {
    "GAS_LEAK": 0.7,          # Gas spreads easily
    "FIRE": 0.5,              # Fire less likely to jump zones
    "SMOKE": 0.8,             # Smoke travels easily via HVAC
    "CHEMICAL_SPILL": 0.4,    # Liquid stays localised
    "ELECTRICAL_FAULT": 0.3,  # Electrical faults rarely propagate
    "TEMPERATURE_ANOMALY": 0.6,
    "PRESSURE_ANOMALY": 0.5,
    "PPE_VIOLATION": 0.0,     # Does not propagate
    "FALL_DETECTED": 0.0,     # Does not propagate
}

# ── Risk level score mapping ──

RISK_LEVEL_SCORES: dict[str, float] = {
    "LOW": 10.0,
    "MEDIUM": 40.0,
    "HIGH": 70.0,
    "CRITICAL": 95.0,
}
