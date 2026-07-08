"""Domain-specific exceptions for the Compound Risk Intelligence module.

Maps to the shared exception hierarchy while providing
compound-risk-specific error codes and messages.
"""

from __future__ import annotations

from app.shared.exceptions.domain_exceptions import DomainError


class CompoundRiskError(DomainError):
    """Base exception for all compound risk errors."""

    error_code: str = "COMPOUND_RISK_ERROR"

    def __init__(self, message: str = "A compound risk error occurred") -> None:
        super().__init__(message)


class CompoundRiskAnalysisFailedError(CompoundRiskError):
    """Raised when compound risk analysis computation fails."""

    error_code = "COMPOUND_RISK_ANALYSIS_FAILED"

    def __init__(self, reason: str) -> None:
        super().__init__(f"Compound risk analysis failed: {reason}")


class InsufficientScenarioDataError(CompoundRiskError):
    """Raised when the scenario does not contain enough data for analysis."""

    error_code = "INSUFFICIENT_SCENARIO_DATA"

    def __init__(self, missing: list[str] | None = None) -> None:
        self.missing_fields = missing or []
        detail = f" Missing: {', '.join(self.missing_fields)}" if self.missing_fields else ""
        super().__init__(f"Insufficient scenario data for compound risk analysis.{detail}")


class InvalidRiskComponentError(CompoundRiskError):
    """Raised when a risk component value is outside valid bounds."""

    error_code = "INVALID_RISK_COMPONENT"

    def __init__(self, component: str, value: float) -> None:
        self.component = component
        self.value = value
        super().__init__(
            f"Invalid risk component '{component}': {value}. "
            f"Must be between 0.0 and 1.0."
        )


class ZoneNotFoundError(CompoundRiskError):
    """Raised when the specified zone does not exist."""

    error_code = "ZONE_NOT_FOUND"

    def __init__(self, zone_id: str) -> None:
        self.zone_id = zone_id
        super().__init__(f"Zone not found: {zone_id}")


class CompoundRiskModelNotLoadedError(CompoundRiskError):
    """Raised when the compound risk model/rules engine is not available."""

    error_code = "COMPOUND_RISK_MODEL_NOT_LOADED"

    def __init__(self, reason: str = "Model not loaded") -> None:
        super().__init__(f"Compound risk model not available: {reason}")
