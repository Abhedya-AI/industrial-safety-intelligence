"""Domain-specific exceptions for the Risk Prediction module.

Maps to the shared exception hierarchy while providing
risk-specific error codes and messages.
"""

from __future__ import annotations

from app.shared.exceptions.domain_exceptions import DomainError


class RiskPredictionError(DomainError):
    """Base exception for all risk prediction errors."""

    error_code: str = "RISK_PREDICTION_ERROR"

    def __init__(self, message: str = "A risk prediction error occurred") -> None:
        super().__init__(message)


class RiskModelNotLoadedError(RiskPredictionError):
    """Raised when the risk model is not available for inference."""

    error_code = "RISK_MODEL_NOT_LOADED"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(
            f"Risk prediction model not loaded: {model_name}"
        )


class RiskPredictionFailedError(RiskPredictionError):
    """Raised when risk prediction computation fails."""

    error_code = "RISK_PREDICTION_FAILED"

    def __init__(self, reason: str) -> None:
        super().__init__(f"Risk prediction failed: {reason}")


class InsufficientFeaturesError(RiskPredictionError):
    """Raised when the feature vector is incomplete or invalid."""

    error_code = "INSUFFICIENT_FEATURES"

    def __init__(self, missing: list[str] | None = None) -> None:
        self.missing_features = missing or []
        detail = f" Missing: {', '.join(self.missing_features)}" if self.missing_features else ""
        super().__init__(f"Insufficient features for risk prediction.{detail}")


class InvalidRiskScoreError(RiskPredictionError):
    """Raised when a computed risk score is outside valid bounds."""

    error_code = "INVALID_RISK_SCORE"

    def __init__(self, score: float) -> None:
        self.score = score
        super().__init__(
            f"Invalid risk score: {score}. Must be between 0 and 100."
        )


class StaleDataError(RiskPredictionError):
    """Raised when input data is too old for a reliable prediction."""

    error_code = "STALE_DATA"

    def __init__(self, sensor_id: str, age_seconds: float) -> None:
        self.sensor_id = sensor_id
        self.age_seconds = age_seconds
        super().__init__(
            f"Data for sensor {sensor_id} is {age_seconds:.0f}s old — "
            f"too stale for reliable prediction."
        )
