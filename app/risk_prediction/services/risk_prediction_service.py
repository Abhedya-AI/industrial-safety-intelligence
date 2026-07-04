"""Risk Prediction service — runtime inference without retraining.

Loads the pre-trained XGBoost model once, then provides synchronous
prediction methods used by the API layer.

Architecture:
    1. Model + scaler + feature metadata are loaded from disk on first use
       (lazy singleton).
    2. Feature vectors are built via the reusable feature_builder module.
    3. Predictions are transformed into structured response objects
       and persisted via the repository layer.
    4. No model training occurs at runtime.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.risk_prediction.domain.exceptions import (
    InsufficientFeaturesError,
    RiskModelNotLoadedError,
    RiskPredictionFailedError,
)
from app.risk_prediction.domain.value_objects import PredictionStatus, RiskLevel
from app.risk_prediction.models.risk_prediction_model import RiskPredictionModel
from app.risk_prediction.preprocessing.feature_builder import (
    RiskFeatureInput,
    build_risk_feature_vector,
    build_risk_features_from_dict,
    build_risk_features_from_input,
    validate_features,
)
from app.risk_prediction.repositories.risk_prediction_repository import (
    RiskPredictionRepository,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_MODEL_PATH = "models/risk_prediction_xgboost.pkl"
DEFAULT_SCALER_PATH = "models/risk_prediction_scaler.pkl"
DEFAULT_FEATURES_META_PATH = "models/risk_prediction_features.json"

MODEL_NAME = "xgboost_risk_prediction"
MODEL_VERSION = "1.0.0"

# Risk level thresholds (from architecture doc)
_RISK_THRESHOLDS: list[tuple[float, RiskLevel]] = [
    (0.75, RiskLevel.CRITICAL),
    (0.50, RiskLevel.HIGH),
    (0.25, RiskLevel.MEDIUM),
    (0.00, RiskLevel.LOW),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model loader (lazy singleton)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _ModelHolder:
    """Process-global singleton holding the loaded model, scaler, and column order.

    Loaded lazily on first prediction, never retrained at runtime.
    """

    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self.column_order: list[str] = []
        self.feature_importances: dict[str, float] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        scaler_path: str | Path = DEFAULT_SCALER_PATH,
        features_meta_path: str | Path = DEFAULT_FEATURES_META_PATH,
    ) -> None:
        """Load model artefacts from disk.  Idempotent — skips if already loaded."""
        if self._loaded:
            return

        import pickle

        model_path = Path(model_path)
        scaler_path = Path(scaler_path)
        features_meta_path = Path(features_meta_path)

        # Load model
        if not model_path.exists():
            raise RiskModelNotLoadedError(str(model_path))
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Loaded risk model from %s", model_path)

        # Load scaler
        if not scaler_path.exists():
            raise RiskModelNotLoadedError(str(scaler_path))
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        logger.info("Loaded risk scaler from %s", scaler_path)

        # Load feature metadata (column order + importances)
        if features_meta_path.exists():
            with open(features_meta_path) as f:
                meta = json.load(f)
            self.column_order = meta.get("column_order", [])
            names = meta.get("feature_names", [])
            imps = meta.get("importances", [])
            self.feature_importances = dict(zip(names, imps))
            logger.info(
                "Loaded feature metadata: %d columns, %d importances",
                len(self.column_order), len(self.feature_importances),
            )
        else:
            logger.warning(
                "Feature metadata not found at %s — using model defaults",
                features_meta_path,
            )
            if hasattr(self.model, "feature_names_in_"):
                self.column_order = list(self.model.feature_names_in_)

        self._loaded = True
        logger.info("Risk prediction model ready (features=%d)", len(self.column_order))

    def reset(self) -> None:
        """Reset for testing purposes."""
        self.model = None
        self.scaler = None
        self.column_order = []
        self.feature_importances = {}
        self._loaded = False


# Global singleton
_model_holder = _ModelHolder()


def get_model_holder() -> _ModelHolder:
    """Provide access to the model holder (useful for DI / testing)."""
    return _model_holder


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pure prediction helpers (no I/O)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def classify_risk_level(probability: float) -> RiskLevel:
    """Map an accident probability to a risk level.

    Uses the architecture-defined thresholds:
        >= 0.75 → CRITICAL
        >= 0.50 → HIGH
        >= 0.25 → MEDIUM
        <  0.25 → LOW
    """
    for threshold, level in _RISK_THRESHOLDS:
        if probability >= threshold:
            return level
    return RiskLevel.LOW


def probability_to_risk_score(probability: float) -> int:
    """Convert probability (0.0–1.0) to a discrete risk score (0–100)."""
    return max(0, min(100, int(round(probability * 100))))


def calculate_confidence(
    probability: float,
    model_holder: Optional[_ModelHolder] = None,
) -> float:
    """Calculate a confidence score for the prediction.

    Uses distance-from-decision-boundary heuristic:
    predictions far from 0.5 are more confident.
    """
    # Distance from decision boundary (0.5)
    distance = abs(probability - 0.5) * 2  # 0.0 → 0.0, 0.5 distance → 1.0
    # Apply sigmoid-like smoothing for more natural distribution
    confidence = 0.5 + 0.5 * distance
    return round(min(1.0, max(0.0, confidence)), 4)


def build_explanation(
    risk_level: RiskLevel,
    probability: float,
    top_features: list[tuple[str, float]],
) -> str:
    """Generate a human-readable risk explanation."""
    feature_parts = []
    for name, importance in top_features[:5]:
        feature_parts.append(f"{name} ({importance:.1%})")

    risk_desc = {
        RiskLevel.LOW: "within normal operating parameters",
        RiskLevel.MEDIUM: "showing elevated risk indicators",
        RiskLevel.HIGH: "at high risk — attention required",
        RiskLevel.CRITICAL: "at critical risk — immediate action recommended",
    }

    desc = risk_desc.get(risk_level, "at undetermined risk")
    features_str = ", ".join(feature_parts) if feature_parts else "no dominant factor"

    return (
        f"The environment is {desc} with an accident probability of "
        f"{probability:.1%}. Key contributing factors: {features_str}."
    )


def get_top_contributing_factors(
    feature_importances: dict[str, float],
    feature_values: dict[str, float],
    n: int = 5,
) -> list[dict[str, Any]]:
    """Get top N contributing factors with their values."""
    sorted_features = sorted(
        feature_importances.items(), key=lambda x: x[1], reverse=True,
    )[:n]

    factors = []
    for name, importance in sorted_features:
        factors.append({
            "factor": name,
            "weight": round(importance, 4),
            "current_value": str(round(feature_values.get(name, 0.0), 4)),
            "contribution": round(importance, 4),
        })
    return factors


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RiskPredictionService
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RiskPredictionService:
    """Service for making risk predictions using the pre-trained XGBoost model.

    Responsibilities:
        1. Load model artefacts (lazy, once)
        2. Build feature vectors from raw input
        3. Predict accident probability
        4. Classify risk level
        5. Compute confidence score
        6. Persist predictions via repository
        7. Return structured response objects

    NOT responsible for:
        - Model training / retraining
        - API routing / HTTP concerns
    """

    def __init__(
        self,
        repository: RiskPredictionRepository,
        *,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        scaler_path: str | Path = DEFAULT_SCALER_PATH,
        features_meta_path: str | Path = DEFAULT_FEATURES_META_PATH,
        model_holder: Optional[_ModelHolder] = None,
    ) -> None:
        self._repo = repository
        self._model_path = model_path
        self._scaler_path = scaler_path
        self._features_meta_path = features_meta_path
        self._holder = model_holder or _model_holder

    # ── Model lifecycle ──

    def _ensure_model(self) -> None:
        """Load model if not yet loaded.  Raises on failure."""
        if not self._holder.is_loaded:
            try:
                self._holder.load(
                    self._model_path,
                    self._scaler_path,
                    self._features_meta_path,
                )
            except Exception as exc:
                logger.exception("Failed to load risk model")
                raise RiskModelNotLoadedError(str(self._model_path)) from exc

    @property
    def is_model_loaded(self) -> bool:
        return self._holder.is_loaded

    # ── Core prediction ──

    async def predict_from_features(
        self,
        features: Dict[str, float],
        *,
        sensor_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        include_breakdown: bool = False,
        include_explanation: bool = False,
        persist: bool = True,
    ) -> RiskPredictionModel:
        """Make a risk prediction from a raw feature dictionary.

        This is the primary entry point for the API layer.

        Args:
            features: Feature name → value mapping.
            sensor_id: Optional sensor context.
            equipment_id: Optional equipment context.
            zone_id: Optional zone context.
            include_breakdown: Attach per-factor breakdown.
            include_explanation: Attach human-readable explanation.
            persist: Save prediction to database.

        Returns:
            Persisted RiskPredictionModel with all fields populated.

        Raises:
            RiskModelNotLoadedError: Model files not available.
            InsufficientFeaturesError: Required features missing/invalid.
            RiskPredictionFailedError: Inference error.
        """
        self._ensure_model()

        # 1. Build feature dict (fill defaults for missing keys)
        feature_dict = build_risk_features_from_dict(features)

        # 2. Validate required features
        issues = validate_features(feature_dict)
        if issues:
            raise InsufficientFeaturesError(issues)

        # 3. Build ordered feature vector
        column_order = self._holder.column_order
        vector = build_risk_feature_vector(feature_dict, column_order)

        # 4. Scale
        try:
            scaled = self._holder.scaler.transform(vector.reshape(1, -1))
            if np.isnan(scaled).any():
                scaled = np.nan_to_num(scaled, nan=0.0)
        except Exception as exc:
            raise RiskPredictionFailedError(f"Feature scaling failed: {exc}") from exc

        # 5. Predict
        try:
            proba = self._holder.model.predict_proba(scaled)
            accident_probability = float(proba[0, 1])
        except Exception as exc:
            raise RiskPredictionFailedError(f"Model inference failed: {exc}") from exc

        # 6. Post-process
        risk_level = classify_risk_level(accident_probability)
        risk_score = probability_to_risk_score(accident_probability)
        confidence = calculate_confidence(accident_probability)
        now = datetime.now(timezone.utc)

        # 7. Build ORM object
        prediction = RiskPredictionModel(
            id=str(uuid.uuid4()),
            sensor_id=sensor_id,
            equipment_id=equipment_id,
            zone_id=zone_id,
            prediction_timestamp=now,
            accident_probability=round(accident_probability, 6),
            predicted_risk_score=risk_score,
            risk_level=risk_level.value,
            confidence_score=confidence,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            status=PredictionStatus.COMPLETED.value,
            created_at=now,
        )

        # Optional enrichment
        if include_breakdown:
            top_factors = get_top_contributing_factors(
                self._holder.feature_importances, feature_dict,
            )
            prediction.top_contributing_factors = json.dumps(top_factors)

        if include_explanation:
            top_features_list = sorted(
                self._holder.feature_importances.items(),
                key=lambda x: x[1], reverse=True,
            )[:5]
            prediction.explanation = build_explanation(
                risk_level, accident_probability, top_features_list,
            )

        # 8. Persist
        if persist:
            prediction = await self._repo.create_prediction(prediction)
            logger.info(
                "Prediction %s: prob=%.4f score=%d level=%s confidence=%.4f",
                prediction.id, accident_probability, risk_score,
                risk_level.value, confidence,
            )

        return prediction

    async def predict_from_input(
        self,
        inp: RiskFeatureInput,
        *,
        sensor_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        include_breakdown: bool = False,
        include_explanation: bool = False,
        persist: bool = True,
    ) -> RiskPredictionModel:
        """Make a risk prediction from a structured RiskFeatureInput.

        Convenience wrapper that converts the dataclass to a feature dict.
        """
        feature_dict = build_risk_features_from_input(inp)
        return await self.predict_from_features(
            feature_dict,
            sensor_id=sensor_id,
            equipment_id=equipment_id,
            zone_id=zone_id or inp.zone_id,
            include_breakdown=include_breakdown,
            include_explanation=include_explanation,
            persist=persist,
        )

    # ── Query helpers ──

    async def get_prediction(self, prediction_id: str) -> Optional[RiskPredictionModel]:
        """Retrieve a prediction by ID."""
        return await self._repo.get_prediction(prediction_id)

    async def get_latest_prediction(
        self,
        sensor_id: Optional[str] = None,
        zone_id: Optional[str] = None,
    ) -> Optional[RiskPredictionModel]:
        """Get the most recent prediction."""
        return await self._repo.get_latest_prediction(
            sensor_id=sensor_id, zone_id=zone_id,
        )

    async def get_prediction_history(
        self,
        sensor_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[RiskPredictionModel], int]:
        """Get paginated prediction history with total count."""
        predictions = await self._repo.get_prediction_history(
            sensor_id=sensor_id, zone_id=zone_id,
            risk_level=risk_level, offset=offset, limit=limit,
        )
        total = await self._repo.count_predictions(
            sensor_id=sensor_id, zone_id=zone_id, risk_level=risk_level,
        )
        return predictions, total

    # ── Model info ──

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for health checks."""
        return {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "is_loaded": self._holder.is_loaded,
            "feature_count": len(self._holder.column_order),
            "model_path": str(self._model_path),
        }
