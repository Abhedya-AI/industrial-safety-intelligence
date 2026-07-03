"""Risk Prediction feature builder.

Collects and assembles a feature vector ready for the XGBoost / LightGBM
risk prediction ensemble.  **Does NOT duplicate** the Sensor Intelligence
preprocessing or feature-engineering logic — it delegates to:

    training.feature_engineering   (rolling, interaction, rate, aggregate)
    training.preprocessing         (constants, scaling helpers)

This module adds the Risk Prediction-specific features on top:

    - Anomaly scores   (Isolation Forest + Autoencoder)
    - Sensor health score
    - Contextual fields (workers, training, experience, age, service days)

Architecture notes:
    - Pure functions — no I/O, no database, no model loading.
    - Designed for both batch (DataFrame) and single-row (dict) usage.
    - Output is a flat numeric vector aligned with the columns produced
      by ``training.feature_engineering.get_risk_prediction_features()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Reuse the Sensor Intelligence feature engineering — NO duplication
from training.feature_engineering import (
    TELEMETRY_COLS,
    add_interaction_features,
    add_rate_features,
    add_rolling_features,
    add_sensor_aggregate_features,
    build_features,
    get_risk_prediction_features,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Core sensor telemetry features (from the raw dataset)
SENSOR_FEATURES: list[str] = [
    "Temp", "Pressure", "Humidity", "Gas", "Vibration", "Speed",
]

# Contextual / workforce features
CONTEXT_FEATURES: list[str] = [
    "Workers", "Age", "Service_Days",
]

# Categorical features (one-hot encoded in dataset)
CATEGORICAL_ONEHOT_PREFIXES: list[str] = [
    "Factory_", "Region_", "Shift_", "Exp_", "Training_", "Alarm_",
]

# Risk-prediction-specific features added by this module
RISK_EXTRA_FEATURES: list[str] = [
    "anomaly_score_if",       # Isolation Forest anomaly score
    "anomaly_score_ae",       # Autoencoder anomaly score
    "sensor_health_score",    # Health score 0–100
]

# Default values for optional features (safe fallbacks)
_DEFAULTS: Dict[str, float] = {
    "anomaly_score_if": 0.0,
    "anomaly_score_ae": 0.0,
    "sensor_health_score": 100.0,
    "Workers": 0.0,
    "Age": 0.0,
    "Service_Days": 0.0,
    "Sparks": 0.0,
    "Training_Yes": 0.0,
    "Training_No": 1.0,
    "Exp_Senior": 0.0,
    "Exp_Junior": 1.0,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data class: single-row input
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class RiskFeatureInput:
    """Structured input for building a single risk feature vector.

    Accepts raw sensor values and optional enrichment fields.
    Any field left as None will receive a safe default.
    """

    # ── Sensor telemetry ──
    temperature: float = 0.0
    pressure: float = 0.0
    humidity: float = 0.0
    gas: float = 0.0
    vibration: float = 0.0
    speed: float = 0.0
    sparks: float = 0.0

    # ── Anomaly enrichment ──
    anomaly_score_if: float = 0.0
    anomaly_score_ae: float = 0.0

    # ── Health enrichment ──
    sensor_health_score: float = 100.0

    # ── Contextual / workforce ──
    workers: int = 0
    age: float = 0.0
    service_days: float = 0.0
    training: bool = False
    experience: str = "Junior"   # "Junior" | "Senior"

    # ── Zone / factory / shift ──
    zone_id: str = "ZONE_A"
    shift: str = "Day"           # "Day" | "Night"
    region: str = "Industrial_Zone"  # "Industrial_Zone" | "Urban" | "Rural"
    alarm: str = "Off"           # "On" | "Off"

    def to_flat_dict(self) -> Dict[str, float]:
        """Convert to the flat numeric dictionary the feature builder expects.

        Maps human-friendly names to dataset column names and one-hot encodes
        the categorical fields.
        """
        d: Dict[str, float] = {
            # Telemetry (dataset column names)
            "Temp": self.temperature,
            "Pressure": self.pressure,
            "Humidity": self.humidity,
            "Gas": self.gas,
            "Vibration": self.vibration,
            "Speed": self.speed,
            "Sparks": self.sparks,
            # Context
            "Workers": float(self.workers),
            "Age": self.age,
            "Service_Days": self.service_days,
            # Anomaly enrichment
            "anomaly_score_if": self.anomaly_score_if,
            "anomaly_score_ae": self.anomaly_score_ae,
            # Health enrichment
            "sensor_health_score": self.sensor_health_score,
        }

        # One-hot: Training
        d["Training_Yes"] = 1.0 if self.training else 0.0
        d["Training_No"] = 0.0 if self.training else 1.0

        # One-hot: Experience
        d["Exp_Senior"] = 1.0 if self.experience == "Senior" else 0.0
        d["Exp_Junior"] = 1.0 if self.experience == "Junior" else 0.0

        # One-hot: Factory / Zone
        for zone in ["ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D"]:
            factory = zone.replace("ZONE", "Factory")
            d[f"Factory_{factory}"] = 1.0 if self.zone_id == zone else 0.0

        # One-hot: Region
        for reg in ["Industrial_Zone", "Urban", "Rural"]:
            d[f"Region_{reg}"] = 1.0 if self.region == reg else 0.0

        # One-hot: Shift
        d["Shift_Day"] = 1.0 if self.shift == "Day" else 0.0
        d["Shift_Night"] = 1.0 if self.shift == "Night" else 0.0

        # One-hot: Alarm
        d["Alarm_On"] = 1.0 if self.alarm == "On" else 0.0
        d["Alarm_Off"] = 1.0 if self.alarm == "Off" else 0.0

        return d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core feature building functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_risk_features_from_dict(
    raw: Dict[str, float],
    *,
    fill_defaults: bool = True,
) -> Dict[str, float]:
    """Build a risk-ready feature dict from a flat key→value map.

    This is the **single-row** path used during real-time inference.
    It applies safe defaults for any missing keys and adds the
    risk-specific extra features.

    Args:
        raw: Flat mapping of feature_name → value.  Column names must
             match the dataset conventions (e.g. ``"Temp"``, ``"Gas"``).
        fill_defaults: If True, fill missing keys with safe defaults.

    Returns:
        A dict containing all numeric features needed by the risk model.
    """
    features = dict(raw)

    if fill_defaults:
        for key, default in _DEFAULTS.items():
            features.setdefault(key, default)
        for col in SENSOR_FEATURES:
            features.setdefault(col, 0.0)

    return features


def build_risk_features_from_input(
    inp: RiskFeatureInput,
) -> Dict[str, float]:
    """Build a risk-ready feature dict from a structured ``RiskFeatureInput``.

    Convenience wrapper around ``build_risk_features_from_dict``.
    """
    return build_risk_features_from_dict(inp.to_flat_dict())


def build_risk_feature_vector(
    features: Dict[str, float],
    column_order: Sequence[str],
) -> np.ndarray:
    """Convert a feature dict to a 1-D numpy array in a fixed column order.

    Missing keys default to 0.0.

    Args:
        features: Feature name → value mapping.
        column_order: The column order expected by the model.

    Returns:
        1-D ndarray of shape (len(column_order),).
    """
    return np.array(
        [features.get(col, 0.0) for col in column_order],
        dtype=np.float64,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Batch feature building (DataFrame path)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_risk_features_batch(
    df: pd.DataFrame,
    *,
    anomaly_scores_if: Optional[Sequence[float]] = None,
    anomaly_scores_ae: Optional[Sequence[float]] = None,
    health_scores: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Build a full risk-ready DataFrame from preprocessed sensor data.

    Delegates **all** rolling, interaction, rate-of-change, and per-sensor
    aggregate features to ``training.feature_engineering.build_features``.
    Then enriches with risk-specific columns.

    Args:
        df: Preprocessed sensor DataFrame (output of ``training.preprocessing``).
        anomaly_scores_if: Per-row Isolation Forest anomaly scores.
        anomaly_scores_ae: Per-row Autoencoder anomaly scores.
        health_scores: Per-row sensor health scores (0–100).

    Returns:
        DataFrame with all ML-ready feature columns.
    """
    # 1) Delegate to Sensor Intelligence feature engineering
    df = build_features(df)

    # 2) Attach risk-specific enrichment columns
    n = len(df)
    df["anomaly_score_if"] = (
        list(anomaly_scores_if) if anomaly_scores_if is not None
        else [0.0] * n
    )
    df["anomaly_score_ae"] = (
        list(anomaly_scores_ae) if anomaly_scores_ae is not None
        else [0.0] * n
    )
    df["sensor_health_score"] = (
        list(health_scores) if health_scores is not None
        else [100.0] * n
    )

    logger.info(
        "Built risk features: %d rows × %d cols "
        "(includes %d risk-specific enrichment cols)",
        len(df), len(df.columns), len(RISK_EXTRA_FEATURES),
    )
    return df


def get_risk_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the full list of numeric feature columns for risk prediction.

    Extends ``get_risk_prediction_features`` with the 3 risk-specific
    enrichment columns.
    """
    base_cols = get_risk_prediction_features(df)
    extra = [c for c in RISK_EXTRA_FEATURES if c in df.columns and c not in base_cols]
    return base_cols + extra


def extract_feature_matrix(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> np.ndarray:
    """Extract a 2-D numpy feature matrix from a DataFrame.

    Args:
        df: Feature-engineered DataFrame.
        columns: Explicit column order.  If None, uses
                 ``get_risk_feature_columns(df)``.

    Returns:
        2-D ndarray of shape (n_rows, n_features).
    """
    cols = columns or get_risk_feature_columns(df)
    present = [c for c in cols if c in df.columns]
    return df[present].fillna(0.0).values.astype(np.float64)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Validation helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def validate_features(
    features: Dict[str, float],
    required: Optional[Sequence[str]] = None,
) -> list[str]:
    """Check that required features are present and finite.

    Args:
        features: Feature dict to validate.
        required: List of required feature names.  Defaults to
                  ``SENSOR_FEATURES``.

    Returns:
        List of missing or invalid feature names (empty = valid).
    """
    required = list(required or SENSOR_FEATURES)
    issues: list[str] = []
    for key in required:
        if key not in features:
            issues.append(key)
        elif not np.isfinite(features[key]):
            issues.append(key)
    return issues
