"""Feature engineering module for Sensor Intelligence ML pipelines.

Transforms the cleaned/preprocessed dataset into an ML-ready feature matrix
consumed by downstream models:
  - Isolation Forest (unsupervised anomaly detection)
  - Autoencoder (reconstruction-based anomaly detection)
  - Risk Prediction (supervised ensemble — XGBoost / LightGBM)

All functions are pure transformations (no I/O, no training logic).
The module is fully reusable at inference time.

Usage:
    from training.feature_engineering import (
        load_processed, build_features, save_features, run_pipeline,
    )
    df = load_processed("datasets/processed/processed_data.csv")
    df = build_features(df)
    save_features(df, "datasets/processed/features.csv")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Core telemetry columns used for rolling / rate features
TELEMETRY_COLS: list[str] = [
    "Temp", "Pressure", "Humidity", "Vibration", "Speed", "Gas", "Sparks",
]

# Columns for which we compute rate-of-change
RATE_COLS: list[str] = ["Temp", "Pressure", "Gas"]

# Rolling window sizes
ROLLING_WINDOWS: list[int] = [5, 10]

# Interaction feature definitions: (col_a, col_b, output_name)
INTERACTIONS: list[tuple[str, str, str]] = [
    ("Temp", "Gas", "Temp_x_Gas"),
    ("Pressure", "Vibration", "Pressure_x_Vibration"),
    ("Speed", "Age", "Speed_x_Age"),
    ("Gas", "Sparks", "Gas_x_Sparks"),
]

# Sensor ID columns for per-sensor aggregation
SENSOR_ID_COLS: list[str] = [
    "sensor_id_temp",
    "sensor_id_pressure",
    "sensor_id_humidity",
    "sensor_id_vibration",
    "sensor_id_gas",
]

# The primary grouping key for per-sensor statistics
# (zone_id acts as a proxy for a multi-sensor station)
SENSOR_GROUP_KEY: str = "zone_id"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Load processed data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_processed(filepath: str | Path) -> pd.DataFrame:
    """Load the preprocessed dataset produced by the preprocessing stage.

    Parses the timestamp column and sorts by it for correct rolling windows.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
    logger.info(
        "Loaded processed data: %d rows × %d cols from %s",
        len(df), len(df.columns), filepath.name,
    )
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Rolling features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def add_rolling_features(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    windows: Sequence[int] | None = None,
    group_col: str = SENSOR_GROUP_KEY,
) -> pd.DataFrame:
    """Compute rolling mean, std, max, and min for telemetry columns.

    Rolling statistics are computed per sensor group (zone_id) to avoid
    cross-sensor contamination. NaN values from the leading edge of the
    rolling window are left in place (downstream models can handle them
    or callers can drop them).

    Args:
        df: Sorted processed DataFrame.
        columns: Telemetry columns to compute rolling features for.
        windows: List of window sizes.
        group_col: Column to group by before rolling.

    Returns:
        DataFrame with new rolling_* columns appended.
    """
    columns = list(columns or TELEMETRY_COLS)
    windows = list(windows or ROLLING_WINDOWS)
    present_cols = [c for c in columns if c in df.columns]

    new_col_count = 0
    for w in windows:
        for col in present_cols:
            grouped = df.groupby(group_col)[col]
            df[f"rolling_mean_{col}_w{w}"] = grouped.transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
            df[f"rolling_std_{col}_w{w}"] = grouped.transform(
                lambda s: s.rolling(w, min_periods=1).std().fillna(0.0)
            )
            df[f"rolling_max_{col}_w{w}"] = grouped.transform(
                lambda s: s.rolling(w, min_periods=1).max()
            )
            df[f"rolling_min_{col}_w{w}"] = grouped.transform(
                lambda s: s.rolling(w, min_periods=1).min()
            )
            new_col_count += 4

    logger.info(
        "Added %d rolling features (%d cols × %d windows × 4 stats)",
        new_col_count, len(present_cols), len(windows),
    )
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Interaction features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def add_interaction_features(
    df: pd.DataFrame,
    interactions: Sequence[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """Compute pairwise interaction (product) features.

    Interactions capture non-linear relationships between sensor channels
    that individual features cannot express.

    Args:
        df: DataFrame with cleaned numerical columns.
        interactions: List of (col_a, col_b, output_name) tuples.

    Returns:
        DataFrame with new interaction columns appended.
    """
    interactions = list(interactions or INTERACTIONS)
    count = 0
    for col_a, col_b, name in interactions:
        if col_a in df.columns and col_b in df.columns:
            df[name] = df[col_a] * df[col_b]
            count += 1
        else:
            logger.warning("Skipping interaction %s: missing column(s)", name)

    logger.info("Added %d interaction feature(s)", count)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Rate-of-change features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def add_rate_features(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    group_col: str = SENSOR_GROUP_KEY,
) -> pd.DataFrame:
    """Compute point-to-point rate of change (first differences) per sensor.

    For each column, creates a `rate_{col}` column with the difference
    between consecutive readings within the same sensor group.
    The first row in each group is filled with 0.0.

    Args:
        df: Sorted processed DataFrame.
        columns: Columns to compute rate of change for.
        group_col: Column to group by.

    Returns:
        DataFrame with new rate_* columns appended.
    """
    columns = list(columns or RATE_COLS)
    present_cols = [c for c in columns if c in df.columns]
    count = 0

    for col in present_cols:
        df[f"rate_{col}"] = df.groupby(group_col)[col].diff().fillna(0.0)
        count += 1

    logger.info("Added %d rate-of-change feature(s)", count)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Per-sensor statistical aggregation features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def add_sensor_aggregate_features(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    group_col: str = SENSOR_GROUP_KEY,
) -> pd.DataFrame:
    """Compute per-sensor aggregate statistics and merge back as features.

    For each telemetry column, computes the global mean, std, min, and max
    across all readings for that sensor group, then joins these as new
    columns. This gives each row context about its sensor's overall profile.

    Args:
        df: Processed DataFrame.
        columns: Telemetry columns to aggregate.
        group_col: Sensor grouping column.

    Returns:
        DataFrame with new sensor_agg_* columns appended.
    """
    columns = list(columns or TELEMETRY_COLS)
    present_cols = [c for c in columns if c in df.columns]
    count = 0

    for col in present_cols:
        agg = df.groupby(group_col)[col].agg(["mean", "std", "min", "max"])
        agg.columns = [
            f"sensor_agg_mean_{col}",
            f"sensor_agg_std_{col}",
            f"sensor_agg_min_{col}",
            f"sensor_agg_max_{col}",
        ]
        # Fill NaN std (single-value groups) with 0
        agg[f"sensor_agg_std_{col}"] = agg[f"sensor_agg_std_{col}"].fillna(0.0)
        df = df.merge(agg, left_on=group_col, right_index=True, how="left")
        count += 4

    logger.info("Added %d per-sensor aggregate feature(s)", count)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. ML-ready output helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of ML-ready numeric feature columns.

    Excludes identifiers, timestamps, targets, and string columns.
    """
    exclude = {
        "timestamp", "zone_id", "equipment_id", "sensor_type",
        "sensor_id_temp", "sensor_id_pressure", "sensor_id_humidity",
        "sensor_id_vibration", "sensor_id_gas",
        "Risk", "Accident",
    }
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)
    ]


def get_isolation_forest_features(df: pd.DataFrame) -> list[str]:
    """Feature subset optimised for Isolation Forest anomaly detection.

    Uses core telemetry + rolling stats + interactions.
    Excludes per-sensor aggregates (IF should detect local anomalies).
    """
    cols = get_feature_columns(df)
    return [c for c in cols if not c.startswith("sensor_agg_")]


def get_autoencoder_features(df: pd.DataFrame) -> list[str]:
    """Feature subset for Autoencoder reconstruction-based anomaly detection.

    Core telemetry + rolling features. Excludes interactions and aggregates
    to keep the reconstruction space focused on raw sensor patterns.
    """
    cols = get_feature_columns(df)
    return [
        c for c in cols
        if not c.startswith("sensor_agg_")
        and not c.endswith(("_x_Gas", "_x_Vibration", "_x_Age", "_x_Sparks"))
    ]


def get_risk_prediction_features(df: pd.DataFrame) -> list[str]:
    """Full feature set for supervised risk prediction (XGBoost / LightGBM).

    Uses everything except the targets (Risk, Accident).
    """
    return get_feature_columns(df)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Build features (orchestrator)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence.

    1. Rolling features (mean, std, max, min) for windows [5, 10]
    2. Interaction features (Temp×Gas, Pressure×Vibration, Speed×Age, Gas×Sparks)
    3. Rate-of-change features (Temp, Pressure, Gas)
    4. Per-sensor aggregate statistics (mean, std, min, max per zone_id)

    Args:
        df: Cleaned/preprocessed DataFrame (from preprocessing stage).

    Returns:
        ML-ready DataFrame with all engineered features appended.
    """
    df = df.copy()
    df = add_rolling_features(df)
    df = add_interaction_features(df)
    df = add_rate_features(df)
    df = add_sensor_aggregate_features(df)

    logger.info(
        "Feature engineering complete: %d rows × %d columns "
        "(IF=%d, AE=%d, Risk=%d features)",
        len(df), len(df.columns),
        len(get_isolation_forest_features(df)),
        len(get_autoencoder_features(df)),
        len(get_risk_prediction_features(df)),
    )
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Save feature dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def save_features(df: pd.DataFrame, filepath: str | Path) -> Path:
    """Save the feature-engineered DataFrame to CSV.

    Creates parent directories if they do not exist.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info("Saved features: %d rows × %d cols → %s", len(df), len(df.columns), filepath)
    return filepath


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_pipeline(
    processed_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Execute the full feature engineering pipeline end-to-end.

    Args:
        processed_path: Path to preprocessed CSV (from preprocessing stage).
        output_path: Path to write the feature-engineered CSV.

    Returns:
        The feature-engineered DataFrame.
    """
    df = load_processed(processed_path)
    df = build_features(df)
    save_features(df, output_path)
    return df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    inp = sys.argv[1] if len(sys.argv) > 1 else "datasets/processed/processed_data.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "datasets/processed/features.csv"

    run_pipeline(inp, out)
