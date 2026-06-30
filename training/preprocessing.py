"""Preprocessing pipeline for the Industrial Fire Risk dataset.

Reusable module that transforms the raw CSV into a clean, feature-engineered
dataset ready for downstream ML modules (Isolation Forest, Autoencoder,
Risk Prediction). Contains NO training logic.

Usage:
    from training.preprocessing import load_raw_data, clean_data, feature_engineer, save_processed

    df = load_raw_data("path/to/raw.csv")
    df = clean_data(df)
    df = feature_engineer(df)
    save_processed(df, "path/to/output.csv")

Architecture alignment:
    Generates derived fields (sensor_id, sensor_type, equipment_id, zone_id)
    required by the SentinelAI platform schemas.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Pressure outlier threshold (values above this are decimal-shifted by 10x)
PRESSURE_OUTLIER_THRESHOLD = 50.0
PRESSURE_CORRECTION_FACTOR = 10.0

# Numerical features to standardize (post-cleaning)
NUMERICAL_FEATURES = [
    "Temp", "Pressure", "Humidity", "Vibration", "Speed", "Gas", "Sparks",
]

# Categorical features to one-hot encode
CATEGORICAL_FEATURES = [
    "Factory", "Region", "Shift", "Exp", "Training", "Alarm",
]

# Factory → zone_id mapping (architecture alignment)
FACTORY_TO_ZONE: dict[str, str] = {
    "Factory_A": "ZONE_A",
    "Factory_B": "ZONE_B",
    "Factory_C": "ZONE_C",
    "Factory_D": "ZONE_D",
}

# Factory → equipment_id mapping (deterministic)
FACTORY_TO_EQUIPMENT: dict[str, str] = {
    "Factory_A": "EQ-001",
    "Factory_B": "EQ-002",
    "Factory_C": "EQ-003",
    "Factory_D": "EQ-004",
}

# Sensor type mappings for derived telemetry fields
SENSOR_TYPE_MAP: dict[str, str] = {
    "Temp": "TEMPERATURE",
    "Pressure": "PRESSURE",
    "Humidity": "HUMIDITY",
    "Vibration": "VIBRATION",
    "Gas": "GAS",
}

# Column for sensor_id generation (deterministic: Factory + sensor_type index)
SENSOR_ID_PREFIX = "S"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Load raw data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load the raw industrial fire risk CSV dataset.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        A pandas DataFrame with all columns as-is from the CSV.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows × %d columns from %s", len(df), len(df.columns), filepath.name)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Clean data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def fix_pressure_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Fix systematic 10x decimal-shift in Pressure readings.

    Values > 50.0 bar are divided by 10.0 to restore the correct scale.
    Based on the dataset analysis report: 500 rows (0.5%) have this issue.
    """
    mask = df["Pressure"] > PRESSURE_OUTLIER_THRESHOLD
    count = mask.sum()
    if count > 0:
        df.loc[mask, "Pressure"] = df.loc[mask, "Pressure"] / PRESSURE_CORRECTION_FACTOR
        logger.info("Corrected %d pressure outlier(s) (divided by %.1f)", count, PRESSURE_CORRECTION_FACTOR)
    return df


def impute_missing_workers(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing Workers values with the column median.

    The Workers column has ~2% missing values (2,000 of 100,000).
    Median imputation preserves the integer nature of the count.
    """
    median_val = df["Workers"].median()
    missing_count = df["Workers"].isna().sum()
    if missing_count > 0:
        df["Workers"] = df["Workers"].fillna(median_val)
        logger.info("Imputed %d missing Workers value(s) with median=%.1f", missing_count, median_val)
    return df


def combine_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Combine Date + Time columns into a single ISO 8601 timestamp.

    Creates a new 'timestamp' column and drops the original Date/Time columns.
    """
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y-%m-%d %H:%M:%S")
    df = df.drop(columns=["Date", "Time"])
    logger.info("Combined Date + Time into 'timestamp' column")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning transformations in sequence.

    1. Fix pressure decimal-shift outliers
    2. Impute missing Workers values
    3. Combine Date + Time into ISO timestamp

    Args:
        df: Raw DataFrame from load_raw_data().

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    df = df.copy()
    df = fix_pressure_outliers(df)
    df = impute_missing_workers(df)
    df = combine_datetime(df)
    logger.info("Data cleaning complete: %d rows × %d columns", len(df), len(df.columns))
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Feature engineering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Generate architecture-aligned derived fields.

    Adds:
        - zone_id: Mapped from Factory (Factory_A → ZONE_A)
        - equipment_id: Mapped from Factory (Factory_A → EQ-001)
        - sensor_id: Deterministic ID per Factory+sensor_type (e.g. S-ZONE_A-TEMP)
        - sensor_type: Primary sensor type based on the highest-risk reading
    """
    # zone_id from Factory
    df["zone_id"] = df["Factory"].map(FACTORY_TO_ZONE)

    # equipment_id from Factory
    df["equipment_id"] = df["Factory"].map(FACTORY_TO_EQUIPMENT)

    # sensor_id: deterministic per factory, one sensor per type per zone
    # Format: S-{ZONE}-{TYPE_ABBREV}  e.g. S-ZONE_A-TEMP
    # We pick the "primary" sensor type as TEMPERATURE for the row-level ID
    # Individual sensor IDs for each telemetry field are generated below
    for field, stype in SENSOR_TYPE_MAP.items():
        col_name = f"sensor_id_{field.lower()}"
        df[col_name] = (
            SENSOR_ID_PREFIX + "-" + df["zone_id"] + "-" + stype
        )

    # Primary sensor_type: the sensor type with the highest relative reading
    # (useful for row-level classification)
    df["sensor_type"] = "MULTI"  # default: multi-sensor row

    logger.info("Added derived fields: zone_id, equipment_id, sensor_id_*, sensor_type")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features.

    Encodes: Factory, Region, Shift, Exp, Training, Alarm.
    Uses drop_first=False to preserve all categories for interpretability.
    """
    present_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if not present_cols:
        logger.warning("No categorical columns found to encode")
        return df

    df = pd.get_dummies(df, columns=present_cols, drop_first=False, dtype=int)
    logger.info("One-hot encoded %d categorical feature(s)", len(present_cols))
    return df


def standardize_numericals(
    df: pd.DataFrame, scaler: Optional[StandardScaler] = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize numerical telemetry features (zero mean, unit variance).

    If a fitted scaler is provided, it is used for transform-only (runtime mode).
    Otherwise a new scaler is fitted (training mode).

    Args:
        df: DataFrame with cleaned numerical columns.
        scaler: Pre-fitted scaler for runtime use. None to fit a new one.

    Returns:
        Tuple of (transformed DataFrame, fitted StandardScaler).
    """
    present_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    if not present_cols:
        logger.warning("No numerical columns found to standardize")
        return df, scaler or StandardScaler()

    if scaler is None:
        scaler = StandardScaler()
        df[present_cols] = scaler.fit_transform(df[present_cols])
        logger.info("Fitted and applied StandardScaler to %d feature(s)", len(present_cols))
    else:
        df[present_cols] = scaler.transform(df[present_cols])
        logger.info("Applied pre-fitted StandardScaler to %d feature(s)", len(present_cols))

    return df, scaler


def feature_engineer(
    df: pd.DataFrame, scaler: Optional[StandardScaler] = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """Apply all feature engineering transformations.

    1. Add architecture-derived fields (zone_id, equipment_id, sensor_id, sensor_type)
    2. One-hot encode categorical variables
    3. Standardize numerical telemetry features

    Args:
        df: Cleaned DataFrame from clean_data().
        scaler: Optional pre-fitted scaler for runtime inference.

    Returns:
        Tuple of (engineered DataFrame, fitted StandardScaler).
    """
    df = df.copy()
    df = add_derived_fields(df)
    df = encode_categoricals(df)
    df, scaler = standardize_numericals(df, scaler)
    logger.info(
        "Feature engineering complete: %d rows × %d columns", len(df), len(df.columns)
    )
    return df, scaler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Save processed data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def save_processed(df: pd.DataFrame, filepath: str | Path) -> Path:
    """Save the processed DataFrame to CSV.

    Creates parent directories if they do not exist.

    Args:
        df: Processed DataFrame.
        filepath: Output CSV path.

    Returns:
        The resolved output Path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info("Saved processed data: %d rows × %d cols → %s", len(df), len(df.columns), filepath)
    return filepath


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_pipeline(
    raw_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Execute the full preprocessing pipeline end-to-end.

    Args:
        raw_path: Path to the raw industrial_fire_risk_data.csv.
        output_path: Path to write the processed CSV.

    Returns:
        The processed DataFrame.
    """
    df = load_raw_data(raw_path)
    df = clean_data(df)
    df, _scaler = feature_engineer(df)
    save_processed(df, output_path)
    return df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    raw = sys.argv[1] if len(sys.argv) > 1 else "../../industrial_fire_risk_data.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "../datasets/processed/processed_data.csv"

    run_pipeline(raw, out)
