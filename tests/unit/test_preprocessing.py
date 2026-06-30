"""Unit tests for the preprocessing pipeline.

Covers:
  - load_raw_data (file existence, column presence)
  - fix_pressure_outliers (correction, no false positives)
  - impute_missing_workers (fills NaN, preserves valid)
  - combine_datetime (ISO format, column removal)
  - clean_data (full pipeline)
  - add_derived_fields (zone_id, equipment_id, sensor_id_*)
  - encode_categoricals (one-hot output)
  - standardize_numericals (zero mean, unit variance, reuse scaler)
  - feature_engineer (full pipeline)
  - save_processed (file creation)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.preprocessing import (
    FACTORY_TO_EQUIPMENT,
    FACTORY_TO_ZONE,
    NUMERICAL_FEATURES,
    PRESSURE_OUTLIER_THRESHOLD,
    add_derived_fields,
    clean_data,
    combine_datetime,
    encode_categoricals,
    feature_engineer,
    fix_pressure_outliers,
    impute_missing_workers,
    load_raw_data,
    save_processed,
    standardize_numericals,
)


# ── Helpers ──


def _sample_df(n: int = 10, *, include_outliers: bool = False) -> pd.DataFrame:
    """Create a minimal sample DataFrame matching the raw CSV schema."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "Date": ["2021-01-01"] * n,
        "Time": [f"{h:02d}:00:00" for h in range(n)],
        "Factory": rng.choice(["Factory_A", "Factory_B", "Factory_C", "Factory_D"], n),
        "Region": rng.choice(["Urban", "Rural", "Industrial_Zone"], n),
        "Shift": rng.choice(["Day", "Night"], n),
        "Workers": rng.choice([10.0, 20.0, 30.0, np.nan], n),
        "Exp": rng.choice(["Senior", "Junior"], n),
        "Training": rng.choice(["Yes", "No"], n),
        "Temp": rng.uniform(15, 40, n).round(2),
        "Pressure": rng.uniform(15, 35, n).round(2),
        "Humidity": rng.uniform(30, 80, n).round(2),
        "Vibration": rng.uniform(0, 5, n).round(2),
        "Speed": rng.uniform(1000, 4000, n).round(2),
        "Age": rng.integers(1, 20, n).astype(float),
        "Service_Days": rng.integers(1, 365, n).astype(float),
        "Gas": rng.uniform(0, 10, n).round(2),
        "Sparks": rng.integers(0, 5, n).astype(float),
        "Alarm": rng.choice(["On", "Off"], n),
        "Risk": rng.uniform(0, 100, n).round(2),
        "Accident": rng.choice([0, 1], n, p=[0.96, 0.04]),
    })
    if include_outliers:
        # Inject 2 pressure outliers (decimal-shifted by 10x)
        df.loc[0, "Pressure"] = 225.0
        df.loc[1, "Pressure"] = 210.5
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# load_raw_data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_load_raw_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_raw_data("/nonexistent/path/data.csv")


def test_load_raw_data_success():
    df = _sample_df(5)
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f, index=False)
        tmp_path = f.name

    loaded = load_raw_data(tmp_path)
    assert len(loaded) == 5
    assert "Date" in loaded.columns
    assert "Pressure" in loaded.columns
    Path(tmp_path).unlink()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# fix_pressure_outliers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_fix_pressure_no_outliers():
    df = _sample_df(5)
    original = df["Pressure"].copy()
    result = fix_pressure_outliers(df)
    pd.testing.assert_series_equal(result["Pressure"], original)


def test_fix_pressure_with_outliers():
    df = _sample_df(5, include_outliers=True)
    assert df.loc[0, "Pressure"] == 225.0
    assert df.loc[1, "Pressure"] == 210.5

    result = fix_pressure_outliers(df)
    assert result.loc[0, "Pressure"] == 22.5
    assert result.loc[1, "Pressure"] == 21.05


def test_fix_pressure_boundary_value():
    """Value exactly at threshold should NOT be corrected."""
    df = _sample_df(3)
    df.loc[0, "Pressure"] = 50.0
    result = fix_pressure_outliers(df)
    assert result.loc[0, "Pressure"] == 50.0  # Not modified (not > 50)


def test_fix_pressure_just_above_threshold():
    df = _sample_df(3)
    df.loc[0, "Pressure"] = 50.01
    result = fix_pressure_outliers(df)
    assert abs(result.loc[0, "Pressure"] - 5.001) < 0.001


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# impute_missing_workers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_impute_workers_fills_nan():
    df = _sample_df(10)
    missing_before = df["Workers"].isna().sum()
    assert missing_before > 0

    result = impute_missing_workers(df)
    assert result["Workers"].isna().sum() == 0


def test_impute_workers_uses_median():
    df = pd.DataFrame({"Workers": [10.0, 20.0, 30.0, np.nan, np.nan]})
    result = impute_missing_workers(df)
    assert result["Workers"].isna().sum() == 0
    # Median of [10, 20, 30] = 20
    assert result.loc[3, "Workers"] == 20.0
    assert result.loc[4, "Workers"] == 20.0


def test_impute_workers_no_missing():
    df = pd.DataFrame({"Workers": [10.0, 20.0, 30.0]})
    result = impute_missing_workers(df)
    assert list(result["Workers"]) == [10.0, 20.0, 30.0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# combine_datetime
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_combine_datetime_creates_timestamp():
    df = pd.DataFrame({
        "Date": ["2021-01-01", "2021-06-15"],
        "Time": ["08:30:00", "14:45:00"],
    })
    result = combine_datetime(df)
    assert "timestamp" in result.columns
    assert "Date" not in result.columns
    assert "Time" not in result.columns


def test_combine_datetime_correct_values():
    df = pd.DataFrame({
        "Date": ["2021-01-01"],
        "Time": ["08:30:00"],
    })
    result = combine_datetime(df)
    ts = result.loc[0, "timestamp"]
    assert ts.year == 2021
    assert ts.month == 1
    assert ts.day == 1
    assert ts.hour == 8
    assert ts.minute == 30


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# clean_data (integrated)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_clean_data_full_pipeline():
    df = _sample_df(20, include_outliers=True)
    result = clean_data(df)

    # Pressure outliers fixed
    assert result["Pressure"].max() <= PRESSURE_OUTLIER_THRESHOLD

    # No missing workers
    assert result["Workers"].isna().sum() == 0

    # Timestamp column present, Date/Time removed
    assert "timestamp" in result.columns
    assert "Date" not in result.columns
    assert "Time" not in result.columns


def test_clean_data_does_not_mutate_input():
    df = _sample_df(5, include_outliers=True)
    original_pressure = df.loc[0, "Pressure"]
    _ = clean_data(df)
    assert df.loc[0, "Pressure"] == original_pressure  # Original unchanged


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# add_derived_fields
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_derived_zone_id():
    df = pd.DataFrame({"Factory": ["Factory_A", "Factory_B", "Factory_C", "Factory_D"]})
    result = add_derived_fields(df)
    assert list(result["zone_id"]) == ["ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D"]


def test_derived_equipment_id():
    df = pd.DataFrame({"Factory": ["Factory_A", "Factory_C"]})
    result = add_derived_fields(df)
    assert list(result["equipment_id"]) == ["EQ-001", "EQ-003"]


def test_derived_sensor_ids():
    df = pd.DataFrame({"Factory": ["Factory_A"]})
    result = add_derived_fields(df)
    assert result.loc[0, "sensor_id_temp"] == "S-ZONE_A-TEMPERATURE"
    assert result.loc[0, "sensor_id_pressure"] == "S-ZONE_A-PRESSURE"
    assert result.loc[0, "sensor_id_humidity"] == "S-ZONE_A-HUMIDITY"
    assert result.loc[0, "sensor_id_vibration"] == "S-ZONE_A-VIBRATION"
    assert result.loc[0, "sensor_id_gas"] == "S-ZONE_A-GAS"


def test_derived_sensor_type_column():
    df = pd.DataFrame({"Factory": ["Factory_B"]})
    result = add_derived_fields(df)
    assert result.loc[0, "sensor_type"] == "MULTI"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# encode_categoricals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_encode_categoricals_creates_dummies():
    df = pd.DataFrame({
        "Factory": ["Factory_A", "Factory_B"],
        "Region": ["Urban", "Rural"],
        "Shift": ["Day", "Night"],
        "Exp": ["Senior", "Junior"],
        "Training": ["Yes", "No"],
        "Alarm": ["On", "Off"],
    })
    result = encode_categoricals(df)
    # Original columns should be removed
    assert "Factory" not in result.columns
    assert "Alarm" not in result.columns
    # Dummy columns should exist
    assert "Factory_Factory_A" in result.columns
    assert "Alarm_On" in result.columns


def test_encode_categoricals_all_integer():
    df = pd.DataFrame({
        "Factory": ["Factory_A", "Factory_B"],
        "Region": ["Urban", "Rural"],
        "Shift": ["Day", "Night"],
        "Exp": ["Senior", "Junior"],
        "Training": ["Yes", "No"],
        "Alarm": ["On", "Off"],
    })
    result = encode_categoricals(df)
    dummy_cols = [c for c in result.columns if "_" in c]
    for col in dummy_cols:
        assert result[col].dtype in (np.int64, np.int32, int), f"{col} is not integer"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# standardize_numericals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_standardize_zero_mean():
    df = _sample_df(100)
    df = fix_pressure_outliers(df)
    result, scaler = standardize_numericals(df)
    for col in NUMERICAL_FEATURES:
        if col in result.columns:
            assert abs(result[col].mean()) < 0.01, f"{col} mean not ~0"


def test_standardize_unit_variance():
    df = _sample_df(100)
    df = fix_pressure_outliers(df)
    result, scaler = standardize_numericals(df)
    for col in NUMERICAL_FEATURES:
        if col in result.columns:
            assert abs(result[col].std(ddof=0) - 1.0) < 0.05, f"{col} std not ~1"


def test_standardize_reuse_scaler():
    """A pre-fitted scaler should produce a transform without re-fitting."""
    df1 = _sample_df(50)
    df1 = fix_pressure_outliers(df1)
    _, scaler = standardize_numericals(df1.copy())

    # Verify the scaler has been fitted (has mean_ attribute)
    assert hasattr(scaler, "mean_")
    assert len(scaler.mean_) == len(NUMERICAL_FEATURES)

    # Apply to new data — should not raise
    df2 = _sample_df(30)
    df2 = fix_pressure_outliers(df2)
    raw_temp_mean = df2["Temp"].mean()
    result, returned_scaler = standardize_numericals(df2, scaler=scaler)

    # Scaler should be the same object passed in
    assert returned_scaler is scaler
    # Transformed values should differ from raw
    assert result["Temp"].mean() != raw_temp_mean


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# feature_engineer (integrated)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_feature_engineer_full_pipeline():
    df = _sample_df(20)
    df = clean_data(df)
    result, scaler = feature_engineer(df)

    # Derived fields present
    assert "zone_id" in result.columns
    assert "equipment_id" in result.columns
    assert "sensor_id_temp" in result.columns

    # Categoricals encoded (originals removed)
    assert "Factory" not in result.columns
    assert "Region" not in result.columns

    # Scaler is returned
    assert scaler is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# save_processed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_save_processed_creates_file():
    df = _sample_df(5)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "subdir" / "output.csv"
        result_path = save_processed(df, out)
        assert result_path.exists()
        loaded = pd.read_csv(result_path)
        assert len(loaded) == 5


def test_save_processed_matches_columns():
    df = _sample_df(5)
    df = clean_data(df)
    df, _ = feature_engineer(df)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "output.csv"
        save_processed(df, out)
        loaded = pd.read_csv(out)
        assert set(loaded.columns) == set(df.columns)
