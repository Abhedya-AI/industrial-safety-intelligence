"""Isolation Forest training pipeline for anomaly detection.

Trains an Isolation Forest model on *normal-only* sensor data (Accident == 0)
so the model learns the distribution of healthy operations. At inference time,
readings that deviate from this learned distribution are flagged as anomalies.

Usage:
    python -m training.train_isolation_forest

    # Or programmatically:
    from training.train_isolation_forest import run_pipeline
    results = run_pipeline()

Outputs:
    models/isolation_forest.pkl   — trained IsolationForest model
    models/scaler.pkl             — fitted StandardScaler (for runtime use)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from training.feature_engineering import get_isolation_forest_features

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_FEATURES_PATH = "datasets/processed/features.csv"
DEFAULT_MODEL_PATH = "models/isolation_forest.pkl"
DEFAULT_SCALER_PATH = "models/scaler.pkl"

# IsolationForest hyperparameters
IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = "auto"
IF_RANDOM_STATE = 42
IF_N_JOBS = -1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result container
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class EvaluationResult:
    """Holds evaluation metrics from comparing IF predictions to Accident labels."""

    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    report: str
    total_samples: int
    predicted_anomalies: int
    actual_accidents: int
    true_positives: int


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Load and prepare data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_features(filepath: str | Path = DEFAULT_FEATURES_PATH) -> pd.DataFrame:
    """Load the feature-engineered dataset."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Features file not found: {filepath}")
    df = pd.read_csv(filepath)
    logger.info("Loaded features: %d rows × %d cols", len(df), len(df.columns))
    return df


def prepare_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    """Split data into normal-only training set and full evaluation set.

    Args:
        df: Full feature DataFrame.

    Returns:
        (train_df, eval_df, eval_labels, feature_columns)
        - train_df: Normal data only (Accident == 0), feature columns only.
        - eval_df: Full dataset, feature columns only.
        - eval_labels: Accident labels for the full dataset.
        - feature_columns: List of selected feature column names.
    """
    feature_cols = get_isolation_forest_features(df)
    logger.info("Selected %d features for Isolation Forest", len(feature_cols))

    # Split: train on normal only
    normal_mask = df["Accident"] == 0
    train_df = df.loc[normal_mask, feature_cols].copy()
    eval_df = df[feature_cols].copy()
    eval_labels = df["Accident"].copy()

    logger.info(
        "Training set: %d normal samples | Eval set: %d total (%d accidents)",
        len(train_df), len(eval_df), eval_labels.sum(),
    )
    return train_df, eval_df, eval_labels, feature_cols


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Train model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def train_model(
    train_df: pd.DataFrame,
    contamination: Optional[float] = None,
) -> tuple[IsolationForest, StandardScaler]:
    """Train an Isolation Forest on normal-only data.

    Args:
        train_df: Training DataFrame (normal samples only, feature columns only).
        contamination: Expected anomaly fraction. If None, auto-computed from
                       the full dataset accident rate (~0.04).

    Returns:
        (model, scaler) — trained IsolationForest and fitted StandardScaler.
    """
    # Fit scaler on normal data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df.values)
    logger.info("Scaler fitted: %d features", X_train.shape[1])

    # Handle any residual NaN (shouldn't exist, but defensive)
    if np.isnan(X_train).any():
        nan_count = np.isnan(X_train).sum()
        logger.warning("Found %d NaN values after scaling — filling with 0", nan_count)
        X_train = np.nan_to_num(X_train, nan=0.0)

    # Set contamination based on dataset distribution
    if contamination is None:
        # The dataset has ~4.09% accident rate; we use a slightly lower
        # contamination since the model trains only on normals and we expect
        # some "near-anomaly" normals at the boundary.
        contamination = 0.04
        logger.info("Auto-set contamination=%.4f (based on dataset accident rate)", contamination)

    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples=IF_MAX_SAMPLES,
        contamination=contamination,
        random_state=IF_RANDOM_STATE,
        n_jobs=IF_N_JOBS,
    )
    model.fit(X_train)
    logger.info(
        "Isolation Forest trained: n_estimators=%d, contamination=%.4f, "
        "training samples=%d",
        IF_N_ESTIMATORS, contamination, len(X_train),
    )
    return model, scaler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Evaluate model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def evaluate_model(
    model: IsolationForest,
    scaler: StandardScaler,
    eval_df: pd.DataFrame,
    eval_labels: pd.Series,
) -> EvaluationResult:
    """Evaluate the trained model by comparing predictions to Accident labels.

    Isolation Forest returns:
        +1  → inlier  (normal)
        -1  → outlier (anomaly)

    We map:
        -1 (anomaly)  → 1 (predicted accident)
        +1 (inlier)   → 0 (predicted normal)

    Then compare against the actual Accident column.

    Args:
        model: Trained IsolationForest.
        scaler: Fitted StandardScaler.
        eval_df: Full evaluation dataset (feature columns only).
        eval_labels: Actual Accident labels (0/1).

    Returns:
        EvaluationResult with precision, recall, F1, confusion matrix, and report.
    """
    X_eval = scaler.transform(eval_df.values)
    if np.isnan(X_eval).any():
        X_eval = np.nan_to_num(X_eval, nan=0.0)

    # Predict: +1 = inlier, -1 = outlier
    raw_predictions = model.predict(X_eval)

    # Map to binary: -1 → 1 (anomaly/accident), +1 → 0 (normal)
    predicted = (raw_predictions == -1).astype(int)
    actual = eval_labels.values

    # Anomaly scores (lower = more anomalous)
    scores = model.decision_function(X_eval)

    # Metrics
    prec = precision_score(actual, predicted, zero_division=0)
    rec = recall_score(actual, predicted, zero_division=0)
    f1 = f1_score(actual, predicted, zero_division=0)
    cm = confusion_matrix(actual, predicted)
    report = classification_report(actual, predicted, target_names=["Normal", "Anomaly"])

    tp = int(((predicted == 1) & (actual == 1)).sum())

    result = EvaluationResult(
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        confusion=cm,
        report=report,
        total_samples=len(actual),
        predicted_anomalies=int(predicted.sum()),
        actual_accidents=int(actual.sum()),
        true_positives=tp,
    )

    logger.info(
        "Evaluation: Precision=%.4f  Recall=%.4f  F1=%.4f  "
        "TP=%d/%d  Predicted anomalies=%d",
        result.precision, result.recall, result.f1,
        result.true_positives, result.actual_accidents,
        result.predicted_anomalies,
    )
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Save model + scaler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def save_model(
    model: IsolationForest,
    scaler: StandardScaler,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
) -> tuple[Path, Path]:
    """Persist the trained model and scaler to disk.

    Args:
        model: Trained IsolationForest.
        scaler: Fitted StandardScaler.
        model_path: Output path for the model pickle.
        scaler_path: Output path for the scaler pickle.

    Returns:
        (model_path, scaler_path) as resolved Path objects.
    """
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model → %s", model_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler → %s", scaler_path)

    return model_path, scaler_path


def load_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
) -> tuple[IsolationForest, StandardScaler]:
    """Load a previously trained model and scaler from disk.

    Args:
        model_path: Path to the model pickle.
        scaler_path: Path to the scaler pickle.

    Returns:
        (model, scaler) tuple.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Loaded model from %s and scaler from %s", model_path, scaler_path)
    return model, scaler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Full pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_pipeline(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    contamination: Optional[float] = None,
) -> EvaluationResult:
    """Execute the full Isolation Forest training pipeline.

    1. Load features
    2. Prepare data (split normal-only train / full eval)
    3. Train model on normal data
    4. Evaluate against Accident labels
    5. Save model + scaler

    Returns:
        EvaluationResult with full metrics.
    """
    df = load_features(features_path)
    train_df, eval_df, eval_labels, feature_cols = prepare_data(df)
    model, scaler = train_model(train_df, contamination=contamination)
    result = evaluate_model(model, scaler, eval_df, eval_labels)
    save_model(model, scaler, model_path, scaler_path)

    print("\n" + "=" * 60)
    print("ISOLATION FOREST TRAINING REPORT")
    print("=" * 60)
    print(f"Training samples (normal only): {len(train_df):,}")
    print(f"Evaluation samples (full):      {len(eval_df):,}")
    print(f"Features used:                  {len(feature_cols)}")
    print(f"Actual accidents:               {result.actual_accidents:,}")
    print(f"Predicted anomalies:            {result.predicted_anomalies:,}")
    print(f"True positives:                 {result.true_positives}")
    print("-" * 60)
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall:    {result.recall:.4f}")
    print(f"F1 Score:  {result.f1:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"  TN={result.confusion[0][0]:,}  FP={result.confusion[0][1]:,}")
    print(f"  FN={result.confusion[1][0]:,}  TP={result.confusion[1][1]:,}")
    print("-" * 60)
    print(result.report)
    print(f"Model saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")
    print("=" * 60)

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_pipeline()
