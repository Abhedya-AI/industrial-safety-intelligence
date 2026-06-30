"""Autoencoder-based anomaly detection training pipeline.

Trains a symmetric dense autoencoder on *normal-only* sensor data
(Accident == 0). Anomalies are detected at inference time by computing
the reconstruction error (MSE) — readings that the model cannot
reconstruct well are flagged as anomalous.

Architecture:
    Input(n) → Dense(64,ReLU) → Dense(32,ReLU) → Dense(16,ReLU)   [Encoder]
             → Dense(8,ReLU)                                       [Bottleneck]
             → Dense(16,ReLU) → Dense(32,ReLU) → Dense(64,ReLU)   [Decoder]
             → Dense(n,Linear)                                     [Output]

Usage:
    python -m training.train_autoencoder

    # Or programmatically:
    from training.train_autoencoder import run_pipeline
    results = run_pipeline()

Outputs:
    models/autoencoder.keras              — trained Keras model
    models/autoencoder_threshold.json     — anomaly threshold (percentile-based)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from training.feature_engineering import get_autoencoder_features

# Suppress TF info/warning logs before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_FEATURES_PATH = "datasets/processed/features.csv"
DEFAULT_MODEL_PATH = "models/autoencoder.keras"
DEFAULT_THRESHOLD_PATH = "models/autoencoder_threshold.json"
DEFAULT_SCALER_PATH = "models/autoencoder_scaler.pkl"

# Architecture
ENCODER_DIMS = [64, 32, 16]
BOTTLENECK_DIM = 8
DECODER_DIMS = [16, 32, 64]

# Training
EPOCHS = 50
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 1e-3

# Threshold: percentile of reconstruction error on normal data
THRESHOLD_PERCENTILE = 95.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result container
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class EvaluationResult:
    """Evaluation metrics from comparing AE anomaly predictions to Accident labels."""

    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    report: str
    total_samples: int
    predicted_anomalies: int
    actual_accidents: int
    true_positives: int
    threshold: float
    mean_normal_error: float
    mean_anomaly_error: float


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Build the autoencoder model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_autoencoder(input_dim: int):
    """Build a symmetric dense autoencoder.

    Args:
        input_dim: Number of input features.

    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf
    from tensorflow import keras

    # Encoder
    inputs = keras.Input(shape=(input_dim,), name="encoder_input")
    x = inputs
    for i, units in enumerate(ENCODER_DIMS):
        x = keras.layers.Dense(units, activation="relu", name=f"encoder_{i}")(x)

    # Bottleneck
    x = keras.layers.Dense(BOTTLENECK_DIM, activation="relu", name="bottleneck")(x)

    # Decoder (symmetric)
    for i, units in enumerate(DECODER_DIMS):
        x = keras.layers.Dense(units, activation="relu", name=f"decoder_{i}")(x)

    # Output: reconstruct original input
    outputs = keras.layers.Dense(input_dim, activation="linear", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Load and prepare data
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """Prepare training and evaluation arrays.

    Args:
        df: Full feature DataFrame.

    Returns:
        (X_train, X_eval, eval_labels, scaler, feature_columns)
        - X_train: Scaled normal-only data (Accident == 0).
        - X_eval: Scaled full dataset.
        - eval_labels: Accident labels (0/1).
        - scaler: Fitted StandardScaler.
        - feature_columns: Selected feature column names.
    """
    feature_cols = get_autoencoder_features(df)
    logger.info("Selected %d features for Autoencoder", len(feature_cols))

    normal_mask = df["Accident"] == 0
    train_raw = df.loc[normal_mask, feature_cols].values
    eval_raw = df[feature_cols].values
    eval_labels = df["Accident"].values

    # Fit scaler on normal data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_raw)
    X_eval = scaler.transform(eval_raw)

    # Handle any residual NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_eval = np.nan_to_num(X_eval, nan=0.0)

    logger.info(
        "Prepared: %d training (normal), %d eval (%d accidents), %d features",
        len(X_train), len(X_eval), eval_labels.sum(), X_train.shape[1],
    )
    return X_train, X_eval, eval_labels, scaler, feature_cols


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Train autoencoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def train_autoencoder(
    X_train: np.ndarray,
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    validation_split: float = VALIDATION_SPLIT,
):
    """Train the autoencoder on normal-only data.

    The model learns to reconstruct normal sensor patterns. Input == Target.

    Args:
        X_train: Scaled normal-only feature array.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        validation_split: Fraction of training data for validation.

    Returns:
        (model, history) — trained Keras model and training history.
    """
    input_dim = X_train.shape[1]
    model = build_autoencoder(input_dim)

    logger.info(
        "Training autoencoder: input_dim=%d, architecture=%s→%d→%s, "
        "epochs=%d, batch=%d",
        input_dim, ENCODER_DIMS, BOTTLENECK_DIM, DECODER_DIMS,
        epochs, batch_size,
    )
    model.summary(print_fn=lambda s: logger.info(s))

    history = model.fit(
        X_train, X_train,  # Input == Target (reconstruction)
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        verbose=1,
    )

    final_loss = history.history["loss"][-1]
    final_val_loss = history.history.get("val_loss", [None])[-1]
    logger.info(
        "Training complete: final_loss=%.6f, final_val_loss=%s",
        final_loss, f"{final_val_loss:.6f}" if final_val_loss else "N/A",
    )
    return model, history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Compute reconstruction error
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_reconstruction_error(model, X: np.ndarray) -> np.ndarray:
    """Compute per-sample reconstruction error (MSE).

    Args:
        model: Trained Keras autoencoder.
        X: Scaled input array.

    Returns:
        1-D array of MSE values (one per sample).
    """
    X_reconstructed = model.predict(X, verbose=0)
    mse = np.mean((X - X_reconstructed) ** 2, axis=1)
    return mse


def compute_threshold(
    errors: np.ndarray,
    percentile: float = THRESHOLD_PERCENTILE,
) -> float:
    """Compute the anomaly threshold as a percentile of reconstruction errors.

    Args:
        errors: Reconstruction errors on normal-only data.
        percentile: Percentile value (e.g. 95 = top 5% flagged).

    Returns:
        Threshold value.
    """
    threshold = float(np.percentile(errors, percentile))
    logger.info(
        "Threshold at %.1f%% percentile: %.6f (min=%.6f, max=%.6f, mean=%.6f)",
        percentile, threshold,
        errors.min(), errors.max(), errors.mean(),
    )
    return threshold


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Evaluate model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def evaluate_model(
    model,
    X_eval: np.ndarray,
    eval_labels: np.ndarray,
    threshold: float,
) -> EvaluationResult:
    """Evaluate the autoencoder by comparing reconstruction-error-based
    predictions against Accident labels.

    Samples with reconstruction error > threshold are predicted as anomalies.

    Args:
        model: Trained Keras autoencoder.
        X_eval: Full scaled evaluation array.
        eval_labels: Actual Accident labels (0/1).
        threshold: Anomaly threshold (from compute_threshold).

    Returns:
        EvaluationResult with full metrics.
    """
    errors = compute_reconstruction_error(model, X_eval)
    predicted = (errors > threshold).astype(int)

    prec = precision_score(eval_labels, predicted, zero_division=0)
    rec = recall_score(eval_labels, predicted, zero_division=0)
    f1 = f1_score(eval_labels, predicted, zero_division=0)
    cm = confusion_matrix(eval_labels, predicted)
    report = classification_report(
        eval_labels, predicted, target_names=["Normal", "Anomaly"]
    )

    tp = int(((predicted == 1) & (eval_labels == 1)).sum())

    # Mean error by class
    normal_mask = eval_labels == 0
    mean_normal = float(errors[normal_mask].mean()) if normal_mask.any() else 0.0
    mean_anomaly = float(errors[~normal_mask].mean()) if (~normal_mask).any() else 0.0

    result = EvaluationResult(
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        confusion=cm,
        report=report,
        total_samples=len(eval_labels),
        predicted_anomalies=int(predicted.sum()),
        actual_accidents=int(eval_labels.sum()),
        true_positives=tp,
        threshold=round(threshold, 6),
        mean_normal_error=round(mean_normal, 6),
        mean_anomaly_error=round(mean_anomaly, 6),
    )

    logger.info(
        "Evaluation: Precision=%.4f  Recall=%.4f  F1=%.4f  "
        "TP=%d/%d  threshold=%.6f  "
        "mean_error(normal)=%.6f  mean_error(anomaly)=%.6f",
        result.precision, result.recall, result.f1,
        result.true_positives, result.actual_accidents,
        threshold, mean_normal, mean_anomaly,
    )
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Save / load
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def save_model(
    model,
    threshold: float,
    scaler: StandardScaler,
    feature_columns: list[str],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold_path: str | Path = DEFAULT_THRESHOLD_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
) -> tuple[Path, Path, Path]:
    """Persist the trained model, threshold, and scaler.

    Args:
        model: Trained Keras autoencoder.
        threshold: Anomaly detection threshold.
        scaler: Fitted StandardScaler.
        feature_columns: List of feature column names used.
        model_path: Output path for the Keras model.
        threshold_path: Output path for the threshold JSON.
        scaler_path: Output path for the scaler pickle.

    Returns:
        (model_path, threshold_path, scaler_path) as Path objects.
    """
    import pickle

    model_path = Path(model_path)
    threshold_path = Path(threshold_path)
    scaler_path = Path(scaler_path)

    for p in (model_path, threshold_path, scaler_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    # Save Keras model
    model.save(str(model_path))
    logger.info("Saved model → %s", model_path)

    # Save threshold + metadata as JSON
    threshold_data = {
        "threshold": threshold,
        "percentile": THRESHOLD_PERCENTILE,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "architecture": {
            "encoder": ENCODER_DIMS,
            "bottleneck": BOTTLENECK_DIM,
            "decoder": DECODER_DIMS,
        },
    }
    with open(threshold_path, "w") as f:
        json.dump(threshold_data, f, indent=2)
    logger.info("Saved threshold → %s", threshold_path)

    # Save scaler
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler → %s", scaler_path)

    return model_path, threshold_path, scaler_path


def load_saved_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold_path: str | Path = DEFAULT_THRESHOLD_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
) -> tuple:
    """Load a previously saved autoencoder, threshold, and scaler.

    Returns:
        (model, threshold, scaler, feature_columns)
    """
    import pickle
    from tensorflow import keras

    model = keras.models.load_model(str(model_path))

    with open(threshold_path, "r") as f:
        threshold_data = json.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    logger.info(
        "Loaded AE model from %s (threshold=%.6f, features=%d)",
        model_path, threshold_data["threshold"], threshold_data["feature_count"],
    )
    return model, threshold_data["threshold"], scaler, threshold_data["feature_columns"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Full pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_pipeline(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold_path: str | Path = DEFAULT_THRESHOLD_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    epochs: int = EPOCHS,
    contamination_percentile: float = THRESHOLD_PERCENTILE,
) -> EvaluationResult:
    """Execute the full autoencoder training pipeline.

    1. Load features
    2. Prepare data (normal-only train / full eval)
    3. Train autoencoder (input == target)
    4. Compute reconstruction errors + threshold
    5. Evaluate against Accident labels
    6. Save model + threshold + scaler

    Returns:
        EvaluationResult with full metrics.
    """
    df = load_features(features_path)
    X_train, X_eval, eval_labels, scaler, feature_cols = prepare_data(df)

    model, history = train_autoencoder(X_train, epochs=epochs)

    # Compute threshold on normal training data
    train_errors = compute_reconstruction_error(model, X_train)
    threshold = compute_threshold(train_errors, percentile=contamination_percentile)

    # Evaluate on full dataset
    result = evaluate_model(model, X_eval, eval_labels, threshold)

    # Save
    save_model(model, threshold, scaler, feature_cols, model_path, threshold_path, scaler_path)

    print("\n" + "=" * 60)
    print("AUTOENCODER TRAINING REPORT")
    print("=" * 60)
    print(f"Architecture:  {X_train.shape[1]}→{ENCODER_DIMS}→{BOTTLENECK_DIM}→{DECODER_DIMS}→{X_train.shape[1]}")
    print(f"Epochs:        {epochs}")
    print(f"Final loss:    {history.history['loss'][-1]:.6f}")
    val_loss = history.history.get("val_loss", [None])[-1]
    if val_loss:
        print(f"Final val_loss:{val_loss:.6f}")
    print(f"Threshold:     {threshold:.6f} (at {contamination_percentile}th percentile)")
    print("-" * 60)
    print(f"Training samples (normal):  {len(X_train):,}")
    print(f"Evaluation samples (full):  {len(X_eval):,}")
    print(f"Features:                   {len(feature_cols)}")
    print(f"Actual accidents:           {result.actual_accidents:,}")
    print(f"Predicted anomalies:        {result.predicted_anomalies:,}")
    print(f"True positives:             {result.true_positives}")
    print(f"Mean error (normal):        {result.mean_normal_error:.6f}")
    print(f"Mean error (anomaly):       {result.mean_anomaly_error:.6f}")
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
    print(f"Model saved:     {model_path}")
    print(f"Threshold saved: {threshold_path}")
    print(f"Scaler saved:    {scaler_path}")
    print("=" * 60)

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_pipeline()
