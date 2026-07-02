"""Runtime Autoencoder anomaly detector.

Loads a pre-trained Keras autoencoder, threshold, and scaler from disk.
Detects anomalies based on reconstruction error exceeding the threshold.

Performs inference-only — NO training logic.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from app.sensor_intelligence.anomaly_detection.base import BaseAnomalyDetector
from app.sensor_intelligence.anomaly_detection.schemas import (
    AnomalyResult,
    AnomalyStatus,
)

# Suppress TF info logs at import time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logger = logging.getLogger(__name__)

# Default paths (relative to project root)
DEFAULT_MODEL_PATH = "models/autoencoder.keras"
DEFAULT_THRESHOLD_PATH = "models/autoencoder_threshold.json"
DEFAULT_SCALER_PATH = "models/autoencoder_scaler.pkl"


class AutoencoderDetector(BaseAnomalyDetector):
    """Autoencoder runtime detector.

    Loads a pre-trained Keras autoencoder, a reconstruction error threshold,
    and a StandardScaler. Classifies samples as anomalies when their
    reconstruction error exceeds the threshold.

    Anomaly score:
        Per-sample MSE between input and reconstruction, normalised
        relative to the threshold so score > 1.0 → anomaly.
    """

    def __init__(self) -> None:
        self._model = None  # Keras model (lazy import)
        self._scaler: Optional[StandardScaler] = None
        self._threshold: float = 0.0
        self._feature_columns: list[str] = []
        self._loaded = False

    # ── Properties ──

    @property
    def name(self) -> str:
        return "autoencoder"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def threshold(self) -> float:
        """Return the anomaly detection threshold."""
        return self._threshold

    # ── Load ──

    def load_model(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        threshold_path: str | Path = DEFAULT_THRESHOLD_PATH,
        scaler_path: str | Path = DEFAULT_SCALER_PATH,
        **kwargs,
    ) -> None:
        """Load the pre-trained autoencoder, threshold, and scaler.

        Args:
            model_path: Path to the saved Keras model (.keras).
            threshold_path: Path to the threshold JSON file.
            scaler_path: Path to the pickled StandardScaler.
        """
        from tensorflow import keras

        model_path = Path(model_path)
        threshold_path = Path(threshold_path)
        scaler_path = Path(scaler_path)

        if not model_path.exists():
            raise FileNotFoundError(f"AE model not found: {model_path}")
        if not threshold_path.exists():
            raise FileNotFoundError(f"AE threshold not found: {threshold_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"AE scaler not found: {scaler_path}")

        # Load Keras model
        self._model = keras.models.load_model(str(model_path))

        # Load threshold metadata
        with open(threshold_path, "r") as f:
            threshold_data = json.load(f)
        self._threshold = threshold_data["threshold"]
        self._feature_columns = threshold_data.get("feature_columns", [])

        # Load scaler
        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)

        self._loaded = True
        logger.info(
            "Loaded Autoencoder model from %s (threshold=%.6f, features=%d)",
            model_path, self._threshold, len(self._feature_columns),
        )

    # ── Reconstruction Error ──

    def _reconstruction_error(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute per-sample MSE between input and reconstruction.

        Args:
            X_scaled: Scaled input array.

        Returns:
            1-D array of MSE values.
        """
        reconstructed = self._model.predict(X_scaled, verbose=0)
        mse = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        return mse

    # ── Predict ──

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return binary predictions: 0 (normal) or 1 (anomaly).

        Samples with reconstruction error > threshold are anomalies.

        Args:
            features: 2-D array (n_samples, n_features) — raw, unscaled.

        Returns:
            1-D array of 0 / 1 labels.
        """
        self._ensure_loaded()
        X = self._scaler.transform(features)
        X = np.nan_to_num(X, nan=0.0)
        errors = self._reconstruction_error(X)
        return (errors > self._threshold).astype(int)

    # ── Anomaly Score ──

    def anomaly_score(self, features: np.ndarray) -> np.ndarray:
        """Compute normalised anomaly scores.

        Score = reconstruction_error / threshold.
        Values > 1.0 are classified as anomalies.

        Args:
            features: 2-D array (n_samples, n_features) — raw, unscaled.

        Returns:
            1-D array of normalised scores.
        """
        self._ensure_loaded()
        X = self._scaler.transform(features)
        X = np.nan_to_num(X, nan=0.0)
        errors = self._reconstruction_error(X)
        # Normalise relative to threshold (> 1.0 = anomaly)
        return errors / self._threshold if self._threshold > 0 else errors

    # ── Classify ──

    def classify(
        self,
        features: np.ndarray,
        sensor_ids: Sequence[str],
    ) -> list[AnomalyResult]:
        """Classify samples and return structured AnomalyResult objects.

        Args:
            features: 2-D array (n_samples, n_features) — raw, unscaled.
            sensor_ids: Sensor ID for each sample.

        Returns:
            List of AnomalyResult (one per sample).
        """
        self._ensure_loaded()
        X = self._scaler.transform(features)
        X = np.nan_to_num(X, nan=0.0)
        errors = self._reconstruction_error(X)
        scores = errors / self._threshold if self._threshold > 0 else errors

        results = []
        for i, sid in enumerate(sensor_ids):
            is_anomaly = errors[i] > self._threshold
            # Confidence: how far from threshold (clamped to [0, 1])
            if is_anomaly:
                confidence = min(1.0, (errors[i] - self._threshold) / self._threshold)
            else:
                confidence = min(1.0, (self._threshold - errors[i]) / self._threshold)

            results.append(
                AnomalyResult(
                    sensor_id=sid,
                    score=float(scores[i]),
                    status=AnomalyStatus.ANOMALY if is_anomaly else AnomalyStatus.NORMAL,
                    detector_type=self.name,
                    confidence=round(float(confidence), 4),
                    threshold=self._threshold,
                    details={
                        "reconstruction_error": round(float(errors[i]), 6),
                    },
                )
            )
        return results
