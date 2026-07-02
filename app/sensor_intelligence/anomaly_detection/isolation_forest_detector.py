"""Runtime Isolation Forest anomaly detector.

Loads a pre-trained sklearn IsolationForest model and scaler from disk.
Performs inference-only predictions — NO training logic.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.sensor_intelligence.anomaly_detection.base import BaseAnomalyDetector
from app.sensor_intelligence.anomaly_detection.schemas import (
    AnomalyResult,
    AnomalyStatus,
)

logger = logging.getLogger(__name__)

# Default model paths (relative to project root)
DEFAULT_MODEL_PATH = "models/isolation_forest.pkl"
DEFAULT_SCALER_PATH = "models/scaler.pkl"


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest runtime detector.

    Loads a pre-trained IsolationForest + StandardScaler and classifies
    samples based on the model's decision function.

    Anomaly score mapping:
        sklearn returns +1 (inlier) / -1 (outlier).
        Decision function returns negative values for anomalies.
        We normalise to [0, 1] where higher = more anomalous.
    """

    def __init__(self) -> None:
        self._model: Optional[IsolationForest] = None
        self._scaler: Optional[StandardScaler] = None
        self._loaded = False

    # ── Properties ──

    @property
    def name(self) -> str:
        return "isolation_forest"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Load ──

    def load_model(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        scaler_path: str | Path = DEFAULT_SCALER_PATH,
        **kwargs,
    ) -> None:
        """Load the pre-trained Isolation Forest model and scaler.

        Args:
            model_path: Path to the pickled IsolationForest.
            scaler_path: Path to the pickled StandardScaler.
        """
        model_path = Path(model_path)
        scaler_path = Path(scaler_path)

        if not model_path.exists():
            raise FileNotFoundError(f"IF model not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"IF scaler not found: {scaler_path}")

        with open(model_path, "rb") as f:
            self._model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)

        self._loaded = True
        logger.info(
            "Loaded IsolationForest model from %s (scaler: %s)",
            model_path, scaler_path,
        )

    # ── Predict ──

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return raw sklearn predictions: +1 (inlier) or -1 (outlier).

        Args:
            features: 2-D array (n_samples, n_features) — raw, unscaled.

        Returns:
            1-D array of +1 / -1 labels.
        """
        self._ensure_loaded()
        X = self._scaler.transform(features)
        X = np.nan_to_num(X, nan=0.0)
        return self._model.predict(X)

    # ── Anomaly Score ──

    def anomaly_score(self, features: np.ndarray) -> np.ndarray:
        """Compute normalised anomaly scores in [0, 1].

        Uses the sklearn decision function (negative = anomalous) and
        applies a sigmoid-like normalisation so higher values indicate
        higher anomaly probability.

        Args:
            features: 2-D array (n_samples, n_features) — raw, unscaled.

        Returns:
            1-D array of scores in [0, 1].
        """
        self._ensure_loaded()
        X = self._scaler.transform(features)
        X = np.nan_to_num(X, nan=0.0)

        # decision_function: negative → anomaly, positive → normal
        raw_scores = self._model.decision_function(X)

        # Normalise via sigmoid: score = 1 / (1 + exp(raw_score * k))
        # k=5 gives a reasonable spread; negative raw → high score
        normalised = 1.0 / (1.0 + np.exp(raw_scores * 5.0))
        return normalised

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
        predictions = self.predict(features)
        scores = self.anomaly_score(features)

        results = []
        for i, sid in enumerate(sensor_ids):
            is_anomaly = predictions[i] == -1
            results.append(
                AnomalyResult(
                    sensor_id=sid,
                    score=float(scores[i]),
                    status=AnomalyStatus.ANOMALY if is_anomaly else AnomalyStatus.NORMAL,
                    detector_type=self.name,
                    confidence=float(scores[i]) if is_anomaly else float(1.0 - scores[i]),
                    threshold=float(self._model.offset_) if self._model else None,
                )
            )
        return results
