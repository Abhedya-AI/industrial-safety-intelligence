"""Base interface for all anomaly detectors.

Every concrete detector (Isolation Forest, Autoencoder, etc.) must
implement this abstract interface. This ensures a uniform API across
all detector types and enables the DetectorFactory pattern.

NO training logic is permitted in implementations of this interface.
"""

from __future__ import annotations

import abc
from typing import Optional, Sequence

import numpy as np

from app.sensor_intelligence.anomaly_detection.schemas import AnomalyResult


class BaseAnomalyDetector(abc.ABC):
    """Abstract base class for runtime anomaly detectors.

    Detectors load pre-trained models from disk and perform inference-only
    predictions. They must NOT contain any training code.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the detector type name (e.g. 'isolation_forest')."""

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded and is ready for inference."""

    @abc.abstractmethod
    def load_model(self, **kwargs) -> None:
        """Load a pre-trained model from disk.

        Args:
            **kwargs: Detector-specific paths (model_path, scaler_path, etc.)
        """

    @abc.abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference and return raw predictions.

        Args:
            features: 2-D array of shape (n_samples, n_features).

        Returns:
            1-D array of predictions (detector-specific semantics).
        """

    @abc.abstractmethod
    def anomaly_score(self, features: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (higher = more anomalous).

        Args:
            features: 2-D array of shape (n_samples, n_features).

        Returns:
            1-D array of anomaly scores.
        """

    @abc.abstractmethod
    def classify(
        self,
        features: np.ndarray,
        sensor_ids: Sequence[str],
    ) -> list[AnomalyResult]:
        """Classify samples and return structured AnomalyResult objects.

        Args:
            features: 2-D array of shape (n_samples, n_features).
            sensor_ids: Corresponding sensor IDs for each sample.

        Returns:
            List of AnomalyResult (one per sample).
        """

    def _ensure_loaded(self) -> None:
        """Guard: raise if model is not loaded."""
        if not self.is_loaded:
            raise RuntimeError(
                f"{self.name} detector model is not loaded. Call load_model() first."
            )
