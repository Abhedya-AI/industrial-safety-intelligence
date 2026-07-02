"""Detector factory — instantiates and configures anomaly detectors.

Usage:
    detector = DetectorFactory.create("isolation_forest")
    detector = DetectorFactory.create("autoencoder")
    detector = DetectorFactory.create("isolation_forest", model_path="custom/path.pkl")
"""

from __future__ import annotations

import logging
from typing import Optional

from app.sensor_intelligence.anomaly_detection.base import BaseAnomalyDetector

logger = logging.getLogger(__name__)

# Registry of available detector types
_DETECTOR_REGISTRY: dict[str, type[BaseAnomalyDetector]] = {}


def _register_defaults() -> None:
    """Lazily register default detectors to avoid import-time side effects."""
    if _DETECTOR_REGISTRY:
        return

    from app.sensor_intelligence.anomaly_detection.autoencoder_detector import (
        AutoencoderDetector,
    )
    from app.sensor_intelligence.anomaly_detection.isolation_forest_detector import (
        IsolationForestDetector,
    )

    _DETECTOR_REGISTRY["isolation_forest"] = IsolationForestDetector
    _DETECTOR_REGISTRY["autoencoder"] = AutoencoderDetector


class DetectorFactory:
    """Factory for creating anomaly detector instances.

    Supports:
        - "isolation_forest" → IsolationForestDetector
        - "autoencoder"      → AutoencoderDetector

    Each call returns a new detector instance with the model loaded
    from the specified (or default) paths.
    """

    @staticmethod
    def create(
        detector_type: str,
        *,
        auto_load: bool = True,
        **kwargs,
    ) -> BaseAnomalyDetector:
        """Create and optionally load an anomaly detector.

        Args:
            detector_type: One of "isolation_forest" or "autoencoder".
            auto_load: If True (default), load the model immediately.
            **kwargs: Passed to load_model() (e.g. model_path, scaler_path).

        Returns:
            A ready-to-use BaseAnomalyDetector instance.

        Raises:
            ValueError: If detector_type is not registered.
        """
        _register_defaults()

        key = detector_type.lower().strip()
        if key not in _DETECTOR_REGISTRY:
            available = ", ".join(sorted(_DETECTOR_REGISTRY.keys()))
            raise ValueError(
                f"Unknown detector type: '{detector_type}'. "
                f"Available: [{available}]"
            )

        detector = _DETECTOR_REGISTRY[key]()
        logger.info("Created detector: %s", detector.name)

        if auto_load:
            detector.load_model(**kwargs)

        return detector

    @staticmethod
    def available_detectors() -> list[str]:
        """Return the list of registered detector type names."""
        _register_defaults()
        return sorted(_DETECTOR_REGISTRY.keys())

    @staticmethod
    def register(name: str, detector_class: type[BaseAnomalyDetector]) -> None:
        """Register a custom detector type.

        Args:
            name: Detector type name (lowercase).
            detector_class: Class that implements BaseAnomalyDetector.
        """
        _register_defaults()
        _DETECTOR_REGISTRY[name.lower().strip()] = detector_class
        logger.info("Registered custom detector: %s", name)
