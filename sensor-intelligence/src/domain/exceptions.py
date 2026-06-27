"""Domain-specific exceptions.

These exceptions represent business rule violations and are caught
by the API layer's error handler middleware to produce proper HTTP responses.
"""


class DomainError(Exception):
    """Base class for all domain exceptions."""

    def __init__(self, message: str = "A domain error occurred") -> None:
        self.message = message
        super().__init__(self.message)


class SensorNotFoundError(DomainError):
    """Raised when a requested sensor does not exist."""

    def __init__(self, sensor_id: str) -> None:
        super().__init__(f"Sensor not found: {sensor_id}")
        self.sensor_id = sensor_id


class DuplicateSensorError(DomainError):
    """Raised when attempting to register a sensor with an existing sensor_id."""

    def __init__(self, sensor_id: str) -> None:
        super().__init__(f"Sensor already exists: {sensor_id}")
        self.sensor_id = sensor_id


class AlertNotFoundError(DomainError):
    """Raised when a requested alert does not exist."""

    def __init__(self, alert_id: str) -> None:
        super().__init__(f"Alert not found: {alert_id}")
        self.alert_id = alert_id


class AnomalyNotFoundError(DomainError):
    """Raised when a requested anomaly does not exist."""

    def __init__(self, anomaly_id: str) -> None:
        super().__init__(f"Anomaly not found: {anomaly_id}")
        self.anomaly_id = anomaly_id


class ThresholdNotFoundError(DomainError):
    """Raised when a requested threshold does not exist."""

    def __init__(self, threshold_id: str) -> None:
        super().__init__(f"Threshold not found: {threshold_id}")
        self.threshold_id = threshold_id


class InvalidReadingError(DomainError):
    """Raised when a sensor reading fails domain validation."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Invalid reading: {reason}")
