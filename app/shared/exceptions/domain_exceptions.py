"""Domain-specific exceptions.

Generic resource exceptions that map to the API specification error codes
(PS1_Detailed_API_Specifications_V2, Error Response Format).
"""


class DomainError(Exception):
    """Base class for all domain exceptions."""

    error_code: str = "DOMAIN_ERROR"

    def __init__(self, message: str = "A domain error occurred") -> None:
        self.message = message
        super().__init__(self.message)


# ── Generic resource errors ──


class ResourceNotFoundError(DomainError):
    """Raised when a requested resource does not exist."""

    error_code = "RESOURCE_NOT_FOUND"

    def __init__(self, resource: str, identifier: str) -> None:
        self.resource = resource
        self.identifier = identifier
        super().__init__(f"{resource} not found: {identifier}")


class DuplicateResourceError(DomainError):
    """Raised when attempting to create an already-existing resource."""

    error_code = "DUPLICATE_RESOURCE"

    def __init__(self, resource: str, identifier: str) -> None:
        self.resource = resource
        self.identifier = identifier
        super().__init__(f"{resource} already exists: {identifier}")


class ValidationError(DomainError):
    """Raised when input fails business validation."""

    error_code = "VALIDATION_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidReadingError(DomainError):
    """Raised when a sensor reading is invalid."""

    error_code = "VALIDATION_ERROR"

    def __init__(self, reason: str) -> None:
        super().__init__(f"Invalid reading: {reason}")


class BusinessRuleViolationError(DomainError):
    """Raised when an operation violates a domain invariant."""

    error_code = "INVALID_REQUEST"

    def __init__(self, message: str) -> None:
        super().__init__(message)


# ── Backwards-compatible aliases ──

SensorNotFoundError = ResourceNotFoundError
DuplicateSensorError = DuplicateResourceError
AlertNotFoundError = ResourceNotFoundError
AnomalyNotFoundError = ResourceNotFoundError
ThresholdNotFoundError = ResourceNotFoundError
