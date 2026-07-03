"""Risk prediction repository interface (port).

Defines the abstract contract for risk prediction persistence.
Concrete implementations (e.g. SQLAlchemy) must implement every method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from app.risk_prediction.models.risk_prediction_model import RiskPredictionModel


class RiskPredictionRepository(ABC):
    """Abstract interface for risk prediction persistence.

    Returns RiskPredictionModel ORM objects. Business validation is NOT
    performed here — that responsibility belongs to the service layer.
    """

    # ── Queries ──

    @abstractmethod
    async def get_prediction(self, prediction_id: str) -> Optional[RiskPredictionModel]:
        """Retrieve a prediction by its UUID primary key.

        Returns None if no prediction matches.
        """
        ...

    @abstractmethod
    async def get_latest_prediction(
        self,
        sensor_id: Optional[str] = None,
        zone_id: Optional[str] = None,
    ) -> Optional[RiskPredictionModel]:
        """Get the most recent prediction, optionally filtered by sensor or zone.

        Returns None if no predictions exist for the given filter.
        """
        ...

    @abstractmethod
    async def get_prediction_history(
        self,
        sensor_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[RiskPredictionModel]:
        """Get paginated prediction history with optional filters.

        Results are ordered by prediction_timestamp descending (newest first).
        """
        ...

    @abstractmethod
    async def count_predictions(
        self,
        sensor_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """Count predictions matching the given filters."""
        ...

    # ── Mutations ──

    @abstractmethod
    async def create_prediction(
        self, prediction: RiskPredictionModel,
    ) -> RiskPredictionModel:
        """Persist a new risk prediction.

        Returns the persisted instance with server-generated defaults populated.
        """
        ...

    @abstractmethod
    async def delete_prediction(self, prediction_id: str) -> bool:
        """Delete a single prediction by ID.

        Returns True if a row was deleted, False if not found.
        """
        ...

    @abstractmethod
    async def delete_prediction_history(
        self,
        sensor_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        before: Optional[datetime] = None,
    ) -> int:
        """Delete predictions matching the given filters.

        At least one filter must be provided to prevent accidental full wipes.
        Returns the number of deleted rows.
        """
        ...
