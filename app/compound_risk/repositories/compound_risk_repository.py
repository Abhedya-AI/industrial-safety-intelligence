"""Compound Risk repository interface (port).

Defines the abstract contract for compound risk persistence.
Concrete implementations (e.g. SQLAlchemy) must implement every method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from app.compound_risk.models.compound_risk_model import CompoundRiskModel


class CompoundRiskRepository(ABC):
    """Abstract interface for compound risk persistence.

    Returns CompoundRiskModel ORM objects. Business validation is NOT
    performed here — that responsibility belongs to the service layer.
    """

    # ── Queries ──

    @abstractmethod
    async def get_by_id(self, analysis_id: str) -> Optional[CompoundRiskModel]:
        """Retrieve a compound risk analysis by its UUID primary key.

        Returns None if no analysis matches.
        """
        ...

    @abstractmethod
    async def get_latest(
        self,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
    ) -> Optional[CompoundRiskModel]:
        """Get the most recent analysis, optionally filtered by zone or equipment.

        Returns None if no analyses exist for the given filter.
        """
        ...

    @abstractmethod
    async def get_history(
        self,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[CompoundRiskModel]:
        """Get paginated analysis history with optional filters.

        Results are ordered by created_at descending (newest first).
        """
        ...

    @abstractmethod
    async def count(
        self,
        zone_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """Count analyses matching the given filters."""
        ...

    # ── Mutations ──

    @abstractmethod
    async def create(
        self, analysis: CompoundRiskModel,
    ) -> CompoundRiskModel:
        """Persist a new compound risk analysis.

        Returns the persisted instance with server-generated defaults populated.
        """
        ...

    @abstractmethod
    async def delete(self, analysis_id: str) -> bool:
        """Delete a single analysis by ID.

        Returns True if a row was deleted, False if not found.
        """
        ...
