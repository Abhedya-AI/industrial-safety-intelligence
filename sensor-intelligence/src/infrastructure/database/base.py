"""SQLAlchemy declarative base for all ORM models.

All ORM models must inherit from this Base class.
Alembic also references this Base for auto-generating migrations.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass
