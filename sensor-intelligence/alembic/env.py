"""Alembic environment configuration for async SQLAlchemy.

This file configures Alembic to work with our async engine
and auto-detect model changes for migration generation.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from src.infrastructure.config.settings import get_settings
from src.infrastructure.database.base import Base

# Import all models so they register with Base.metadata
from src.infrastructure.database.models.sensor_model import SensorModel  # noqa: F401
from src.infrastructure.database.models.reading_model import ReadingModel  # noqa: F401
from src.infrastructure.database.models.anomaly_model import AnomalyModel  # noqa: F401
from src.infrastructure.database.models.alert_model import AlertModel  # noqa: F401
from src.infrastructure.database.models.threshold_model import ThresholdModel  # noqa: F401

# Alembic Config object
config = context.config

# Set up logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate
target_metadata = Base.metadata

# Override sqlalchemy.url from settings if available
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Generates SQL scripts without connecting to the database.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # Required for SQLite ALTER TABLE support
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Run migrations with the given connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,  # Required for SQLite ALTER TABLE support
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    # For async, we need to strip the async driver from the URL
    # and use a sync driver for Alembic
    connectable_url = settings.database_url
    if "aiosqlite" in connectable_url:
        connectable_url = connectable_url.replace("sqlite+aiosqlite", "sqlite")
    elif "asyncpg" in connectable_url:
        connectable_url = connectable_url.replace("postgresql+asyncpg", "postgresql")

    from sqlalchemy import create_engine

    connectable = create_engine(connectable_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        do_run_migrations(connection)

    connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
