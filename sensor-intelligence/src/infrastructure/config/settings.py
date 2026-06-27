"""Application settings driven by environment variables.

Uses pydantic-settings to load configuration from .env files
and environment variables with type validation.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration.

    All settings can be overridden via environment variables or .env file.
    Naming convention: UPPER_SNAKE_CASE in env, lower_snake_case in code.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──
    app_name: str = "SensorIntelligence"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True

    # ── Database ──
    database_url: str = "sqlite+aiosqlite:///./sensor_intelligence.db"

    # ── API ──
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ── Messaging ──
    event_broker: str = "noop"  # "noop" | "kafka" | "mqtt"
    kafka_bootstrap_servers: str = "localhost:9092"
    mqtt_broker_url: str = "mqtt://localhost:1883"

    # ── Logging ──
    log_level: str = "INFO"

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite (affects engine options)."""
        return "sqlite" in self.database_url

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton.

    Settings are loaded once and cached for the lifetime of the process.
    """
    return Settings()
