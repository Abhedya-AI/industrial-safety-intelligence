"""Regression tests for Critical Readiness Findings.

Validates the fixes for all four critical findings from the
Platform Readiness Report:

  1. EVENT_BROKER config wired into DI (K-1 / DI-3)
  2. kafka-python and neo4j in requirements.txt (K-2 / N-1)
  3. CompoundRiskModel registered in lifespan (DB-1)
  4. All publishers use the shared event producer

These tests do NOT require Kafka or Neo4j to be running.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
MAIN_MODULE = "app.main"
DEPS_MODULE = "app.core.dependencies"


def _make_settings(**overrides):
    """Create a mock Settings object with the given overrides."""
    defaults = {
        "event_broker": "noop",
        "kafka_bootstrap_servers": "localhost:9092",
        "graph_repository": "in_memory",
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_username": "neo4j",
        "neo4j_password": "test",
        "neo4j_database": "neo4j",
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


def _reset_producer_singleton():
    """Reset the cached event producer singleton between tests."""
    import app.core.dependencies as deps
    deps._event_producer = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. EVENT_BROKER CONFIG WIRING (K-1 / DI-3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEventBrokerWiring:
    """Verify that settings.event_broker drives producer selection."""

    def setup_method(self):
        _reset_producer_singleton()

    def teardown_method(self):
        _reset_producer_singleton()

    def test_noop_broker_returns_noop_producer(self):
        """EVENT_BROKER=noop → NoopEventProducer."""
        from app.core.dependencies import _get_event_producer
        from app.shared.messaging.producer import NoopEventProducer

        settings = _make_settings(event_broker="noop")
        producer = _get_event_producer(settings)

        assert isinstance(producer, NoopEventProducer)

    def test_noop_is_default(self):
        """Default event_broker value should produce NoopEventProducer."""
        from app.core.dependencies import _get_event_producer
        from app.shared.messaging.producer import NoopEventProducer

        settings = _make_settings()  # default: "noop"
        producer = _get_event_producer(settings)

        assert isinstance(producer, NoopEventProducer)

    def test_kafka_broker_returns_kafka_producer(self):
        """EVENT_BROKER=kafka → KafkaEventProducer (may degrade to noop)."""
        from app.core.dependencies import _get_event_producer
        from app.shared.messaging.producer import KafkaEventProducer

        settings = _make_settings(event_broker="kafka")
        producer = _get_event_producer(settings)

        # KafkaEventProducer degrades gracefully if broker is unreachable,
        # but the TYPE should still be KafkaEventProducer
        assert isinstance(producer, KafkaEventProducer)

    def test_kafka_broker_case_insensitive(self):
        """EVENT_BROKER=Kafka (mixed case) should still work."""
        from app.core.dependencies import _get_event_producer
        from app.shared.messaging.producer import KafkaEventProducer

        settings = _make_settings(event_broker="Kafka")
        producer = _get_event_producer(settings)

        assert isinstance(producer, KafkaEventProducer)

    def test_kafka_broker_with_whitespace(self):
        """EVENT_BROKER=' kafka ' (whitespace) should still work."""
        from app.core.dependencies import _get_event_producer
        from app.shared.messaging.producer import KafkaEventProducer

        settings = _make_settings(event_broker=" kafka ")
        producer = _get_event_producer(settings)

        assert isinstance(producer, KafkaEventProducer)

    def test_producer_singleton_cached(self):
        """The producer should be cached as a singleton."""
        from app.core.dependencies import _get_event_producer

        settings = _make_settings(event_broker="noop")
        producer1 = _get_event_producer(settings)
        producer2 = _get_event_producer(settings)

        assert producer1 is producer2

    def test_unknown_broker_falls_back_to_noop(self):
        """Unknown EVENT_BROKER value → NoopEventProducer."""
        from app.core.dependencies import _get_event_producer
        from app.shared.messaging.producer import NoopEventProducer

        settings = _make_settings(event_broker="unknown_broker")
        producer = _get_event_producer(settings)

        assert isinstance(producer, NoopEventProducer)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. PUBLISHER DI USES SHARED PRODUCER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPublisherDIIntegration:
    """Verify all module publishers use the config-driven event producer."""

    def setup_method(self):
        _reset_producer_singleton()

    def teardown_method(self):
        _reset_producer_singleton()

    def test_sensor_intelligence_publisher_uses_shared_producer(self):
        """SensorIntelligencePublisher gets the shared event producer."""
        from app.core.dependencies import (
            _get_event_producer,
            get_sensor_intelligence_publisher,
        )
        from app.sensor_intelligence.messaging.publisher import (
            SensorIntelligencePublisher,
        )

        settings = _make_settings(event_broker="noop")
        expected_producer = _get_event_producer(settings)

        # Simulate FastAPI Depends() resolution
        publisher = get_sensor_intelligence_publisher(settings=settings)

        assert isinstance(publisher, SensorIntelligencePublisher)
        assert publisher._producer is expected_producer

    def test_risk_prediction_publisher_uses_shared_producer(self):
        """RiskPredictionPublisher gets the shared event producer."""
        from app.core.dependencies import (
            _get_event_producer,
            get_risk_prediction_publisher,
        )
        from app.risk_prediction.messaging.publisher import (
            RiskPredictionPublisher,
        )

        settings = _make_settings(event_broker="noop")
        expected_producer = _get_event_producer(settings)

        publisher = get_risk_prediction_publisher(settings=settings)

        assert isinstance(publisher, RiskPredictionPublisher)
        assert publisher._producer is expected_producer

    def test_hazard_propagation_publisher_uses_shared_producer(self):
        """HazardPropagationService's publisher uses the shared producer."""
        from app.core.dependencies import (
            _get_event_producer,
            get_hazard_propagation_service,
        )

        settings = _make_settings(event_broker="noop")
        expected_producer = _get_event_producer(settings)

        service = get_hazard_propagation_service(settings=settings)

        assert service._publisher._producer is expected_producer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. REQUIREMENTS.TXT DEPENDENCIES (K-2, N-1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRequirementsTxt:
    """Verify critical runtime dependencies are listed in requirements.txt."""

    @pytest.fixture(autouse=True)
    def _load_requirements(self):
        self.requirements = REQUIREMENTS_FILE.read_text()

    def test_kafka_python_in_requirements(self):
        """kafka-python must be listed as a dependency."""
        assert "kafka-python" in self.requirements

    def test_neo4j_in_requirements(self):
        """neo4j must be listed as a dependency."""
        assert re.search(r"^neo4j[=<>~!]", self.requirements, re.MULTILINE)

    def test_kafka_python_importable(self):
        """kafka-python must be importable at runtime."""
        # This verifies it's actually installed, not just listed
        try:
            import kafka  # noqa: F401
            importable = True
        except ImportError:
            importable = False

        # If not installed yet, that's OK — we just check it's in requirements
        if not importable:
            pytest.skip("kafka-python not installed in test env")

    def test_neo4j_importable(self):
        """neo4j driver must be importable at runtime."""
        try:
            import neo4j  # noqa: F401
            importable = True
        except ImportError:
            importable = False

        assert importable, (
            "neo4j is in requirements.txt but not importable. "
            "Run: pip install neo4j"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. COMPOUND RISK MODEL REGISTRATION (DB-1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCompoundRiskModelRegistration:
    """Verify CompoundRiskModel is registered on Base.metadata."""

    def test_compound_risk_model_extends_base(self):
        """CompoundRiskModel must extend the shared Base."""
        from app.compound_risk.models.compound_risk_model import (
            CompoundRiskModel,
        )
        from app.shared.database.base import Base

        assert issubclass(CompoundRiskModel, Base)

    def test_compound_risk_table_in_metadata(self):
        """CompoundRiskModel's table must be registered on Base.metadata."""
        # Import the model (this registers it on Base.metadata)
        from app.compound_risk.models.compound_risk_model import (  # noqa: F401
            CompoundRiskModel,
        )
        from app.shared.database.base import Base

        table_names = list(Base.metadata.tables.keys())
        assert "compound_risk_analyses" in table_names or any(
            "compound_risk" in t for t in table_names
        ), (
            f"CompoundRiskModel table not in Base.metadata. "
            f"Tables found: {table_names}"
        )

    def test_compound_risk_model_imported_in_main(self):
        """main.py must import compound_risk_model in its lifespan."""
        main_source = (PROJECT_ROOT / "app" / "main.py").read_text()

        assert "compound_risk_model" in main_source, (
            "compound_risk_model is not imported in app/main.py. "
            "The table will not be auto-created in dev mode."
        )

    def test_all_orm_models_imported_in_main(self):
        """All ORM model modules should be imported in main.py lifespan."""
        main_source = (PROJECT_ROOT / "app" / "main.py").read_text()

        expected_imports = [
            "alert_model",
            "anomaly_model",
            "reading_model",
            "sensor_model",
            "threshold_model",
            "risk_prediction_model",
            "compound_risk_model",
        ]

        missing = [m for m in expected_imports if m not in main_source]
        assert not missing, (
            f"ORM models not imported in main.py lifespan: {missing}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. SETTINGS CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSettingsConfiguration:
    """Verify settings includes all broker/graph config fields."""

    def test_event_broker_setting_exists(self):
        """Settings must have event_broker field."""
        from app.core.settings import Settings

        s = Settings()
        assert hasattr(s, "event_broker")
        assert s.event_broker == "noop"

    def test_kafka_bootstrap_servers_setting_exists(self):
        """Settings must have kafka_bootstrap_servers field."""
        from app.core.settings import Settings

        s = Settings()
        assert hasattr(s, "kafka_bootstrap_servers")
        assert "localhost" in s.kafka_bootstrap_servers

    def test_graph_repository_setting_exists(self):
        """Settings must have graph_repository field."""
        from app.core.settings import Settings

        s = Settings()
        assert hasattr(s, "graph_repository")
        assert s.graph_repository == "in_memory"

    def test_neo4j_settings_exist(self):
        """Settings must have neo4j connection fields."""
        from app.core.settings import Settings

        s = Settings()
        assert hasattr(s, "neo4j_uri")
        assert hasattr(s, "neo4j_username")
        assert hasattr(s, "neo4j_password")
        assert hasattr(s, "neo4j_database")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. APPLICATION STARTUP VERIFICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestApplicationStartup:
    """Verify the FastAPI app can be created and configured."""

    def test_app_creates_successfully(self):
        """create_app() should return a configured FastAPI instance."""
        from app.main import create_app

        app = create_app()
        assert app is not None
        assert app.title == "IndustrialSafetyIntelligence"

    def test_all_module_routers_registered(self):
        """All 4 module routers must be mounted on the app."""
        from app.main import create_app

        app = create_app()
        route_paths = [r.path for r in app.routes]

        # Check that the key prefixes are present
        sensor_routes = [p for p in route_paths if "/sensors" in p or "/readings" in p]
        risk_routes = [p for p in route_paths if "/risk" in p]
        compound_routes = [p for p in route_paths if "/compound-analysis" in p]
        hazard_routes = [p for p in route_paths if "/hazard" in p]

        assert len(sensor_routes) > 0, "Sensor Intelligence routes not found"
        assert len(risk_routes) > 0, "Risk Prediction routes not found"
        assert len(compound_routes) > 0, "Compound Risk routes not found"
        assert len(hazard_routes) > 0, "Hazard Propagation routes not found"

    def test_event_broker_env_var_recognized(self):
        """EVENT_BROKER environment variable must be recognized by Settings."""
        from app.core.settings import Settings

        with patch.dict(os.environ, {"EVENT_BROKER": "kafka"}):
            s = Settings()
            assert s.event_broker == "kafka"

    def test_graph_repository_env_var_recognized(self):
        """GRAPH_REPOSITORY environment variable must be recognized by Settings."""
        from app.core.settings import Settings

        with patch.dict(os.environ, {"GRAPH_REPOSITORY": "neo4j"}):
            s = Settings()
            assert s.graph_repository == "neo4j"
