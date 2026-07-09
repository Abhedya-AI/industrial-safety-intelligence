"""FastAPI dependency injection container (Composition Root).

Provides application settings, database sessions, event publishers,
and service-layer dependencies.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import Settings, get_settings
from app.shared.database.connection import get_async_session

# Event publisher (generic, shared)
from app.sensor_intelligence.repositories.noop_publisher import NoOpPublisher


def get_app_settings() -> Settings:
    """Provide application settings."""
    return get_settings()


async def get_db_session(
    session: AsyncSession = Depends(get_async_session),
) -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session for the request lifecycle."""
    yield session


# ── Event Publisher ──

_event_publisher = None


def _get_publisher(settings: Settings):
    global _event_publisher
    if _event_publisher is None:
        _event_publisher = NoOpPublisher()
    return _event_publisher


def get_event_publisher(
    settings: Settings = Depends(get_app_settings),
):
    """Provide the configured event publisher."""
    return _get_publisher(settings)


# ── Sensor Intelligence Dependencies ──


def get_sensor_service(
    session: AsyncSession = Depends(get_async_session),
):
    """Provide a SensorService wired to a SQLAlchemy repository."""
    from app.sensor_intelligence.repositories.sqlalchemy_sensor_repo import (
        SQLAlchemySensorRepository,
    )
    from app.sensor_intelligence.services.sensor_service import SensorService

    repo = SQLAlchemySensorRepository(session)
    return SensorService(repo, session)


def get_reading_service(
    session: AsyncSession = Depends(get_async_session),
):
    """Provide a ReadingService wired to SQLAlchemy repositories."""
    from app.sensor_intelligence.repositories.sqlalchemy_reading_repo import (
        SQLAlchemyReadingRepository,
    )
    from app.sensor_intelligence.repositories.sqlalchemy_sensor_repo import (
        SQLAlchemySensorRepository,
    )
    from app.sensor_intelligence.services.reading_service import ReadingService

    reading_repo = SQLAlchemyReadingRepository(session)
    sensor_repo = SQLAlchemySensorRepository(session)
    return ReadingService(reading_repo, sensor_repo)


# ── Risk Prediction Dependencies ──


def get_risk_prediction_service(
    session: AsyncSession = Depends(get_async_session),
):
    """Provide a RiskPredictionService wired to a SQLAlchemy repository."""
    from app.risk_prediction.repositories.sqlalchemy_risk_prediction_repo import (
        SQLAlchemyRiskPredictionRepository,
    )
    from app.risk_prediction.services.risk_prediction_service import (
        RiskPredictionService,
    )

    repo = SQLAlchemyRiskPredictionRepository(session)
    return RiskPredictionService(repo)


def get_risk_prediction_publisher():
    """Provide a RiskPredictionPublisher wired to a shared Kafka producer.

    Uses NoopEventProducer in development; switch to KafkaEventProducer in
    production via environment configuration.
    """
    from app.risk_prediction.messaging.publisher import RiskPredictionPublisher
    from app.shared.messaging.producer import NoopEventProducer

    return RiskPredictionPublisher(NoopEventProducer())


# ── Compound Risk Intelligence Dependencies ──


def get_compound_risk_service(
    session: AsyncSession = Depends(get_async_session),
):
    """Provide a CompoundRiskService wired to all sub-components.

    Injects:
      - SQLAlchemy repository → Aggregation service
      - Default rule engine (configurable thresholds)
      - Explainability service
      - Noop Kafka publisher (switched to real in production)
    """
    from app.compound_risk.messaging.publisher import CompoundRiskPublisher
    from app.compound_risk.repositories.sqlalchemy_compound_risk_repo import (
        SQLAlchemyCompoundRiskRepository,
    )
    from app.compound_risk.rules.rule_engine import (
        CompoundRiskRuleEngine,
        create_default_rules,
    )
    from app.compound_risk.services.compound_risk_facade import (
        CompoundRiskService,
    )
    from app.compound_risk.services.compound_risk_service import (
        CompoundRiskAggregationService,
    )
    from app.compound_risk.services.explainability_service import (
        ExplainabilityService,
    )
    from app.shared.messaging.producer import NoopEventProducer

    repo = SQLAlchemyCompoundRiskRepository(session)
    aggregation = CompoundRiskAggregationService(repo)
    rule_engine = CompoundRiskRuleEngine(create_default_rules())
    explainability = ExplainabilityService()
    publisher = CompoundRiskPublisher(NoopEventProducer())

    return CompoundRiskService(
        aggregation_service=aggregation,
        rule_engine=rule_engine,
        explainability_service=explainability,
        publisher=publisher,
    )


# ── Sensor Intelligence Kafka Publisher ──


def get_sensor_intelligence_publisher():
    """Provide a SensorIntelligencePublisher wired to a shared Kafka producer.

    Uses NoopEventProducer in development; switch to KafkaEventProducer in
    production via environment configuration.
    """
    from app.sensor_intelligence.messaging.publisher import (
        SensorIntelligencePublisher,
    )
    from app.shared.messaging.producer import NoopEventProducer

    return SensorIntelligencePublisher(NoopEventProducer())


# ── Hazard Propagation Dependencies ──

# Cached graph repository singleton (created once, reused)
_graph_repository = None


def _get_graph_repository(settings: Settings):
    """Select graph repository implementation based on configuration.

    Reads ``settings.graph_repository`` to choose between:
      - ``"in_memory"`` → InMemoryGraphRepository (default, no external deps)
      - ``"neo4j"``     → Neo4jGraphRepository (requires running Neo4j instance)

    The repository is cached as a module-level singleton so that all
    requests share the same graph state.
    """
    global _graph_repository
    if _graph_repository is not None:
        return _graph_repository

    repo_type = settings.graph_repository.lower().strip()

    if repo_type == "neo4j":
        try:
            from app.hazard_propagation.repositories.neo4j_graph_repo import (
                Neo4jGraphRepository,
            )

            _graph_repository = Neo4jGraphRepository(
                uri=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=settings.neo4j_password,
                database=settings.neo4j_database,
            )
            import logging
            logging.getLogger(__name__).info(
                "Graph repository: Neo4jGraphRepository (%s)",
                settings.neo4j_uri,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to create Neo4jGraphRepository (%s), "
                "falling back to InMemoryGraphRepository: %s",
                settings.neo4j_uri, exc,
            )
            from app.hazard_propagation.repositories.in_memory_graph_repo import (
                InMemoryGraphRepository,
            )
            _graph_repository = InMemoryGraphRepository()
    else:
        from app.hazard_propagation.repositories.in_memory_graph_repo import (
            InMemoryGraphRepository,
        )
        _graph_repository = InMemoryGraphRepository()
        import logging
        logging.getLogger(__name__).info(
            "Graph repository: InMemoryGraphRepository",
        )

    return _graph_repository


def get_hazard_propagation_service(
    settings: Settings = Depends(get_app_settings),
):
    """Provide a HazardPropagationService wired to all sub-components.

    Injects:
      - Graph repository selected by ``GRAPH_REPOSITORY`` env var:
        ``"in_memory"`` (default) or ``"neo4j"``
      - HazardPropagationEngine (BFS propagation)
      - HazardPropagationPublisher (Noop in dev; real Kafka in production)
    """
    from app.hazard_propagation.messaging.publisher import (
        HazardPropagationPublisher,
    )
    from app.hazard_propagation.services.config import PropagationConfig
    from app.hazard_propagation.services.hazard_propagation_service import (
        HazardPropagationService,
    )
    from app.shared.messaging.producer import NoopEventProducer

    graph_repo = _get_graph_repository(settings)
    config = PropagationConfig()
    publisher = HazardPropagationPublisher(NoopEventProducer())

    return HazardPropagationService(
        graph_repo=graph_repo,
        publisher=publisher,
        config=config,
    )
