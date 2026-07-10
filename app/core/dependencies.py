"""FastAPI dependency injection container (Composition Root).

Provides application settings, database sessions, event publishers,
and service-layer dependencies.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import Settings, get_settings
from app.shared.database.connection import get_async_session

# Event publisher (generic, shared)
from app.sensor_intelligence.repositories.noop_publisher import NoOpPublisher

logger = logging.getLogger(__name__)


def get_app_settings() -> Settings:
    """Provide application settings."""
    return get_settings()


async def get_db_session(
    session: AsyncSession = Depends(get_async_session),
) -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session for the request lifecycle."""
    yield session


# ── Event Producer (Kafka / Noop) ──

_event_producer = None
_event_publisher = None


def _get_event_producer(settings: Settings):
    """Select event producer implementation based on configuration.

    Reads ``settings.event_broker`` to choose between:
      - ``"noop"``  → NoopEventProducer (default, logs events only)
      - ``"kafka"`` → KafkaEventProducer (connects to Kafka cluster)

    The producer is cached as a module-level singleton so that all
    publishers share the same connection.
    """
    global _event_producer
    if _event_producer is not None:
        return _event_producer

    broker = settings.event_broker.lower().strip()

    if broker == "kafka":
        try:
            from app.shared.messaging.producer import KafkaEventProducer

            _event_producer = KafkaEventProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                enabled=True,
            )
            logger.info(
                "Event producer: KafkaEventProducer (%s)",
                settings.kafka_bootstrap_servers,
            )
        except Exception as exc:
            from app.shared.messaging.producer import NoopEventProducer

            logger.warning(
                "Failed to create KafkaEventProducer (%s), "
                "falling back to NoopEventProducer: %s",
                settings.kafka_bootstrap_servers, exc,
            )
            _event_producer = NoopEventProducer()
    else:
        from app.shared.messaging.producer import NoopEventProducer

        _event_producer = NoopEventProducer()
        logger.info("Event producer: NoopEventProducer (event_broker=%s)", broker)

    return _event_producer


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


def get_risk_prediction_publisher(
    settings: Settings = Depends(get_app_settings),
):
    """Provide a RiskPredictionPublisher wired to the shared event producer.

    Uses NoopEventProducer when ``EVENT_BROKER=noop`` (default).
    Uses KafkaEventProducer when ``EVENT_BROKER=kafka``.
    """
    from app.risk_prediction.messaging.publisher import RiskPredictionPublisher

    return RiskPredictionPublisher(_get_event_producer(settings))


# ── Compound Risk Intelligence Dependencies ──


def get_compound_risk_service(
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_app_settings),
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
    repo = SQLAlchemyCompoundRiskRepository(session)
    aggregation = CompoundRiskAggregationService(repo)
    rule_engine = CompoundRiskRuleEngine(create_default_rules())
    explainability = ExplainabilityService()
    publisher = CompoundRiskPublisher(_get_event_producer(settings))

    return CompoundRiskService(
        aggregation_service=aggregation,
        rule_engine=rule_engine,
        explainability_service=explainability,
        publisher=publisher,
    )


# ── Sensor Intelligence Kafka Publisher ──


def get_sensor_intelligence_publisher(
    settings: Settings = Depends(get_app_settings),
):
    """Provide a SensorIntelligencePublisher wired to the shared event producer.

    Uses NoopEventProducer when ``EVENT_BROKER=noop`` (default).
    Uses KafkaEventProducer when ``EVENT_BROKER=kafka``.
    """
    from app.sensor_intelligence.messaging.publisher import (
        SensorIntelligencePublisher,
    )

    return SensorIntelligencePublisher(_get_event_producer(settings))


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

    graph_repo = _get_graph_repository(settings)
    config = PropagationConfig()
    publisher = HazardPropagationPublisher(_get_event_producer(settings))

    return HazardPropagationService(
        graph_repo=graph_repo,
        publisher=publisher,
        config=config,
    )


# ── Digital Twin Dependencies ──

# Cached TwinStateManager singleton (created once, reused)
_twin_state_manager = None


def _get_twin_state_manager(settings: Settings):
    """Get or create the TwinStateManager singleton.

    Shares the same GraphRepository singleton used by
    HazardPropagationService.
    """
    global _twin_state_manager
    if _twin_state_manager is not None:
        return _twin_state_manager

    from app.digital_twin.services.twin_state_manager import TwinStateManager

    graph_repo = _get_graph_repository(settings)
    _twin_state_manager = TwinStateManager(graph_repo=graph_repo)
    logger.info("Digital Twin state manager created")
    return _twin_state_manager


def get_digital_twin_service(
    settings: Settings = Depends(get_app_settings),
):
    """Provide the TwinStateManager for API endpoints.

    Reuses the shared GraphRepository singleton.
    """
    return _get_twin_state_manager(settings)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Kafka Consumer Infrastructure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_event_consumer = None
_consumer_thread = None


def _get_all_consumer_topics() -> list[str]:
    """Collect all topics that our consumers need to subscribe to."""
    from app.compound_risk.messaging.consumer import (
        COMPOUND_RISK_SUBSCRIBED_TOPICS,
    )
    from app.digital_twin.messaging.consumer import (
        DIGITAL_TWIN_SUBSCRIBED_TOPICS,
    )
    from app.hazard_propagation.messaging.consumer import (
        HAZARD_PROPAGATION_SUBSCRIBED_TOPICS,
    )

    all_topics = list(set(
        COMPOUND_RISK_SUBSCRIBED_TOPICS
        + HAZARD_PROPAGATION_SUBSCRIBED_TOPICS
        + DIGITAL_TWIN_SUBSCRIBED_TOPICS
    ))
    return all_topics


def _get_event_consumer(settings: Settings):
    """Select event consumer implementation based on configuration.

    Reads ``settings.event_broker`` to choose between:
      - ``"noop"``  → NoopEventConsumer (default, no Kafka connection)
      - ``"kafka"`` → KafkaEventConsumer (connects to Kafka cluster)

    The consumer is cached as a module-level singleton.
    """
    global _event_consumer
    if _event_consumer is not None:
        return _event_consumer

    broker = settings.event_broker.lower().strip()

    if broker == "kafka":
        try:
            from app.shared.messaging.consumer import KafkaEventConsumer

            topics = _get_all_consumer_topics()
            _event_consumer = KafkaEventConsumer(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                group_id=settings.kafka_consumer_group_id,
                topics=topics,
                enabled=True,
                auto_offset_reset=settings.kafka_auto_offset_reset,
            )
            logger.info(
                "Event consumer: KafkaEventConsumer (group=%s, topics=%s)",
                settings.kafka_consumer_group_id, topics,
            )
        except Exception as exc:
            from app.shared.messaging.consumer import NoopEventConsumer

            logger.warning(
                "Failed to create KafkaEventConsumer (%s), "
                "falling back to NoopEventConsumer: %s",
                settings.kafka_bootstrap_servers, exc,
            )
            _event_consumer = NoopEventConsumer()
    else:
        from app.shared.messaging.consumer import NoopEventConsumer

        _event_consumer = NoopEventConsumer()
        logger.info("Event consumer: NoopEventConsumer (event_broker=%s)", broker)

    return _event_consumer


def _register_consumer_handlers(settings: Settings) -> None:
    """Register all module-specific event handlers on the shared consumer.

    Creates handler instances with proper dependencies and registers
    them via each module's ConsumerSetup class.
    """
    consumer = _get_event_consumer(settings)
    producer = _get_event_producer(settings)

    # ── Compound Risk Consumer ──
    from app.compound_risk.messaging.consumer import CompoundRiskConsumerSetup
    from app.compound_risk.messaging.handler import CompoundRiskEventHandler
    from app.compound_risk.messaging.publisher import CompoundRiskPublisher
    from app.compound_risk.repositories.session_scoped_compound_risk_repo import (
        SessionScopedCompoundRiskRepository,
    )
    from app.compound_risk.rules.rule_engine import (
        CompoundRiskRuleEngine,
        create_default_rules,
    )
    from app.compound_risk.services.compound_risk_service import (
        CompoundRiskAggregationService,
    )
    from app.compound_risk.services.explainability_service import (
        ExplainabilityService,
    )
    from app.shared.database.connection import async_session_factory

    cr_rule_engine = CompoundRiskRuleEngine(create_default_rules())
    cr_explainability = ExplainabilityService()
    cr_publisher = CompoundRiskPublisher(producer)

    # Consumer runs in a background thread with asyncio.run() per event,
    # so it needs a session-scoped repository (fresh session per operation)
    # rather than a request-scoped session from FastAPI's Depends().
    cr_repo = SessionScopedCompoundRiskRepository(async_session_factory)
    cr_handler = CompoundRiskEventHandler(
        aggregation_service=CompoundRiskAggregationService(repository=cr_repo),
        rule_engine=cr_rule_engine,
        explainability_service=cr_explainability,
        publisher=cr_publisher,
    )
    cr_setup = CompoundRiskConsumerSetup(consumer, cr_handler)
    cr_setup.register()

    # ── Hazard Propagation Consumer ──
    from app.hazard_propagation.messaging.consumer import (
        HazardPropagationConsumerSetup,
    )
    from app.hazard_propagation.messaging.handler import (
        HazardPropagationEventHandler,
    )
    from app.hazard_propagation.messaging.publisher import (
        HazardPropagationPublisher,
    )
    from app.hazard_propagation.services.propagation_engine import (
        HazardPropagationEngine,
    )

    graph_repo = _get_graph_repository(settings)
    hp_engine = HazardPropagationEngine(graph_repo=graph_repo)
    hp_publisher = HazardPropagationPublisher(producer)

    hp_handler = HazardPropagationEventHandler(
        propagation_engine=hp_engine,
        publisher=hp_publisher,
        graph_repo=graph_repo,
    )
    hp_setup = HazardPropagationConsumerSetup(consumer, hp_handler)
    hp_setup.register()

    # ── Digital Twin Consumer ──
    from app.digital_twin.messaging.consumer import DigitalTwinConsumerSetup
    from app.digital_twin.messaging.handler import DigitalTwinEventHandler

    dt_state = _get_twin_state_manager(settings)
    dt_handler = DigitalTwinEventHandler(state_manager=dt_state)
    dt_setup = DigitalTwinConsumerSetup(consumer, dt_handler)
    dt_setup.register()

    logger.info(
        "All consumer handlers registered: "
        "compound_risk=%s, hazard_propagation=%s, digital_twin=%s",
        cr_setup.is_registered, hp_setup.is_registered,
        dt_setup.is_registered,
    )


def start_consumers(settings: Settings) -> None:
    """Start Kafka consumer loop in a background thread.

    Called during application startup (lifespan). The consumer loop
    runs in a daemon thread so it doesn't block the ASGI event loop
    and automatically terminates on process exit.

    Safe to call when EVENT_BROKER=noop — will log and return.
    """
    global _consumer_thread

    broker = settings.event_broker.lower().strip()
    if broker != "kafka":
        logger.info(
            "Kafka consumers not started (event_broker=%s)", broker,
        )
        return

    # Register handlers before starting the loop
    _register_consumer_handlers(settings)

    consumer = _get_event_consumer(settings)
    if not consumer.is_enabled or not consumer.is_connected:
        logger.warning(
            "Kafka consumer is not connected — skipping consumer start.",
        )
        return

    import threading

    _consumer_thread = threading.Thread(
        target=consumer.start,
        name="kafka-consumer-loop",
        daemon=True,
    )
    _consumer_thread.start()
    logger.info(
        "Kafka consumer loop started in background thread "
        "(group=%s, thread=%s)",
        settings.kafka_consumer_group_id,
        _consumer_thread.name,
    )


def stop_consumers() -> None:
    """Stop Kafka consumer loop and close the consumer connection.

    Called during application shutdown (lifespan).
    """
    global _event_consumer, _consumer_thread

    if _event_consumer is not None:
        _event_consumer.close()
        logger.info("Kafka consumer closed.")
        _event_consumer = None

    if _consumer_thread is not None and _consumer_thread.is_alive():
        _consumer_thread.join(timeout=5.0)
        logger.info("Kafka consumer thread joined.")
        _consumer_thread = None
