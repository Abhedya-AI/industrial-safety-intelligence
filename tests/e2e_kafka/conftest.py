"""Shared fixtures for Kafka E2E verification tests.

Provides real Kafka connectivity, topic management, event payloads,
graph topology, database sessions, and offset tracking utilities.

All fixtures connect to the REAL Kafka broker at localhost:9092
(started via docker-compose). Tests auto-skip if Kafka is unreachable.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.shared.database.base import Base

# Import all ORM models to register on Base.metadata
from app.sensor_intelligence.models.sensor_model import SensorModel  # noqa: F401
from app.sensor_intelligence.models.reading_model import ReadingModel  # noqa: F401
from app.sensor_intelligence.models.anomaly_model import AnomalyModel  # noqa: F401
from app.sensor_intelligence.models.alert_model import AlertModel  # noqa: F401
from app.sensor_intelligence.models.threshold_model import ThresholdModel  # noqa: F401
from app.sensor_intelligence.models.sensor_health_model import SensorHealthModel  # noqa: F401
from app.sensor_intelligence.models.sensor_baseline_model import SensorBaselineModel  # noqa: F401
from app.risk_prediction.models.risk_prediction_model import RiskPredictionModel  # noqa: F401
from app.compound_risk.models.compound_risk_model import CompoundRiskModel  # noqa: F401

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = "localhost:9092"

# Topics used in the E2E pipeline
E2E_TOPICS = [
    "sensor.reading.anomaly",
    "compound.risk.detected",
    "hazard.propagated",
    "hazard.detected",
    "risk.assessment.generated",
    "risk.score.updated",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Kafka connectivity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _kafka_reachable(host: str = "localhost", port: int = 9092, timeout: float = 3.0) -> bool:
    """Check if Kafka broker is reachable via TCP."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


@pytest.fixture(scope="session")
def kafka_available():
    """Skip entire test session if Kafka is not reachable."""
    if not _kafka_reachable():
        pytest.skip(
            "Kafka broker not reachable at localhost:9092. "
            "Run: docker-compose up -d zookeeper kafka"
        )
    return True


@pytest.fixture(scope="session")
def kafka_bootstrap():
    """Return the Kafka bootstrap server address."""
    return KAFKA_BOOTSTRAP


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Kafka producer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture(scope="module")
def kafka_producer(kafka_available, kafka_bootstrap):
    """Provide a real KafkaEventProducer connected to the broker."""
    from app.shared.messaging.producer import KafkaEventProducer

    producer = KafkaEventProducer(
        bootstrap_servers=kafka_bootstrap,
        enabled=True,
    )
    assert producer.is_connected, "KafkaEventProducer failed to connect"
    yield producer
    producer.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Kafka admin / topic management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture(scope="module")
def kafka_admin(kafka_available, kafka_bootstrap):
    """Provide a KafkaAdminClient for topic and offset management."""
    from kafka.admin import KafkaAdminClient

    admin = KafkaAdminClient(bootstrap_servers=kafka_bootstrap)
    yield admin
    admin.close()


@pytest.fixture(scope="module")
def ensure_topics(kafka_admin, kafka_bootstrap):
    """Ensure all E2E topics exist on the broker."""
    from kafka.admin import NewTopic

    existing = kafka_admin.list_topics()
    to_create = [
        NewTopic(name=t, num_partitions=1, replication_factor=1)
        for t in E2E_TOPICS
        if t not in existing
    ]
    if to_create:
        kafka_admin.create_topics(new_topics=to_create)
        logger.info("Created topics: %s", [t.name for t in to_create])
        # Wait for topic metadata to propagate
        time.sleep(2)
    return E2E_TOPICS


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Verification consumer factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture(scope="module")
def verification_consumer_factory(kafka_available, kafka_bootstrap):
    """Factory for short-lived Kafka consumers with unique group IDs.

    Each consumer subscribes to specified topics and uses 'earliest'
    offset reset so it can read messages published during the test.
    """
    consumers = []

    def _create(topics: List[str], group_id: Optional[str] = None):
        from kafka import KafkaConsumer
        from app.shared.messaging.serialization import (
            kafka_value_deserializer,
            kafka_key_deserializer,
        )

        gid = group_id or f"e2e-verify-{uuid.uuid4().hex[:8]}"
        consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=kafka_bootstrap,
            group_id=gid,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=kafka_value_deserializer,
            key_deserializer=kafka_key_deserializer,
            consumer_timeout_ms=15000,
            max_poll_interval_ms=30000,
        )
        consumers.append(consumer)
        return consumer

    yield _create

    for c in consumers:
        try:
            c.close()
        except Exception:
            pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Database session (in-memory SQLite)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


E2E_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

_e2e_engine = create_async_engine(
    E2E_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

e2e_session_factory = async_sessionmaker(
    bind=_e2e_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest_asyncio.fixture(scope="function")
async def db_session():
    """Provide a clean database session for each test."""
    async with _e2e_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with e2e_session_factory() as session:
        yield session

    async with _e2e_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph repository with test topology
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture(scope="module")
def populated_graph_repo():
    """InMemoryGraphRepository pre-loaded with a 3-zone facility graph.

    Topology:
        ZONE_A ↔ ZONE_B ↔ ZONE_C

    Each zone has equipment and sensors. Workers assigned for
    propagation simulation.
    """
    from app.hazard_propagation.repositories.in_memory_graph_repo import (
        InMemoryGraphRepository,
    )
    from app.hazard_propagation.graph.entities import (
        EquipmentNode,
        SensorNode,
        ZoneNode,
    )

    repo = InMemoryGraphRepository()

    async def _build():
        # Create zones with workers
        zone_a = ZoneNode(
            zone_id="ZONE_A",
            zone_name="Reactor Hall A",
            risk_level_baseline="MEDIUM",
            worker_capacity=20,
            current_worker_count=12,
        )
        zone_b = ZoneNode(
            zone_id="ZONE_B",
            zone_name="Processing Bay B",
            risk_level_baseline="HIGH",
            worker_capacity=15,
            current_worker_count=8,
        )
        zone_c = ZoneNode(
            zone_id="ZONE_C",
            zone_name="Storage Area C",
            risk_level_baseline="LOW",
            worker_capacity=10,
            current_worker_count=3,
        )

        await repo.create_zone(zone_a)
        await repo.create_zone(zone_b)
        await repo.create_zone(zone_c)

        # Connect zones: A ↔ B ↔ C
        await repo.create_connection("ZONE_A", "ZONE_B", weight=0.8)
        await repo.create_connection("ZONE_B", "ZONE_C", weight=0.6)

        # Add equipment to zones
        eq_a = EquipmentNode(
            equipment_id="EQ_REACTOR_01",
            equipment_type="REACTOR",
            health_score=85.0,
        )
        eq_b = EquipmentNode(
            equipment_id="EQ_PUMP_01",
            equipment_type="PUMP",
            health_score=70.0,
        )
        eq_c = EquipmentNode(
            equipment_id="EQ_TANK_01",
            equipment_type="STORAGE_TANK",
            health_score=95.0,
        )

        await repo.create_equipment("ZONE_A", eq_a)
        await repo.create_equipment("ZONE_B", eq_b)
        await repo.create_equipment("ZONE_C", eq_c)

        # Add sensors
        sensor_a = SensorNode(
            sensor_id="SENS_TEMP_A1",
            sensor_type="TEMPERATURE",
            unit_of_measurement="celsius",
        )
        sensor_b = SensorNode(
            sensor_id="SENS_GAS_B1",
            sensor_type="GAS",
            unit_of_measurement="ppm",
        )
        await repo.create_sensor("EQ_REACTOR_01", sensor_a)
        await repo.create_sensor("EQ_PUMP_01", sensor_b)

    asyncio.get_event_loop().run_until_complete(_build())
    return repo


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Canonical event payloads
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def anomaly_event_payload() -> Dict[str, Any]:
    """Canonical sensor.reading.anomaly event with high anomaly scores."""
    return {
        "event_type": "sensor.reading.anomaly",
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_system": "sensor_intelligence",
        "version": "1.0",
        "correlation_id": str(uuid.uuid4()),
        "data": {
            "sensor_id": "SENS_TEMP_A1",
            "equipment_id": "EQ_REACTOR_01",
            "zone_id": "ZONE_A",
            "reading_type": "TEMPERATURE",
            "value": 95.5,
            "unit": "celsius",
            "isolation_forest_score": 0.85,
            "autoencoder_score": 0.78,
            "sensor_health_score": 72.0,
            "temperature_celsius": 95.5,
            "gas_level_ppm": 120.0,
            "pressure_bar": 4.5,
            "vibration_level": 0.6,
            "threshold_violation_count": 3,
            "active_alert_count": 2,
            "alert_severity_max": 0.8,
        },
    }


@pytest.fixture
def compound_risk_event_payload() -> Dict[str, Any]:
    """Canonical compound.risk.detected event above propagation threshold."""
    return {
        "event_type": "compound.risk.detected",
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_system": "compound_risk_intelligence",
        "version": "1.0",
        "correlation_id": str(uuid.uuid4()),
        "data": {
            "analysis_id": str(uuid.uuid4()),
            "equipment_id": "EQ_REACTOR_01",
            "zone_id": "ZONE_A",
            "anomaly_score": 0.85,
            "accident_probability": 0.65,
            "risk_score": 72.0,
            "sensor_health_score": 72.0,
            "compound_risk_score": 68.5,
            "risk_level": "HIGH",
            "confidence_score": 0.82,
            "contributing_factors": {
                "gas_risk": 0.7,
                "temperature_risk": 0.6,
                "pressure_risk": 0.3,
            },
            "component_scores": {
                "risk_prediction": 65.0,
                "isolation_forest": 85.0,
                "autoencoder": 78.0,
                "sensor_health": 28.0,
                "alert": 40.0,
                "threshold_violation": 30.0,
            },
            "recommendation": "Immediate inspection required",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Offset tracking utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_topic_offsets(bootstrap_servers: str, topics: List[str]) -> Dict[str, Dict[int, int]]:
    """Get the latest offsets for given topics.

    Returns: {topic: {partition: offset}}
    """
    from kafka import KafkaConsumer, TopicPartition

    consumer = KafkaConsumer(
        bootstrap_servers=bootstrap_servers,
        group_id=f"offset-check-{uuid.uuid4().hex[:8]}",
        enable_auto_commit=False,
    )
    try:
        offsets = {}
        for topic in topics:
            partitions = consumer.partitions_for_topic(topic)
            if partitions:
                tps = [TopicPartition(topic, p) for p in partitions]
                end_offsets = consumer.end_offsets(tps)
                offsets[topic] = {
                    tp.partition: offset for tp, offset in end_offsets.items()
                }
            else:
                offsets[topic] = {}
        return offsets
    finally:
        consumer.close()


def get_consumer_group_offsets(
    bootstrap_servers: str,
    group_id: str,
    topics: List[str],
) -> Dict[str, Dict[int, int]]:
    """Get committed offsets for a consumer group.

    Returns: {topic: {partition: committed_offset}}
    """
    from kafka import KafkaConsumer, TopicPartition
    from kafka.structs import OffsetAndMetadata

    consumer = KafkaConsumer(
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        enable_auto_commit=False,
    )
    try:
        offsets = {}
        for topic in topics:
            partitions = consumer.partitions_for_topic(topic)
            if partitions:
                tps = [TopicPartition(topic, p) for p in partitions]
                for tp in tps:
                    committed = consumer.committed(tp)
                    if topic not in offsets:
                        offsets[topic] = {}
                    offsets[topic][tp.partition] = committed if committed is not None else -1
            else:
                offsets[topic] = {}
        return offsets
    finally:
        consumer.close()
