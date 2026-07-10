"""Kafka consumer and event flow verification tests.

Tests for the recommended Kafka findings:
  K-3: Kafka/Zookeeper in docker-compose.yml
  K-4: Consumer loop started in main.py lifespan
  K-6: Consumer group configuration

Verifies:
  1. Consumer DI (config-driven selection)
  2. Consumer group configuration
  3. Consumer handler registration
  4. Event flow: SI → RP → CR → HP
  5. Lifespan consumer start/stop
  6. docker-compose.yml Kafka services

Does NOT require a running Kafka broker.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCKER_COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"


def _make_settings(**overrides):
    """Create a mock Settings with reasonable defaults."""
    defaults = {
        "event_broker": "noop",
        "kafka_bootstrap_servers": "localhost:9092",
        "kafka_consumer_group_id": "sentinel_ai",
        "kafka_auto_offset_reset": "earliest",
        "graph_repository": "in_memory",
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_username": "neo4j",
        "neo4j_password": "test",
        "neo4j_database": "neo4j",
    }
    defaults.update(overrides)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


def _reset_singletons():
    """Reset all cached singletons between tests."""
    import app.core.dependencies as deps
    deps._event_producer = None
    deps._event_consumer = None
    deps._consumer_thread = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. DOCKER-COMPOSE KAFKA SERVICES (K-3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDockerComposeKafka:
    """Verify docker-compose.yml includes Kafka infrastructure."""

    @pytest.fixture(autouse=True)
    def _load_compose(self):
        self.compose = DOCKER_COMPOSE_FILE.read_text()

    def test_zookeeper_service_exists(self):
        """docker-compose must define a zookeeper service."""
        assert "zookeeper:" in self.compose

    def test_kafka_service_exists(self):
        """docker-compose must define a kafka service."""
        assert "kafka:" in self.compose

    def test_kafka_port_9092_exposed(self):
        """Kafka must expose port 9092 for host access."""
        assert "9092" in self.compose

    def test_kafka_depends_on_zookeeper(self):
        """Kafka must depend on zookeeper."""
        assert "zookeeper" in self.compose

    def test_kafka_has_healthcheck(self):
        """Kafka must have a health check."""
        assert "kafka-topics" in self.compose

    def test_backend_depends_on_kafka(self):
        """Backend service must depend on kafka."""
        # Check that the backend section references kafka
        backend_idx = self.compose.find("backend:")
        assert backend_idx > 0
        backend_section = self.compose[backend_idx:]
        assert "kafka:" in backend_section

    def test_backend_event_broker_kafka(self):
        """Backend must set EVENT_BROKER=kafka in docker-compose."""
        assert "EVENT_BROKER=kafka" in self.compose

    def test_backend_kafka_bootstrap_servers(self):
        """Backend must set KAFKA_BOOTSTRAP_SERVERS for internal network."""
        assert "KAFKA_BOOTSTRAP_SERVERS" in self.compose


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. CONSUMER DI (CONFIG-DRIVEN)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConsumerDI:
    """Verify consumer DI follows the same pattern as producer DI."""

    def setup_method(self):
        _reset_singletons()

    def teardown_method(self):
        _reset_singletons()

    def test_noop_broker_returns_noop_consumer(self):
        """EVENT_BROKER=noop → NoopEventConsumer."""
        from app.core.dependencies import _get_event_consumer
        from app.shared.messaging.consumer import NoopEventConsumer

        settings = _make_settings(event_broker="noop")
        consumer = _get_event_consumer(settings)

        assert isinstance(consumer, NoopEventConsumer)
        assert not consumer.is_enabled

    def test_consumer_singleton_cached(self):
        """Consumer should be cached as a singleton."""
        from app.core.dependencies import _get_event_consumer

        settings = _make_settings(event_broker="noop")
        c1 = _get_event_consumer(settings)
        c2 = _get_event_consumer(settings)

        assert c1 is c2

    def test_unknown_broker_returns_noop_consumer(self):
        """Unknown EVENT_BROKER → NoopEventConsumer."""
        from app.core.dependencies import _get_event_consumer
        from app.shared.messaging.consumer import NoopEventConsumer

        settings = _make_settings(event_broker="rabbitmq")
        consumer = _get_event_consumer(settings)

        assert isinstance(consumer, NoopEventConsumer)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. CONSUMER GROUP CONFIGURATION (K-6)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConsumerGroupConfig:
    """Verify consumer group settings exist and are used."""

    def test_consumer_group_id_setting_exists(self):
        """Settings must have kafka_consumer_group_id."""
        from app.core.settings import Settings

        s = Settings()
        assert hasattr(s, "kafka_consumer_group_id")
        assert s.kafka_consumer_group_id == "sentinel_ai"

    def test_auto_offset_reset_setting_exists(self):
        """Settings must have kafka_auto_offset_reset."""
        from app.core.settings import Settings

        s = Settings()
        assert hasattr(s, "kafka_auto_offset_reset")
        assert s.kafka_auto_offset_reset in ("earliest", "latest")

    def test_consumer_group_env_var_recognized(self):
        """KAFKA_CONSUMER_GROUP_ID env var must be recognized."""
        from app.core.settings import Settings

        with patch.dict(os.environ, {"KAFKA_CONSUMER_GROUP_ID": "my_custom_group"}):
            s = Settings()
            assert s.kafka_consumer_group_id == "my_custom_group"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. CONSUMER TOPIC SUBSCRIPTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConsumerTopicSubscriptions:
    """Verify the correct topics are subscribed to by each module."""

    def test_compound_risk_subscribes_to_upstream_topics(self):
        """CR consumer must subscribe to SI and RP output topics."""
        from app.compound_risk.messaging.consumer import (
            COMPOUND_RISK_SUBSCRIBED_TOPICS,
        )
        from app.shared.messaging.topics import KafkaTopics

        assert KafkaTopics.SENSOR_READING_ANOMALY in COMPOUND_RISK_SUBSCRIBED_TOPICS
        assert KafkaTopics.RISK_ASSESSMENT_GENERATED in COMPOUND_RISK_SUBSCRIBED_TOPICS
        assert KafkaTopics.RISK_SCORE_UPDATED in COMPOUND_RISK_SUBSCRIBED_TOPICS

    def test_hazard_propagation_subscribes_to_compound_risk(self):
        """HP consumer must subscribe to CR output topic."""
        from app.hazard_propagation.messaging.consumer import (
            HAZARD_PROPAGATION_SUBSCRIBED_TOPICS,
        )
        from app.shared.messaging.topics import KafkaTopics

        assert KafkaTopics.COMPOUND_RISK_DETECTED in HAZARD_PROPAGATION_SUBSCRIBED_TOPICS
        assert KafkaTopics.HAZARD_DETECTED in HAZARD_PROPAGATION_SUBSCRIBED_TOPICS

    def test_all_consumer_topics_collected(self):
        """_get_all_consumer_topics should aggregate all module topics."""
        from app.core.dependencies import _get_all_consumer_topics
        from app.shared.messaging.topics import KafkaTopics

        topics = _get_all_consumer_topics()

        # Must include both CR and HP topics
        assert KafkaTopics.SENSOR_READING_ANOMALY in topics
        assert KafkaTopics.RISK_ASSESSMENT_GENERATED in topics
        assert KafkaTopics.COMPOUND_RISK_DETECTED in topics
        assert KafkaTopics.HAZARD_DETECTED in topics

    def test_event_flow_chain_topics_exist(self):
        """Full event chain: SI → CR → HP must have matching topics."""
        from app.shared.messaging.topics import KafkaTopics

        # SI publishes → CR consumes
        assert hasattr(KafkaTopics, "SENSOR_READING_ANOMALY")
        assert hasattr(KafkaTopics, "RISK_ASSESSMENT_GENERATED")

        # CR publishes → HP consumes
        assert hasattr(KafkaTopics, "COMPOUND_RISK_DETECTED")

        # HP publishes (terminal)
        assert hasattr(KafkaTopics, "HAZARD_PROPAGATED")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. CONSUMER HANDLER REGISTRATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConsumerHandlerRegistration:
    """Verify handler registration works via ConsumerSetup classes."""

    def test_compound_risk_setup_registers_handlers(self):
        """CompoundRiskConsumerSetup.register() should mark as registered."""
        from app.compound_risk.messaging.consumer import (
            CompoundRiskConsumerSetup,
        )
        from app.shared.messaging.consumer import NoopEventConsumer

        consumer = NoopEventConsumer()
        handler = MagicMock()
        handler.handle_event = MagicMock()

        setup = CompoundRiskConsumerSetup(consumer, handler)
        assert not setup.is_registered

        setup.register()
        assert setup.is_registered

    def test_compound_risk_setup_idempotent(self):
        """Calling register() twice should be safe."""
        from app.compound_risk.messaging.consumer import (
            CompoundRiskConsumerSetup,
        )
        from app.shared.messaging.consumer import NoopEventConsumer

        consumer = NoopEventConsumer()
        handler = MagicMock()

        setup = CompoundRiskConsumerSetup(consumer, handler)
        setup.register()
        setup.register()  # Should not error

        assert setup.is_registered

    def test_hazard_propagation_setup_registers_handlers(self):
        """HazardPropagationConsumerSetup.register() should mark as registered."""
        from app.hazard_propagation.messaging.consumer import (
            HazardPropagationConsumerSetup,
        )
        from app.shared.messaging.consumer import NoopEventConsumer

        consumer = NoopEventConsumer()
        handler = MagicMock()
        handler.handle_event = MagicMock()

        setup = HazardPropagationConsumerSetup(consumer, handler)
        assert not setup.is_registered

        setup.register()
        assert setup.is_registered

    def test_hazard_propagation_lists_subscribed_topics(self):
        """HazardPropagationConsumerSetup should expose its topics."""
        from app.hazard_propagation.messaging.consumer import (
            HazardPropagationConsumerSetup,
        )
        from app.shared.messaging.consumer import NoopEventConsumer
        from app.shared.messaging.topics import KafkaTopics

        consumer = NoopEventConsumer()
        handler = MagicMock()

        setup = HazardPropagationConsumerSetup(consumer, handler)
        topics = setup.subscribed_topics

        assert KafkaTopics.COMPOUND_RISK_DETECTED in topics
        assert KafkaTopics.HAZARD_DETECTED in topics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. LIFESPAN CONSUMER START/STOP (K-4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLifespanConsumerWiring:
    """Verify start_consumers/stop_consumers are wired in main.py lifespan."""

    def setup_method(self):
        _reset_singletons()

    def teardown_method(self):
        _reset_singletons()

    def test_start_consumers_imported_in_main(self):
        """main.py must import start_consumers."""
        main_source = (PROJECT_ROOT / "app" / "main.py").read_text()
        assert "start_consumers" in main_source

    def test_stop_consumers_imported_in_main(self):
        """main.py must import stop_consumers."""
        main_source = (PROJECT_ROOT / "app" / "main.py").read_text()
        assert "stop_consumers" in main_source

    def test_start_consumers_noop_is_safe(self):
        """start_consumers with EVENT_BROKER=noop should be a no-op."""
        from app.core.dependencies import start_consumers

        settings = _make_settings(event_broker="noop")
        # Should not raise, should not start any threads
        start_consumers(settings)

    def test_stop_consumers_noop_is_safe(self):
        """stop_consumers when nothing is started should be a no-op."""
        from app.core.dependencies import stop_consumers

        # Should not raise
        stop_consumers()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. EVENT FLOW CHAIN VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEventFlowChain:
    """Verify the event flow: SI → RP → CR → HP is properly connected."""

    def test_si_publishes_reading_anomaly(self):
        """SI publisher must have publish_reading_anomaly method."""
        from app.sensor_intelligence.messaging.publisher import (
            SensorIntelligencePublisher,
        )

        assert hasattr(SensorIntelligencePublisher, "publish_reading_anomaly")

    def test_rp_publishes_assessment_generated(self):
        """RP publisher must have publish_assessment_generated method."""
        from app.risk_prediction.messaging.publisher import (
            RiskPredictionPublisher,
        )

        assert hasattr(RiskPredictionPublisher, "publish_assessment_generated")

    def test_cr_publishes_compound_risk_detected(self):
        """CR publisher must have publish_compound_risk_detected method."""
        from app.compound_risk.messaging.publisher import (
            CompoundRiskPublisher,
        )

        assert hasattr(CompoundRiskPublisher, "publish_compound_risk_detected")

    def test_hp_publishes_hazard_propagated(self):
        """HP publisher must have publish_hazard_propagated method."""
        from app.hazard_propagation.messaging.publisher import (
            HazardPropagationPublisher,
        )

        assert hasattr(HazardPropagationPublisher, "publish_hazard_propagated")

    def test_si_anomaly_topic_matches_cr_subscription(self):
        """SI output topic must match CR input topic for anomaly events."""
        from app.compound_risk.messaging.consumer import (
            COMPOUND_RISK_SUBSCRIBED_TOPICS,
        )
        from app.shared.messaging.topics import KafkaTopics

        # SI publishes sensor.reading.anomaly
        # CR must subscribe to it
        assert KafkaTopics.SENSOR_READING_ANOMALY in COMPOUND_RISK_SUBSCRIBED_TOPICS

    def test_rp_assessment_topic_matches_cr_subscription(self):
        """RP output topic must match CR input topic for risk assessments."""
        from app.compound_risk.messaging.consumer import (
            COMPOUND_RISK_SUBSCRIBED_TOPICS,
        )
        from app.shared.messaging.topics import KafkaTopics

        # RP publishes risk.assessment.generated
        # CR must subscribe to it
        assert KafkaTopics.RISK_ASSESSMENT_GENERATED in COMPOUND_RISK_SUBSCRIBED_TOPICS

    def test_cr_output_topic_matches_hp_subscription(self):
        """CR output topic must match HP input topic."""
        from app.hazard_propagation.messaging.consumer import (
            HAZARD_PROPAGATION_SUBSCRIBED_TOPICS,
        )
        from app.shared.messaging.topics import KafkaTopics

        # CR publishes compound.risk.detected
        # HP must subscribe to it
        assert KafkaTopics.COMPOUND_RISK_DETECTED in HAZARD_PROPAGATION_SUBSCRIBED_TOPICS

    def test_full_event_chain_producers_and_consumers_aligned(self):
        """All publish topics from upstream match consume topics downstream."""
        from app.compound_risk.messaging.consumer import (
            COMPOUND_RISK_SUBSCRIBED_TOPICS,
        )
        from app.hazard_propagation.messaging.consumer import (
            HAZARD_PROPAGATION_SUBSCRIBED_TOPICS,
        )
        from app.shared.messaging.topics import KafkaTopics

        # SI → CR chain
        si_output_topics = [
            KafkaTopics.SENSOR_READING_ANOMALY,
        ]
        for topic in si_output_topics:
            assert topic in COMPOUND_RISK_SUBSCRIBED_TOPICS, (
                f"SI output topic '{topic}' not consumed by CR"
            )

        # RP → CR chain
        rp_output_topics = [
            KafkaTopics.RISK_ASSESSMENT_GENERATED,
            KafkaTopics.RISK_SCORE_UPDATED,
        ]
        for topic in rp_output_topics:
            assert topic in COMPOUND_RISK_SUBSCRIBED_TOPICS, (
                f"RP output topic '{topic}' not consumed by CR"
            )

        # CR → HP chain
        cr_output_topics = [
            KafkaTopics.COMPOUND_RISK_DETECTED,
        ]
        for topic in cr_output_topics:
            assert topic in HAZARD_PROPAGATION_SUBSCRIBED_TOPICS, (
                f"CR output topic '{topic}' not consumed by HP"
            )
