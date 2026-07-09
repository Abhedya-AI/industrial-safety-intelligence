"""Integration tests for Risk Prediction Kafka integration.

Verifies:
  1. RiskPredictionPublisher publishes risk.assessment.generated correctly
  2. RiskPredictionPublisher publishes risk.score.updated correctly
  3. publish_prediction_events() fires both events from a model
  4. Optional fields (sensor_id, zone_id, equipment_id) handling
  5. Event payload format compliance (PS-1 v2.0)
  6. Publisher failure does not crash business operations
  7. Metrics tracking (published_count, failed_count)
  8. Topic constants match team-standard names
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from app.risk_prediction.messaging.publisher import RiskPredictionPublisher
from app.shared.messaging.producer import NoopEventProducer
from app.shared.messaging.topics import KafkaTopics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def publish_log() -> List[Dict[str, Any]]:
    return []


@pytest.fixture
def tracking_producer(publish_log) -> NoopEventProducer:
    """Producer that tracks all published events."""
    prod = NoopEventProducer()
    original = prod.publish

    def tracking(topic, data, **kw):
        event = original(topic, data, **kw)
        publish_log.append({"topic": topic, "data": data, "event": event})
        return event

    prod.publish = tracking
    return prod


@pytest.fixture
def publisher(tracking_producer) -> RiskPredictionPublisher:
    return RiskPredictionPublisher(tracking_producer)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fake prediction model for publish_prediction_events()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class FakePrediction:
    """Mimics RiskPredictionModel for testing convenience."""

    def __init__(self, **kwargs):
        defaults = {
            "id": "PRED-001",
            "accident_probability": 0.72,
            "predicted_risk_score": 72,
            "risk_level": "HIGH",
            "confidence_score": 0.85,
            "model_name": "xgboost_risk_prediction",
            "model_version": "1.0.0",
            "sensor_id": "S001",
            "equipment_id": "EQ001",
            "zone_id": "ZONE_A",
            "prediction_timestamp": datetime.now(timezone.utc),
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. risk.assessment.generated
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPublishAssessmentGenerated:
    def test_publishes_to_correct_topic(self, publisher, publish_log):
        publisher.publish_assessment_generated(
            prediction_id="P001",
            accident_probability=0.65,
            risk_score=65,
            risk_level="HIGH",
            confidence_score=0.8,
            model_name="xgboost",
            model_version="1.0.0",
        )
        assert len(publish_log) == 1
        assert publish_log[0]["topic"] == KafkaTopics.RISK_ASSESSMENT_GENERATED

    def test_event_data_fields(self, publisher, publish_log):
        publisher.publish_assessment_generated(
            prediction_id="P002",
            accident_probability=0.42,
            risk_score=42,
            risk_level="MEDIUM",
            confidence_score=0.9,
            model_name="xgboost_risk_prediction",
            model_version="1.0.0",
            sensor_id="S002",
            zone_id="ZONE_B",
        )
        data = publish_log[0]["data"]
        assert data["prediction_id"] == "P002"
        assert data["accident_probability"] == 0.42
        assert data["risk_score"] == 42
        assert data["risk_level"] == "MEDIUM"
        assert data["confidence_score"] == 0.9
        assert data["model_name"] == "xgboost_risk_prediction"
        assert data["model_version"] == "1.0.0"
        assert data["sensor_id"] == "S002"
        assert data["zone_id"] == "ZONE_B"

    def test_optional_fields_omitted_when_none(self, publisher, publish_log):
        publisher.publish_assessment_generated(
            prediction_id="P003",
            accident_probability=0.1,
            risk_score=10,
            risk_level="LOW",
            confidence_score=0.95,
            model_name="xgboost",
            model_version="1.0.0",
        )
        data = publish_log[0]["data"]
        assert "sensor_id" not in data
        assert "equipment_id" not in data
        assert "zone_id" not in data

    def test_includes_prediction_timestamp(self, publisher, publish_log):
        publisher.publish_assessment_generated(
            prediction_id="P004",
            accident_probability=0.5,
            risk_score=50,
            risk_level="MEDIUM",
            confidence_score=0.7,
            model_name="xgboost",
            model_version="1.0.0",
            prediction_timestamp="2026-07-08T12:00:00+00:00",
        )
        data = publish_log[0]["data"]
        assert data["prediction_timestamp"] == "2026-07-08T12:00:00+00:00"

    def test_returns_base_event(self, publisher):
        result = publisher.publish_assessment_generated(
            prediction_id="P005",
            accident_probability=0.3,
            risk_score=30,
            risk_level="MEDIUM",
            confidence_score=0.85,
            model_name="xgboost",
            model_version="1.0.0",
        )
        assert result is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. risk.score.updated
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPublishScoreUpdated:
    def test_publishes_to_correct_topic(self, publisher, publish_log):
        publisher.publish_score_updated(
            prediction_id="P001",
            risk_score=72,
            risk_level="HIGH",
            accident_probability=0.72,
            confidence_score=0.85,
        )
        assert publish_log[0]["topic"] == KafkaTopics.RISK_SCORE_UPDATED

    def test_event_data_fields(self, publisher, publish_log):
        publisher.publish_score_updated(
            prediction_id="P002",
            risk_score=88,
            risk_level="CRITICAL",
            accident_probability=0.88,
            confidence_score=0.92,
            sensor_id="S005",
            zone_id="ZONE_C",
            equipment_id="EQ005",
        )
        data = publish_log[0]["data"]
        assert data["prediction_id"] == "P002"
        assert data["risk_score"] == 88
        assert data["risk_level"] == "CRITICAL"
        assert data["accident_probability"] == 0.88
        assert data["confidence_score"] == 0.92
        assert data["sensor_id"] == "S005"
        assert data["zone_id"] == "ZONE_C"
        assert data["equipment_id"] == "EQ005"

    def test_optional_fields_omitted_when_none(self, publisher, publish_log):
        publisher.publish_score_updated(
            prediction_id="P003",
            risk_score=25,
            risk_level="LOW",
            accident_probability=0.25,
            confidence_score=0.95,
        )
        data = publish_log[0]["data"]
        assert "sensor_id" not in data
        assert "equipment_id" not in data
        assert "zone_id" not in data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. publish_prediction_events() — convenience
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPublishPredictionEvents:
    def test_fires_both_events(self, publisher, publish_log):
        pred = FakePrediction()
        publisher.publish_prediction_events(pred)
        assert len(publish_log) == 2
        topics = [e["topic"] for e in publish_log]
        assert KafkaTopics.RISK_ASSESSMENT_GENERATED in topics
        assert KafkaTopics.RISK_SCORE_UPDATED in topics

    def test_event_data_from_model(self, publisher, publish_log):
        pred = FakePrediction(
            id="P100",
            accident_probability=0.55,
            predicted_risk_score=55,
            risk_level="HIGH",
            confidence_score=0.78,
            sensor_id="S100",
            zone_id="ZONE_X",
        )
        publisher.publish_prediction_events(pred)

        # risk.assessment.generated data
        assessment_data = publish_log[0]["data"]
        assert assessment_data["prediction_id"] == "P100"
        assert assessment_data["accident_probability"] == 0.55
        assert assessment_data["risk_score"] == 55
        assert assessment_data["sensor_id"] == "S100"

        # risk.score.updated data
        score_data = publish_log[1]["data"]
        assert score_data["prediction_id"] == "P100"
        assert score_data["risk_score"] == 55
        assert score_data["zone_id"] == "ZONE_X"

    def test_handles_none_optional_fields(self, publisher, publish_log):
        pred = FakePrediction(
            sensor_id=None, equipment_id=None, zone_id=None,
        )
        publisher.publish_prediction_events(pred)
        assert len(publish_log) == 2
        for entry in publish_log:
            assert "sensor_id" not in entry["data"]
            assert "equipment_id" not in entry["data"]
            assert "zone_id" not in entry["data"]

    def test_count_incremented_by_two(self, publisher):
        pred = FakePrediction()
        publisher.publish_prediction_events(pred)
        assert publisher.published_count == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Failure handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPublishFailure:
    def test_handles_producer_failure(self):
        failing_producer = NoopEventProducer()
        failing_producer.publish = MagicMock(
            side_effect=Exception("Kafka unreachable"),
        )
        pub = RiskPredictionPublisher(failing_producer)
        result = pub.publish_assessment_generated(
            prediction_id="P001",
            accident_probability=0.5,
            risk_score=50,
            risk_level="MEDIUM",
            confidence_score=0.8,
            model_name="xgboost",
            model_version="1.0.0",
        )
        assert result is None
        assert pub.failed_count == 1
        assert pub.published_count == 0

    def test_prediction_events_partial_failure(self):
        """If one event fails, the other should still be attempted."""
        producer = NoopEventProducer()
        call_count = [0]
        original = producer.publish

        def fail_first(*args, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("First publish fails")
            return original(*args, **kw)

        producer.publish = fail_first
        pub = RiskPredictionPublisher(producer)
        pred = FakePrediction()
        pub.publish_prediction_events(pred)
        assert pub.failed_count == 1
        assert pub.published_count == 1

    def test_multiple_failures_tracked(self):
        failing_producer = NoopEventProducer()
        failing_producer.publish = MagicMock(
            side_effect=Exception("Kafka unreachable"),
        )
        pub = RiskPredictionPublisher(failing_producer)
        for _ in range(4):
            pub.publish_score_updated(
                prediction_id="P001",
                risk_score=50,
                risk_level="MEDIUM",
                accident_probability=0.5,
                confidence_score=0.8,
            )
        assert pub.failed_count == 4


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Event format compliance (PS-1 v2.0)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEventFormat:
    def test_assessment_event_has_standard_fields(self, publisher, publish_log):
        publisher.publish_assessment_generated(
            prediction_id="P001",
            accident_probability=0.5,
            risk_score=50,
            risk_level="MEDIUM",
            confidence_score=0.8,
            model_name="xgboost",
            model_version="1.0.0",
        )
        event = publish_log[0]["event"]
        assert event.event_type == KafkaTopics.RISK_ASSESSMENT_GENERATED
        assert event.event_id  # Non-empty UUID
        assert event.timestamp  # ISO 8601
        assert event.source_system == "risk_prediction"
        assert event.data is not None

    def test_score_event_has_standard_fields(self, publisher, publish_log):
        publisher.publish_score_updated(
            prediction_id="P001",
            risk_score=72,
            risk_level="HIGH",
            accident_probability=0.72,
            confidence_score=0.85,
        )
        event = publish_log[0]["event"]
        assert event.event_type == KafkaTopics.RISK_SCORE_UPDATED
        assert event.source_system == "risk_prediction"

    def test_timestamp_is_iso_8601(self, publisher, publish_log):
        publisher.publish_assessment_generated(
            prediction_id="P001",
            accident_probability=0.5,
            risk_score=50,
            risk_level="MEDIUM",
            confidence_score=0.8,
            model_name="xgboost",
            model_version="1.0.0",
        )
        event = publish_log[0]["event"]
        from datetime import datetime
        # Should not raise
        datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMetrics:
    def test_tracks_success_count(self, publisher):
        for i in range(3):
            publisher.publish_score_updated(
                prediction_id=f"P{i}",
                risk_score=50 + i,
                risk_level="MEDIUM",
                accident_probability=0.5,
                confidence_score=0.8,
            )
        assert publisher.published_count == 3
        assert publisher.failed_count == 0

    def test_initial_counts_zero(self, publisher):
        assert publisher.published_count == 0
        assert publisher.failed_count == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Topic constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTopicConstants:
    def test_assessment_topic_matches_standard(self):
        assert KafkaTopics.RISK_ASSESSMENT_GENERATED == "risk.assessment.generated"

    def test_score_topic_matches_standard(self):
        assert KafkaTopics.RISK_SCORE_UPDATED == "risk.score.updated"

    def test_both_in_risk_topics(self):
        topics = KafkaTopics.risk_topics()
        assert KafkaTopics.RISK_ASSESSMENT_GENERATED in topics
        assert KafkaTopics.RISK_SCORE_UPDATED in topics
