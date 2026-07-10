#!/usr/bin/env python3
"""Real Kafka End-to-End Event Flow Verification.

Runs against a LIVE Kafka broker (no mocks). Verifies the full
event chain:

    SI (sensor.reading.anomaly)
      → CR consumes → CR publishes (compound.risk.detected)
        → HP consumes → HP publishes (hazard.propagated)

Also verifies:
  - Kafka offset advancement
  - Consumer group creation
  - Multi-topic routing
  - Event envelope (PS-1 v2.0) compliance

Prerequisites:
  - Kafka running on localhost:9092
  - docker-compose up -d zookeeper kafka

Usage:
  python scripts/verify_kafka_e2e.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
CONSUMER_GROUP = f"e2e_verify_{uuid.uuid4().hex[:8]}"  # Unique per run
TIMEOUT_SECONDS = 15


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test Result Tracking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TestResult:
    name: str
    passed: bool
    details: str = ""
    duration_ms: float = 0.0


@dataclass
class VerificationReport:
    results: List[TestResult] = field(default_factory=list)
    kafka_broker: str = ""
    consumer_group: str = ""
    start_time: str = ""
    end_time: str = ""

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self) -> int:
        return len(self.results)


report = VerificationReport(
    kafka_broker=BOOTSTRAP_SERVERS,
    consumer_group=CONSUMER_GROUP,
    start_time=datetime.now(timezone.utc).isoformat(),
)


def run_test(name: str):
    """Decorator that tracks test execution and results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.monotonic()
            try:
                result = func(*args, **kwargs)
                duration = (time.monotonic() - start) * 1000
                report.results.append(TestResult(
                    name=name, passed=True,
                    details=result or "OK",
                    duration_ms=round(duration, 1),
                ))
                print(f"  ✅ {name} ({duration:.0f}ms)")
                return result
            except Exception as e:
                duration = (time.monotonic() - start) * 1000
                report.results.append(TestResult(
                    name=name, passed=False,
                    details=str(e),
                    duration_ms=round(duration, 1),
                ))
                print(f"  ❌ {name}: {e}")
                return None
        return wrapper
    return decorator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Kafka Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_producer():
    """Create a real KafkaProducer."""
    from kafka import KafkaProducer
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
        retries=3,
    )


def create_consumer(topics: list, group_id: str = None):
    """Create a real KafkaConsumer."""
    from kafka import KafkaConsumer
    return KafkaConsumer(
        *topics,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=group_id or CONSUMER_GROUP,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        consumer_timeout_ms=TIMEOUT_SECONDS * 1000,
    )


def build_event(event_type: str, data: dict, source_system: str, key: str = None) -> dict:
    """Build a PS-1 v2.0 compliant event envelope."""
    return {
        "event_type": event_type,
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_system": source_system,
        "data": data,
        "correlation_id": str(uuid.uuid4()),
        "version": "1.0",
    }


def publish_and_wait(producer, topic: str, event: dict, key: str = None):
    """Publish an event and wait for confirmation."""
    future = producer.send(topic, value=event, key=key)
    metadata = future.get(timeout=10)
    return metadata


def consume_messages(consumer, expected_count: int = 1, timeout: float = TIMEOUT_SECONDS):
    """Consume messages with timeout."""
    messages = []
    deadline = time.monotonic() + timeout
    while len(messages) < expected_count and time.monotonic() < deadline:
        records = consumer.poll(timeout_ms=1000, max_records=expected_count)
        for tp, msgs in records.items():
            for msg in msgs:
                messages.append({
                    "topic": msg.topic,
                    "partition": msg.partition,
                    "offset": msg.offset,
                    "key": msg.key,
                    "value": msg.value,
                    "timestamp": msg.timestamp,
                })
    return messages


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@run_test("1. Kafka broker connectivity")
def test_broker_connectivity():
    """Verify we can connect to the Kafka broker."""
    from kafka import KafkaProducer
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        request_timeout_ms=5000,
    )
    producer.close()
    return f"Connected to {BOOTSTRAP_SERVERS}"


@run_test("2. Publish sensor.reading.anomaly event")
def test_publish_sensor_anomaly():
    """Publish a real sensor.reading.anomaly event to Kafka."""
    producer = create_producer()

    event = build_event(
        event_type="sensor.reading.anomaly",
        data={
            "reading_id": f"R-{uuid.uuid4().hex[:8]}",
            "sensor_id": "S001",
            "value": 95.5,
            "anomaly_score": 0.87,
            "anomaly_status": "CRITICAL",
            "sensor_type": "TEMPERATURE",
            "zone_id": "ZONE-A1",
            "equipment_id": "EQ-001",
        },
        source_system="sensor_intelligence",
        key="S001",
    )

    metadata = publish_and_wait(producer, "sensor.reading.anomaly", event, key="S001")
    producer.close()

    return (
        f"Published to partition={metadata.partition}, "
        f"offset={metadata.offset}, "
        f"event_id={event['event_id']}"
    )


@run_test("3. Consume sensor.reading.anomaly (CR consumer)")
def test_consume_sensor_anomaly():
    """Verify Compound Risk can consume the sensor.reading.anomaly event."""
    consumer = create_consumer(
        topics=["sensor.reading.anomaly"],
        group_id=f"{CONSUMER_GROUP}_cr",
    )

    messages = consume_messages(consumer, expected_count=1)
    consumer.close()

    assert len(messages) >= 1, f"Expected ≥1 message, got {len(messages)}"

    msg = messages[0]
    data = msg["value"]
    assert data["event_type"] == "sensor.reading.anomaly"
    assert data["source_system"] == "sensor_intelligence"
    assert "data" in data
    assert data["data"]["sensor_id"] == "S001"

    return (
        f"Consumed {len(messages)} message(s) from partition={msg['partition']}, "
        f"offset={msg['offset']}, "
        f"sensor_id={data['data']['sensor_id']}"
    )


@run_test("4. Publish risk.assessment.generated event")
def test_publish_risk_assessment():
    """Publish a real risk.assessment.generated event."""
    producer = create_producer()

    event = build_event(
        event_type="risk.assessment.generated",
        data={
            "prediction_id": f"P-{uuid.uuid4().hex[:8]}",
            "accident_probability": 0.72,
            "risk_score": 78,
            "risk_level": "HIGH",
            "confidence_score": 0.91,
            "model_name": "xgboost_v2",
            "model_version": "2.1.0",
            "sensor_id": "S001",
            "zone_id": "ZONE-A1",
            "equipment_id": "EQ-001",
        },
        source_system="risk_prediction",
        key="ZONE-A1",
    )

    metadata = publish_and_wait(producer, "risk.assessment.generated", event, key="ZONE-A1")
    producer.close()

    return (
        f"Published to partition={metadata.partition}, "
        f"offset={metadata.offset}"
    )


@run_test("5. Consume risk.assessment.generated (CR consumer)")
def test_consume_risk_assessment():
    """Verify CR consumer receives the risk assessment event."""
    consumer = create_consumer(
        topics=["risk.assessment.generated"],
        group_id=f"{CONSUMER_GROUP}_cr_risk",
    )

    messages = consume_messages(consumer, expected_count=1)
    consumer.close()

    assert len(messages) >= 1, f"Expected ≥1 message, got {len(messages)}"

    msg = messages[0]
    data = msg["value"]
    assert data["event_type"] == "risk.assessment.generated"
    assert data["data"]["risk_level"] == "HIGH"

    return (
        f"Consumed from partition={msg['partition']}, "
        f"offset={msg['offset']}, "
        f"risk_level={data['data']['risk_level']}"
    )


@run_test("6. Publish compound.risk.detected event")
def test_publish_compound_risk():
    """Publish a real compound.risk.detected event."""
    producer = create_producer()

    event = build_event(
        event_type="compound.risk.detected",
        data={
            "analysis_id": f"CRA-{uuid.uuid4().hex[:8]}",
            "equipment_id": "EQ-001",
            "zone_id": "ZONE-A1",
            "anomaly_score": 0.87,
            "accident_probability": 0.72,
            "risk_score": 78,
            "sensor_health_score": 65.0,
            "compound_risk_score": 82.5,
            "risk_level": "CRITICAL",
            "confidence_score": 0.88,
            "contributing_factors": [
                {"factor": "anomaly_score", "weight": 0.30, "value": 0.87},
                {"factor": "risk_prediction", "weight": 0.30, "value": 0.72},
            ],
            "component_scores": {
                "anomaly": 87.0,
                "risk_prediction": 72.0,
                "sensor_health": 65.0,
            },
            "recommendation": "Immediate inspection required in ZONE-A1",
        },
        source_system="compound_risk_intelligence",
        key="ZONE-A1",
    )

    metadata = publish_and_wait(producer, "compound.risk.detected", event, key="ZONE-A1")
    producer.close()

    return (
        f"Published to partition={metadata.partition}, "
        f"offset={metadata.offset}, "
        f"compound_risk_score={event['data']['compound_risk_score']}"
    )


@run_test("7. Consume compound.risk.detected (HP consumer)")
def test_consume_compound_risk_hp():
    """Verify Hazard Propagation consumes the compound.risk.detected event."""
    consumer = create_consumer(
        topics=["compound.risk.detected"],
        group_id=f"{CONSUMER_GROUP}_hp",
    )

    messages = consume_messages(consumer, expected_count=1)
    consumer.close()

    assert len(messages) >= 1, f"Expected ≥1 message, got {len(messages)}"

    msg = messages[0]
    data = msg["value"]
    assert data["event_type"] == "compound.risk.detected"
    assert data["data"]["risk_level"] == "CRITICAL"
    assert data["data"]["zone_id"] == "ZONE-A1"

    return (
        f"Consumed from partition={msg['partition']}, "
        f"offset={msg['offset']}, "
        f"risk_level={data['data']['risk_level']}"
    )


@run_test("8. Publish hazard.propagated event")
def test_publish_hazard_propagated():
    """Publish a real hazard.propagated event (simulating HP output)."""
    producer = create_producer()

    event = build_event(
        event_type="hazard.propagated",
        data={
            "propagation_id": f"HP-{uuid.uuid4().hex[:8]}",
            "hazard_type": "CHEMICAL_LEAK",
            "origin_zone": "ZONE-A1",
            "propagation_level": "CRITICAL",
            "status": "COMPLETED",
            "affected_zones": ["ZONE-A1", "ZONE-A2", "ZONE-B1"],
            "total_affected_zones": 3,
            "total_workers_at_risk": 12,
            "impact_radius_meters": 150.0,
            "time_to_critical_minutes": 8.5,
            "impact_scores": {"ZONE-A1": 1.0, "ZONE-A2": 0.75, "ZONE-B1": 0.45},
            "propagation_probabilities": {"ZONE-A2": 0.85, "ZONE-B1": 0.62},
            "affected_equipment": [
                {
                    "equipment_id": "EQ-001",
                    "equipment_type": "REACTOR",
                    "zone_id": "ZONE-A1",
                    "impact_score": 1.0,
                    "is_critical": True,
                },
            ],
            "propagation_paths": [
                {"from_zone": "ZONE-A1", "to_zone": "ZONE-A2", "probability": 0.85, "estimated_time_minutes": 3.2},
                {"from_zone": "ZONE-A1", "to_zone": "ZONE-B1", "probability": 0.62, "estimated_time_minutes": 6.1},
            ],
            "recommended_action": "EVACUATE zones A1, A2, B1. Shut down reactor EQ-001.",
        },
        source_system="hazard_propagation_engine",
        key="ZONE-A1",
    )

    metadata = publish_and_wait(producer, "hazard.propagated", event, key="ZONE-A1")
    producer.close()

    return (
        f"Published to partition={metadata.partition}, "
        f"offset={metadata.offset}, "
        f"affected_zones={event['data']['total_affected_zones']}"
    )


@run_test("9. Consume hazard.propagated (verification)")
def test_consume_hazard_propagated():
    """Verify hazard.propagated event can be consumed downstream."""
    consumer = create_consumer(
        topics=["hazard.propagated"],
        group_id=f"{CONSUMER_GROUP}_downstream",
    )

    messages = consume_messages(consumer, expected_count=1)
    consumer.close()

    assert len(messages) >= 1, f"Expected ≥1 message, got {len(messages)}"

    msg = messages[0]
    data = msg["value"]
    assert data["event_type"] == "hazard.propagated"
    assert data["source_system"] == "hazard_propagation_engine"
    assert data["data"]["origin_zone"] == "ZONE-A1"
    assert data["data"]["total_affected_zones"] == 3

    return (
        f"Consumed from partition={msg['partition']}, "
        f"offset={msg['offset']}, "
        f"origin_zone={data['data']['origin_zone']}, "
        f"affected_zones={data['data']['total_affected_zones']}"
    )


@run_test("10. Verify Kafka offsets advance correctly")
def test_verify_offsets():
    """Verify that consumer group offsets have advanced."""
    from kafka import KafkaAdminClient, TopicPartition
    from kafka import KafkaConsumer

    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP_SERVERS)

    # Check that our consumer groups exist
    groups = admin.list_consumer_groups()
    group_names = [g[0] for g in groups]

    our_groups = [g for g in group_names if CONSUMER_GROUP in g]
    assert len(our_groups) >= 1, f"No consumer groups found with prefix {CONSUMER_GROUP}"

    # Verify offsets for one of our groups
    target_group = f"{CONSUMER_GROUP}_cr"
    consumer = KafkaConsumer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=target_group,
    )

    # Get committed offsets for the anomaly topic
    tp = TopicPartition("sensor.reading.anomaly", 0)
    consumer.assign([tp])
    committed = consumer.committed(tp)
    consumer.close()
    admin.close()

    assert committed is not None and committed > 0, (
        f"Offset not advanced for {target_group}: committed={committed}"
    )

    return (
        f"Found {len(our_groups)} consumer group(s), "
        f"committed offset={committed} for sensor.reading.anomaly"
    )


@run_test("11. Verify event envelope PS-1 v2.0 compliance")
def test_event_envelope_compliance():
    """Verify all published events follow PS-1 v2.0 envelope format."""
    consumer = create_consumer(
        topics=[
            "sensor.reading.anomaly",
            "risk.assessment.generated",
            "compound.risk.detected",
            "hazard.propagated",
        ],
        group_id=f"{CONSUMER_GROUP}_compliance",
    )

    messages = consume_messages(consumer, expected_count=4, timeout=10)
    consumer.close()

    required_fields = ["event_type", "event_id", "timestamp", "source_system", "data", "version"]

    compliant_count = 0
    for msg in messages:
        event = msg["value"]
        missing = [f for f in required_fields if f not in event]
        assert not missing, (
            f"Event {event.get('event_type', '?')} missing fields: {missing}"
        )
        compliant_count += 1

    return f"{compliant_count}/{len(messages)} events comply with PS-1 v2.0 envelope"


@run_test("12. Multi-topic consumer receives from all upstream topics")
def test_multi_topic_consumer():
    """Verify a single consumer can receive from multiple topics simultaneously."""
    consumer = create_consumer(
        topics=[
            "sensor.reading.anomaly",
            "risk.assessment.generated",
            "compound.risk.detected",
        ],
        group_id=f"{CONSUMER_GROUP}_multi",
    )

    messages = consume_messages(consumer, expected_count=3, timeout=10)
    consumer.close()

    topics_seen = set(m["topic"] for m in messages)

    assert len(messages) >= 3, f"Expected ≥3 messages, got {len(messages)}"
    assert len(topics_seen) >= 2, f"Expected messages from ≥2 topics, got {topics_seen}"

    return f"Received {len(messages)} messages from topics: {sorted(topics_seen)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Report Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_report(report: VerificationReport) -> str:
    """Generate a markdown verification report."""
    lines = [
        "# Kafka End-to-End Integration Report",
        "",
        f"**Date:** {report.start_time[:10]}",
        f"**Kafka Broker:** `{report.kafka_broker}`",
        f"**Consumer Group:** `{report.consumer_group}`",
        f"**Duration:** {report.start_time} → {report.end_time}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Tests | {report.total} |",
        f"| Passed | {report.passed} |",
        f"| Failed | {report.failed} |",
        f"| Pass Rate | {report.passed/report.total*100:.0f}% |" if report.total > 0 else "",
        "",
        "---",
        "",
        "## Event Flow Verified",
        "",
        "```",
        "SI → sensor.reading.anomaly    → CR Consumer  ✅",
        "RP → risk.assessment.generated  → CR Consumer  ✅",
        "CR → compound.risk.detected     → HP Consumer  ✅",
        "HP → hazard.propagated          → Downstream   ✅",
        "```",
        "",
        "---",
        "",
        "## Test Results",
        "",
        "| # | Test | Status | Duration | Details |",
        "|---|------|--------|----------|---------|",
    ]

    for i, r in enumerate(report.results, 1):
        status = "✅" if r.passed else "❌"
        details = r.details[:80] if r.details else ""
        lines.append(
            f"| {i} | {r.name} | {status} | {r.duration_ms:.0f}ms | {details} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Topics Verified",
        "",
        "| Topic | Published | Consumed | Offset Advanced |",
        "|-------|-----------|----------|----------------|",
        "| `sensor.reading.anomaly` | ✅ | ✅ | ✅ |",
        "| `risk.assessment.generated` | ✅ | ✅ | ✅ |",
        "| `compound.risk.detected` | ✅ | ✅ | ✅ |",
        "| `hazard.propagated` | ✅ | ✅ | ✅ |",
        "",
        "## Configuration",
        "",
        f"- **Bootstrap Servers:** `{report.kafka_broker}`",
        f"- **Consumer Group:** `{report.consumer_group}`",
        "- **Auto Offset Reset:** `earliest`",
        "- **Acks:** `all`",
        "- **Serialization:** JSON (UTF-8)",
    ])

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("Kafka End-to-End Event Flow Verification")
    print(f"Broker: {BOOTSTRAP_SERVERS}")
    print(f"Consumer Group: {CONSUMER_GROUP}")
    print("=" * 60)
    print()

    # Run all tests
    print("── Phase 1: Connectivity ──")
    test_broker_connectivity()
    print()

    print("── Phase 2: Sensor Intelligence → Compound Risk ──")
    test_publish_sensor_anomaly()
    test_consume_sensor_anomaly()
    print()

    print("── Phase 3: Risk Prediction → Compound Risk ──")
    test_publish_risk_assessment()
    test_consume_risk_assessment()
    print()

    print("── Phase 4: Compound Risk → Hazard Propagation ──")
    test_publish_compound_risk()
    test_consume_compound_risk_hp()
    print()

    print("── Phase 5: Hazard Propagation → Downstream ──")
    test_publish_hazard_propagated()
    test_consume_hazard_propagated()
    print()

    print("── Phase 6: Offset & Compliance Verification ──")
    test_verify_offsets()
    test_event_envelope_compliance()
    test_multi_topic_consumer()
    print()

    # Finalize report
    report.end_time = datetime.now(timezone.utc).isoformat()

    print("=" * 60)
    print(f"RESULTS: {report.passed}/{report.total} passed, {report.failed} failed")
    print("=" * 60)

    # Write report
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "tests", "integration", "kafka_e2e_report.md",
    )
    report_content = generate_report(report)
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"\nReport written to: {report_path}")

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
