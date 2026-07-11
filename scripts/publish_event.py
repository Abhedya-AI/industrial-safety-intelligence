"""Helper script to publish test Kafka events to the Digital Twin.

Enables you to inject custom events directly from your host machine
(macOS) to see the Digital Twin update in real-time.
"""

import json
import uuid
from datetime import datetime, timezone
import argparse
from kafka import KafkaProducer


def publish_event(topic: str, payload: dict):
    # Kafka is mapped to localhost:9092 on your MacBook
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    event_id = str(uuid.uuid4())
    event = {
        "event_id": event_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": payload,
    }

    print(f"Publishing to '{topic}':")
    print(json.dumps(event, indent=2))

    producer.send(topic, event)
    producer.flush()
    producer.close()
    print("✓ Event successfully published to Kafka!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish test Kafka events.")
    parser.add_argument(
        "--type",
        choices=["sensor", "risk", "hazard"],
        default="sensor",
        help="Type of event to inject (default: sensor)",
    )
    args = parser.parse_args()

    if args.type == "sensor":
        # Inject an anomaly reading for S001 in ZONE_A
        publish_event(
            topic="sensor.reading.anomaly",
            payload={
                "sensor_id": "S001",
                "zone_id": "ZONE_A",
                "value": 185.5,
                "unit": "ppm",
                "anomaly_score": -0.92,
            },
        )
    elif args.type == "risk":
        # Inject a compound risk update for ZONE_A
        publish_event(
            topic="compound.risk.detected",
            payload={
                "zone_id": "ZONE_A",
                "compound_risk_score": 88.5,
                "risk_level": "CRITICAL",
                "contributing_factors": {
                    "gas_leak": 0.9,
                    "high_temperature": 0.8,
                },
            },
        )
    elif args.type == "hazard":
        # Inject a hazard detected in ZONE_B
        publish_event(
            topic="hazard.detected",
            payload={
                "hazard_id": f"HAZ-{uuid.uuid4().hex[:6].upper()}",
                "zone_id": "ZONE_B",
                "hazard_type": "FIRE",
                "severity": "CRITICAL",
                "affected_zones": ["ZONE_B", "ZONE_C"],
            },
        )
