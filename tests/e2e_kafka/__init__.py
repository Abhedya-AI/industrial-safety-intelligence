"""Kafka end-to-end verification test suite.

Tests the full event pipeline against a real Kafka broker:
  sensor.reading.anomaly → compound.risk.detected → hazard.propagated

Two verification modes:
  Mode 1 (Manual Handler): Direct handler invocation with real Kafka I/O.
  Mode 2 (Full Consumer):  Application consumer loop via start_consumers().
"""
