"""Shared messaging infrastructure for Kafka-based event streaming.

Provides reusable Kafka producer, consumer, topic constants, base event
schema, and serialization utilities. All modules must use this shared
infrastructure — no module-specific Kafka code.
"""
