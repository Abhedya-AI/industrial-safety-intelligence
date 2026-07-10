"""Digital Twin module for Sentinel AI.

Maintains a real-time virtual replica of the facility by consuming
events from all upstream modules:

  - Sensor Intelligence
  - Risk Prediction
  - Compound Risk Intelligence
  - Hazard Propagation

The Digital Twin is a **pure consumer** — it does not publish events.
It serves as the single source of truth for facility state.
"""
