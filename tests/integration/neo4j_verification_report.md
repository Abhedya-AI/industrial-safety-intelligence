# Hazard Propagation — Neo4j Verification Report

**Date:** 2026-07-09 10:43:54 UTC  
**Neo4j URI:** `bolt://localhost:7687`  
**Neo4j Connected:** ✅ Yes  
**Total Execution Time:** 3.24s

---

## ✅ Test Summary: 77/77 passed, 0 failed

## 1. Dependency Injection Verification

### Repository Selection Mechanism

The DI container in [`dependencies.py`](file:///Users/alisha/Hackathon/industrial-safety-intelligence/app/core/dependencies.py) selects the graph repository based on:

| Setting | Env Var | Default | Effect |
|---------|---------|---------|--------|
| `graph_repository` | `GRAPH_REPOSITORY` | `in_memory` | `in_memory` → InMemoryGraphRepository, `neo4j` → Neo4jGraphRepository |
| `neo4j_uri` | `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection URI |
| `neo4j_username` | `NEO4J_USERNAME` | `neo4j` | Neo4j authentication username |
| `neo4j_password` | `NEO4J_PASSWORD` | `sentinelai_test` | Neo4j authentication password |
| `neo4j_database` | `NEO4J_DATABASE` | `neo4j` | Neo4j database name |

### DI Test Results

| Test | Result | Details |
|------|--------|---------|
| [DI] Default config selects InMemoryGraphRepository | ✅ PASS | Type: InMemoryGraphRepository |
| [DI] Neo4j config selects Neo4jGraphRepository | ✅ PASS | Type: Neo4jGraphRepository |
| [DI] Invalid Neo4j URI → graceful handling | ✅ PASS | Type: Neo4jGraphRepository (no crash) |
| [DI] Repository singleton caching | ✅ PASS | Same object: True |
| [DI] GRAPH_REPOSITORY env var recognized | ✅ PASS | Value: in_memory |

## 2. Graph Data Seeding

### Facility Graph Topology

```
    ZONE_A (Main Processing Hall, MEDIUM, 12 workers)
    /    \
  ZONE_B  ZONE_C (Boiler Room / Chemical Storage, HIGH)
    \    /
    ZONE_D (Assembly Line, MEDIUM, 18 workers)
      |
    ZONE_E (Warehouse, LOW, 6 workers)
      |
    ZONE_F (Loading Dock, LOW, 3 workers)
```

### Entity Counts

| Entity | Neo4j | InMemory | Match |
|--------|-------|----------|-------|
| Zones | 6 | 6 | ✅ |
| Equipment | 8 | 8 | ✅ |
| Sensors | 11 | 11 | ✅ |
| Edges | 52 | 33 | ⚠️ |
| Hazards | 1 | 1 | ✅ |

## 3. Propagation Scenario Results

### Scenario 1: GAS_LEAK from ZONE_A (High Risk)

**Input:** `GAS_LEAK` from `ZONE_A`, compound_risk_score=80.0

| Metric | Neo4j | InMemory | Match |
|--------|-------|----------|-------|
| Propagation Level | EMERGENCY | EMERGENCY | ✅ |
| Affected Zones | 5 | 5 | ✅ |
| Workers at Risk | 42 | 42 | ✅ |
| Impact Radius (m) | 200 | 200 | ✅ |
| Time to Critical (min) | 0.0 | 0.0 | ✅ |
| Processing Time (ms) | 235.0 | 0.8 | — |

**Zone Impact Scores (Neo4j):**

| Zone | Risk Score | Probability | Workers |
|------|-----------|-------------|---------|
| ZONE_A | 56.00 | 1.0000 | 12 |
| ZONE_B | 46.48 | 0.7000 | 4 |
| ZONE_C | 36.96 | 0.7000 | 2 |
| ZONE_D | 29.79 | 0.4900 | 18 |
| ZONE_E | 24.15 | 0.3430 | 6 |

**Propagation Paths (Neo4j):**

| From | To | Probability | ETA (min) |
|------|----|-------------|-----------|
| ZONE_A | ZONE_C | 0.7000 | 5 |
| ZONE_A | ZONE_B | 0.7000 | 5 |
| ZONE_C | ZONE_D | 0.4900 | 10 |
| ZONE_B | ZONE_D | 0.4900 | 10 |
| ZONE_D | ZONE_E | 0.3430 | 15 |

---

### Scenario 2: FIRE from ZONE_D (Critical)

**Input:** `FIRE` from `ZONE_D`, compound_risk_score=95.0

| Metric | Neo4j | InMemory | Match |
|--------|-------|----------|-------|
| Propagation Level | EMERGENCY | EMERGENCY | ✅ |
| Affected Zones | 6 | 6 | ✅ |
| Workers at Risk | 45 | 45 | ✅ |
| Impact Radius (m) | 150 | 150 | ✅ |
| Time to Critical (min) | 0.0 | 0.0 | ✅ |
| Processing Time (ms) | 263.2 | 0.8 | — |

**Zone Impact Scores (Neo4j):**

| Zone | Risk Score | Probability | Workers |
|------|-----------|-------------|---------|
| ZONE_A | 16.62 | 0.2500 | 12 |
| ZONE_B | 39.42 | 0.5000 | 4 |
| ZONE_C | 31.35 | 0.5000 | 2 |
| ZONE_D | 72.20 | 1.0000 | 18 |
| ZONE_E | 41.80 | 0.5000 | 6 |
| ZONE_F | 13.06 | 0.2500 | 3 |

**Propagation Paths (Neo4j):**

| From | To | Probability | ETA (min) |
|------|----|-------------|-----------|
| ZONE_D | ZONE_E | 0.5000 | 5 |
| ZONE_D | ZONE_C | 0.5000 | 5 |
| ZONE_D | ZONE_B | 0.5000 | 5 |
| ZONE_E | ZONE_F | 0.2500 | 10 |
| ZONE_C | ZONE_A | 0.2500 | 10 |
| ZONE_B | ZONE_A | 0.2500 | 10 |

---

### Scenario 3: CHEMICAL_SPILL from ZONE_F (Edge Zone)

**Input:** `CHEMICAL_SPILL` from `ZONE_F`, compound_risk_score=60.0

| Metric | Neo4j | InMemory | Match |
|--------|-------|----------|-------|
| Propagation Level | SPREADING | SPREADING | ✅ |
| Affected Zones | 3 | 3 | ✅ |
| Workers at Risk | 27 | 27 | ✅ |
| Impact Radius (m) | 150 | 150 | ✅ |
| Time to Critical (min) | 2.5 | 2.5 | ✅ |
| Processing Time (ms) | 188.5 | 0.5 | — |

**Zone Impact Scores (Neo4j):**

| Zone | Risk Score | Probability | Workers |
|------|-----------|-------------|---------|
| ZONE_D | 7.30 | 0.1600 | 18 |
| ZONE_E | 21.12 | 0.4000 | 6 |
| ZONE_F | 33.00 | 1.0000 | 3 |

**Propagation Paths (Neo4j):**

| From | To | Probability | ETA (min) |
|------|----|-------------|-----------|
| ZONE_F | ZONE_E | 0.4000 | 5 |
| ZONE_E | ZONE_D | 0.1600 | 10 |

---

### Scenario 4: SMOKE from ZONE_A (High Decay Factor)

**Input:** `SMOKE` from `ZONE_A`, compound_risk_score=50.0

| Metric | Neo4j | InMemory | Match |
|--------|-------|----------|-------|
| Propagation Level | EMERGENCY | EMERGENCY | ✅ |
| Affected Zones | 5 | 5 | ✅ |
| Workers at Risk | 42 | 42 | ✅ |
| Impact Radius (m) | 200 | 200 | ✅ |
| Time to Critical (min) | 5.6 | 5.6 | ✅ |
| Processing Time (ms) | 365.6 | 0.6 | — |

**Zone Impact Scores (Neo4j):**

| Zone | Risk Score | Probability | Workers |
|------|-----------|-------------|---------|
| ZONE_A | 35.00 | 1.0000 | 12 |
| ZONE_B | 33.20 | 0.8000 | 4 |
| ZONE_C | 26.40 | 0.8000 | 2 |
| ZONE_D | 24.32 | 0.6400 | 18 |
| ZONE_E | 22.53 | 0.5120 | 6 |

**Propagation Paths (Neo4j):**

| From | To | Probability | ETA (min) |
|------|----|-------------|-----------|
| ZONE_A | ZONE_C | 0.8000 | 5 |
| ZONE_A | ZONE_B | 0.8000 | 5 |
| ZONE_C | ZONE_D | 0.6400 | 10 |
| ZONE_B | ZONE_D | 0.6400 | 10 |
| ZONE_D | ZONE_E | 0.5120 | 15 |

---

### Scenario 5: ELECTRICAL_FAULT from ZONE_B (Low Propagation)

**Input:** `ELECTRICAL_FAULT` from `ZONE_B`, compound_risk_score=70.0

| Metric | Neo4j | InMemory | Match |
|--------|-------|----------|-------|
| Propagation Level | SPREADING | SPREADING | ✅ |
| Affected Zones | 3 | 3 | ✅ |
| Workers at Risk | 34 | 34 | ✅ |
| Impact Radius (m) | 100 | 100 | ✅ |
| Time to Critical (min) | 0.6 | 0.6 | ✅ |
| Processing Time (ms) | 156.7 | 0.4 | — |

**Zone Impact Scores (Neo4j):**

| Zone | Risk Score | Probability | Workers |
|------|-----------|-------------|---------|
| ZONE_A | 14.70 | 0.3000 | 12 |
| ZONE_B | 58.10 | 1.0000 | 4 |
| ZONE_D | 15.96 | 0.3000 | 18 |

**Propagation Paths (Neo4j):**

| From | To | Probability | ETA (min) |
|------|----|-------------|-----------|
| ZONE_B | ZONE_D | 0.3000 | 5 |
| ZONE_B | ZONE_A | 0.3000 | 5 |

---

## 4. Validation Test Results

### Scenario Validation

| # | Test | Result | Details |
|---|------|--------|---------|
| 1 | [GAS_LEAK from ZONE_A (High Risk)] Origin zone probability = 1.0 | ✅ | Origin probability: 1.0 |
| 2 | [GAS_LEAK from ZONE_A (High Risk)] Origin zone in affected zones | ✅ | Affected zones: ['ZONE_A', 'ZONE_C', 'ZONE_B', 'ZONE_D', 'ZONE_E'] |
| 3 | [GAS_LEAK from ZONE_A (High Risk)] Minimum affected zones | ✅ | Expected ≥ 3, got 5 |
| 4 | [GAS_LEAK from ZONE_A (High Risk)] Impact scores in [0, 100] | ✅ | Scores: {'ZONE_A': 56.0, 'ZONE_C': 36.96, 'ZONE_B': 46.48, 'ZONE_D': 29.79, 'ZON |
| 5 | [GAS_LEAK from ZONE_A (High Risk)] Propagation probabilities in [0, 1] | ✅ | Probabilities: {'ZONE_A': 1.0, 'ZONE_C': 0.7, 'ZONE_B': 0.7, 'ZONE_D': 0.4899999 |
| 6 | [GAS_LEAK from ZONE_A (High Risk)] Probability decay from origin | ✅ | All non-origin probabilities ≤ origin |
| 7 | [GAS_LEAK from ZONE_A (High Risk)] Propagation paths reference valid zones | ✅ | Paths: [('ZONE_A', 'ZONE_C'), ('ZONE_A', 'ZONE_B'), ('ZONE_C', 'ZONE_D'), ('ZONE |
| 8 | [GAS_LEAK from ZONE_A (High Risk)] Workers at risk ≥ 0 | ✅ | Workers at risk: 42 |
| 9 | [GAS_LEAK from ZONE_A (High Risk)] Impact radius ≥ 0 | ✅ | Impact radius: 200.0m |
| 10 | [GAS_LEAK from ZONE_A (High Risk)] Recommended action is non-empty | ✅ | Action: EMERGENCY: Initiate facility-wide evacuation. Affected zones... |
| 11 | [FIRE from ZONE_D (Critical)] Origin zone probability = 1.0 | ✅ | Origin probability: 1.0 |
| 12 | [FIRE from ZONE_D (Critical)] Origin zone in affected zones | ✅ | Affected zones: ['ZONE_D', 'ZONE_E', 'ZONE_C', 'ZONE_B', 'ZONE_F', 'ZONE_A'] |
| 13 | [FIRE from ZONE_D (Critical)] Minimum affected zones | ✅ | Expected ≥ 3, got 6 |
| 14 | [FIRE from ZONE_D (Critical)] Impact scores in [0, 100] | ✅ | Scores: {'ZONE_D': 72.2, 'ZONE_E': 41.8, 'ZONE_C': 31.35, 'ZONE_B': 39.42, 'ZONE |
| 15 | [FIRE from ZONE_D (Critical)] Propagation probabilities in [0, 1] | ✅ | Probabilities: {'ZONE_D': 1.0, 'ZONE_E': 0.5, 'ZONE_C': 0.5, 'ZONE_B': 0.5, 'ZON |
| 16 | [FIRE from ZONE_D (Critical)] Probability decay from origin | ✅ | All non-origin probabilities ≤ origin |
| 17 | [FIRE from ZONE_D (Critical)] Propagation paths reference valid zones | ✅ | Paths: [('ZONE_D', 'ZONE_E'), ('ZONE_D', 'ZONE_C'), ('ZONE_D', 'ZONE_B'), ('ZONE |
| 18 | [FIRE from ZONE_D (Critical)] Workers at risk ≥ 0 | ✅ | Workers at risk: 45 |
| 19 | [FIRE from ZONE_D (Critical)] Impact radius ≥ 0 | ✅ | Impact radius: 150.0m |
| 20 | [FIRE from ZONE_D (Critical)] Recommended action is non-empty | ✅ | Action: EMERGENCY: Initiate facility-wide evacuation. Affected zones... |
| 21 | [CHEMICAL_SPILL from ZONE_F (Edge Zone)] Origin zone probability = 1.0 | ✅ | Origin probability: 1.0 |
| 22 | [CHEMICAL_SPILL from ZONE_F (Edge Zone)] Origin zone in affected zones | ✅ | Affected zones: ['ZONE_F', 'ZONE_E', 'ZONE_D'] |
| 23 | [CHEMICAL_SPILL from ZONE_F (Edge Zone)] Minimum affected zones | ✅ | Expected ≥ 1, got 3 |
| 24 | [CHEMICAL_SPILL from ZONE_F (Edge Zone)] Impact scores in [0, 100] | ✅ | Scores: {'ZONE_F': 33.0, 'ZONE_E': 21.12, 'ZONE_D': 7.3} |
| 25 | [CHEMICAL_SPILL from ZONE_F (Edge Zone)] Propagation probabilities in [0, 1] | ✅ | Probabilities: {'ZONE_F': 1.0, 'ZONE_E': 0.4, 'ZONE_D': 0.16000000000000003} |
| 26 | [CHEMICAL_SPILL from ZONE_F (Edge Zone)] Probability decay from origin | ✅ | All non-origin probabilities ≤ origin |
| 27 | [CHEMICAL_SPILL from ZONE_F (Edge Zone)] Propagation paths reference valid zones | ✅ | Paths: [('ZONE_F', 'ZONE_E'), ('ZONE_E', 'ZONE_D')] |
| 28 | [CHEMICAL_SPILL from ZONE_F (Edge Zone)] Workers at risk ≥ 0 | ✅ | Workers at risk: 27 |
| 29 | [CHEMICAL_SPILL from ZONE_F (Edge Zone)] Impact radius ≥ 0 | ✅ | Impact radius: 150.0m |
| 30 | [CHEMICAL_SPILL from ZONE_F (Edge Zone)] Recommended action is non-empty | ✅ | Action: WARNING: Hazard spreading from ZONE_F. Restrict access to ad... |
| 31 | [SMOKE from ZONE_A (High Decay Factor)] Origin zone probability = 1.0 | ✅ | Origin probability: 1.0 |
| 32 | [SMOKE from ZONE_A (High Decay Factor)] Origin zone in affected zones | ✅ | Affected zones: ['ZONE_A', 'ZONE_C', 'ZONE_B', 'ZONE_D', 'ZONE_E'] |
| 33 | [SMOKE from ZONE_A (High Decay Factor)] Minimum affected zones | ✅ | Expected ≥ 3, got 5 |
| 34 | [SMOKE from ZONE_A (High Decay Factor)] Impact scores in [0, 100] | ✅ | Scores: {'ZONE_A': 35.0, 'ZONE_C': 26.4, 'ZONE_B': 33.2, 'ZONE_D': 24.32, 'ZONE_ |
| 35 | [SMOKE from ZONE_A (High Decay Factor)] Propagation probabilities in [0, 1] | ✅ | Probabilities: {'ZONE_A': 1.0, 'ZONE_C': 0.8, 'ZONE_B': 0.8, 'ZONE_D': 0.6400000 |
| 36 | [SMOKE from ZONE_A (High Decay Factor)] Probability decay from origin | ✅ | All non-origin probabilities ≤ origin |
| 37 | [SMOKE from ZONE_A (High Decay Factor)] Propagation paths reference valid zones | ✅ | Paths: [('ZONE_A', 'ZONE_C'), ('ZONE_A', 'ZONE_B'), ('ZONE_C', 'ZONE_D'), ('ZONE |
| 38 | [SMOKE from ZONE_A (High Decay Factor)] Workers at risk ≥ 0 | ✅ | Workers at risk: 42 |
| 39 | [SMOKE from ZONE_A (High Decay Factor)] Impact radius ≥ 0 | ✅ | Impact radius: 200.0m |
| 40 | [SMOKE from ZONE_A (High Decay Factor)] Recommended action is non-empty | ✅ | Action: EMERGENCY: Initiate facility-wide evacuation. Affected zones... |
| 41 | [ELECTRICAL_FAULT from ZONE_B (Low Propagation)] Origin zone probability = 1.0 | ✅ | Origin probability: 1.0 |
| 42 | [ELECTRICAL_FAULT from ZONE_B (Low Propagation)] Origin zone in affected zones | ✅ | Affected zones: ['ZONE_B', 'ZONE_D', 'ZONE_A'] |
| 43 | [ELECTRICAL_FAULT from ZONE_B (Low Propagation)] Minimum affected zones | ✅ | Expected ≥ 1, got 3 |
| 44 | [ELECTRICAL_FAULT from ZONE_B (Low Propagation)] Impact scores in [0, 100] | ✅ | Scores: {'ZONE_B': 58.1, 'ZONE_D': 15.96, 'ZONE_A': 14.7} |
| 45 | [ELECTRICAL_FAULT from ZONE_B (Low Propagation)] Propagation probabilities in [0, 1] | ✅ | Probabilities: {'ZONE_B': 1.0, 'ZONE_D': 0.3, 'ZONE_A': 0.3} |
| 46 | [ELECTRICAL_FAULT from ZONE_B (Low Propagation)] Probability decay from origin | ✅ | All non-origin probabilities ≤ origin |
| 47 | [ELECTRICAL_FAULT from ZONE_B (Low Propagation)] Propagation paths reference valid zones | ✅ | Paths: [('ZONE_B', 'ZONE_D'), ('ZONE_B', 'ZONE_A')] |
| 48 | [ELECTRICAL_FAULT from ZONE_B (Low Propagation)] Workers at risk ≥ 0 | ✅ | Workers at risk: 34 |
| 49 | [ELECTRICAL_FAULT from ZONE_B (Low Propagation)] Impact radius ≥ 0 | ✅ | Impact radius: 100.0m |
| 50 | [ELECTRICAL_FAULT from ZONE_B (Low Propagation)] Recommended action is non-empty | ✅ | Action: WARNING: Hazard spreading from ZONE_B. Restrict access to ad... |
| 51 | [Neo4j] Hazard nodes persisted after propagation | ✅ | Found 6 hazard nodes (expected ≥ 6) |
| 52 | [Neo4j] AFFECTS relationships persisted | ✅ | Found 24 AFFECTS relationships |

### Neo4j ↔ InMemory Parity

| # | Test | Result | Details |
|---|------|--------|---------|
| 1 | [Parity: GAS_LEAK from ZONE_A (High Risk)] Propagation level matches | ✅ | Neo4j=EMERGENCY, InMemory=EMERGENCY |
| 2 | [Parity: GAS_LEAK from ZONE_A (High Risk)] Affected zones match | ✅ | Neo4j=['ZONE_A', 'ZONE_B', 'ZONE_C', 'ZONE_D', 'ZONE_E'], InMemory=['ZONE_A', 'Z |
| 3 | [Parity: GAS_LEAK from ZONE_A (High Risk)] Workers at risk match | ✅ | Neo4j=42, InMemory=42 |
| 4 | [Parity: GAS_LEAK from ZONE_A (High Risk)] Impact scores match (±1.0) | ✅ | All scores match |
| 5 | [Parity: FIRE from ZONE_D (Critical)] Propagation level matches | ✅ | Neo4j=EMERGENCY, InMemory=EMERGENCY |
| 6 | [Parity: FIRE from ZONE_D (Critical)] Affected zones match | ✅ | Neo4j=['ZONE_A', 'ZONE_B', 'ZONE_C', 'ZONE_D', 'ZONE_E', 'ZONE_F'], InMemory=['Z |
| 7 | [Parity: FIRE from ZONE_D (Critical)] Workers at risk match | ✅ | Neo4j=45, InMemory=45 |
| 8 | [Parity: FIRE from ZONE_D (Critical)] Impact scores match (±1.0) | ✅ | All scores match |
| 9 | [Parity: CHEMICAL_SPILL from ZONE_F (Edge Zone)] Propagation level matches | ✅ | Neo4j=SPREADING, InMemory=SPREADING |
| 10 | [Parity: CHEMICAL_SPILL from ZONE_F (Edge Zone)] Affected zones match | ✅ | Neo4j=['ZONE_D', 'ZONE_E', 'ZONE_F'], InMemory=['ZONE_D', 'ZONE_E', 'ZONE_F'] |
| 11 | [Parity: CHEMICAL_SPILL from ZONE_F (Edge Zone)] Workers at risk match | ✅ | Neo4j=27, InMemory=27 |
| 12 | [Parity: CHEMICAL_SPILL from ZONE_F (Edge Zone)] Impact scores match (±1.0) | ✅ | All scores match |
| 13 | [Parity: SMOKE from ZONE_A (High Decay Factor)] Propagation level matches | ✅ | Neo4j=EMERGENCY, InMemory=EMERGENCY |
| 14 | [Parity: SMOKE from ZONE_A (High Decay Factor)] Affected zones match | ✅ | Neo4j=['ZONE_A', 'ZONE_B', 'ZONE_C', 'ZONE_D', 'ZONE_E'], InMemory=['ZONE_A', 'Z |
| 15 | [Parity: SMOKE from ZONE_A (High Decay Factor)] Workers at risk match | ✅ | Neo4j=42, InMemory=42 |
| 16 | [Parity: SMOKE from ZONE_A (High Decay Factor)] Impact scores match (±1.0) | ✅ | All scores match |
| 17 | [Parity: ELECTRICAL_FAULT from ZONE_B (Low Propagation)] Propagation level matches | ✅ | Neo4j=SPREADING, InMemory=SPREADING |
| 18 | [Parity: ELECTRICAL_FAULT from ZONE_B (Low Propagation)] Affected zones match | ✅ | Neo4j=['ZONE_A', 'ZONE_B', 'ZONE_D'], InMemory=['ZONE_A', 'ZONE_B', 'ZONE_D'] |
| 19 | [Parity: ELECTRICAL_FAULT from ZONE_B (Low Propagation)] Workers at risk match | ✅ | Neo4j=34, InMemory=34 |
| 20 | [Parity: ELECTRICAL_FAULT from ZONE_B (Low Propagation)] Impact scores match (±1.0) | ✅ | All scores match |

## 5. Neo4j Startup Requirements & Deployment Configuration

### Docker Compose

```bash
# Start Neo4j
docker compose up -d neo4j

# Wait for health check (auto in docker-compose)
# Neo4j Browser: http://localhost:7474
# Bolt endpoint: bolt://localhost:7687
```

### Environment Variables

```bash
# .env file — switch to Neo4j
GRAPH_REPOSITORY=neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=sentinelai_test
NEO4J_DATABASE=neo4j
```

### Fallback Behavior

> [!NOTE]
> If `GRAPH_REPOSITORY=neo4j` but Neo4j is unavailable, the DI container
> logs a warning and falls back to `InMemoryGraphRepository` automatically.
> No application crash occurs.

### Python Dependencies

```bash
pip install neo4j  # Required only when GRAPH_REPOSITORY=neo4j
```
