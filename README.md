# Industrial Safety Intelligence Platform

A production-grade FastAPI backend for the ET AI Hackathon SentinelAI Industrial Safety Intelligence Platform. This monolith hosts modules for Sensor Intelligence, Risk Prediction, Compound Risk, and Hazard Propagation.

## Quick Start

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 2. Run
uvicorn app.main:app --reload

# 3. Swagger docs
open http://localhost:8000/docs
```

## Interactive API Testing with Swagger

1. **Register a Sensor**: `POST /api/v1/sensors`
   ```json
   {
     "sensor_id": "S001",
     "sensor_name": "Zone A Gas Detector",
     "sensor_type": "GAS",
     "location_zone": "ZONE_A",
     "unit": "ppm",
     "min_value": 0.0,
     "max_value": 10000.0
   }
   ```
2. **Ingest Readings**: `POST /api/v1/readings/ingest`
   ```json
   {
     "sensor_id": "S001",
     "value": 45.2,
     "timestamp": "2026-06-27T16:30:00+05:30",
     "confidence": 98.5
   }
   ```
3. **Retrieve**: `GET /api/v1/readings/latest/{sensor_id}` or `GET /api/v1/readings/{sensor_id}`

---

# Sensor Intelligence — Technical Documentation

> **Version**: 1.0.0 | **Last Updated**: 2026-07-02 | **Module**: `app/sensor_intelligence`

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Interactions](#2-component-interactions)
3. [Database Models](#3-database-models)
4. [Repository Layer](#4-repository-layer)
5. [Service Layer](#5-service-layer)
6. [ML Pipeline](#6-ml-pipeline)
7. [Runtime Anomaly Detection](#7-runtime-anomaly-detection)
8. [Alert Management](#8-alert-management)
9. [Sensor Health Monitoring](#9-sensor-health-monitoring)
10. [Baseline Learning](#10-baseline-learning)
11. [Model Monitoring](#11-model-monitoring)
12. [API Endpoints](#12-api-endpoints)
13. [Configuration](#13-configuration)
14. [Deployment](#14-deployment)
15. [Testing Strategy](#15-testing-strategy)

---

## 1. Architecture Overview

The Sensor Intelligence module is a complete end-to-end system for industrial IoT sensor data processing, anomaly detection, and safety alerting. It is part of the broader Industrial Safety Intelligence platform.

### Design Principles

- **Hexagonal Architecture**: Domain logic is isolated from infrastructure (DB, API, ML)
- **Pure Functions for Business Logic**: Statistics, health scoring, and feature engineering are side-effect-free
- **Repository Pattern**: All persistence is abstracted behind async interfaces
- **Composition Root**: Dependencies are wired via FastAPI's `Depends()` in `core/dependencies.py`
- **Training/Runtime Separation**: ML models are trained offline and loaded at runtime (no training in production)

### High-Level Architecture

```mermaid
graph TB
    subgraph "API Layer"
        A1["/api/v1/sensors"]
        A2["/api/v1/readings"]
        A3["/api/v1/alerts"]
        A4["/api/v1/health"]
    end

    subgraph "Service Layer"
        S1["SensorService"]
        S2["ReadingService"]
        S3["AlertService"]
        S4["StatisticsService"]
        S5["SensorHealthService"]
        S6["BaselineLearningService"]
        S7["ModelMonitoringService"]
    end

    subgraph "Anomaly Detection"
        D1["DetectorFactory"]
        D2["IsolationForestDetector"]
        D3["AutoencoderDetector"]
    end

    subgraph "Analysis"
        AN1["statistics.py"]
        AN2["health_scoring.py"]
    end

    subgraph "Repository Layer"
        R1["SensorRepository"]
        R2["ReadingRepository"]
        R3["AlertRepository"]
        R4["SensorHealthRepository"]
        R5["BaselineRepository"]
    end

    subgraph "Database"
        DB[(SQLite / PostgreSQL)]
    end

    subgraph "ML Models"
        ML1["isolation_forest.pkl"]
        ML2["autoencoder.keras"]
        ML3["scaler.pkl"]
    end

    A1 --> S1
    A2 --> S2
    A3 --> S3
    A4 --> S5

    S2 --> D1
    D1 --> D2
    D1 --> D3
    D2 --> ML1
    D3 --> ML2

    S2 --> R2
    S1 --> R1
    S3 --> R3
    S5 --> R4
    S6 --> R5

    S4 --> AN1
    S5 --> AN2

    R1 --> DB
    R2 --> DB
    R3 --> DB
    R4 --> DB
    R5 --> DB
```

### Directory Structure

```
app/sensor_intelligence/
├── analysis/
│   └── statistics.py             # Pure statistical functions
├── anomaly_detection/
│   ├── base.py                   # BaseAnomalyDetector ABC
│   ├── isolation_forest_detector.py
│   ├── autoencoder_detector.py
│   ├── factory.py                # DetectorFactory
│   └── schemas.py                # AnomalyResult, AnomalyStatus
├── api/
│   ├── router.py                 # Aggregated v1 router
│   ├── sensors.py                # CRUD + current + history
│   ├── readings.py               # Ingestion + retrieval + stats
│   ├── alerts.py                 # Alert endpoints
│   ├── health.py                 # Health check endpoint
│   ├── anomalies.py              # Anomaly endpoints
│   └── thresholds.py             # Threshold endpoints
├── domain/
│   ├── entities/
│   │   └── alert.py              # Alert domain entity
│   └── value_objects/
│       ├── alert_level.py        # WARNING | CRITICAL | EMERGENCY
│       ├── sensor_status.py      # NORMAL | WARNING | CRITICAL | OFFLINE
│       └── sensor_type.py        # TEMPERATURE | GAS | PRESSURE | VIBRATION
├── models/                       # SQLAlchemy ORM models
│   ├── sensor_model.py
│   ├── reading_model.py
│   ├── anomaly_model.py
│   ├── alert_model.py
│   ├── threshold_model.py
│   ├── sensor_health_model.py
│   └── sensor_baseline_model.py
├── repositories/                 # Data access abstractions
│   ├── sensor_repository.py      # Interface
│   ├── reading_repository.py     # Interface
│   ├── alert_repository.py       # Interface
│   ├── sqlalchemy_sensor_repo.py # Implementation
│   ├── sqlalchemy_reading_repo.py
│   ├── sqlalchemy_alert_repo.py
│   └── noop_publisher.py         # Event publisher
├── schemas/                      # Pydantic request/response models
│   ├── reading_schemas.py
│   ├── sensor_schemas.py
│   └── common.py
└── services/                     # Business logic orchestration
    ├── reading_service.py
    ├── sensor_service.py
    ├── alert_service.py
    ├── statistics_service.py
    ├── sensor_health_service.py
    ├── baseline_service.py
    ├── model_monitoring_service.py
    └── health_scoring.py         # Pure scoring functions

training/                         # Offline ML training
├── preprocessing.py
├── feature_engineering.py
├── train_isolation_forest.py
└── train_autoencoder.py

models/                           # Serialized ML artifacts
├── isolation_forest.pkl
├── scaler.pkl
├── autoencoder.keras
├── autoencoder_scaler.pkl
└── autoencoder_threshold.json
```

---

## 2. Component Interactions

### Reading Ingestion Sequence

```mermaid
sequenceDiagram
    participant Client
    participant API as ReadingsAPI
    participant RS as ReadingService
    participant SR as SensorRepo
    participant RR as ReadingRepo
    participant AD as AnomalyDetector
    participant AS as AlertService

    Client->>API: POST /readings/ingest
    API->>RS: ingest_reading(request)
    RS->>SR: get_sensor_by_code(sensor_id)
    SR-->>RS: SensorModel

    Note over RS: Validation Pipeline
    RS->>RS: validate_sensor_active()
    RS->>RS: validate_value()
    RS->>RS: validate_timestamp()
    RS->>RS: validate_range()
    RS->>RS: validate_confidence()
    RS->>RR: check_duplicate()

    RS->>AD: classify(features, [sensor_id])
    AD-->>RS: AnomalyResult(score, status)

    RS->>RS: set anomaly_score & anomaly_status
    RS->>RR: create_reading(reading)
    RR-->>RS: ReadingModel (persisted)

    RS-->>API: ReadingModel
    API-->>Client: 201 SingleReadingResponse

    Note over AS: Post-ingestion (if integrated)
    RS->>AS: evaluate_reading(context)
    AS-->>RS: list[AlertModel]
```

### Alert Lifecycle

```mermaid
sequenceDiagram
    participant RS as ReadingService
    participant AS as AlertService
    participant AR as AlertRepo
    participant EP as EventPublisher

    RS->>AS: evaluate_reading(ReadingContext)

    Note over AS: Check thresholds
    AS->>AS: _check_temperature(value)
    AS->>AS: _check_anomaly_score(score)
    AS->>AS: _check_gas_concentration(value)

    alt Threshold Exceeded
        AS->>AR: get_active_for_sensor(sensor_id, title)
        AR-->>AS: None (no duplicate)
        AS->>AR: create(AlertModel)
        AR-->>AS: AlertModel (persisted)
        AS->>EP: publish("alert.created", alert)
    end

    alt Values Return to Normal
        AS->>AR: get_active_for_sensor(sensor_id)
        AR-->>AS: list[AlertModel]
        AS->>AR: resolve(alert_id)
        AS->>EP: publish("alert.resolved", alert)
    end
```

### Health Scoring Flow

```mermaid
sequenceDiagram
    participant Caller
    participant HS as SensorHealthService
    participant RR as ReadingRepo
    participant SR as SensorRepo
    participant SC as health_scoring.py
    participant HR as HealthRepo

    Caller->>HS: update_sensor_health(sensor)
    HS->>RR: get_stats(sensor_id, window)
    RR-->>HS: ReadingStats
    HS->>RR: get_sensor_history(sensor_id, window)
    RR-->>HS: list[ReadingModel]
    HS->>HS: count anomalies

    HS->>SC: calculate_health_score(HealthFactors)
    Note over SC: Pure function
    SC->>SC: calibration_score (25%)
    SC->>SC: anomaly_score (25%)
    SC->>SC: uptime_score (20%)
    SC->>SC: stability_score (15%)
    SC->>SC: missing_readings_score (15%)
    SC-->>HS: HealthScoreResult

    HS->>HR: save/update(SensorHealthModel)
    HR-->>HS: SensorHealthModel
    HS-->>Caller: SensorHealthModel
```

---

## 3. Database Models

### Entity-Relationship Diagram

```mermaid
erDiagram
    sensors ||--o{ sensor_readings : has
    sensors ||--o{ anomalies : has
    sensors ||--o{ alerts : has
    sensors ||--o{ thresholds : has
    sensors ||--o| sensor_health : has
    sensors ||--o| sensor_baselines : has
    sensor_readings ||--o{ anomalies : triggers
    anomalies ||--o{ alerts : triggers

    sensors {
        string id PK
        string sensor_id UK
        string sensor_name
        string sensor_type
        string status
        string location_zone
        string equipment_id
        string unit
        float min_value
        float max_value
        date installation_date
        date last_calibration
        date next_calibration_due
        datetime created_at
        datetime updated_at
    }

    sensor_readings {
        string id PK
        string sensor_id FK
        float value
        datetime timestamp
        float confidence
        float anomaly_score
        string anomaly_status
        text raw_metadata
        datetime received_at
    }

    anomalies {
        string id PK
        string reading_id FK
        string sensor_id FK
        string anomaly_type
        float severity_score
        float confidence
        text description
        boolean is_resolved
        datetime detected_at
        datetime resolved_at
    }

    alerts {
        string id PK
        string anomaly_id FK
        string sensor_id FK
        string level
        string title
        text message
        boolean is_acknowledged
        string acknowledged_by
        datetime acknowledged_at
        datetime created_at
    }

    thresholds {
        string id PK
        string sensor_id FK
        string metric
        float warning_value
        float critical_value
        float emergency_value
    }

    sensor_health {
        string id PK
        string sensor_id FK
        float health_score
        string health_status
        float calibration_score
        float anomaly_score
        float uptime_score
        float stability_score
        float missing_readings_score
        int total_readings
        int anomaly_count
        text details
        datetime calculated_at
    }

    sensor_baselines {
        string id PK
        string sensor_id FK
        float mean
        float median
        float std_dev
        float variance
        float normal_range_low
        float normal_range_high
        text rolling_avg_5
        text rolling_avg_10
        text hourly_pattern
        string trend_direction
        float trend_slope
        int sample_count
        int window_days
        datetime learned_at
    }
```

### Key Indexes

| Table | Index | Purpose |
|---|---|---|
| `sensors` | `sensor_id (unique)` | Business ID lookup |
| `sensor_readings` | `(sensor_id, timestamp)` | Time-range queries per sensor |
| `anomalies` | `(sensor_id, detected_at)` | Unresolved anomalies per sensor |
| `alerts` | `(is_acknowledged, created_at)` | Dashboard: unacknowledged alerts |
| `sensor_health` | `sensor_id` | Health lookup per sensor |
| `sensor_baselines` | `sensor_id (unique)` | One baseline per sensor |

---

## 4. Repository Layer

### Interface / Implementation Pattern

Each repository follows the same pattern: an **abstract interface** defining the contract, and a **SQLAlchemy implementation** handling persistence.

| Interface | Implementation | Table |
|---|---|---|
| `SensorRepository` | `SQLAlchemySensorRepository` | `sensors` |
| `ReadingRepository` | `SQLAlchemyReadingRepository` | `sensor_readings` |
| `AlertRepository` | `SQLAlchemyAlertRepository` | `alerts` |
| — (inline) | `SensorHealthRepository` | `sensor_health` |
| — (inline) | `BaselineRepository` | `sensor_baselines` |

### Key Repository Methods

**ReadingRepository**:
- `create_reading(reading)` — persist a single reading
- `create_readings_batch(readings)` — atomic batch insert
- `get_latest_reading(sensor_pk)` — most recent reading
- `get_sensor_history(sensor_pk, start, end, limit)` — time-range query
- `get_stats(sensor_pk, start, end)` → `ReadingStats` — aggregated stats
- `count_for_sensor(sensor_pk)` — total reading count

**SensorRepository**:
- `create_sensor(sensor)` / `update_sensor(sensor)` / `delete_sensor(pk)`
- `get_sensor_by_code(sensor_id)` — lookup by business ID
- `list_sensors(sensor_type, status, zone, offset, limit)` — paginated list

**AlertRepository**:
- `create(alert)` / `update(alert)`
- `get_active_for_sensor(sensor_pk, title)` — duplicate check
- `get_active_alerts(offset, limit)` — dashboard query
- `get_alerts_for_sensor(sensor_pk)` — per-sensor history

---

## 5. Service Layer

### Service Dependency Map

```mermaid
graph LR
    RS[ReadingService] --> RR[ReadingRepo]
    RS --> SR[SensorRepo]
    RS --> AD[AnomalyDetector]

    SS[SensorService] --> SR
    SS --> RR
    SS --> StatS[StatisticsService]

    AS[AlertService] --> AR[AlertRepo]
    AS --> EP[EventPublisher]

    StatS --> RR
    StatS --> SR
    StatS --> STATS["statistics.py"]

    HS[HealthService] --> HR[HealthRepo]
    HS --> SR
    HS --> RR
    HS --> HSC["health_scoring.py"]

    BLS[BaselineService] --> BR[BaselineRepo]
    BLS --> SR
    BLS --> RR
    BLS --> STATS

    MMS[MonitoringService]
```

### Service Descriptions

#### ReadingService

`app/sensor_intelligence/services/reading_service.py`

The central ingestion orchestrator. Enforces 8 validation rules:

| Rule | Description | Exception |
|---|---|---|
| 1 | Sensor must exist | `ResourceNotFoundError` |
| 2 | Sensor must not be OFFLINE | `BusinessRuleViolationError` |
| 3 | Value must be finite (no NaN/Inf) | `InvalidReadingError` |
| 4 | Timestamp must not be future (>5 min) | `InvalidReadingError` |
| 5 | Value within sensor min/max range | `InvalidReadingError` |
| 6 | Confidence 0–100 | `InvalidReadingError` |
| 7 | No duplicate (same sensor + timestamp) | `BusinessRuleViolationError` |
| 8 | Batch: all-or-nothing semantics | (any of above) |

After validation, the reading is scored by the anomaly detector (if configured) and persisted.

#### SensorService

`app/sensor_intelligence/services/sensor_service.py`

CRUD operations plus composite queries:
- `create_sensor()` / `update_sensor()` / `delete_sensor()`
- `get_current_sensors()` — all sensors with latest readings, stats, and anomalies
- `get_sensor_history()` — sensor metadata + historical readings + statistics

#### AlertService

`app/sensor_intelligence/services/alert_service.py`

Evaluates readings against configurable thresholds:

| Alert Type | Trigger | Severity Levels |
|---|---|---|
| `HIGH_TEMPERATURE` | Temp > threshold | WARNING / CRITICAL / EMERGENCY |
| `HIGH_GAS_CONCENTRATION` | Gas > threshold | WARNING / CRITICAL / EMERGENCY |
| `HIGH_PRESSURE` | Pressure > threshold | WARNING / CRITICAL / EMERGENCY |
| `ANOMALY_SCORE` | Score > 0.7 | CRITICAL (>0.7) / EMERGENCY (>0.9) |

Features:
- **Duplicate prevention**: One active alert per condition per sensor
- **Auto-resolution**: Alerts cleared when values return to normal
- **Acknowledgment**: Manual operator acknowledgment with audit trail
- **Event publishing**: Publishes `alert.created` / `alert.resolved` events

#### StatisticsService

`app/sensor_intelligence/services/statistics_service.py`

Bridges the pure `statistics.py` module with repository data:
- `compute_statistics(sensor_id, time_range)` → descriptive stats, trend, rolling avg/std
- `compute_descriptive(sensor_id, start, end)` → lightweight stats
- `compute_trend(sensor_id, start, end)` → trend direction + slope

#### SensorHealthService

`app/sensor_intelligence/services/sensor_health_service.py`

Computes a 0–100 health score from 5 weighted factors:

| Factor | Weight | Score Logic |
|---|---|---|
| Calibration | 25% | Days overdue from next_calibration_due |
| Anomaly | 25% | Anomaly rate (0% = 100, ≥20% = 0) |
| Uptime | 20% | Based on sensor status (NORMAL=100, OFFLINE=0) |
| Stability | 15% | Coefficient of variation of recent readings |
| Missing Readings | 15% | actual/expected reading ratio |

Classification: **EXCELLENT** (≥90) → **GOOD** (70–89) → **FAIR** (50–69) → **POOR** (30–49) → **CRITICAL** (<30)

#### BaselineLearningService

`app/sensor_intelligence/services/baseline_service.py`

Learns normal operating statistics per sensor:
- `learn_baseline(sensor_pk)` → `BaselineResult` (mean, median, std_dev, range, rolling avg, hourly pattern, trend)
- `update_baseline(sensor_pk)` → upserts to DB
- `is_within_normal_range(sensor_id, value)` → `True | False | None`
- `update_all_baselines()` → batch recomputation

Normal range: `mean ± sigma_multiplier × std_dev` (default σ=2)

#### ModelMonitoringService

`app/sensor_intelligence/services/model_monitoring_service.py`

Thread-safe, in-memory monitoring for deployed ML models:
- `register_model(metadata)` — register model for tracking
- `record_inference(model, sensor, score, is_anomaly, latency)` — per-inference stats
- `record_loading_failure(model, error)` — failure tracking
- `get_model_health(model)` → `ModelHealthReport` (5 health checks)
- `validate_model(model)` → validation result with pass/fail checks
- `get_system_summary()` → aggregate overview

---

## 6. ML Pipeline

### Training Workflow

```mermaid
graph LR
    subgraph "Data"
        CSV["datasets/processed/features.csv"]
    end

    subgraph "Training - Offline"
        FE["Feature Engineering"]
        PP["Preprocessing"]
        IF["Isolation Forest"]
        AE["Autoencoder"]
    end

    subgraph "Artifacts"
        M1["isolation_forest.pkl"]
        M2["scaler.pkl"]
        M3["autoencoder.keras"]
        M4["autoencoder_scaler.pkl"]
        M5["autoencoder_threshold.json"]
    end

    CSV --> FE
    FE --> PP
    PP --> IF --> M1
    PP --> IF --> M2
    PP --> AE --> M3
    PP --> AE --> M4
    PP --> AE --> M5
```

### Preprocessing

`training/preprocessing.py`

- Loads raw features CSV
- Handles missing values (mean imputation)
- Filters to **normal data only** (`Accident == 0`)
- Applies `StandardScaler`
- Train/test split

### Feature Engineering

`training/feature_engineering.py`

- Selects relevant features for each model type
- Creates derived features (rolling stats, interactions)
- Returns feature arrays for model training

### Isolation Forest Training

`training/train_isolation_forest.py`

```bash
python -m training.train_isolation_forest
```

- Trains `sklearn.IsolationForest` on normal data
- Tunes `contamination` based on dataset distribution
- Evaluates against held-out data with Accident labels
- Saves: `models/isolation_forest.pkl` + `models/scaler.pkl`

Key functions:
- `train_model(X_train, contamination)` → fitted IsolationForest
- `evaluate_model(model, scaler, X_test, y_test)` → metrics dict
- `save_model(model, scaler, paths)` → serialize to disk

### Autoencoder Training

`training/train_autoencoder.py`

```bash
python -m training.train_autoencoder
```

- Builds encoder-decoder architecture (Dense layers, bottleneck)
- Trains on normal data with MSE loss
- Computes reconstruction error threshold (percentile-based)
- Saves: `models/autoencoder.keras` + `models/autoencoder_scaler.pkl` + `models/autoencoder_threshold.json`

Key functions:
- `train_autoencoder(X_train, epochs, batch_size)` → trained model
- `compute_reconstruction_error(model, X)` → error array
- `evaluate_model(model, threshold, X_test, y_test)` → metrics dict

---

## 7. Runtime Anomaly Detection

### Class Hierarchy

```mermaid
classDiagram
    class BaseAnomalyDetector {
        <<abstract>>
        +name: str
        +is_loaded: bool
        +load_model(**kwargs)
        +predict(features) ndarray
        +anomaly_score(features) ndarray
        +classify(features, sensor_ids) list~AnomalyResult~
    }

    class IsolationForestDetector {
        -_model: IsolationForest
        -_scaler: StandardScaler
        +load_model(model_path, scaler_path)
        +predict(features) ndarray
        +anomaly_score(features) ndarray
        +classify(features, sensor_ids) list~AnomalyResult~
    }

    class AutoencoderDetector {
        -_model: keras.Model
        -_scaler: StandardScaler
        -_threshold: float
        +load_model(model_path, threshold_path, scaler_path)
        +predict(features) ndarray
        +anomaly_score(features) ndarray
        +classify(features, sensor_ids) list~AnomalyResult~
    }

    class DetectorFactory {
        +create(detector_type, auto_load) BaseAnomalyDetector
        +available_detectors() list~str~
        +register(name, cls)
    }

    BaseAnomalyDetector <|-- IsolationForestDetector
    BaseAnomalyDetector <|-- AutoencoderDetector
    DetectorFactory ..> BaseAnomalyDetector : creates
```

### DetectorFactory

`app/sensor_intelligence/anomaly_detection/factory.py`

```python
# Create and auto-load a detector
detector = DetectorFactory.create("isolation_forest")
detector = DetectorFactory.create("autoencoder")

# Custom model paths
detector = DetectorFactory.create("isolation_forest",
    model_path="custom/model.pkl",
    scaler_path="custom/scaler.pkl")

# List available detectors
print(DetectorFactory.available_detectors())  # ["autoencoder", "isolation_forest"]
```

### AnomalyResult

`app/sensor_intelligence/anomaly_detection/schemas.py`

```python
@dataclass(frozen=True)
class AnomalyResult:
    sensor_id: str            # Sensor evaluated
    score: float              # Anomaly score (higher = more anomalous)
    status: AnomalyStatus     # NORMAL | ANOMALY
    detector_type: str        # "isolation_forest" | "autoencoder"
    confidence: float | None  # 0.0–1.0
    threshold: float | None   # Classification threshold
    details: dict             # Detector-specific metadata
```

### Score Semantics

| Detector | Score Range | Anomaly Condition |
|---|---|---|
| Isolation Forest | [0, 1] (sigmoid-normalised) | Higher = more anomalous |
| Autoencoder | [0, ∞) (reconstruction_error / threshold) | > 1.0 = anomaly |

### Graceful Degradation

If the detector fails at runtime (model not loaded, inference error), the `ReadingService` catches the exception and defaults to `anomaly_score=0.0, anomaly_status="NORMAL"`. This ensures ingestion is never blocked by ML failures.

---

## 8. Alert Management

### AlertService

`app/sensor_intelligence/services/alert_service.py`

#### Configurable Thresholds

```python
DEFAULT_THRESHOLDS = {
    "TEMPERATURE": SensorThreshold(warning=80, critical=120, emergency=200),
    "GAS":         SensorThreshold(warning=500, critical=1000, emergency=2000),
    "PRESSURE":    SensorThreshold(warning=150, critical=250, emergency=400),
}

ANOMALY_SCORE_THRESHOLDS = {
    "critical": 0.7,
    "emergency": 0.9,
}
```

#### Alert Flow

1. **Reading Context** → `AlertService.evaluate_reading(ctx)`
2. **Threshold Checks** → generate alerts for each violation
3. **Duplicate Check** → skip if active alert exists for same condition
4. **Multi-Violation Escalation** → multiple simultaneous triggers → EMERGENCY
5. **Auto-Resolution** → normal reading clears active alerts for that sensor
6. **Event Publishing** → `alert.created` / `alert.resolved` events

#### Public API

| Method | Description |
|---|---|
| `evaluate_reading(ctx)` | Run all threshold checks, return new alerts |
| `get_active_alerts()` | Unacknowledged alerts for dashboard |
| `acknowledge_alert(id, by)` | Operator acknowledgment |
| `get_alerts_for_sensor(id)` | Alert history per sensor |

---

## 9. Sensor Health Monitoring

### Health Scoring (Pure Functions)

`app/sensor_intelligence/services/health_scoring.py`

```python
# All pure functions, no I/O
factors = HealthFactors(
    last_calibration=date(2026, 6, 1),
    next_calibration_due=date(2026, 12, 1),
    total_readings=500,
    anomaly_count=10,
    sensor_status="NORMAL",
    reading_std_dev=3.5,
    reading_mean=42.0,
    expected_readings=672,
    actual_readings=500,
)
result = calculate_health_score(factors)
# → HealthScoreResult(health_score=78.5, health_status=GOOD, ...)
```

### Scoring Formula

```
health_score = Σ (weight_i × score_i)
```

Where each score_i is 0–100 and weights sum to 1.0.

---

## 10. Baseline Learning

### BaselineLearningService

`app/sensor_intelligence/services/baseline_service.py`

```python
# Learn baseline for a sensor
baseline = await baseline_service.update_baseline(sensor.id)

# Check if a value is within normal range
is_normal = await baseline_service.is_within_normal_range("S001", 42.5)

# Batch recompute all baselines (scheduled job)
results = await baseline_service.update_all_baselines()
```

### What Gets Learned

| Statistic | Source | Storage |
|---|---|---|
| Mean, Median, Std Dev, Variance | `statistics.describe()` | Direct columns |
| Min, Max | `statistics.describe()` | Direct columns |
| Normal Range (low, high) | `mean ± σ × std_dev` | Direct columns |
| Rolling Avg (window 5, 10) | `statistics.rolling_average()` | JSON text |
| Hourly Pattern (24 entries) | `_compute_hourly_pattern()` | JSON text |
| Trend Direction & Slope | `statistics.trend()` | Direct columns |

---

## 11. Model Monitoring

### ModelMonitoringService

`app/sensor_intelligence/services/model_monitoring_service.py`

#### Tracked Metrics

| Metric | Type | Description |
|---|---|---|
| `prediction_count` | Counter | Total inferences |
| `anomaly_count` | Counter | Anomaly classifications |
| `normal_count` | Counter | Normal classifications |
| `avg_anomaly_score` | Gauge | Running average score |
| `min/max_anomaly_score` | Gauge | Score range |
| `loading_failure_count` | Counter | Model load errors |
| `last_inference_at` | Timestamp | Recency check |
| `active_detector` | String | Currently active model |

#### Health Checks

| Check | Pass Condition |
|---|---|
| `model_registered` | Model is registered for monitoring |
| `model_file_exists` | Model file exists on disk |
| `no_loading_failures` | Zero loading failures |
| `recent_inference` | Inference within last hour |
| `anomaly_rate_normal` | Anomaly rate < 50% |

Status: **HEALTHY** (all pass) → **DEGRADED** (1+ non-critical fail) → **UNHEALTHY** (critical fail)

---

## 12. API Endpoints

### Base URL: `/api/v1`

#### Health Checks

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/health/ready` | Readiness check (DB connection) |

#### Sensors

| Method | Path | Description |
|---|---|---|
| `POST` | `/sensors` | Register a new sensor |
| `GET` | `/sensors` | List sensors (paginated, filterable) |
| `GET` | `/sensors/{sensor_id}` | Get sensor by ID |
| `PUT` | `/sensors/{sensor_id}` | Update sensor metadata |
| `DELETE` | `/sensors/{sensor_id}` | Delete sensor |
| `GET` | `/sensors/current` | All sensors with current readings |
| `GET` | `/sensors/{sensor_id}/history` | Sensor history + statistics |

#### Readings

| Method | Path | Description |
|---|---|---|
| `POST` | `/readings/ingest` | Ingest single reading |
| `POST` | `/readings/ingest/batch` | Batch ingest (all-or-nothing) |
| `GET` | `/readings/latest/{sensor_id}` | Latest reading for sensor |
| `GET` | `/readings/{sensor_id}` | Historical readings |
| `GET` | `/readings/{sensor_id}/stats` | Aggregated statistics |

#### Example: Ingest Reading

```json
POST /api/v1/readings/ingest
{
    "sensor_id": "S001",
    "value": 125.7,
    "timestamp": "2026-06-27T08:30:00Z",
    "confidence": 98.5,
    "metadata": {"equipment_id": "EQ-BOILER-001"}
}
```

Response (201):
```json
{
    "success": true,
    "reading": {
        "id": "a1b2c3d4-...",
        "sensor_id": "S001",
        "value": 125.7,
        "timestamp": "2026-06-27T08:30:00Z",
        "confidence": 98.5,
        "received_at": "2026-06-27T08:30:01Z"
    }
}
```

---

## 13. Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | IndustrialSafetyIntelligence | Application name |
| `APP_VERSION` | 0.1.0 | Version string |
| `APP_ENV` | development | Environment (development/production) |
| `DEBUG` | true | Enable debug logging |
| `DATABASE_URL` | sqlite+aiosqlite:///./sensor_intelligence.db | Async DB URL |
| `API_PREFIX` | /api/v1 | API prefix |
| `CORS_ORIGINS` | ["http://localhost:3000"] | CORS allowed origins |
| `EVENT_BROKER` | noop | Event broker (noop/kafka/mqtt) |
| `LOG_LEVEL` | INFO | Logging level |

### Model Paths (Hardcoded Defaults)

| Model | Default Path |
|---|---|
| Isolation Forest | `models/isolation_forest.pkl` |
| IF Scaler | `models/scaler.pkl` |
| Autoencoder | `models/autoencoder.keras` |
| AE Scaler | `models/autoencoder_scaler.pkl` |
| AE Threshold | `models/autoencoder_threshold.json` |

---

## 14. Deployment

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Access Swagger docs
open http://localhost:8000/docs
```

### Production

```bash
# Set environment
export APP_ENV=production
export DATABASE_URL=postgresql+asyncpg://user:pass@host/dbname
export DEBUG=false

# Run with gunicorn + uvicorn workers
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### ML Model Deployment

```bash
# Train models (offline)
python -m training.train_isolation_forest
python -m training.train_autoencoder

# Models are saved to models/ directory
# They are loaded at runtime by DetectorFactory
```

### Database Migrations

```bash
# Auto-generate migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
```

> **Important**: In development mode, tables are auto-created at startup via `Base.metadata.create_all`. In production, always use Alembic migrations.

---

## 15. Testing Strategy

### Test Pyramid

```
                    ┌────────────────────┐
                    │   E2E Integration  │  26 tests
                    │   (Full pipeline)  │
                    ├────────────────────┤
                    │   API Endpoint     │  86 tests
                    │   (HTTP layer)     │
                    ├────────────────────┤
                    │                    │
                    │   Unit Tests       │  374 tests
                    │   (Services,       │
                    │    Analysis,       │
                    │    Training)       │
                    │                    │
                    └────────────────────┘
```

### Test Infrastructure

- **Database**: In-memory async SQLite via `aiosqlite`
- **Session**: Per-test transactional rollback (no test pollution)
- **ML Models**: Stub detectors (`StubNormalDetector`, `StubAnomalyDetector`, `StubFailingDetector`)
- **Async**: `pytest-asyncio` with `asyncio_mode=auto`
- **Coverage**: `pytest-cov` targeting 70% minimum (actual: 92%)

### Running Tests

```bash
# Full suite
pytest tests/ --override-ini="asyncio_mode=auto"

# With coverage
pytest tests/ --override-ini="asyncio_mode=auto" --cov=app --cov-report=term-missing

# E2E only
pytest tests/e2e/test_pipeline_integration.py -v --override-ini="asyncio_mode=auto"

# Specific module
pytest tests/unit/test_alert_service.py -v --override-ini="asyncio_mode=auto"
```

### E2E Pipeline Scenarios

| Scenario | What's Verified |
|---|---|
| Normal readings | Full pipeline: register → ingest → validate → score → persist → respond |
| Anomalous readings | High anomaly score triggers alerts |
| Invalid readings | NaN, Inf, range, future timestamp, nonexistent sensor — all rejected |
| Duplicate readings | Same sensor + timestamp → `BusinessRuleViolationError` |
| Sensor offline | OFFLINE status blocks ingestion |
| Model failures | Detector crash → graceful fallback to NORMAL |
| Batch ingestion | All-or-nothing: one bad reading rejects entire batch |
| Full pipeline | All 6 stages exercised in sequence |

### Test Results Summary

| Suite | Tests | Coverage |
|---|---|---|
| Unit: Services | 210 | 90–100% |
| Unit: Analysis | 24 | 98% |
| Unit: Training | 65 | 95%+ |
| E2E: Pipeline | 26 | — |
| E2E: API Endpoints | 86 | — |
| E2E: Health | 1 | — |
| **Total** | **486** | **92.14%** |