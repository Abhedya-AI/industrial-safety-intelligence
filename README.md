# Industrial Safety Intelligence Platform

A production-grade FastAPI backend for the ET AI Hackathon SentinelAI Industrial Safety Intelligence Platform. This monolith hosts modules for Sensor Intelligence, Risk Prediction, Compound Risk, and Hazard Propagation.

## Repository Structure

```
industrial-safety-intelligence/
├── alembic/                      # Database migrations
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│
├── app/                          # Main application code
│   ├── main.py                   # App startup & lifecycle entry point
│   │
│   ├── core/                     # Core cross-cutting infrastructure
│   │   ├── settings.py           # Configuration environment settings
│   │   ├── logging.py            # Log configurations
│   │   ├── dependencies.py       # Dependency Injection container
│   │   ├── middleware.py         # Middlewares (CORS, Logging, Error Handlers)
│   │   └── security.py           # Auth and security helpers
│   │
│   ├── sensor_intelligence/      # Sensor Intelligence Module
│   │   ├── api/                  # Route handlers
│   │   ├── services/             # Use cases and domain logic
│   │   ├── repositories/         # Ports (interfaces) and concrete adapters
│   │   ├── models/               # SQLAlchemy ORM models
│   │   ├── schemas/              # Pydantic schemas (Request/Response validation)
│   │   ├── domain/               # Domain Entities & Value Objects
│   │   ├── validators/           # Custom validation utilities
│   │   ├── preprocessing/        # Input preprocessing logic
│   │   └── anomaly_detection/    # Anomaly models and scoring algorithms
│   │
│   ├── risk_prediction/          # Risk Prediction Module
│   │   ├── api/
│   │   ├── services/
│   │   ├── models/
│   │   └── ml/
│   │
│   ├── compound_risk/            # Compound Risk Module
│   │   ├── api/
│   │   ├── services/
│   │   └── rules/
│   │
│   ├── hazard_propagation/       # Hazard Propagation Module
│   │   ├── api/
│   │   ├── services/
│   │   └── graph/
│   │
│   └── shared/                   # Infrastructure shared across all modules
│       ├── database/             # Database async connection & Base models
│       ├── exceptions/           # Global domain-specific exceptions
│       ├── dto/                  # Shared data transfer objects
│       ├── utils/                # General utility helper functions
│       └── constants/            # Common domain constants
│
├── training/                     # ML training pipelines
│   ├── dataset_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate.py
│
├── datasets/                     # Local data storage for ML datasets
├── models/                       # Stored model weights/checkpoints
│
├── tests/                        # Comprehensive test suite
│   ├── conftest.py
│   ├── unit/                     # Business logic unit tests
│   ├── integration/              # DB repository integration tests
│   └── e2e/                      # HTTP endpoint request tests
│
├── Dockerfile                    # Production Docker configuration
├── docker-compose.yml            # Local development orchestration
├── requirements.txt              # Pinned python dependencies
├── pyproject.toml                # Project configurations & lint settings
└── README.md                     # Documentation
```

## Getting Started

### Local Setup

1. **Set Up Python Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   Copy the example environment settings to your local configuration:
   ```bash
   cp .env.example .env
   ```

3. **Running the Application**
   ```bash
   uvicorn app.main:app --reload
   ```
   The interactive Swagger documentation will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### Interactive API Testing with Swagger

You can test the ingestion and retrieval flow directly from the Swagger UI:

1. **Register a Sensor**:
   Expand `POST /api/v1/sensors` and register a new sensor (e.g. `S001`):
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
2. **Ingest Readings**:
   Expand `POST /api/v1/readings/ingest`. Ingestion validates if the sensor is registered and is not `OFFLINE`.
   
   *Timezone Note*: If your local timezone has an offset (e.g., IST `+05:30`), ensure the timestamp reflects UTC or explicitly includes your timezone offset so it is not marked as a future timestamp by the validation rules:
   ```json
   {
     "sensor_id": "S001",
     "value": 45.2,
     "timestamp": "2026-06-27T16:30:00+05:30",
     "confidence": 98.5
   }
   ```
3. **Retrieve and Verify**:
   Use `GET /api/v1/readings/latest/{sensor_id}` or `GET /api/v1/readings/{sensor_id}` to fetch the ingested records.

---

### Swagger API Directory

The following endpoints are currently available in the Swagger interface (`/docs`):

#### 🩺 Health Checks
* **`GET /api/v1/health`** — Liveness check (checks if service is running).
* **`GET /api/v1/health/ready`** — Readiness check (verifies database connection).

#### 📟 Sensors
* **`GET /api/v1/sensors/current`** — Retrieves current reading, trend, thresholds, health, and anomaly details for all sensors.
* **`GET /api/v1/sensors/{sensor_id}/history`** — Detailed sensor history including stats, readings range, anomalies, and forecasting.
* **`GET /api/v1/sensors`** — Paginated list of registered sensors (supports filtering by `sensor_type`, `status`, and `zone_id`).
* **`POST /api/v1/sensors`** — Register a new sensor with metadata (threshold bounds, manufacturer, model, units).
* **`GET /api/v1/sensors/{sensor_id}`** — Fetch a single sensor metadata by its business ID.
* **`PUT /api/v1/sensors/{sensor_id}`** — Update an existing sensor's metadata.
* **`DELETE /api/v1/sensors/{sensor_id}`** — Unregister/delete a sensor.

#### 📊 Readings (Telemetry)
* **`POST /api/v1/readings/ingest`** — Ingest a single reading with validations (sensor status active, values within limits, unique timestamp).
* **`POST /api/v1/readings/ingest/batch`** — Atomically ingest multiple readings (all-or-nothing validation).
* **`GET /api/v1/readings/latest/{sensor_id}`** — Get the most recent reading for a sensor.
* **`GET /api/v1/readings/{sensor_id}`** — Query historical readings for a sensor with limit and time range filters.
* **`GET /api/v1/readings/{sensor_id}/stats`** — Query statistics (mean, std dev, min, max, count) for a sensor's readings over a time window.

*(Note: Alerts, Anomalies, and Thresholds management routes are currently stubbed for downstream integrations).*

### Verification

To run the verification test suite:
```bash
python -m pytest tests/ -v
```