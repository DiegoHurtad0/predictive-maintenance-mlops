# Predictive Maintenance — MLOps Pipeline

> **PoC** → From raw sensor CSVs to a fully containerized, production-grade ML system .

A complete end-to-end MLOps pipeline that predicts equipment failures **24 hours in advance** across a fleet of 100 machines, using 5 years of telemetry, error, and maintenance history.

---

## Goal

Prevent unplanned downtime by flagging at-risk machines before they fail — giving maintenance teams a 24-hour window to intervene. The system ingests raw sensor readings and outputs a ranked priority list with failure probabilities and risk tiers (`HIGH` / `MEDIUM` / `LOW`).

---

## Results

| Metric | LightGBM (Primary) | Logistic Regression (Baseline) |
|--------|-------------------|-------------------------------|
| F1 Score (test) | **0.990** | 0.62 |
| ROC-AUC (test) | **0.9999** | ~0.85 |
| Precision | 0.985 | — |
| Recall | 0.994 | — |

> Evaluation uses a **strict temporal split** (train → Jan–Sep 2015, val → Oct–Nov, test → Dec) — zero data leakage by construction.

---

## Project Structure

```
.
├── api/                        # FastAPI serving layer
│   ├── main.py                 #   /predict, /health, /metrics endpoints
│   ├── dependencies.py         #   Model loading logic
│   └── schemas.py              #   Pydantic request/response contracts
│
├── configs/
│   ├── main_config.yaml        # Single source of truth — all hyperparams & paths
│   └── prometheus.yml          # Prometheus scrape config
│
├── data/                       # Raw CSVs (5 files, ~876K telemetry rows)
│   └── processed/              # Engineered feature matrix (Parquet)
│
├── monitoring/
│   └── grafana/
│       ├── dashboards/         # Pre-built Grafana dashboard JSONs
│       └── provisioning/       # Auto-provisioned datasource + dashboard provider
│
├── notebooks/
│   ├── eda.ipynb               # Exploratory Data Analysis
│   └── orchestration.ipynb     # Single-pane-of-glass pipeline runner
│
├── outputs/
│   ├── artifacts/              # Trained model pipeline (joblib)
│   ├── metrics/                # Feature importance, SHAP plots, confusion matrix
│   ├── mlruns/                 # MLflow experiment store
│   └── predictions/            # Batch predictions + maintenance priority report
│
├── src/                        # Core ML library
│   ├── config.py               #   Pydantic config loader
│   ├── data_processing.py      #   Load, validate & label 5 raw CSVs
│   ├── feature_engineering.py  #   47 temporal features (zero leakage)
│   ├── training.py             #   Train, evaluate, log to MLflow
│   ├── inference.py            #   Batch scoring + maintenance report
│   └── pipeline.py             #   Master orchestrator script
│
├── tests/                      # Pytest suite (data, model, API)
├── docker-compose.yml          # Full stack: API + MLflow + Prometheus + Grafana
├── Dockerfile.api              # FastAPI container
├── Dockerfile.pipeline         # Training pipeline container
└── Makefile                    # CLI shortcuts
```

---

## Services

All services start with a single command — no environment setup required.

```bash
docker compose up -d
```

| Service | URL | Purpose |
|---------|-----|---------|
| **FastAPI** | http://localhost:8000/docs | REST API — real-time & batch predictions |
| **MLflow** | http://localhost:5050 | Experiment tracking UI |
| **Prometheus** | http://localhost:9090 | Metrics collection |
| **Grafana** | http://localhost:3000 | Live dashboard (no login required) |

---

## Quick Start

### Prerequisites
- Docker Desktop installed and running
- Python 3.11+ (for local runs)

### Option A — Docker (recommended, zero setup)

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-handle>/predictive-maintenance-mlops.git
cd predictive-maintenance-mlops

# 2. Train the model
docker compose --profile train run pipeline

# 3. Start all services
docker compose up -d

# 4. Open the interfaces
open http://localhost:8000/docs   # API docs
open http://localhost:5050        # MLflow experiments
open http://localhost:3000        # Grafana dashboard
```

### Option B — Local Python

```bash
# 1. Create virtualenv and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the full pipeline (train → evaluate → batch predict)
make train

# 3. Run the orchestration notebook end-to-end
jupyter notebook notebooks/orchestration.ipynb

# 4. Run the test suite
make test
```

---

## Pipeline Stages

```
Raw CSVs (5 files)
    │
    ▼
data_processing.py   → validate schemas, engineer 24h failure labels
    │
    ▼
feature_engineering.py → 47 features: rolling stats, lag diffs,
    │                     error counts, maintenance recency, metadata
    ▼
training.py          → temporal split → LightGBM + LR baseline
    │                   → MLflow logging (params, metrics, model artifact)
    ▼
inference.py         → batch score all machines
    │                   → failure_predictions.csv + maintenance_report.csv
    ▼
api/main.py          → /predict endpoint (real-time + bulk)
                        → Prometheus metrics → Grafana dashboard
```

---

## Features Engineered (47 total)

| Group | Count | Examples |
|-------|-------|---------|
| Rolling telemetry (3h / 12h / 24h mean + std) | 24 | `volt_mean_3h`, `vibration_std_24h` |
| Lag differences (24h delta) | 4 | `volt_lag24`, `rotate_lag24` |
| Error frequency (last 24h) | 5 | `error1_count_24h` … `error5_count_24h` |
| Maintenance recency | 4 | `hours_since_maint_comp1` … `comp4` |
| Machine metadata | 4 | `age`, `model2`, `model3`, `model4` |
| Temporal | 2 | `hour_of_day`, `day_of_week` |
| Raw telemetry | 4 | `volt`, `rotate`, `pressure`, `vibration` |

---

## Makefile Commands

```bash
make train      # Run full training pipeline
make predict    # Run batch inference on latest model
make test       # Run pytest suite
make serve      # Start FastAPI locally (no Docker)
make lint       # Run Ruff linter
make typecheck  # Run MyPy type checker
make clean      # Remove generated artifacts
```

---

## Data

Five Microsoft Azure PdM dataset CSVs placed in `data/`:

| File | Rows | Description |
|------|------|-------------|
| `PdM_telemetry.csv` | 876,100 | Hourly sensor readings (volt, rotate, pressure, vibration) |
| `PdM_errors.csv` | 3,919 | Error events (error1–error5) |
| `PdM_maint.csv` | 3,286 | Maintenance records per component |
| `PdM_failures.csv` | 761 | Component failure events |
| `PdM_machines.csv` | 100 | Machine metadata (model, age) |

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **Temporal split** over K-fold | Time-series data — random splits cause catastrophic label leakage |
| **LightGBM** over neural nets | Tabular data + interpretability requirement + fast iteration |
| **Atomic `joblib` artifact** | Scaler + classifier in one file — no version mismatch possible at serving time |
| **Pydantic config validation** | Fail fast at load time, not silently during training |
| **SHAP explanations** | Maintenance teams need to know *why* a machine is flagged, not just *that* it's flagged |
| **`gunicorn` direct invocation** | MLflow 2.x's `mlflow server` wrapper binds gunicorn to `127.0.0.1` (bypasses Docker NAT) — direct gunicorn call on `0.0.0.0` fixes this |

---

## Future Work

- **Real-time streaming**: Kafka consumer for live telemetry → sub-minute predictions
- **Automated retraining**: Airflow DAG triggered when F1 drops below 0.95 on a rolling window
- **Drift monitoring**: Evidently AI PSI checks on feature distributions
- **Multi-class failure prediction**: Predict *which* component (comp1–4) will fail, not just *whether*
- **Cost-sensitive threshold**: Tune the 0.5 decision threshold using maintenance cost vs. downtime cost curves

---

