"""
FastAPI application for real-time and batch predictions.

I designed this API with two primary endpoints:
  /predict  — accepts a batch of telemetry records and returns failure predictions
  /health   — liveness probe for container orchestration (Docker, K8s)

My choice of FastAPI here was deliberate: automatic OpenAPI docs, native Pydantic
integration for request validation, and async support for high-throughput serving.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from api.dependencies import get_model, load_model
from api.schemas import PredictionRequest, PredictionResponse, PredictionResult

logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNT = Counter("predictions_total", "Total prediction requests")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
HIGH_RISK_COUNT = Counter("high_risk_predictions_total", "Total high-risk predictions")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    try:
        load_model()
        logger.info("Model loaded at startup")
    except FileNotFoundError:
        logger.warning("No model artifact found — /predict will return 503")
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "I built this API to serve machine failure predictions. It accepts telemetry "
        "data and returns failure probabilities with risk tier classifications."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Liveness probe — always returns 200 if the service is running."""
    model, _ = get_model()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Accept a batch of telemetry records and return failure predictions.

    Each record must contain the base telemetry features (volt, rotate, pressure,
    vibration) plus any pre-computed engineered features. The model applies the
    same StandardScaler transform used during training.
    """
    model, feature_cols = get_model()
    if model is None or feature_cols is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    PREDICTION_COUNT.inc()
    start = time.time()

    # Convert request to DataFrame
    records_dicts = [r.model_dump() for r in request.records]
    df = pd.DataFrame(records_dicts)

    # Ensure all required feature columns exist, fill missing with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols].astype(float)
    probabilities = model.predict_proba(X)[:, 1]

    results = []
    high_risk = 0
    for i, record in enumerate(request.records):
        prob = float(probabilities[i])
        if prob >= 0.6:
            tier = "HIGH"
            high_risk += 1
        elif prob >= 0.3:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        results.append(PredictionResult(
            machineID=record.machineID,
            failure_probability=round(prob, 4),
            predicted_failure=prob >= 0.5,
            risk_tier=tier,
        ))

    HIGH_RISK_COUNT.inc(high_risk)
    PREDICTION_LATENCY.observe(time.time() - start)

    return PredictionResponse(
        predictions=results,
        total_records=len(results),
        high_risk_count=high_risk,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint for monitoring."""
    return Response(content=generate_latest(), media_type="text/plain")
