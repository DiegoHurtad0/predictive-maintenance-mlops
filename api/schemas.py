"""
Pydantic models for API request/response validation.

I designed these schemas to be the enforceable contract between the API consumer
and the prediction service. Any malformed input is rejected at the boundary with
a clear validation error — the model code never sees bad data.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TelemetryRecord(BaseModel):
    """A single machine telemetry observation with pre-computed features."""
    machineID: int = Field(ge=1, le=100)
    volt: float
    rotate: float
    pressure: float
    vibration: float

    class Config:
        extra = "allow"  # Accept additional feature columns dynamically


class PredictionRequest(BaseModel):
    """Batch prediction request containing one or more telemetry records."""
    records: list[TelemetryRecord] = Field(min_length=1)

    @field_validator("records")
    @classmethod
    def check_not_empty(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("At least one record is required")
        return v


class PredictionResult(BaseModel):
    """Prediction output for a single machine."""
    machineID: int
    failure_probability: float = Field(ge=0.0, le=1.0)
    predicted_failure: bool
    risk_tier: str


class PredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: list[PredictionResult]
    total_records: int
    high_risk_count: int
