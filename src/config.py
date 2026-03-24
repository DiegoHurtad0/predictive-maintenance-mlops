"""
Type-safe configuration loading via Pydantic v2.

I designed this module as the single entry point for all configuration. Every downstream
module imports `load_config()` and receives a fully validated, typed object — no raw dicts,
no KeyError surprises. If the YAML is malformed or a value violates its constraint, the
pipeline fails fast with a clear Pydantic validation error before any compute starts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Nested config sections
# ---------------------------------------------------------------------------
class DataConfig(BaseModel):
    raw_dir: str
    processed_dir: str
    telemetry_file: str
    errors_file: str
    maintenance_file: str
    failures_file: str
    machines_file: str
    output_features_file: str
    output_labels_file: str


class FeaturesConfig(BaseModel):
    telemetry_columns: list[str]
    rolling_windows_hours: list[int]
    lag_periods_hours: list[int]
    error_types: list[str]
    component_types: list[str]
    error_lookback_hours: int = Field(ge=1)
    failure_horizon_hours: int = Field(ge=1)


class SplitConfig(BaseModel):
    train_end: str
    val_end: str

    @field_validator("train_end", "val_end")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        from datetime import datetime

        datetime.strptime(v, "%Y-%m-%d")
        return v


class ModelParams(BaseModel):
    n_estimators: int = 500
    max_depth: int = 7
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    is_unbalance: bool = True
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = -1


class ModelConfig(BaseModel):
    type: str
    random_seed: int = 42
    params: ModelParams = ModelParams()


class BaselineParams(BaseModel):
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str = "balanced"
    random_state: int = 42
    solver: str = "lbfgs"


class BaselineConfig(BaseModel):
    type: str = "logistic_regression"
    params: BaselineParams = BaselineParams()


class EvaluationConfig(BaseModel):
    cv_folds: int = Field(default=3, ge=2)
    primary_metric: str = "f1"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    metrics: list[str] = ["f1", "precision", "recall", "roc_auc", "average_precision"]


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
    register_model: bool = False
    registry_model_name: str = "pdm_failure_classifier"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_file: str = "logs/pipeline.log"
    format: str = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"


class InferenceConfig(BaseModel):
    model_artifact_path: str
    batch_output_dir: str
    batch_output_file: str


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "outputs/model_pipeline.joblib"


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------
class PipelineConfig(BaseModel):
    data: DataConfig
    features: FeaturesConfig
    split: SplitConfig
    model: ModelConfig
    baseline: BaselineConfig = BaselineConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    mlflow: MLflowConfig
    logging: LoggingConfig = LoggingConfig()
    inference: InferenceConfig
    api: APIConfig = APIConfig()


def load_config(config_path: str | Path = "configs/main_config.yaml") -> PipelineConfig:
    """Load and validate the pipeline configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return PipelineConfig(**raw)
