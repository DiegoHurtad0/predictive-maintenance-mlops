"""
API dependency injection: model loading and shared resources.

I designed this module to handle the heavy lifting of model loading at startup,
not per-request. The model is loaded once into memory and shared across all
concurrent requests via FastAPI's dependency injection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline as SkPipeline

from src.config import PipelineConfig, load_config

logger = logging.getLogger(__name__)

# Module-level singletons
_model: SkPipeline | None = None
_feature_cols: list[str] | None = None
_config: PipelineConfig | None = None


def get_config() -> PipelineConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def load_model() -> tuple[SkPipeline, list[str]]:
    """Load model at startup — called once, cached globally."""
    global _model, _feature_cols

    if _model is not None and _feature_cols is not None:
        return _model, _feature_cols

    cfg = get_config()
    model_path = Path(cfg.inference.model_artifact_path)
    feat_path = model_path.parent / "feature_columns.json"

    if not model_path.exists():
        logger.warning("Model artifact not found at %s", model_path)
        raise FileNotFoundError(f"Model not found: {model_path}")

    _model = joblib.load(model_path)
    with open(feat_path) as f:
        _feature_cols = json.load(f)

    logger.info("Model loaded successfully (%d features)", len(_feature_cols))
    return _model, _feature_cols


def get_model() -> tuple[SkPipeline | None, list[str] | None]:
    """Return cached model or None if not loaded."""
    return _model, _feature_cols
