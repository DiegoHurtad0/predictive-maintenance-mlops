"""
Batch inference module.

I designed this module for the primary production use case: given a set of machines
and their recent telemetry/error/maintenance data, generate failure predictions for
the next 24 hours. This simulates the batch job that would run on a schedule
(e.g., every hour via cron or Airflow) to produce an actionable maintenance report.

The module loads the saved atomic Pipeline artifact (scaler + model) and applies it
to new data, ensuring the exact same preprocessing transformations used during training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline as SkPipeline

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


def load_model_artifact(cfg: PipelineConfig) -> tuple[SkPipeline, list[str]]:
    """Load the trained model pipeline and feature column list."""
    model_path = Path(cfg.inference.model_artifact_path)
    feature_path = model_path.parent / "feature_columns.json"

    logger.info("Loading model from: %s", model_path)
    model = joblib.load(model_path)

    logger.info("Loading feature columns from: %s", feature_path)
    with open(feature_path) as f:
        feature_cols = json.load(f)

    return model, feature_cols


def run_batch_inference(
    feature_df: pd.DataFrame,
    cfg: PipelineConfig,
    model: SkPipeline | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run batch predictions on the full feature matrix.

    Produces a DataFrame with columns:
      - datetime, machineID
      - failure_probability (continuous 0-1)
      - predicted_failure (binary, using configured threshold)

    This output is the actionable artifact — maintenance teams use it to
    prioritize which machines to inspect in the next 24 hours.
    """
    if model is None or feature_cols is None:
        model, feature_cols = load_model_artifact(cfg)

    logger.info("Running batch inference on %d rows...", len(feature_df))

    X = feature_df[feature_cols]
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= cfg.evaluation.threshold).astype(int)

    results = pd.DataFrame({
        "datetime": feature_df["datetime"],
        "machineID": feature_df["machineID"],
        "failure_probability": probabilities,
        "predicted_failure": predictions,
    })

    # Summary stats
    n_alerts = predictions.sum()
    machines_at_risk = results[results["predicted_failure"] == 1]["machineID"].nunique()
    logger.info(
        "Batch inference complete — %d alerts across %d machines (of %d total rows)",
        n_alerts, machines_at_risk, len(results),
    )

    return results


def generate_maintenance_report(
    predictions: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Aggregate batch predictions into a per-machine maintenance priority report.
    Ranks machines by their maximum failure probability in the prediction window.
    """
    report = (
        predictions.groupby("machineID")
        .agg(
            max_failure_prob=("failure_probability", "max"),
            mean_failure_prob=("failure_probability", "mean"),
            alert_hours=("predicted_failure", "sum"),
            total_hours=("predicted_failure", "count"),
        )
        .reset_index()
        .sort_values("max_failure_prob", ascending=False)
    )

    report["risk_tier"] = pd.cut(
        report["max_failure_prob"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"],
        include_lowest=True,
    )

    logger.info("Maintenance priority report:\n%s", report.head(20).to_string())

    return report


def save_predictions(predictions: pd.DataFrame, cfg: PipelineConfig) -> Path:
    """Save batch predictions to CSV."""
    output_dir = Path(cfg.inference.batch_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / cfg.inference.batch_output_file

    predictions.to_csv(output_path, index=False)
    logger.info("Predictions saved to: %s", output_path)
    return output_path
