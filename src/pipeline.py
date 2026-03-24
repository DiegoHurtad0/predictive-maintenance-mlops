"""
Master pipeline orchestrator.

I designed this as the single entry point that chains every stage of the ML lifecycle:
  data loading → validation → feature engineering → training → evaluation → inference

Running `python -m src.pipeline` (or `make train`) executes the entire pipeline
end-to-end. Each stage is independently importable for debugging or notebook use,
but this script is the canonical, reproducible execution path.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

from src.config import PipelineConfig, load_config
from src.data_processing import prepare_data
from src.feature_engineering import build_all_features
from src.inference import (
    generate_maintenance_report,
    run_batch_inference,
    save_predictions,
)
from src.training import train_and_evaluate

logger = logging.getLogger(__name__)


def setup_logging(cfg: PipelineConfig) -> None:
    """Configure structured logging for the entire pipeline."""
    log_path = Path(cfg.logging.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w"),
        ],
    )
    # Suppress noisy third-party loggers
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)


def run_pipeline(config_path: str = "configs/main_config.yaml") -> None:
    """Execute the full training + inference pipeline."""
    t_start = time.time()

    # 1. Load config
    cfg = load_config(config_path)
    setup_logging(cfg)

    logger.info("=" * 70)
    logger.info("PREDICTIVE MAINTENANCE PIPELINE — STARTING")
    logger.info("=" * 70)
    logger.info("Config loaded from: %s", config_path)

    # Set global random seed
    np.random.seed(cfg.model.random_seed)

    # 2. Data loading & validation
    logger.info("\n>>> STAGE 1: Data Loading & Validation")
    data, labels = prepare_data(cfg)

    # 3. Feature engineering
    logger.info("\n>>> STAGE 2: Feature Engineering")
    feature_df = build_all_features(data, labels, cfg)

    # Save processed features for reproducibility
    processed_dir = Path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    feature_path = processed_dir / cfg.data.output_features_file
    feature_df.to_parquet(feature_path, index=False)
    logger.info("Feature matrix saved to: %s", feature_path)

    # 4. Model training & evaluation
    logger.info("\n>>> STAGE 3: Model Training & Evaluation")
    model, metrics = train_and_evaluate(feature_df, cfg)

    # 5. Batch inference (on the test period — simulates production batch job)
    logger.info("\n>>> STAGE 4: Batch Inference")
    predictions = run_batch_inference(feature_df, cfg, model=model,
                                      feature_cols=None)
    # Load feature cols for inference
    import json
    feature_meta = Path(cfg.inference.model_artifact_path).parent / "feature_columns.json"
    with open(feature_meta) as f:
        feat_cols = json.load(f)
    predictions = run_batch_inference(feature_df, cfg, model=model, feature_cols=feat_cols)

    save_predictions(predictions, cfg)

    # 6. Maintenance priority report
    logger.info("\n>>> STAGE 5: Maintenance Report")
    report = generate_maintenance_report(predictions, cfg)
    report_path = Path(cfg.inference.batch_output_dir) / "maintenance_report.csv"
    report.to_csv(report_path, index=False)
    logger.info("Maintenance report saved to: %s", report_path)

    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE — Total time: %.1f seconds", elapsed)
    logger.info("=" * 70)


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/main_config.yaml"
    run_pipeline(config_file)
