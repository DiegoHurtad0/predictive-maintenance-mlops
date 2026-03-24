"""
Model training, evaluation, and experiment tracking.

I designed this module around three non-negotiable principles:
1. Temporal train/val/test split — the model never sees the future during training
2. All preprocessing (scaling) is fit ONLY on training data
3. Every experiment is logged to MLflow for reproducibility and comparison

The training pipeline produces a single atomic artifact: a scikit-learn Pipeline
containing the StandardScaler + the trained model, serialized via joblib. This
means inference only needs one file — no separate scaler, no version mismatch risk.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns that are model features (exclude metadata + label)."""
    exclude = {"datetime", "machineID", "label", "failure_type"}
    return [c for c in df.columns if c not in exclude]


def temporal_split(
    df: pd.DataFrame, cfg: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by temporal boundaries defined in config."""
    train_end = pd.Timestamp(cfg.split.train_end)
    val_end = pd.Timestamp(cfg.split.val_end)

    train = df[df["datetime"] < train_end].copy()
    val = df[(df["datetime"] >= train_end) & (df["datetime"] < val_end)].copy()
    test = df[df["datetime"] >= val_end].copy()

    logger.info("Temporal split:")
    logger.info("  Train: %d rows (< %s)", len(train), train_end)
    logger.info("  Val:   %d rows ([%s, %s))", len(val), train_end, val_end)
    logger.info("  Test:  %d rows (>= %s)", len(test), val_end)
    logger.info(
        "  Train label dist: %s", train["label"].value_counts().to_dict()
    )
    logger.info(
        "  Val label dist:   %s", val["label"].value_counts().to_dict()
    )
    logger.info(
        "  Test label dist:  %s", test["label"].value_counts().to_dict()
    )

    return train, val, test


def build_model(cfg: PipelineConfig, model_type: str = "lightgbm") -> SkPipeline:
    """
    Build a scikit-learn Pipeline with StandardScaler + chosen classifier.
    The Pipeline ensures the scaler is always fit on training data and applied
    consistently during both training and inference — zero leakage by construction.
    """
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        params = cfg.model.params.model_dump()
        classifier = LGBMClassifier(**params)
        logger.info("Building LightGBM pipeline with params: %s", params)
    elif model_type == "logistic_regression":
        params = cfg.baseline.params.model_dump()
        classifier = LogisticRegression(**params)
        logger.info("Building Logistic Regression baseline with params: %s", params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = SkPipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])
    return pipeline


def evaluate_model(
    model: SkPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all evaluation metrics for a given split."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        f"{split_name}_f1": f1_score(y, y_pred, zero_division=0),
        f"{split_name}_precision": precision_score(y, y_pred, zero_division=0),
        f"{split_name}_recall": recall_score(y, y_pred, zero_division=0),
        f"{split_name}_roc_auc": roc_auc_score(y, y_prob) if y.nunique() > 1 else 0.0,
        f"{split_name}_avg_precision": average_precision_score(y, y_prob) if y.nunique() > 1 else 0.0,
    }

    logger.info("--- %s Metrics ---", split_name.upper())
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    logger.info(
        "\n%s Classification Report:\n%s",
        split_name.upper(),
        classification_report(y, y_pred, zero_division=0),
    )
    logger.info(
        "%s Confusion Matrix:\n%s",
        split_name.upper(),
        confusion_matrix(y, y_pred),
    )

    return metrics


def train_and_evaluate(
    feature_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> tuple[SkPipeline, dict[str, float]]:
    """
    Full training + evaluation loop:
    1. Temporal split
    2. Build model pipeline (scaler + classifier)
    3. Fit on training data
    4. Evaluate on val and test
    5. Log everything to MLflow
    6. Save atomic model artifact
    """
    logger.info("=" * 60)
    logger.info("MODEL TRAINING — START")
    logger.info("=" * 60)

    # 1. Split
    train, val, test = temporal_split(feature_df, cfg)
    feature_cols = _get_feature_columns(train)
    logger.info("Using %d features: %s", len(feature_cols), feature_cols[:10])

    X_train, y_train = train[feature_cols], train["label"]
    X_val, y_val = val[feature_cols], val["label"]
    X_test, y_test = test[feature_cols], test["label"]

    # 2. Setup MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    logger.info("MLflow tracking URI: %s", tracking_uri)
    logger.info("MLflow experiment:   %s", cfg.mlflow.experiment_name)

    all_metrics: dict[str, float] = {}

    # 3. Train & evaluate LightGBM (primary model)
    with mlflow.start_run(run_name="lightgbm_primary") as run:
        model = build_model(cfg, model_type="lightgbm")

        logger.info("Fitting LightGBM on %d training samples...", len(X_train))
        model.fit(X_train, y_train)

        val_metrics = evaluate_model(model, X_val, y_val, "val", cfg.evaluation.threshold)
        test_metrics = evaluate_model(model, X_test, y_test, "test", cfg.evaluation.threshold)
        all_metrics.update(val_metrics)
        all_metrics.update(test_metrics)

        # Log parameters — model hyperparameters + dataset context
        mlflow.log_params(cfg.model.params.model_dump())
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("train_positive_rate", float(y_train.mean()))
        mlflow.log_param("threshold", cfg.evaluation.threshold)
        mlflow.log_param("train_end", str(cfg.split.train_end))
        mlflow.log_param("val_end", str(cfg.split.val_end))

        # Log metrics
        mlflow.log_metrics(all_metrics)

        # Feature importance
        lgbm_model = model.named_steps["classifier"]
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": lgbm_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        logger.info("Top 15 features by importance:\n%s", importance.head(15).to_string())

        importance_path = Path("outputs/metrics/feature_importance.csv")
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))

        # Log the trained pipeline as an MLflow model artifact (the key missing piece)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=(
                cfg.mlflow.registry_model_name if cfg.mlflow.register_model else None
            ),
        )
        logger.info("MLflow model artifact logged to run: %s", run.info.run_id)

        # Log confusion matrix as a text artifact
        y_test_pred = (model.predict_proba(X_test)[:, 1] >= cfg.evaluation.threshold).astype(int)
        cm = confusion_matrix(y_test, y_test_pred)
        cm_path = Path("outputs/metrics/confusion_matrix.txt")
        cm_text = (
            f"Confusion Matrix (test set, threshold={cfg.evaluation.threshold}):\n"
            f"               Predicted 0   Predicted 1\n"
            f"  Actual 0     {cm[0, 0]:>10}   {cm[0, 1]:>10}\n"
            f"  Actual 1     {cm[1, 0]:>10}   {cm[1, 1]:>10}\n"
        )
        cm_path.write_text(cm_text)
        mlflow.log_artifact(str(cm_path))

        # Tag the run for easy filtering
        mlflow.set_tags({
            "model_type": "lightgbm",
            "pipeline_stage": "primary",
            "split_strategy": "temporal",
            "weekend_sprint": "true",
        })

        primary_run_id = run.info.run_id

    # 4. Train baseline for comparison (Logistic Regression)
    with mlflow.start_run(run_name="logistic_regression_baseline") as bl_run:
        baseline = build_model(cfg, model_type="logistic_regression")
        logger.info("Fitting Logistic Regression baseline...")
        baseline.fit(X_train, y_train)

        bl_val = evaluate_model(baseline, X_val, y_val, "baseline_val", cfg.evaluation.threshold)
        bl_test = evaluate_model(baseline, X_test, y_test, "baseline_test", cfg.evaluation.threshold)
        mlflow.log_params(cfg.baseline.params.model_dump())
        mlflow.log_metrics({**bl_val, **bl_test})

        # Log baseline model too
        mlflow.sklearn.log_model(baseline, artifact_path="model")
        mlflow.set_tags({
            "model_type": "logistic_regression",
            "pipeline_stage": "baseline",
            "split_strategy": "temporal",
        })

        logger.info("MLflow baseline run ID: %s", bl_run.info.run_id)

    # 5. Save primary model artifact locally (for Docker/API serving)
    artifact_path = Path(cfg.inference.model_artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    logger.info("Model artifact saved to: %s", artifact_path)

    # Also save feature column list for inference
    feature_meta_path = artifact_path.parent / "feature_columns.json"
    with open(feature_meta_path, "w") as f:
        json.dump(feature_cols, f)
    logger.info("Feature columns saved to: %s", feature_meta_path)

    # 6. Save a summary JSON with MLflow run references
    summary = {
        "primary_run_id": primary_run_id,
        "baseline_run_id": bl_run.info.run_id,
        "tracking_uri": tracking_uri,
        "experiment_name": cfg.mlflow.experiment_name,
        "test_f1": all_metrics.get("test_f1"),
        "test_roc_auc": all_metrics.get("test_roc_auc"),
    }
    summary_path = Path("outputs/metrics/training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training summary saved to: %s", summary_path)

    logger.info("MODEL TRAINING — COMPLETE")
    logger.info("=" * 60)

    return model, all_metrics
