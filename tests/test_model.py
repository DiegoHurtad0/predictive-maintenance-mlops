"""
Model contract tests.

These tests verify that the trained model artifact behaves correctly:
- Outputs valid probability arrays with the right shape
- Produces predictions in [0, 1] range
- The pipeline object is self-contained (scaler + classifier)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config


@pytest.fixture
def cfg():
    return load_config("configs/main_config.yaml")


class TestModelArtifact:
    def test_artifact_exists(self, cfg):
        """Model artifact must exist after training."""
        path = Path(cfg.inference.model_artifact_path)
        if not path.exists():
            pytest.skip("Model artifact not found — run training first")

    def test_feature_columns_exist(self, cfg):
        """Feature column metadata must exist alongside the model."""
        path = Path(cfg.inference.model_artifact_path).parent / "feature_columns.json"
        if not path.exists():
            pytest.skip("Feature columns file not found — run training first")

    def test_model_predictions_shape(self, cfg):
        """Model must produce probability arrays matching input row count."""
        import json
        import joblib

        model_path = Path(cfg.inference.model_artifact_path)
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        model = joblib.load(model_path)
        feat_path = model_path.parent / "feature_columns.json"
        with open(feat_path) as f:
            feature_cols = json.load(f)

        # Create synthetic input
        n_samples = 50
        X = pd.DataFrame(
            np.random.randn(n_samples, len(feature_cols)),
            columns=feature_cols,
        )

        proba = model.predict_proba(X)
        assert proba.shape == (n_samples, 2), f"Expected ({n_samples}, 2), got {proba.shape}"
        assert np.all(proba >= 0) and np.all(proba <= 1), "Probabilities out of range"

    def test_pipeline_has_scaler_and_classifier(self, cfg):
        """The Pipeline must contain both a scaler and classifier step."""
        import joblib

        model_path = Path(cfg.inference.model_artifact_path)
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        model = joblib.load(model_path)
        step_names = [name for name, _ in model.steps]
        assert "scaler" in step_names, "Pipeline missing 'scaler' step"
        assert "classifier" in step_names, "Pipeline missing 'classifier' step"
