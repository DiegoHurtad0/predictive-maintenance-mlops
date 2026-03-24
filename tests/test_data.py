"""
Data validation tests.

I designed these tests to enforce the data contracts that the pipeline depends on.
If the raw data changes shape or schema, these tests catch it immediately —
before any compute is wasted on training with corrupted inputs.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config


@pytest.fixture
def cfg():
    return load_config("configs/main_config.yaml")


@pytest.fixture
def raw_dir(cfg):
    return Path(cfg.data.raw_dir)


class TestTelemetryData:
    def test_file_exists(self, raw_dir, cfg):
        assert (raw_dir / cfg.data.telemetry_file).exists()

    def test_schema(self, raw_dir, cfg):
        df = pd.read_csv(raw_dir / cfg.data.telemetry_file, nrows=10)
        expected = {"datetime", "machineID", "volt", "rotate", "pressure", "vibration"}
        assert expected.issubset(set(df.columns))

    def test_no_nulls(self, raw_dir, cfg):
        df = pd.read_csv(raw_dir / cfg.data.telemetry_file)
        assert df.isna().sum().sum() == 0, "Unexpected nulls in telemetry"

    def test_machine_count(self, raw_dir, cfg):
        df = pd.read_csv(raw_dir / cfg.data.telemetry_file)
        assert df["machineID"].nunique() == 100


class TestFailuresData:
    def test_file_exists(self, raw_dir, cfg):
        assert (raw_dir / cfg.data.failures_file).exists()

    def test_schema(self, raw_dir, cfg):
        df = pd.read_csv(raw_dir / cfg.data.failures_file, nrows=10)
        expected = {"datetime", "machineID", "failure"}
        assert expected.issubset(set(df.columns))

    def test_valid_failure_types(self, raw_dir, cfg):
        df = pd.read_csv(raw_dir / cfg.data.failures_file)
        valid_types = set(cfg.features.component_types)
        actual_types = set(df["failure"].unique())
        assert actual_types.issubset(valid_types), f"Unknown failure types: {actual_types - valid_types}"


class TestMachinesData:
    def test_exactly_100_machines(self, raw_dir, cfg):
        df = pd.read_csv(raw_dir / cfg.data.machines_file)
        assert len(df) == 100
        assert df["machineID"].nunique() == 100
