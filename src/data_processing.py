"""
Data loading, validation, and label engineering.

I designed this module to handle the complete data ingestion lifecycle:
1. Load all 5 raw CSV files with correct dtypes
2. Validate schemas via lightweight assertions (Pydantic-style contracts)
3. Engineer the binary failure label with a configurable forecast horizon
4. Merge all sources into a single analysis-ready DataFrame

The critical invariant enforced here is **zero future leakage** in label creation:
for each (machineID, datetime) pair, the label looks only at failures that occur
*after* that timestamp, within the configured horizon window.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------
def load_raw_data(cfg: PipelineConfig) -> dict[str, pd.DataFrame]:
    """Load all raw CSV files and parse datetime columns."""
    raw_dir = Path(cfg.data.raw_dir)

    loaders = {
        "telemetry": (cfg.data.telemetry_file, ["datetime"]),
        "errors": (cfg.data.errors_file, ["datetime"]),
        "maintenance": (cfg.data.maintenance_file, ["datetime"]),
        "failures": (cfg.data.failures_file, ["datetime"]),
        "machines": (cfg.data.machines_file, []),
    }

    data: dict[str, pd.DataFrame] = {}
    for name, (filename, date_cols) in loaders.items():
        path = raw_dir / filename
        logger.info("Loading %s from %s", name, path)
        df = pd.read_csv(path, parse_dates=date_cols if date_cols else False)
        logger.info("  -> %d rows, %d columns", len(df), len(df.columns))
        data[name] = df

    return data


def validate_raw_data(data: dict[str, pd.DataFrame]) -> None:
    """Run schema assertions on raw data — fail fast on unexpected formats."""
    tel = data["telemetry"]
    expected_tel_cols = {"datetime", "machineID", "volt", "rotate", "pressure", "vibration"}
    assert expected_tel_cols.issubset(set(tel.columns)), (
        f"Telemetry missing columns: {expected_tel_cols - set(tel.columns)}"
    )
    assert tel["datetime"].dtype == "datetime64[ns]", "Telemetry datetime not parsed"
    assert tel.isna().sum().sum() == 0, "Unexpected nulls in telemetry"

    err = data["errors"]
    assert {"datetime", "machineID", "errorID"}.issubset(set(err.columns))

    maint = data["maintenance"]
    assert {"datetime", "machineID", "comp"}.issubset(set(maint.columns))

    fail = data["failures"]
    assert {"datetime", "machineID", "failure"}.issubset(set(fail.columns))

    mach = data["machines"]
    assert {"machineID", "model", "age"}.issubset(set(mach.columns))
    assert mach["machineID"].nunique() == len(mach), "Duplicate machine IDs"

    logger.info("All raw data schema validations passed.")


# ---------------------------------------------------------------------------
# Label engineering
# ---------------------------------------------------------------------------
def create_failure_labels(
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
    horizon_hours: int,
) -> pd.DataFrame:
    """
    For each (machineID, datetime) row in telemetry, create a binary label:
      label = 1  if a failure occurs for that machine within the next `horizon_hours`
      label = 0  otherwise

    Also creates `failure_type` for multi-class analysis (comp1–comp4 or 'none').

    CRITICAL: The label looks strictly FORWARD in time — no leakage.
    """
    logger.info(
        "Engineering failure labels with %d-hour forecast horizon...", horizon_hours
    )

    horizon_td = pd.Timedelta(hours=horizon_hours)

    # Build a set of (machineID, failure_datetime, failure_type) for fast lookup
    failures_sorted = failures.sort_values(["machineID", "datetime"]).copy()

    labels = []
    for machine_id in telemetry["machineID"].unique():
        tel_machine = telemetry[telemetry["machineID"] == machine_id].copy()
        fail_machine = failures_sorted[failures_sorted["machineID"] == machine_id]

        tel_machine["label"] = 0
        tel_machine["failure_type"] = "none"

        for _, fail_row in fail_machine.iterrows():
            fail_time = fail_row["datetime"]
            window_start = fail_time - horizon_td
            # Mark all telemetry rows in [fail_time - horizon, fail_time) as positive
            mask = (tel_machine["datetime"] >= window_start) & (
                tel_machine["datetime"] < fail_time
            )
            tel_machine.loc[mask, "label"] = 1
            tel_machine.loc[mask, "failure_type"] = fail_row["failure"]

        labels.append(tel_machine[["datetime", "machineID", "label", "failure_type"]])

    result = pd.concat(labels, ignore_index=True)
    pos = result["label"].sum()
    neg = len(result) - pos
    logger.info(
        "Label distribution — positive: %d (%.2f%%), negative: %d (%.2f%%)",
        pos, 100 * pos / len(result), neg, 100 * neg / len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Full data preparation
# ---------------------------------------------------------------------------
def prepare_data(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end data preparation: load → validate → label → return (telemetry, labels).
    Downstream feature engineering merges additional tables.
    """
    data = load_raw_data(cfg)
    validate_raw_data(data)

    labels = create_failure_labels(
        telemetry=data["telemetry"],
        failures=data["failures"],
        horizon_hours=cfg.features.failure_horizon_hours,
    )

    return data, labels
