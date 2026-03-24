"""
Feature engineering for predictive maintenance.

I designed this module around one core principle: every feature must be computable
using ONLY past data relative to the prediction timestamp. This guarantees zero
data leakage when the model is deployed for real-time or batch inference.

Feature groups:
1. Rolling telemetry statistics (mean, std over 3h/12h/24h windows)
2. Lag differences (24h delta for each sensor)
3. Error frequency counts (per error type, last 24h)
4. Maintenance recency (hours since last maintenance per component)
5. Machine metadata (age + one-hot model)
6. Temporal (hour of day, day of week)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


def build_rolling_telemetry_features(
    telemetry: pd.DataFrame,
    columns: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """
    Compute rolling mean and std for each telemetry column per machine.
    Uses strictly backward-looking windows (no future information).
    """
    logger.info("Building rolling telemetry features (windows=%s)...", windows)
    frames = []

    for machine_id, group in telemetry.groupby("machineID"):
        group = group.sort_values("datetime").copy()

        for col in columns:
            for w in windows:
                roll = group[col].rolling(window=w, min_periods=1)
                group[f"{col}_mean_{w}h"] = roll.mean()
                group[f"{col}_std_{w}h"] = roll.std().fillna(0)

        frames.append(group)

    result = pd.concat(frames, ignore_index=True)
    logger.info("  -> Rolling features shape: %s", result.shape)
    return result


def build_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lag_periods: list[int],
) -> pd.DataFrame:
    """Compute lag differences (current value - value N hours ago) per machine."""
    logger.info("Building lag difference features (lags=%s)...", lag_periods)

    for machine_id, group in df.groupby("machineID"):
        idx = group.index
        for col in columns:
            for lag in lag_periods:
                df.loc[idx, f"{col}_diff_{lag}h"] = group[col].diff(periods=lag).values

    # Fill NaN from diff at the start of each machine's series
    lag_cols = [c for c in df.columns if "_diff_" in c]
    df[lag_cols] = df[lag_cols].fillna(0)
    return df


def build_error_count_features(
    telemetry: pd.DataFrame,
    errors: pd.DataFrame,
    error_types: list[str],
    lookback_hours: int,
) -> pd.DataFrame:
    """
    For each (machineID, datetime), count errors of each type in the last N hours.
    Uses a merge-and-filter approach for correctness over performance.
    """
    logger.info(
        "Building error count features (lookback=%dh, types=%s)...",
        lookback_hours, error_types,
    )

    lookback_td = pd.Timedelta(hours=lookback_hours)

    # One-hot encode error types
    errors_ohe = errors.copy()
    for etype in error_types:
        errors_ohe[etype] = (errors_ohe["errorID"] == etype).astype(int)

    # For each machine, compute cumulative error counts then subtract lagged values
    for etype in error_types:
        telemetry[f"error_{etype}_count_{lookback_hours}h"] = 0

    for machine_id in telemetry["machineID"].unique():
        tel_idx = telemetry["machineID"] == machine_id
        tel_times = telemetry.loc[tel_idx, "datetime"]
        err_machine = errors_ohe[errors_ohe["machineID"] == machine_id].sort_values("datetime")

        if err_machine.empty:
            continue

        for etype in error_types:
            col_name = f"error_{etype}_count_{lookback_hours}h"
            err_times = err_machine.loc[err_machine[etype] == 1, "datetime"].values

            if len(err_times) == 0:
                continue

            counts = []
            for t in tel_times.values:
                window_start = t - lookback_td
                count = int(np.sum((err_times >= window_start) & (err_times <= t)))
                counts.append(count)

            telemetry.loc[tel_idx, col_name] = counts

    logger.info("  -> Error features added.")
    return telemetry


def build_maintenance_recency_features(
    telemetry: pd.DataFrame,
    maintenance: pd.DataFrame,
    component_types: list[str],
) -> pd.DataFrame:
    """
    For each (machineID, datetime), compute hours since last maintenance of each component.
    If no prior maintenance exists, uses a large sentinel value (10000 hours ≈ 14 months).
    """
    logger.info("Building maintenance recency features...")

    sentinel = 10_000.0  # ~14 months, signals "never maintained"

    for comp in component_types:
        telemetry[f"hours_since_maint_{comp}"] = sentinel

    for machine_id in telemetry["machineID"].unique():
        tel_idx = telemetry["machineID"] == machine_id
        tel_times = telemetry.loc[tel_idx, "datetime"].values

        maint_machine = maintenance[maintenance["machineID"] == machine_id]

        for comp in component_types:
            col_name = f"hours_since_maint_{comp}"
            comp_maint = maint_machine[maint_machine["comp"] == comp]["datetime"].sort_values()

            if comp_maint.empty:
                continue

            maint_times = comp_maint.values
            hours_since = []
            for t in tel_times:
                past_maint = maint_times[maint_times <= t]
                if len(past_maint) > 0:
                    delta = (t - past_maint[-1]) / np.timedelta64(1, "h")
                    hours_since.append(float(delta))
                else:
                    hours_since.append(sentinel)

            telemetry.loc[tel_idx, col_name] = hours_since

    logger.info("  -> Maintenance recency features added.")
    return telemetry


def build_machine_metadata_features(
    telemetry: pd.DataFrame,
    machines: pd.DataFrame,
) -> pd.DataFrame:
    """Merge static machine metadata (age + one-hot model) into telemetry."""
    logger.info("Building machine metadata features...")

    machines_ohe = pd.get_dummies(machines, columns=["model"], prefix="model", dtype=int)
    # Drop one dummy to avoid multicollinearity
    model_cols = [c for c in machines_ohe.columns if c.startswith("model_")]
    if len(model_cols) > 1:
        machines_ohe = machines_ohe.drop(columns=[model_cols[-1]])

    result = telemetry.merge(machines_ohe, on="machineID", how="left")
    logger.info("  -> Machine metadata merged. Shape: %s", result.shape)
    return result


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour-of-day and day-of-week from datetime."""
    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    return df


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------
def build_all_features(
    data: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Orchestrate the full feature engineering pipeline.
    Returns a single DataFrame with all features + the binary label.
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING — START")
    logger.info("=" * 60)

    tel = data["telemetry"].copy()

    # 1. Rolling telemetry stats
    tel = build_rolling_telemetry_features(
        tel,
        columns=cfg.features.telemetry_columns,
        windows=cfg.features.rolling_windows_hours,
    )

    # 2. Lag differences
    tel = build_lag_features(
        tel,
        columns=cfg.features.telemetry_columns,
        lag_periods=cfg.features.lag_periods_hours,
    )

    # 3. Error counts
    tel = build_error_count_features(
        tel,
        errors=data["errors"],
        error_types=cfg.features.error_types,
        lookback_hours=cfg.features.error_lookback_hours,
    )

    # 4. Maintenance recency
    tel = build_maintenance_recency_features(
        tel,
        maintenance=data["maintenance"],
        component_types=cfg.features.component_types,
    )

    # 5. Machine metadata
    tel = build_machine_metadata_features(tel, machines=data["machines"])

    # 6. Temporal features
    tel = build_temporal_features(tel)

    # 7. Merge labels
    tel = tel.merge(
        labels[["datetime", "machineID", "label", "failure_type"]],
        on=["datetime", "machineID"],
        how="left",
    )
    tel["label"] = tel["label"].fillna(0).astype(int)

    logger.info("Final feature matrix shape: %s", tel.shape)
    logger.info("FEATURE ENGINEERING — COMPLETE")
    logger.info("=" * 60)

    return tel
