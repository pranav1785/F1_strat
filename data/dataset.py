"""Dataset builders for TFT, LSTM, and XGBoost models."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer, NaNLabelEncoder
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


def _add_global_lap_index(df: pd.DataFrame) -> pd.DataFrame:
    """Create a globally unique lap number within each session."""
    out = df.copy()
    session_keys = ["season", "event_name", "session_type"]
    session_offsets = {
        key: idx * 1000 for idx, key in enumerate(out[session_keys].drop_duplicates().itertuples(index=False, name=None))
    }
    out["lap_number_global"] = out.apply(
        lambda row: int(row["LapNumber"]) + session_offsets[(row["season"], row["event_name"], row["session_type"])],
        axis=1,
    )
    return out


def _prepare_tft_frame(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Adapt engineered features to the schema expected by ``TimeSeriesDataSet``."""
    out = _add_global_lap_index(df)

    alias_map = {
        "air_temp": "AirTemp",
        "humidity": "Humidity",
    }
    for target_col, source_col in alias_map.items():
        if target_col not in out.columns and source_col in out.columns:
            out[target_col] = out[source_col]

    stable_categorical_sources = {
        "circuit_id": "event_name",
        "compound_type": "Compound",
        "team_id": "Team",
    }
    for target_col, source_col in stable_categorical_sources.items():
        if source_col in out.columns:
            out[target_col] = out[source_col].fillna("unknown").astype(str)

    if "tire_age_laps" in out.columns and out["tire_age_laps"].isna().any():
        fallback_age = out.groupby("driver_stint_id").cumcount() + 1
        out["tire_age_laps"] = out["tire_age_laps"].fillna(fallback_age.astype(float))
    if "tire_age_sq" in out.columns and out["tire_age_sq"].isna().any():
        out["tire_age_sq"] = out["tire_age_sq"].fillna(out["tire_age_laps"] ** 2)

    required_real_cols = (
        cfg["tft"]["static_reals"]
        + cfg["tft"]["known_time_varying_reals"]
        + cfg["tft"]["unknown_time_varying_reals"]
        + [cfg["tft"]["target_col"]]
    )
    for col in dict.fromkeys(required_real_cols):
        if col not in out.columns:
            continue
        null_count = int(out[col].isna().sum())
        if null_count:
            LOGGER.warning("Imputing %s nulls in TFT real column %s with 0.0", null_count, col)
            out[col] = out[col].fillna(0.0)

    for col in cfg["tft"]["static_categoricals"]:
        if col in out.columns:
            out[col] = out[col].astype(str)

    return out


def build_tft_dataset(
    df: pd.DataFrame,
    cfg: dict,
    reference_dataset: TimeSeriesDataSet | None = None,
    predict: bool = False,
) -> TimeSeriesDataSet:
    """Build the TFT timeseries dataset.

    Args:
        df: Feature dataframe.
        cfg: Project configuration.
        reference_dataset: Optional training dataset whose fitted encoders should be reused.
        predict: Whether to create a prediction-style dataset from the reference dataset.

    Returns:
        Configured ``TimeSeriesDataSet``.
    """
    out = _prepare_tft_frame(df, cfg)
    if reference_dataset is not None:
        return TimeSeriesDataSet.from_dataset(
            reference_dataset,
            out,
            stop_randomization=True,
            predict=predict,
            allow_missing_timesteps=True,
        )
    return TimeSeriesDataSet(
        out,
        time_idx="lap_number_global",
        target=cfg["tft"]["target_col"],
        group_ids=["driver_stint_id"],
        max_encoder_length=cfg["tft"]["encoder_length"],
        max_prediction_length=cfg["tft"]["prediction_length"],
        static_categoricals=cfg["tft"]["static_categoricals"],
        static_reals=cfg["tft"]["static_reals"],
        time_varying_known_reals=cfg["tft"]["known_time_varying_reals"],
        time_varying_unknown_reals=cfg["tft"]["unknown_time_varying_reals"],
        categorical_encoders={col: NaNLabelEncoder(add_nan=True) for col in cfg["tft"]["static_categoricals"]},
        target_normalizer=EncoderNormalizer(method="standard"),
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=True,
    )


def build_lstm_windows(
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows for the causal LSTM classifier.

    Args:
        df: Feature dataframe with labels.
        cfg: Project configuration.

    Returns:
        Tuple ``(X_seq, X_static, y, loss_weights)``.
    """
    seq_len = cfg["lstm"]["seq_len"]
    features = cfg["lstm"]["features"]
    static_feats = cfg["lstm"]["static_features"]
    prepared = df.copy()
    if "tire_age_laps" in prepared.columns and prepared["tire_age_laps"].isna().any():
        fallback_age = prepared.groupby("driver_stint_id").cumcount() + 1
        prepared["tire_age_laps"] = prepared["tire_age_laps"].fillna(fallback_age.astype(float))
    if "tire_age_sq" in prepared.columns and prepared["tire_age_sq"].isna().any():
        prepared["tire_age_sq"] = prepared["tire_age_sq"].fillna(prepared["tire_age_laps"] ** 2)
    required_cols = list(dict.fromkeys(features + static_feats))
    missing_mask = prepared[required_cols].isna()
    if missing_mask.any().any():
        missing_counts = {
            col: int(count)
            for col, count in missing_mask.sum().items()
            if int(count) > 0
        }
        LOGGER.warning("Imputing LSTM window nulls with 0.0 for columns: %s", missing_counts)
        prepared[required_cols] = prepared[required_cols].fillna(0.0)

    X_seq: list[np.ndarray] = []
    X_static: list[np.ndarray] = []
    y: list[int] = []
    loss_weights: list[float] = []

    for _, group in prepared.groupby("driver_stint_id"):
        group_sorted = group.sort_values("LapNumber").reset_index(drop=True)
        for start in range(0, len(group_sorted) - seq_len):
            pred_row = group_sorted.iloc[start + seq_len]
            if pred_row["LapNumber"] <= 3:
                continue
            X_seq.append(group_sorted.iloc[start : start + seq_len][features].to_numpy(dtype=np.float32))
            X_static.append(pred_row[static_feats].to_numpy(dtype=np.float32))
            y.append(int(pred_row["pit_label"]))
            loss_weights.append(float(pred_row["loss_weight"]))

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(X_static, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(loss_weights, dtype=np.float32),
    )


def fit_lstm_scaler(
    X_seq_train: np.ndarray,
    X_static_train: np.ndarray,
    cfg: dict,
) -> tuple[StandardScaler, StandardScaler]:
    """Fit LSTM scalers on training data only.

    Args:
        X_seq_train: Sequential training windows.
        X_static_train: Static training windows.
        cfg: Project configuration.

    Returns:
        Tuple of fitted sequence and static scalers.
    """
    seq_scaler = StandardScaler()
    static_scaler = StandardScaler()
    seq_scaler.fit(X_seq_train.reshape(-1, X_seq_train.shape[-1]))
    static_scaler.fit(X_static_train)
    output_path = Path(cfg["paths"]["scaler_lstm"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"seq": seq_scaler, "static": static_scaler}, output_path)
    LOGGER.info("Saved LSTM scaler to %s", output_path)
    return seq_scaler, static_scaler


def transform_lstm_windows(
    X_seq: np.ndarray,
    X_static: np.ndarray,
    seq_scaler: StandardScaler,
    static_scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform windows with previously fitted training-only scalers."""
    X_seq_scaled = seq_scaler.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape).astype(np.float32)
    X_static_scaled = static_scaler.transform(X_static).astype(np.float32)
    return X_seq_scaled, X_static_scaled


def build_xgb_dataset(
    qualifying_df: pd.DataFrame,
    fingerprint_df: pd.DataFrame,
    cfg: dict,
    return_metadata: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build the XGBoost dataset for compound recommendation.

    Args:
        qualifying_df: Qualifying lap dataframe.
        fingerprint_df: Circuit fingerprint dataframe.
        cfg: Project configuration.

    Returns:
        Tuple of features and target labels.
    """
    merged = qualifying_df.copy()
    fp_features = fingerprint_df.reset_index()
    merge_keys = ["event_name"] if "event_name" in fp_features.columns and "event_name" in merged.columns else ["circuit_id"]
    merged = qualifying_df.merge(
        fp_features,
        on=merge_keys,
        how="left",
        suffixes=("", "_fp"),
    )
    pace_cols = []
    if {"LapTime", "LapTime_s"}.intersection(merged.columns):
        if "LapTime_s" not in merged.columns and "LapTime" in merged.columns:
            merged["LapTime_s"] = merged["LapTime"].dt.total_seconds()
        merged["event_fastest_lap_s"] = merged.groupby(["season", "event_name"])["LapTime_s"].transform("min")
        merged["qualifying_pace_delta_s"] = merged["LapTime_s"] - merged["event_fastest_lap_s"]
        pace_cols.append("qualifying_pace_delta_s")
    weather_cols = [col for col in ["AirTemp", "Humidity", "TrackTemp"] if col in merged.columns]
    tire_cols = [col for col in ["FreshTyre", "TyreLife"] if col in merged.columns]
    base_cols = [
        "avg_deg_rate",
        "compound_variance",
        "track_temp_percentile",
        "high_speed_fraction",
        "hist_soft_stint_len",
        "sc_vsc_frequency",
        "lap_count_norm",
    ]
    feature_cols = [col for col in base_cols + weather_cols + tire_cols + pace_cols if col in merged.columns]
    X = merged[feature_cols].fillna(0.0)

    target_col = cfg["xgboost"]["target_col"]
    if target_col not in merged.columns:
        fallback_target = "q2_compound" if "q2_compound" in merged.columns and merged["q2_compound"].notna().any() else "Compound"
        LOGGER.warning("XGBoost target column %s missing; using %s instead.", target_col, fallback_target)
        merged[target_col] = merged[fallback_target]

    compound_map = {
        "HARD": 0,
        "SUPERHARD": 0,
        "MEDIUM": 1,
        "SOFT": 2,
        "SUPERSOFT": 2,
        "ULTRASOFT": 2,
        "HYPERSOFT": 2,
    }
    target_series = merged[target_col].fillna("").astype(str).str.upper()
    target_series = target_series.map(lambda value: compound_map.get(value, np.nan))
    valid_mask = target_series.notna()
    if not valid_mask.any():
        raise ValueError("No valid XGBoost target labels were found in the qualifying dataset.")
    if (~valid_mask).any():
        LOGGER.warning("Dropping %s qualifying rows with unsupported target compounds.", int((~valid_mask).sum()))
    X = X.loc[valid_mask].reset_index(drop=True)
    y = target_series.loc[valid_mask].astype(int).reset_index(drop=True)
    if not return_metadata:
        return X, y
    metadata_cols = [col for col in ["season", "event_name", "Driver"] if col in merged.columns]
    metadata = merged.loc[valid_mask, metadata_cols].reset_index(drop=True)
    return X, y, metadata
