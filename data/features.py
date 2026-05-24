"""Feature engineering utilities for F1 strategy modeling."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LOGGER = logging.getLogger(__name__)


def _to_seconds(series: pd.Series) -> pd.Series:
    """Convert a numeric or timedelta-like series to float seconds.

    Args:
        series: Input series that may already be numeric or may contain timedeltas.

    Returns:
        Float-valued seconds series.
    """
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.astype(float)
    return pd.Series(np.nan, index=series.index, dtype=float)


def _track_status_tokens(series: pd.Series) -> pd.Series:
    """Normalize FastF1 track-status strings into per-lap token sets."""
    normalized = series.fillna("").astype(str)
    return normalized.apply(lambda value: {char for char in value if char.isdigit()})


def _compute_gap_to_leader_seconds(df: pd.DataFrame, group_keys: list[str]) -> pd.Series:
    """Return gap-to-leader in seconds from an explicit column or lap-end time."""
    if "GapToLeader" in df.columns:
        return _to_seconds(df["GapToLeader"])
    if "Time" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    time_s = _to_seconds(df["Time"])
    leader_time_s = time_s.groupby([df[key] for key in group_keys], dropna=False).transform("min")
    return (time_s - leader_time_s).clip(lower=0.0)


def create_stint_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create the unique driver-stint identifier used throughout the pipeline.

    Args:
        df: Lap-level dataframe.

    Returns:
        Copy of ``df`` with ``driver_stint_id`` added.
    """
    out = df.copy()
    out["driver_stint_id"] = (
        out["season"].astype(str)
        + "_"
        + out["event_name"].astype(str)
        + "_"
        + out["Driver"].astype(str)
        + "_"
        + out["Stint"].astype(str)
    )
    return out


def compute_fuel_load(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute the estimated fuel load for each lap.

    Args:
        df: Lap-level dataframe.
        cfg: Project configuration.

    Returns:
        Copy of ``df`` with fuel load features.
    """
    out = df.copy()
    out["fuel_load_kg"] = (
        cfg["data"]["fuel_kg_start"]
        - (out["LapNumber"] * cfg["data"]["fuel_consumption_per_lap"])
    )
    out["fuel_load_kg"] = out["fuel_load_kg"].clip(lower=0.0)
    return out


def compute_lap_time_delta(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute fuel-corrected lap time deltas within each stint.

    Args:
        df: Lap-level dataframe with ``driver_stint_id``.
        cfg: Project configuration.

    Returns:
        Copy of ``df`` with ``LapTime_s`` and ``lap_time_delta_s``.
    """
    out = df.copy()
    out["LapTime_s"] = _to_seconds(out["LapTime"])
    out = out.sort_values(["driver_stint_id", "LapNumber"]).copy()
    within_stint = out.groupby("driver_stint_id").cumcount() + 1
    out["_lap_number_within_stint"] = within_stint
    out["_corrected_lap_time"] = out["LapTime_s"] - (
        out["_lap_number_within_stint"] * cfg["data"]["fuel_lap_time_effect"]
    )

    def _baseline(group: pd.DataFrame) -> pd.Series:
        corrected = group["_corrected_lap_time"].astype(float)
        reference = corrected.where(group["_lap_number_within_stint"] >= 2)
        valid_reference = reference.dropna()
        if valid_reference.empty:
            out_series = corrected.copy()
            out_series.attrs = {}
            return out_series

        center = float(valid_reference.median())
        mad = float((valid_reference - center).abs().median())
        clip_width = max(3.0 * mad, 0.25)
        clipped_reference = reference.clip(lower=center - clip_width, upper=center + clip_width)
        baseline = clipped_reference.expanding(min_periods=1).median()
        out_series = baseline.bfill().fillna(corrected)
        out_series.attrs = {}
        return out_series

    baseline_parts: list[pd.Series] = []
    for _, group in out.groupby("driver_stint_id", sort=False):
        baseline_parts.append(_baseline(group))
    out["_baseline"] = pd.concat(baseline_parts).sort_index()
    out["lap_time_delta_s"] = out["_corrected_lap_time"] - out["_baseline"]
    return out.drop(columns=["_lap_number_within_stint", "_corrected_lap_time", "_baseline"])


def compute_degradation_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """Compute first and second lap-time degradation derivatives.

    Args:
        df: Lap-level dataframe with ``lap_time_delta_s``.

    Returns:
        Copy of ``df`` with degradation derivative columns.
    """
    out = df.copy()
    out["deg_velocity"] = (
        out.groupby("driver_stint_id")["lap_time_delta_s"].diff(1).fillna(0.0)
    )
    out["deg_acceleration"] = (
        out.groupby("driver_stint_id")["deg_velocity"].diff(1).fillna(0.0)
    )
    return out


def compute_tire_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create tire age features.

    Args:
        df: Lap-level dataframe.

    Returns:
        Copy of ``df`` with tire age features.
    """
    out = df.copy()
    out["tire_age_laps"] = out["TyreLife"].astype(float)
    out["tire_age_sq"] = out["tire_age_laps"] ** 2
    return out


def compute_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute gap to the car ahead and its short-term trend.

    Args:
        df: Lap-level dataframe.

    Returns:
        Copy of ``df`` with gap features.
    """
    out = df.copy()
    session_keys = ["season", "event_name", "session_type", "LapNumber"]

    # Vectorize the per-lap "car ahead" calculation instead of iterating row-by-row.
    # This keeps the semantics from the spec while avoiding Python-level loops over
    # tens of thousands of laps.
    out["_position_numeric"] = pd.to_numeric(out["Position"], errors="coerce")
    out["_gap_to_leader_s"] = _compute_gap_to_leader_seconds(out, session_keys)
    out = out.sort_values(session_keys + ["_position_numeric", "Driver"], kind="mergesort").copy()

    out["gap_ahead_s"] = out.groupby(session_keys)["_gap_to_leader_s"].diff().clip(lower=0.0)
    leader_mask = out.groupby(session_keys).cumcount() == 0
    out.loc[leader_mask, "gap_ahead_s"] = 0.0
    out.loc[out["_gap_to_leader_s"].isna(), "gap_ahead_s"] = 0.0
    out["gap_ahead_s"] = out["gap_ahead_s"].fillna(0.0)
    out["gap_ahead_trend"] = (
        out.groupby("driver_stint_id")["gap_ahead_s"].diff(3).fillna(0.0)
    )
    return out.drop(columns=["_position_numeric", "_gap_to_leader_s"]).sort_index()


def compute_undercut_threat(session_laps: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute whether a nearby rival who recently pitted creates an undercut threat.

    Args:
        session_laps: Single-session lap dataframe.
        cfg: Project configuration.

    Returns:
        Copy of ``session_laps`` with ``undercut_threat``.
    """
    df = session_laps.copy()
    df["undercut_threat"] = 0
    gap_thresh = cfg["data"]["undercut_threat_gap_s"]
    look_back = cfg["data"]["undercut_threat_laps"]
    session_keys = [key for key in ["season", "event_name", "session_type"] if key in df.columns]
    gap_group_keys = session_keys + ["LapNumber"]
    df["_gap_to_leader_s"] = _compute_gap_to_leader_seconds(df, gap_group_keys)

    recent_pits = (
        df.loc[df["PitInTime"].notna(), session_keys + ["LapNumber", "Driver"]]
        .rename(columns={"LapNumber": "pit_lap", "Driver": "rival_driver"})
        .copy()
    )
    if recent_pits.empty:
        return df.drop(columns=["_gap_to_leader_s"])

    current = df[session_keys + ["LapNumber", "Driver", "_gap_to_leader_s"]].copy()
    current["source_index"] = current.index

    if session_keys:
        candidate_pairs = current.merge(recent_pits, on=session_keys, how="left")
    else:
        current["_join_key"] = 1
        recent_pits["_join_key"] = 1
        candidate_pairs = current.merge(recent_pits, on="_join_key", how="left").drop(columns="_join_key")

    candidate_pairs = candidate_pairs[
        candidate_pairs["rival_driver"].notna()
        & (candidate_pairs["Driver"] != candidate_pairs["rival_driver"])
        & (candidate_pairs["pit_lap"] < candidate_pairs["LapNumber"])
        & ((candidate_pairs["LapNumber"] - candidate_pairs["pit_lap"]) <= look_back)
    ].copy()
    if candidate_pairs.empty:
        return df.drop(columns=["_gap_to_leader_s"])

    rival_gap_lookup = (
        df[session_keys + ["LapNumber", "Driver", "_gap_to_leader_s"]]
        .rename(columns={"Driver": "rival_driver", "_gap_to_leader_s": "rival_gap_to_leader_s"})
    )
    candidate_pairs = candidate_pairs.merge(
        rival_gap_lookup,
        on=session_keys + ["LapNumber", "rival_driver"],
        how="left",
    )
    threat_indices = candidate_pairs.loc[
        candidate_pairs["rival_gap_to_leader_s"].notna()
        & candidate_pairs["_gap_to_leader_s"].notna()
        & (candidate_pairs["rival_gap_to_leader_s"] < candidate_pairs["_gap_to_leader_s"])
        & ((candidate_pairs["_gap_to_leader_s"] - candidate_pairs["rival_gap_to_leader_s"]) < gap_thresh),
        "source_index",
    ].unique()
    df.loc[threat_indices, "undercut_threat"] = 1
    return df.drop(columns=["_gap_to_leader_s"])


def compute_sc_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute safety car and VSC flags from track status.

    Args:
        df: Lap-level dataframe.
        cfg: Project configuration.

    Returns:
        Copy of ``df`` with SC/VSC flags.
    """
    out = df.copy()
    status_tokens = _track_status_tokens(out["TrackStatus"])
    explicit_sc_codes = set(cfg["data"].get("sc_track_statuses", ["4", "5", "6", "7"]))
    safety_codes = {"4", "5"}
    vsc_codes = {"6", "7"}
    caution_codes = explicit_sc_codes | {"2"}

    out["TrackStatus"] = out["TrackStatus"].fillna("").astype(str)
    out["safety_car_flag"] = status_tokens.apply(lambda tokens: int(bool(tokens & safety_codes)))
    out["vsc_flag"] = status_tokens.apply(lambda tokens: int(bool(tokens & vsc_codes)))
    out["sc_or_vsc_flag"] = status_tokens.apply(lambda tokens: int(bool(tokens & caution_codes)))
    return out


def compute_track_evolution(session_laps: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Estimate track evolution relative to the lap-3 baseline.

    Args:
        session_laps: Single-session lap dataframe.
        cfg: Project configuration.

    Returns:
        Copy of ``session_laps`` with track evolution deltas.
    """
    df = session_laps.copy()
    fuel_effect = cfg["data"]["track_evolution_fuel_effect"]
    df["_corrected"] = df["LapTime_s"] + (df["LapNumber"] * fuel_effect)
    lap3_median = df[df["LapNumber"] == 3]["_corrected"].median()
    per_lap_median = df.groupby("LapNumber")["_corrected"].transform("median")
    df["track_evolution_delta"] = lap3_median - per_lap_median
    return df.drop(columns=["_corrected"])


def compute_lap_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize lap progress and race length features.

    Args:
        df: Lap-level dataframe.

    Returns:
        Copy of ``df`` with normalized lap counters.
    """
    out = df.copy()
    total_laps = out.groupby(["season", "event_name"])["LapNumber"].transform("max")
    out["lap_norm"] = out["LapNumber"] / total_laps
    out["lap_count_norm"] = total_laps / total_laps.max()
    return out


def _transform_with_reference_encoder(series: pd.Series, encoder: LabelEncoder) -> pd.Series:
    """Apply a fitted encoder with an unknown-category fallback."""
    lookup = {label: idx for idx, label in enumerate(encoder.classes_)}
    return series.astype(str).map(lookup).fillna(-1).astype(int)


def encode_categoricals(
    df: pd.DataFrame,
    reference_encoders: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Encode categorical features for downstream models.

    Args:
        df: Lap-level dataframe.

    Returns:
        Tuple of transformed dataframe and encoder metadata.
    """
    out = df.copy()
    encoders: dict[str, Any] = {}

    if reference_encoders and "circuit_id" in reference_encoders:
        circuit_encoder = reference_encoders["circuit_id"]
        out["circuit_id_enc"] = _transform_with_reference_encoder(out["event_name"], circuit_encoder)
    else:
        circuit_encoder = LabelEncoder()
        out["circuit_id_enc"] = circuit_encoder.fit_transform(out["event_name"].astype(str))
    out["circuit_id"] = out["circuit_id_enc"]
    encoders["circuit_id"] = circuit_encoder

    compound_map = {"HARD": 0, "MEDIUM": 1, "SOFT": 2}
    out["compound_type"] = out["Compound"].map(compound_map).fillna(-1).astype(int)
    out["compound_enc"] = out["compound_type"]

    if reference_encoders and "team_id" in reference_encoders:
        team_encoder = reference_encoders["team_id"]
        out["team_id_enc"] = _transform_with_reference_encoder(out["Team"], team_encoder)
    else:
        team_encoder = LabelEncoder()
        out["team_id_enc"] = team_encoder.fit_transform(out["Team"].astype(str))
    out["team_id"] = out["team_id_enc"]
    encoders["team_id"] = team_encoder

    temp = out["TrackTemp"] if "TrackTemp" in out.columns else out.get("AirTemp", pd.Series(25.0, index=out.index))
    out["track_temp_category"] = np.select(
        [temp < 30.0, temp.between(30.0, 45.0, inclusive="both"), temp > 45.0],
        [0, 1, 2],
        default=1,
    ).astype(int)

    quali_pos = out["GridPosition"] if "GridPosition" in out.columns else out.get("Position", 20)
    out["starting_position_norm"] = pd.to_numeric(quali_pos, errors="coerce").fillna(20.0) / 20.0
    return out, encoders


def compute_circuit_softness(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Map abrasion scores to normalized circuit softness values.

    Args:
        df: Lap-level dataframe.
        cfg: Project configuration.

    Returns:
        Copy of ``df`` with ``circuit_softness``.
    """
    out = df.copy()
    abrasion_map = cfg["circuits"]["abrasion"]
    max_abrasion = max(abrasion_map.values())
    out["circuit_softness"] = out["event_name"].map(
        {k: v / max_abrasion for k, v in abrasion_map.items()}
    ).fillna(0.5)
    return out


def compute_historical_sc_prob(df: pd.DataFrame) -> pd.DataFrame:
    """Compute historical SC deployment probability from training seasons only.

    Args:
        df: Lap-level dataframe. A training-only reference dataframe may be provided
            via ``df.attrs['historical_sc_reference']``.

    Returns:
        Copy of ``df`` with ``historical_sc_prob``.
    """
    out = df.copy()
    reference = df.attrs.get("historical_sc_reference", df)
    if "sc_or_vsc_flag" not in reference.columns:
        if "safety_car_flag" in reference.columns:
            reference = reference.copy()
            reference["sc_or_vsc_flag"] = reference["safety_car_flag"]
        else:
            raise KeyError("historical_sc_reference must include sc_or_vsc_flag or safety_car_flag.")
    train_seasons = set(df.attrs.get("train_seasons", reference["season"].unique().tolist()))
    assert set(reference["season"].unique()).issubset(train_seasons), (
        "historical_sc_prob must be computed from training seasons only."
    )
    race_level = (
        reference.groupby(["season", "event_name"])["sc_or_vsc_flag"]
        .max()
        .reset_index(name="had_sc")
    )
    sc_prob = race_level.groupby("event_name")["had_sc"].mean().rename("historical_sc_prob")
    out = out.merge(sc_prob, on="event_name", how="left")
    out["historical_sc_prob"] = out["historical_sc_prob"].fillna(sc_prob.mean() if not sc_prob.empty else 0.0)
    return out


def _log_shape_and_nulls(step_name: str, df: pd.DataFrame) -> None:
    """Log pipeline progress after each feature step."""
    null_count = int(df.isna().sum().sum())
    LOGGER.info("%s -> shape=%s nulls=%s", step_name, df.shape, null_count)


def _concat_preserving_attrs(frames: list[pd.DataFrame], attrs: dict[str, Any]) -> pd.DataFrame:
    """Concatenate frames without triggering ambiguous equality on rich attrs values."""
    sanitized_frames: list[pd.DataFrame] = []
    for frame in frames:
        clean = frame.copy()
        clean.attrs = {}
        sanitized_frames.append(clean)
    out = pd.concat(sanitized_frames, ignore_index=False).sort_index()
    out.attrs = attrs.copy()
    return out


def run_full_feature_pipeline(
    df: pd.DataFrame,
    cfg: dict,
    is_session_level: bool = False,
) -> pd.DataFrame:
    """Run the full deterministic feature engineering pipeline.

    Args:
        df: Input lap dataframe.
        cfg: Project configuration.
        is_session_level: Retained for API compatibility; session-level transforms
            are always handled per session group inside this function.

    Returns:
        Fully engineered dataframe.
    """
    del is_session_level
    pipeline_attrs = df.attrs.copy()
    out = create_stint_id(df)
    out.attrs = pipeline_attrs.copy()
    _log_shape_and_nulls("create_stint_id", out)
    out = compute_fuel_load(out, cfg)
    _log_shape_and_nulls("compute_fuel_load", out)
    out = compute_lap_time_delta(out, cfg)
    _log_shape_and_nulls("compute_lap_time_delta", out)
    out = compute_degradation_derivatives(out)
    _log_shape_and_nulls("compute_degradation_derivatives", out)
    out = compute_tire_age_features(out)
    _log_shape_and_nulls("compute_tire_age_features", out)
    out = compute_gap_features(out)
    _log_shape_and_nulls("compute_gap_features", out)
    out = compute_sc_flags(out, cfg)
    _log_shape_and_nulls("compute_sc_flags", out)

    session_frames: list[pd.DataFrame] = []
    for _, session_df in out.groupby(["season", "event_name", "session_type"], dropna=False):
        evolved = compute_track_evolution(session_df, cfg)
        threatened = compute_undercut_threat(evolved, cfg)
        session_frames.append(threatened)
    out = _concat_preserving_attrs(session_frames, pipeline_attrs)
    _log_shape_and_nulls("session_level_features", out)

    out = compute_lap_norm(out)
    _log_shape_and_nulls("compute_lap_norm", out)
    reference_encoders = pipeline_attrs.get("reference_encoders")
    out, encoders = encode_categoricals(out, reference_encoders=reference_encoders)
    out.attrs["encoders"] = encoders
    out.attrs.update({key: value for key, value in pipeline_attrs.items() if key != "encoders"})
    _log_shape_and_nulls("encode_categoricals", out)
    out = compute_circuit_softness(out, cfg)
    _log_shape_and_nulls("compute_circuit_softness", out)
    out = compute_historical_sc_prob(out)
    out.attrs.update({key: value for key, value in pipeline_attrs.items() if key != "encoders"})
    out.attrs["encoders"] = encoders
    _log_shape_and_nulls("compute_historical_sc_prob", out)

    p1_cols = ["lap_time_delta_s", "deg_acceleration", "tire_age_laps", "undercut_threat"]
    for col in p1_cols:
        null_fraction = float(out[col].isna().mean())
        assert null_fraction <= 0.05, f"{col} exceeds 5% null threshold: {null_fraction:.3f}"
    return out
