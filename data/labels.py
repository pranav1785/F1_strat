"""Label engineering and cleaning utilities."""

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _pit_lap_candidates(driver_df: pd.DataFrame) -> pd.Series:
    """Infer pit laps from explicit pit times or stint transitions."""
    explicit = driver_df.loc[driver_df["PitInTime"].notna(), "LapNumber"]
    if not explicit.empty:
        return explicit
    if "Stint" not in driver_df.columns:
        return pd.Series(dtype=driver_df["LapNumber"].dtype)
    ordered = driver_df.sort_values("LapNumber").copy()
    next_stint = ordered["Stint"].shift(-1)
    transition_mask = next_stint.notna() & (next_stint != ordered["Stint"])
    return ordered.loc[transition_mask, "LapNumber"]


def encode_pit_labels(race_laps: pd.DataFrame) -> pd.DataFrame:
    """Encode 3-class pit timing labels with SC/VSC masking.

    Args:
        race_laps: Race lap dataframe.

    Returns:
        Copy of ``race_laps`` with ``pit_label`` and ``loss_weight``.
    """
    df = race_laps.copy()
    df["pit_label"] = 0
    df["loss_weight"] = 1.0
    df.loc[df["sc_or_vsc_flag"] == 1, "loss_weight"] = 0.0

    group_keys = [key for key in ["season", "event_name", "session_type", "Driver"] if key in df.columns]
    for group_values, driver_df in df.groupby(group_keys if group_keys else ["Driver"]):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_filter = pd.Series(True, index=df.index)
        for key, value in zip(group_keys if group_keys else ["Driver"], group_values):
            group_filter &= df[key] == value
        pit_laps = _pit_lap_candidates(driver_df).dropna().astype(int).unique()
        for pit_lap in pit_laps:
            df.loc[group_filter & (df["LapNumber"] == pit_lap), "pit_label"] = 2
            for offset in [1, 2, 3]:
                warn_lap = pit_lap - offset
                mask = (
                    group_filter
                    & (df["LapNumber"] == warn_lap)
                    & (df["pit_label"] == 0)
                )
                df.loc[mask, "pit_label"] = 1
    return df


def _team_matches(team_name: str, reference_name: str) -> bool:
    """Match team names using lowercased substring checks in both directions."""
    team_norm = team_name.lower()
    ref_norm = reference_name.lower()
    return team_norm in ref_norm or ref_norm in team_norm


def filter_top4_constructors(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Mark the configured constructor subset and optionally filter training rows.

    Args:
        df: Lap-level dataframe.
        cfg: Project configuration.

    Returns:
        Dataframe filtered to the configured constructor subset when enabled.
    """
    out = df.copy()
    selected_constructors = cfg["data"].get(
        "selected_constructors",
        ["Red Bull", "Mercedes", "Ferrari", "McLaren"],
    )
    out["is_top4"] = out["Team"].fillna("").astype(str).apply(
        lambda team: any(_team_matches(team, ref) for ref in selected_constructors)
    )
    if cfg["data"]["top4_filter"]:
        return out[out["is_top4"]].copy()
    return out


def apply_cleaning_masks(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply the requested lap-cleaning rules in the specified order.

    Args:
        df: Lap-level dataframe.
        cfg: Project configuration.

    Returns:
        Cleaned dataframe.
    """
    out = df.copy()
    pit_out_mask = out["PitOutTime"].notna() if "PitOutTime" in out.columns else pd.Series(False, index=out.index)
    out["is_pit_out_lap"] = pit_out_mask.astype(int)

    before = len(out)
    out = out[out["LapNumber"] > 1]
    LOGGER.info("Cleaning rule 1 removed %s rows", before - len(out))

    before = len(out)
    out = out[~out["Compound"].isin(cfg["data"]["wet_compounds"])]
    LOGGER.info("Cleaning rule 2 removed %s rows", before - len(out))

    before = len(out)
    stint_medians = out.groupby("driver_stint_id")["LapTime_s"].transform("median")
    out = out[(out["LapTime_s"] <= stint_medians * cfg["data"]["outlier_threshold"]) | pit_out_mask.loc[out.index]]
    LOGGER.info("Cleaning rule 3 removed %s rows", before - len(out))

    LOGGER.info("Cleaning rule 4 preserved %s pit-out laps via is_pit_out_lap flag", int(out["is_pit_out_lap"].sum()))

    before = len(out)
    out = out[out["LapTime_s"].notna() & (out["LapTime_s"] > 0)]
    LOGGER.info("Cleaning rule 5 removed %s rows", before - len(out))

    return out.copy()
