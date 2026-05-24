"""Counterfactual race outcome analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _counterfactual_delta_single(
    lstm_predictions: pd.Series,
    tft_p50_curves: dict,
    actual_laps: pd.DataFrame,
    pit_stop_time_loss_s: float,
    strategy_policy: str = "conservative_advisor",
    min_time_gain_s: float = 0.0,
    allow_late_divergence: bool = False,
) -> float:
    """Compute the capped counterfactual delta for a single evaluation unit."""
    df = actual_laps.copy().reset_index(drop=True)
    preds = lstm_predictions.reset_index(drop=True)
    delta_pos = 0.0

    for driver_stint_id, stint_df in df.groupby("driver_stint_id", sort=False):
        stint_df = stint_df.sort_values("LapNumber").reset_index()
        curve = tft_p50_curves.get(driver_stint_id)
        actual_pit_rows = stint_df[stint_df["pit_label"] == 2]
        for _, pit_row in actual_pit_rows.iterrows():
            history = stint_df[stint_df["LapNumber"] <= pit_row["LapNumber"]]
            if history.empty:
                continue
            sc_pred = history[
                (history["sc_or_vsc_flag"] == 1)
                & (history["index"].isin(preds[preds == 2].index))
            ]
            if not sc_pred.empty:
                continue
            valid_pred_indices = [
                idx
                for idx in history["index"].tolist()
                if idx < len(preds) and int(preds.iloc[idx]) == 2 and int(df.loc[idx, "sc_or_vsc_flag"]) == 0
            ]
            if valid_pred_indices:
                pred_idx = valid_pred_indices[0]
                pred_lap = int(df.loc[pred_idx, "LapNumber"])
                if pred_lap < int(pit_row["LapNumber"]):
                    model_lap = (
                        float(curve.loc[pred_lap])
                        if curve is not None and pred_lap in curve.index
                        else float(df.loc[pred_idx, "lap_time_delta_s"])
                    )
                    time_saved_s = max(float(pit_row["lap_time_delta_s"]) - model_lap, 0.0)
                    if time_saved_s >= min_time_gain_s:
                        delta_pos += time_saved_s / pit_stop_time_loss_s
                continue
            pit_idx = int(pit_row["index"])
            pred_label = int(preds.iloc[pit_idx]) if pit_idx < len(preds) else 0
            if (
                strategy_policy == "aggressive"
                and allow_late_divergence
                and pred_label == 0
                and int(pit_row["sc_or_vsc_flag"]) == 0
            ):
                fresh_tire_time = (
                    float(curve.loc[int(pit_row["LapNumber"])])
                    if curve is not None and int(pit_row["LapNumber"]) in curve.index
                    else float(pit_row["lap_time_delta_s"])
                )
                time_lost_s = max(float(pit_row["lap_time_delta_s"]) - fresh_tire_time, 0.0)
                delta_pos -= time_lost_s / pit_stop_time_loss_s
    return float(np.clip(delta_pos, -5.0, 5.0))


def counterfactual_delta_position(
    race_id: str,
    lstm_predictions: pd.Series,
    tft_p50_curves: dict,
    actual_laps: pd.DataFrame,
    cfg: dict,
) -> float:
    """Compute net counterfactual position delta for one race."""
    del race_id
    return _counterfactual_delta_single(
        lstm_predictions=lstm_predictions,
        tft_p50_curves=tft_p50_curves,
        actual_laps=actual_laps,
        pit_stop_time_loss_s=float(cfg["eval"]["pit_stop_time_loss_s"]),
        strategy_policy=str(cfg["eval"].get("strategy_policy", "conservative_advisor")),
        min_time_gain_s=float(cfg["eval"].get("min_time_gain_s", 0.0)),
        allow_late_divergence=bool(cfg["eval"].get("allow_late_divergence", False)),
    )


def aggregate_counterfactual_by_entity(records: list[dict], entity_name: str) -> dict:
    """Aggregate per-entity counterfactual deltas into summary statistics."""
    if not records:
        return {
            "mean_delta_pos": 0.0,
            "std_delta_pos": 0.0,
            "median_delta_pos": 0.0,
            f"{entity_name}s_positive": 0,
            f"{entity_name}s_negative": 0,
            f"{entity_name}s_neutral": 0,
            f"dry_{entity_name}_mean": 0.0,
            f"street_{entity_name}_mean": 0.0,
            f"wet_{entity_name}_mean": 0.0,
            f"per_{entity_name}_deltas": {},
        }
    values = np.asarray([record["delta_pos"] for record in records], dtype=float)
    dry_vals = [record["delta_pos"] for record in records if record.get("is_wet") is False]
    wet_vals = [record["delta_pos"] for record in records if record.get("is_wet") is True]
    street_vals = [record["delta_pos"] for record in records if record.get("is_street") is True]
    entity_key = f"{entity_name}_id"
    return {
        "mean_delta_pos": float(np.mean(values)),
        "std_delta_pos": float(np.std(values)),
        "median_delta_pos": float(np.median(values)),
        f"{entity_name}s_positive": int(np.sum(values > 0.1)),
        f"{entity_name}s_negative": int(np.sum(values < -0.1)),
        f"{entity_name}s_neutral": int(np.sum(np.abs(values) < 0.1)),
        f"dry_{entity_name}_mean": float(np.mean(dry_vals)) if dry_vals else 0.0,
        f"street_{entity_name}_mean": float(np.mean(street_vals)) if street_vals else 0.0,
        f"wet_{entity_name}_mean": float(np.mean(wet_vals)) if wet_vals else 0.0,
        f"per_{entity_name}_deltas": {
            str(record.get(entity_key, f"{entity_name}_{idx}")): float(record["delta_pos"])
            for idx, record in enumerate(records)
        },
    }


def aggregate_counterfactual(all_race_deltas: list, cfg: dict) -> dict:
    """Aggregate per-race position deltas into the legacy race summary."""
    del cfg
    if not all_race_deltas:
        return {
            "mean_delta_pos": 0.0,
            "std_delta_pos": 0.0,
            "median_delta_pos": 0.0,
            "races_positive": 0,
            "races_negative": 0,
            "races_neutral": 0,
            "dry_race_mean": 0.0,
            "street_circuit_mean": 0.0,
            "wet_race_mean": 0.0,
            "per_race_deltas": {},
        }
    if isinstance(all_race_deltas[0], dict):
        records = all_race_deltas
    else:
        records = [{"race_id": f"race_{idx}", "delta_pos": float(val)} for idx, val in enumerate(all_race_deltas)]
    summary = aggregate_counterfactual_by_entity(records, entity_name="race")
    return {
        "mean_delta_pos": summary["mean_delta_pos"],
        "std_delta_pos": summary["std_delta_pos"],
        "median_delta_pos": summary["median_delta_pos"],
        "races_positive": summary["races_positive"],
        "races_negative": summary["races_negative"],
        "races_neutral": summary["races_neutral"],
        "dry_race_mean": summary["dry_race_mean"],
        "street_circuit_mean": summary["street_race_mean"],
        "wet_race_mean": summary["wet_race_mean"],
        "per_race_deltas": summary["per_race_deltas"],
    }
