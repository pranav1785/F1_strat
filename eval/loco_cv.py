"""Leave-one-circuit-out validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.circuit_fingerprint import FINGERPRINT_COLS, fingerprint_distance


def run_loco_cv(full_df: pd.DataFrame, fingerprint_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run leave-one-circuit-out evaluation on the 2023 test season.

    Args:
        full_df: Full lap dataframe.
        fingerprint_df: Circuit fingerprint dataframe.
        cfg: Project configuration.

    Returns:
        Per-circuit LOCO results with correlation in attrs.
    """
    test_df = full_df[full_df["season"] == cfg["data"]["test_season"]]
    train_df = full_df[full_df["season"].isin(cfg["data"]["train_seasons"])]
    rows = []
    training_vectors = fingerprint_df[FINGERPRINT_COLS]
    for circuit_name, held_out in test_df.groupby("event_name"):
        circuit_fp = fingerprint_df[fingerprint_df["event_name"] == circuit_name]
        if circuit_fp.empty or training_vectors.empty:
            continue
        distances = training_vectors.apply(
            lambda row: fingerprint_distance(
                circuit_fp[FINGERPRINT_COLS].iloc[0].to_numpy(),
                row.to_numpy(),
            ),
            axis=1,
        )
        nearest_distance = float(distances.min()) if not distances.empty else 0.0
        train_subset = train_df[train_df["event_name"] != circuit_name]
        baseline = train_subset["lap_time_delta_s"].median()
        mae_p50 = float(np.mean(np.abs(held_out["lap_time_delta_s"] - baseline)))
        rows.append(
            {
                "circuit": circuit_name,
                "mae_p50": mae_p50,
                "fingerprint_distance": nearest_distance,
                "circuit_type": "street" if circuit_name in cfg["eval"]["street_circuits"] else "permanent",
            }
        )
    result = pd.DataFrame(rows)
    if len(result) >= 2:
        result.attrs["pearson_r"] = float(result["mae_p50"].corr(result["fingerprint_distance"]))
    else:
        result.attrs["pearson_r"] = 0.0
    return result
