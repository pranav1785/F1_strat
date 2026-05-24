"""Circuit fingerprint computation and persistence."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

LOGGER = logging.getLogger(__name__)

HIGH_SPEED_FRACTION = {
    "Bahrain": 0.35,
    "Saudi Arabia": 0.55,
    "Australia": 0.30,
    "Japan": 0.45,
    "China": 0.38,
    "Miami": 0.42,
    "Emilia Romagna": 0.40,
    "Monaco": 0.05,
    "Canada": 0.25,
    "Spain": 0.35,
    "Austria": 0.50,
    "Britain": 0.60,
    "Hungary": 0.15,
    "Belgium": 0.65,
    "Netherlands": 0.38,
    "Italy": 0.70,
    "Singapore": 0.08,
    "Qatar": 0.52,
    "United States": 0.40,
    "Mexico": 0.45,
    "Brazil": 0.42,
    "Las Vegas": 0.55,
    "Abu Dhabi": 0.35,
}

FINGERPRINT_COLS = [
    "avg_deg_rate",
    "compound_variance",
    "track_temp_percentile",
    "high_speed_fraction",
    "hist_soft_stint_len",
    "sc_vsc_frequency",
]


def compute_circuit_fingerprints(train_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute six-dimensional circuit fingerprints from training data only.

    Args:
        train_df: Training-only lap dataframe.
        cfg: Project configuration.

    Returns:
        Dataframe indexed by integer ``circuit_id``.
    """
    expected_train = set(cfg["data"]["train_seasons"])
    observed = set(train_df["season"].unique())
    assert observed.issubset(expected_train), "Circuit fingerprints must use training data only."

    mean_airtemp = train_df.groupby("event_name")["AirTemp"].mean()
    max_airtemp = float(mean_airtemp.max()) if not mean_airtemp.empty else 1.0
    rows: list[dict[str, float | int | str]] = []

    for circuit_id, circuit_df in train_df.groupby("circuit_id"):
        event_name = str(circuit_df["event_name"].mode().iloc[0])
        slopes: list[float] = []
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            compound_df = circuit_df[circuit_df["Compound"] == compound]
            if len(compound_df) > 5:
                slope, _, _, _, _ = linregress(
                    compound_df["tire_age_laps"],
                    compound_df["lap_time_delta_s"],
                )
                slopes.append(float(slope))
        avg_deg_rate = float(np.mean(slopes)) if slopes else 0.0
        compound_means = (
            circuit_df.groupby("Compound")["lap_time_delta_s"].mean().reindex(["SOFT", "MEDIUM", "HARD"])
        )
        compound_variance = float(compound_means.std(skipna=True)) if compound_means.notna().any() else 0.0
        if not np.isfinite(compound_variance):
            compound_variance = 0.0
        hist_soft_stint_len = float(
            circuit_df[circuit_df["Compound"] == "SOFT"]["TyreLife"].median()
        ) if (circuit_df["Compound"] == "SOFT").any() else 0.0
        race_sc = (
            circuit_df.groupby(["season", "event_name"])["sc_or_vsc_flag"]
            .max()
            .mean()
        )
        rows.append(
            {
                "circuit_id": int(circuit_id),
                "event_name": event_name,
                "avg_deg_rate": avg_deg_rate,
                "compound_variance": compound_variance,
                "track_temp_percentile": float(mean_airtemp.get(event_name, 0.0) / max_airtemp),
                "high_speed_fraction": HIGH_SPEED_FRACTION.get(event_name, 0.35),
                "hist_soft_stint_len": hist_soft_stint_len,
                "sc_vsc_frequency": float(race_sc),
            }
        )
    fp_df = pd.DataFrame(rows).set_index("circuit_id").sort_index()
    fp_df.attrs["feature_ranges"] = (fp_df[FINGERPRINT_COLS].max() - fp_df[FINGERPRINT_COLS].min()).replace(0, 1.0)
    return fp_df


def fingerprint_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute normalized Euclidean distance between two fingerprint vectors.

    Args:
        fp1: First six-dimensional fingerprint vector.
        fp2: Second six-dimensional fingerprint vector.

    Returns:
        Normalized Euclidean distance.
    """
    fp1 = np.asarray(fp1, dtype=float)
    fp2 = np.asarray(fp2, dtype=float)
    stacked = np.vstack([fp1, fp2])
    ranges = np.ptp(stacked, axis=0)
    ranges[ranges == 0.0] = 1.0
    return float(np.linalg.norm((fp1 - fp2) / ranges))


def save_fingerprints(fp_df: pd.DataFrame, path: str) -> None:
    """Persist circuit fingerprints to parquet.

    Args:
        fp_df: Fingerprint dataframe.
        path: Output parquet path.

    Returns:
        None.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    persist_df = fp_df.copy()
    persist_df.attrs = {}
    persist_df.to_parquet(output_path)
    LOGGER.info("Saved circuit fingerprints to %s", output_path)


def load_fingerprints(path: str) -> pd.DataFrame:
    """Load circuit fingerprints from parquet.

    Args:
        path: Input parquet path.

    Returns:
        Fingerprint dataframe.
    """
    return pd.read_parquet(Path(path))
