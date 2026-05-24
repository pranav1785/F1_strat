"""Unit tests for circuit fingerprints."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from data.circuit_fingerprint import FINGERPRINT_COLS, compute_circuit_fingerprints, fingerprint_distance, save_fingerprints


def _fingerprint_df() -> pd.DataFrame:
    rows = []
    specs = {
        "Monaco": {"circuit_id": 0, "slope": 0.02, "air": 25.0, "sc": 1},
        "Qatar": {"circuit_id": 1, "slope": 0.30, "air": 35.0, "sc": 0},
        "Spain": {"circuit_id": 2, "slope": 0.15, "air": 30.0, "sc": 0},
    }
    for event, spec in specs.items():
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            for lap in range(1, 9):
                rows.append(
                    {
                        "season": 2021,
                        "event_name": event,
                        "circuit_id": spec["circuit_id"],
                        "Compound": compound,
                        "tire_age_laps": float(lap),
                        "lap_time_delta_s": spec["slope"] * lap,
                        "AirTemp": spec["air"],
                        "TyreLife": lap + (0 if compound == "SOFT" else 2),
                        "sc_or_vsc_flag": spec["sc"],
                    }
                )
    return pd.DataFrame(rows)


def test_all_fingerprint_dimensions_finite() -> None:
    cfg = {"data": {"train_seasons": [2021]}}
    fp = compute_circuit_fingerprints(_fingerprint_df(), cfg)
    assert np.isfinite(fp[FINGERPRINT_COLS].to_numpy()).all()


def test_monaco_has_lowest_avg_deg_rate() -> None:
    cfg = {"data": {"train_seasons": [2021]}}
    fp = compute_circuit_fingerprints(_fingerprint_df(), cfg)
    event_map = fp.reset_index().set_index("event_name")
    assert event_map.loc["Monaco", "avg_deg_rate"] == event_map["avg_deg_rate"].min()


def test_qatar_has_highest_avg_deg_rate() -> None:
    cfg = {"data": {"train_seasons": [2021]}}
    fp = compute_circuit_fingerprints(_fingerprint_df(), cfg)
    event_map = fp.reset_index().set_index("event_name")
    assert event_map.loc["Qatar", "avg_deg_rate"] == event_map["avg_deg_rate"].max()


def test_fingerprint_distance_zero_for_identical_vectors() -> None:
    vec = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    assert fingerprint_distance(vec, vec) == 0.0


def test_fingerprint_distance_is_symmetric() -> None:
    a = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    b = np.array([2, 3, 4, 5, 6, 7], dtype=float)
    assert fingerprint_distance(a, b) == fingerprint_distance(b, a)


def test_save_fingerprints_ignores_non_serializable_attrs() -> None:
    cfg = {"data": {"train_seasons": [2021]}}
    fp = compute_circuit_fingerprints(_fingerprint_df(), cfg)
    output_path = Path("processed") / "test_fingerprints_tmp.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fingerprints(fp, str(output_path))
    assert output_path.exists()
