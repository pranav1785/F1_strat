"""Smoke tests for the LightGBM pit-window module."""

from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from models.pit_classifier.train import (
    build_pit_window_dataset,
    evaluate_pit_lightgbm,
    find_thresh_max_f1,
    load_pit_model,
    train_pit_lightgbm,
)


def _frame(n: int, event_name: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    tire_age = np.arange(1, n + 1, dtype=float)
    pit_label = np.where(tire_age >= n - 2, 2, np.where(tire_age >= n - 5, 1, 0))
    return pd.DataFrame(
        {
            "event_name": event_name,
            "pit_label": pit_label,
            "loss_weight": 1.0,
            "tire_age_laps": tire_age,
            "tire_age_sq": tire_age**2,
            "compound_enc": rng.integers(0, 3, size=n),
            "circuit_id_enc": 0,
            "team_id_enc": rng.integers(0, 4, size=n),
            "fuel_load_kg": 100.0 - tire_age,
            "lap_norm": tire_age / n,
            "gap_ahead_s": rng.normal(2.0, 0.1, size=n),
            "undercut_threat": (tire_age > n - 6).astype(int),
            "sc_or_vsc_flag": 0,
            "lap_time_delta_s": tire_age * 0.03,
            "deg_velocity": 0.03,
            "deg_acceleration": 0.0,
        }
    )


def test_build_pit_window_dataset_maps_warning_and_pit_labels_to_positive() -> None:
    cfg = {"lgb_pit": {"features": ["tire_age_laps", "compound_enc"]}}
    X, y, sub = build_pit_window_dataset(_frame(12, "Bahrain"), cfg)

    assert X.columns.tolist() == ["tire_age_laps", "compound_enc"]
    assert len(sub) == 12
    assert y.tolist().count(1) == 6


def test_find_thresh_max_f1_returns_useful_cutoff() -> None:
    y = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.2, 0.7, 0.9])

    assert 0.2 < find_thresh_max_f1(y, probabilities) <= 0.7


def test_train_pit_lightgbm_persists_and_evaluates() -> None:
    model_path = Path(f"checkpoints/test_lgb_pit_{uuid.uuid4().hex}.pkl")
    calibration_path = Path(f"checkpoints/test_lgb_pit_{uuid.uuid4().hex}.json")
    cfg = {
        "lgb_pit": {
            "features": [
                "tire_age_laps",
                "tire_age_sq",
                "compound_enc",
                "circuit_id_enc",
                "team_id_enc",
                "fuel_load_kg",
                "lap_norm",
                "gap_ahead_s",
                "undercut_threat",
                "sc_or_vsc_flag",
                "lap_time_delta_s",
                "deg_velocity",
                "deg_acceleration",
            ],
            "n_estimators": 20,
            "learning_rate": 0.1,
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 2,
            "early_stopping_rounds": 5,
            "seed": 42,
        },
        "paths": {
            "lgb_pit_model": str(model_path),
            "lgb_pit_calibration": str(calibration_path),
        },
        "viz": {"degradation_circuits": ["Bahrain", "Monaco"]},
    }

    train_df = pd.concat([_frame(30, "Bahrain"), _frame(30, "Monaco")], ignore_index=True)
    val_df = pd.concat([_frame(24, "Bahrain"), _frame(24, "Monaco")], ignore_index=True)
    model = train_pit_lightgbm(train_df, val_df, cfg)
    loaded = load_pit_model(cfg)
    metrics = evaluate_pit_lightgbm(model, val_df, cfg)

    assert model_path.exists()
    assert calibration_path.exists()
    assert loaded.feature_names == model.feature_names
    assert metrics["average_precision"] >= 0.0
    assert set(model.circuit_thresholds) == {"Bahrain", "Monaco"}
