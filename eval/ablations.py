"""LightGBM pit-window ablation study orchestration."""

from __future__ import annotations

import copy

import pandas as pd

from models.pit_classifier.train import evaluate_pit_lightgbm, train_pit_lightgbm

ABLATIONS = [
    {
        "name": "remove_degradation",
        "description": "Remove lap-time degradation derivatives",
        "features": ["lap_time_delta_s", "deg_velocity", "deg_acceleration"],
    },
    {
        "name": "remove_undercut_threat",
        "description": "Remove undercut threat signal",
        "features": ["undercut_threat"],
    },
    {
        "name": "remove_gap_features",
        "description": "Remove gap and traffic features",
        "features": ["gap_ahead_s", "gap_ahead_trend"],
    },
    {
        "name": "remove_circuit_context",
        "description": "Remove circuit identity and prior context",
        "features": ["circuit_id_enc", "circuit_softness", "historical_sc_prob"],
    },
    {
        "name": "all_teams_vs_top4",
        "description": "Train on all teams instead of top-4 only",
        "modify_cfg": lambda cfg: cfg["data"].update({"top4_filter": False}),
    },
]


def _drop_features(cfg: dict, features: list[str]) -> None:
    current = cfg["lgb_pit"]["features"]
    cfg["lgb_pit"]["features"] = [feature for feature in current if feature not in features]


def run_all_ablations(train_data, val_data, test_data, base_cfg: dict) -> pd.DataFrame:
    """Run LightGBM feature ablations and report metric deltas."""
    base_model = train_pit_lightgbm(train_data, val_data, base_cfg)
    base_metrics = evaluate_pit_lightgbm(base_model, test_data, base_cfg)
    rows = []
    for ablation in ABLATIONS:
        cfg = copy.deepcopy(base_cfg)
        if "modify_cfg" in ablation:
            ablation["modify_cfg"](cfg)
        if "features" in ablation:
            _drop_features(cfg, ablation["features"])
        model = train_pit_lightgbm(train_data, val_data, cfg)
        metrics = evaluate_pit_lightgbm(model, test_data, cfg)
        rows.append(
            {
                "ablation_name": ablation["name"],
                "description": ablation["description"],
                "average_precision": metrics["average_precision"],
                "auc": metrics["auc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "delta_average_precision": metrics["average_precision"] - base_metrics["average_precision"],
                "delta_f1": metrics["f1"] - base_metrics["f1"],
            }
        )
    return pd.DataFrame(rows)
