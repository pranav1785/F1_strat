"""Central metric registry for all model families."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_score,
    recall_score,
)


def compute_all_lstm_metrics(y_true, y_pred) -> dict:
    """Single function that returns all LSTM metrics at once."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "pit_now_recall": float(recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)),
        "pit_now_precision": float(precision_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)),
        "pit_soon_recall": float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "stay_out_precision": float(precision_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def _pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Compute quantile pinball loss."""
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def compute_all_tft_metrics(y_true, y_pred_p10, y_pred_p50, y_pred_p90) -> dict:
    """Single function that returns all TFT metrics at once."""
    return {
        "p50_mae_overall": float(mean_absolute_error(y_true, y_pred_p50)),
        "p90_coverage": float(np.mean(np.asarray(y_true) <= np.asarray(y_pred_p90))),
        "p10_coverage": float(np.mean(np.asarray(y_true) >= np.asarray(y_pred_p10))),
        "pinball_loss": float(
            np.mean(
                [
                    _pinball_loss(y_true, y_pred_p10, 0.1),
                    _pinball_loss(y_true, y_pred_p50, 0.5),
                    _pinball_loss(y_true, y_pred_p90, 0.9),
                ]
            )
        ),
    }


def compute_xgb_metrics(y_true, y_pred, shap_heuristic_results: dict) -> dict:
    """Single function that returns all XGBoost metrics at once."""
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    return {
        "accuracy": accuracy,
        "heuristics_recovered": shap_heuristic_results.get("summary", "0/4 heuristics recovered"),
        "shap_heuristics": shap_heuristic_results,
    }


def print_metric_report(all_metrics: dict[str, Any], cfg: dict) -> None:
    """Pretty-print all metrics with target vs achieved columns."""
    target_map = {
        "p50_mae_overall": cfg["eval"]["tft_mae_target"],
        "p90_coverage": cfg["eval"]["tft_p90_coverage_target"],
        "cliff_lead_time_mean": cfg["eval"]["tft_cliff_lead_target"],
        "macro_f1": cfg["eval"]["lstm_macro_f1_target"],
        "pit_now_recall": cfg["eval"]["lstm_pit_now_recall_target"],
        "pit_now_precision": cfg["eval"]["lstm_pit_now_precision_target"],
        "mcc": cfg["eval"]["lstm_mcc_target"],
        "accuracy": cfg["eval"]["xgb_accuracy_target"],
    }
    print(f"{'metric':<24} {'target':<12} {'achieved':<12}")
    print("-" * 52)
    for key, value in all_metrics.items():
        if isinstance(value, (dict, list, np.ndarray)):
            continue
        target = target_map.get(key, "-")
        print(f"{key:<24} {target!s:<12} {value:<12.4f}" if isinstance(value, (int, float)) else f"{key:<24} {target!s:<12} {value}")
    print(json.dumps({k: v for k, v in all_metrics.items() if isinstance(v, (int, float, str))}, indent=2))
