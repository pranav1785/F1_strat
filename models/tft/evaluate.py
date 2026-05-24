"""Evaluation helpers for TFT predictions."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from eval.metrics import compute_all_tft_metrics

LOGGER = logging.getLogger(__name__)


def compute_cliff_lead_time(predictions: pd.DataFrame, actuals: pd.DataFrame, cfg: dict) -> tuple[float, float]:
    """Compute warning lead time for degradation cliffs.

    Args:
        predictions: Prediction dataframe with ``driver_stint_id``, ``lap_number_global``, and ``p90``.
        actuals: Actual dataframe with ``driver_stint_id``, ``lap_number_global``, and target values.
        cfg: Project configuration.

    Returns:
        Mean and standard deviation of cliff lead time.
    """
    threshold = cfg["tft"]["cliff_threshold_s"]
    lead_times: list[float] = []
    merged = actuals.merge(
        predictions[["driver_stint_id", "lap_number_global", "p90"]],
        on=["driver_stint_id", "lap_number_global"],
        how="left",
    )
    for _, stint_df in merged.groupby("driver_stint_id"):
        actual_cliffs = stint_df[stint_df[cfg["tft"]["target_col"]] > threshold]
        if actual_cliffs.empty:
            continue
        cliff_lap = float(actual_cliffs["lap_number_global"].iloc[0])
        warned = stint_df[stint_df["p90"] > threshold]
        warning_lap = float(warned["lap_number_global"].iloc[0]) if not warned.empty else cliff_lap
        lead_times.append(max(cliff_lap - warning_lap, 0.0))
    if not lead_times:
        return 0.0, 0.0
    return float(np.mean(lead_times)), float(np.std(lead_times))


def evaluate_tft(model, test_dataloader, cfg: dict) -> dict:
    """Evaluate the TFT on the test set.

    Args:
        model: Trained TFT model or wrapper.
        test_dataloader: Test dataloader.
        cfg: Project configuration.

    Returns:
        TFT metric dictionary.
    """
    predictor = model.base_model if hasattr(model, "base_model") else model
    raw_predictions = predictor.predict(test_dataloader, mode="raw", return_x=True)
    preds = raw_predictions.output.prediction.detach().cpu().numpy()
    target = raw_predictions.x["decoder_target"].detach().cpu().numpy()
    compounds = raw_predictions.x["decoder_cat"][:, :, 1].detach().cpu().numpy() if "decoder_cat" in raw_predictions.x else None

    y_true = target.reshape(-1)
    y_pred_p10 = preds[..., 0].reshape(-1)
    y_pred_p50 = preds[..., 1].reshape(-1)
    y_pred_p90 = preds[..., 2].reshape(-1)

    mask = np.ones_like(y_true, dtype=bool)
    if "decoder_cont" in raw_predictions.x:
        decoder_cont = raw_predictions.x["decoder_cont"].detach().cpu().numpy()
        sc_idx = cfg["tft"]["known_time_varying_reals"] + cfg["tft"]["unknown_time_varying_reals"]
        if "safety_car_flag" in sc_idx:
            feat_idx = sc_idx.index("safety_car_flag")
            mask = decoder_cont[..., feat_idx].reshape(-1) == 0
    assert mask.any(), "SC/VSC masking removed all TFT evaluation samples."

    metrics = compute_all_tft_metrics(
        y_true[mask],
        y_pred_p10[mask],
        y_pred_p50[mask],
        y_pred_p90[mask],
    )

    result = {
        "p50_mae_overall": float(metrics["p50_mae_overall"]),
        "p50_mae_soft": float("nan"),
        "p50_mae_medium": float("nan"),
        "p50_mae_hard": float("nan"),
        "p90_coverage": float(metrics["p90_coverage"]),
        "p10_coverage": float(metrics["p10_coverage"]),
        "pinball_loss": float(metrics["pinball_loss"]),
        "cliff_lead_time_mean": 0.0,
        "cliff_lead_time_std": 0.0,
        "per_circuit_mae": {},
    }
    if compounds is not None:
        compound_flat = compounds.reshape(-1)[mask]
        for compound_name, compound_idx in {"hard": 0, "medium": 1, "soft": 2}.items():
            compound_mask = compound_flat == compound_idx
            if compound_mask.any():
                result[f"p50_mae_{compound_name}"] = float(
                    np.mean(np.abs(y_true[mask][compound_mask] - y_pred_p50[mask][compound_mask]))
                )

    prediction_df = pd.DataFrame(
        {
            "driver_stint_id": np.repeat(np.arange(preds.shape[0]), preds.shape[1]),
            "lap_number_global": np.tile(np.arange(preds.shape[1]), preds.shape[0]),
            "p90": preds[..., 2].reshape(-1),
        }
    )
    actual_df = pd.DataFrame(
        {
            "driver_stint_id": np.repeat(np.arange(target.shape[0]), target.shape[1]),
            "lap_number_global": np.tile(np.arange(target.shape[1]), target.shape[0]),
            cfg["tft"]["target_col"]: target.reshape(-1),
        }
    )
    lead_mean, lead_std = compute_cliff_lead_time(prediction_df, actual_df, cfg)
    result["cliff_lead_time_mean"] = lead_mean
    result["cliff_lead_time_std"] = lead_std
    return result
