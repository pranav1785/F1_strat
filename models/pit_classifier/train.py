"""Training utilities for the LightGBM pit-window classifier."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from models.pit_classifier.model import PitWindowLightGBM

LOGGER = logging.getLogger(__name__)


DEFAULT_PIT_FEATURES = [
    "tire_age_laps",
    "tire_age_sq",
    "compound_enc",
    "circuit_id_enc",
    "team_id_enc",
    "fuel_load_kg",
    "lap_norm",
    "lap_count_norm",
    "TrackTemp",
    "AirTemp",
    "Humidity",
    "track_temp_category",
    "gap_ahead_s",
    "gap_ahead_trend",
    "undercut_threat",
    "sc_or_vsc_flag",
    "Position",
    "starting_position_norm",
    "circuit_softness",
    "historical_sc_prob",
    "lap_time_delta_s",
    "deg_velocity",
    "deg_acceleration",
]


def _cfg(cfg: dict) -> dict:
    """Return the LightGBM pit config, accepting legacy notebook names."""
    return cfg.get("lgb_pit", cfg.get("lstm", {}))


def pit_feature_names(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Resolve configured LightGBM pit features that exist in a dataframe."""
    configured = _cfg(cfg).get("features", DEFAULT_PIT_FEATURES)
    return [feature for feature in configured if feature in df.columns]


def build_pit_window_dataset(
    df: pd.DataFrame,
    cfg: dict,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Build the point-in-time binary pit-window dataset.

    The notebook's v3 target is binary: a positive row means a pit stop is due
    now or within the warning horizon. The repository already encodes this as
    ``pit_label`` 1 or 2, so both labels become class 1 for LightGBM.
    """
    features = feature_names or pit_feature_names(df, cfg)
    if not features:
        raise ValueError("No LightGBM pit features were found in the dataframe.")
    loss_weight_col = "loss_weight" if "loss_weight" in df.columns else None
    mask = df["pit_label"].isin([0, 1, 2])
    if loss_weight_col is not None:
        mask &= df[loss_weight_col] > 0
    sub = df.loc[mask].copy()
    X = sub[features].fillna(0.0)
    y = sub["pit_label"].isin([1, 2]).astype(int).to_numpy()
    return X, y, sub


def _build_estimator(cfg: dict) -> lgb.LGBMClassifier:
    params = _cfg(cfg)
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", 5),
        num_leaves=params.get("num_leaves", 31),
        learning_rate=params.get("learning_rate", 0.03),
        min_child_samples=params.get("min_child_samples", 25),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 2.0),
        class_weight=params.get("class_weight", "balanced"),
        random_state=params.get("seed", 42),
        verbose=-1,
    )


def find_thresh_max_f1(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the probability threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2.0 * precision * recall / (precision + recall + 1.0e-9)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def tune_circuit_thresholds(
    validation_frame: pd.DataFrame,
    y_val: np.ndarray,
    probabilities: np.ndarray,
    cfg: dict,
) -> tuple[dict[str, float], float]:
    """Tune a global and per-circuit F1 threshold on validation data."""
    global_threshold = find_thresh_max_f1(y_val, probabilities)
    circuit_col = "event_name"
    circuits = validation_frame[circuit_col].astype(str).unique().tolist() if circuit_col in validation_frame.columns else []
    configured = cfg.get("viz", {}).get("degradation_circuits", [])
    thresholds: dict[str, float] = {}
    for circuit in sorted(set(circuits) | set(configured)):
        mask = validation_frame[circuit_col].astype(str).to_numpy() == circuit if circuit_col in validation_frame.columns else np.zeros(len(y_val), dtype=bool)
        if mask.sum() == 0 or y_val[mask].sum() == 0:
            thresholds[circuit] = global_threshold
            continue
        thresholds[circuit] = find_thresh_max_f1(y_val[mask], probabilities[mask])
    return thresholds, global_threshold


def save_pit_calibration(model: PitWindowLightGBM, cfg: dict) -> None:
    """Persist calibration metadata as JSON for auditability."""
    output_path = Path(cfg["paths"].get("lgb_pit_calibration", cfg["paths"].get("lstm_calibration", "./checkpoints/lgb_pit_calibration.json")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": model.feature_names,
        "global_threshold": model.global_threshold,
        "circuit_thresholds": model.circuit_thresholds,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_pit_model(cfg: dict) -> PitWindowLightGBM:
    """Load the persisted LightGBM pit-window model."""
    model_path = Path(cfg["paths"].get("lgb_pit_model", cfg["paths"].get("lstm_checkpoint", "./checkpoints/lgb_pit.pkl")))
    if not model_path.exists():
        raise FileNotFoundError(f"LightGBM pit model not found at {model_path}")
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def train_pit_lightgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
) -> PitWindowLightGBM:
    """Train, calibrate, threshold-tune, and persist the LightGBM pit model."""
    features = pit_feature_names(train_df, cfg)
    X_train, y_train, _ = build_pit_window_dataset(train_df, cfg, features)
    X_val, y_val, val_sub = build_pit_window_dataset(val_df, cfg, features)
    estimator = _build_estimator(cfg)
    callbacks = [
        lgb.early_stopping(_cfg(cfg).get("early_stopping_rounds", 50), verbose=False),
    ]
    estimator.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)

    val_raw = estimator.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw, y_val)
    val_cal = calibrator.transform(val_raw)
    thresholds, global_threshold = tune_circuit_thresholds(val_sub, y_val, val_cal, cfg)

    model = PitWindowLightGBM(
        estimator=estimator,
        calibrator=calibrator,
        feature_names=features,
        circuit_thresholds=thresholds,
        global_threshold=global_threshold,
    )
    model_path = Path(cfg["paths"].get("lgb_pit_model", cfg["paths"].get("lstm_checkpoint", "./checkpoints/lgb_pit.pkl")))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    save_pit_calibration(model, cfg)
    LOGGER.info("Saved LightGBM pit model to %s", model_path)
    return model


def evaluate_pit_lightgbm(
    model: PitWindowLightGBM,
    df: pd.DataFrame,
    cfg: dict,
) -> dict:
    """Evaluate calibrated binary pit-window predictions."""
    X, y_true, sub = build_pit_window_dataset(df, cfg, model.feature_names)
    probabilities = model.predict_proba(X)
    labels = model.predict_labels(sub)
    y_pred = (labels == 2).astype(int)
    has_pos = y_true.sum() > 0
    has_neg = (y_true == 0).sum() > 0
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)) if has_pos else 0.0,
        "auc": float(roc_auc_score(y_true, probabilities)) if has_pos and has_neg else 0.5,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision),
        "recall": float(recall),
        "brier": float(brier_score_loss(y_true, probabilities)) if len(y_true) else 0.0,
        "global_threshold": float(model.global_threshold),
        "circuit_thresholds": model.circuit_thresholds,
    }


# Backwards-compatible names for the previous module-2 CLI/tests.
train_lstm = train_pit_lightgbm
evaluate_lstm = evaluate_pit_lightgbm
load_lstm_calibration = lambda cfg: json.loads(Path(cfg["paths"].get("lgb_pit_calibration", cfg["paths"].get("lstm_calibration", "./checkpoints/lgb_pit_calibration.json"))).read_text(encoding="utf-8")) if Path(cfg["paths"].get("lgb_pit_calibration", cfg["paths"].get("lstm_calibration", "./checkpoints/lgb_pit_calibration.json"))).exists() else None
