"""SHAP analysis helpers for the compound recommender."""

from __future__ import annotations

import numpy as np
import shap
import xgboost as xgb


def _normalize_xgb_contribs(contribs: np.ndarray, n_features: int) -> np.ndarray:
    """Normalize XGBoost pred_contribs output to ``(n_samples, n_features, n_classes)``."""
    if contribs.ndim == 2:
        return contribs[:, :n_features][:, :, None]
    if contribs.ndim == 3:
        if contribs.shape[1] == n_features + 1:
            return np.transpose(contribs[:, :n_features, :], (0, 1, 2))
        if contribs.shape[2] == n_features + 1:
            return np.transpose(contribs[:, :, :n_features], (0, 2, 1))
    raise ValueError(f"Unexpected pred_contribs shape: {contribs.shape}")


def compute_shap_values(model, X_test: np.ndarray, feature_names: list) -> np.ndarray:
    """Compute SHAP values using TreeExplainer.

    Args:
        model: Trained tree model.
        X_test: Test feature matrix.
        feature_names: Feature names corresponding to columns in ``X_test``.

    Returns:
        SHAP values stacked to shape ``(n_samples, n_features, n_classes)``.
    """
    n_features = len(feature_names)
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        dtest = xgb.DMatrix(X_test, feature_names=feature_names)
        contribs = booster.predict(dtest, pred_contribs=True, strict_shape=True)
        return _normalize_xgb_contribs(np.asarray(contribs), n_features)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            return np.stack(shap_values, axis=-1)
        if shap_values.ndim == 2:
            return shap_values[:, :, None]
        return shap_values
    except ValueError as exc:
        if "base_score" not in str(exc):
            raise
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        dtest = xgb.DMatrix(X_test, feature_names=feature_names)
        contribs = booster.predict(dtest, pred_contribs=True, strict_shape=True)
        return _normalize_xgb_contribs(np.asarray(contribs), n_features)


def check_heuristic_recovery(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    circuit_indices: dict,
    feature_names: list,
    cfg: dict,
) -> dict:
    """Check whether SHAP patterns recover domain heuristics."""
    del X_test
    SOFT_CLASS_IDX = 2
    MEDIUM_CLASS_IDX = 1
    results = {}
    feat_idx = {name: i for i, name in enumerate(feature_names)}
    for circuit, indices in circuit_indices.items():
        circuit_shap = shap_values[indices, :, :]
        mean_shap = circuit_shap.mean(axis=0)
        if circuit == "Spain":
            val = mean_shap[feat_idx["avg_deg_rate"], SOFT_CLASS_IDX]
            threshold = cfg["eval"]["heuristic_circuits"]["barcelona_soft_deg_threshold"]
            results["Barcelona"] = {"passed": bool(val < -threshold), "shap_value": float(val)}
        elif circuit == "Monaco":
            val = mean_shap[feat_idx["hist_soft_stint_len"], SOFT_CLASS_IDX]
            threshold = cfg["eval"]["heuristic_circuits"]["monaco_soft_positive_threshold"]
            results["Monaco"] = {"passed": bool(val > threshold), "shap_value": float(val)}
        elif circuit == "Singapore":
            val = mean_shap[feat_idx["sc_vsc_frequency"], MEDIUM_CLASS_IDX]
            threshold = cfg["eval"]["heuristic_circuits"]["singapore_medium_threshold"]
            results["Singapore"] = {"passed": bool(val > threshold), "shap_value": float(val)}
        elif circuit == "Qatar":
            val1 = mean_shap[feat_idx["avg_deg_rate"], SOFT_CLASS_IDX]
            val2 = mean_shap[feat_idx["circuit_softness"], SOFT_CLASS_IDX]
            threshold = cfg["eval"]["heuristic_circuits"]["qatar_soft_neg_threshold"]
            results["Qatar"] = {
                "passed": bool(val1 < -threshold and val2 < -threshold),
                "shap_avg_deg": float(val1),
                "shap_softness": float(val2),
            }
    n_passed = sum(r["passed"] for r in results.values())
    results["summary"] = f"{n_passed}/4 heuristics recovered"
    return results
