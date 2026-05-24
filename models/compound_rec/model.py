"""XGBoost model builder for compound recommendation."""

from __future__ import annotations

import xgboost as xgb


def build_xgb_model(cfg: dict) -> xgb.XGBClassifier:
    """Build the configured XGBoost classifier."""
    return xgb.XGBClassifier(
        n_estimators=cfg["xgboost"]["n_estimators"],
        max_depth=cfg["xgboost"]["max_depth"],
        learning_rate=cfg["xgboost"]["learning_rate"],
        subsample=cfg["xgboost"]["subsample"],
        colsample_bytree=cfg["xgboost"]["colsample_bytree"],
        min_child_weight=cfg["xgboost"]["min_child_weight"],
        gamma=cfg["xgboost"]["gamma"],
        tree_method=cfg["xgboost"]["tree_method"],
        eval_metric=cfg["xgboost"]["eval_metric"],
        early_stopping_rounds=cfg["xgboost"]["early_stopping_rounds"],
        random_state=cfg["xgboost"]["seed"],
        n_jobs=-1,
    )


def train_xgb(X_train, y_train, X_val, y_val, cfg):
    """Train and persist the XGBoost classifier."""
    model = build_xgb_model(cfg)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    model.save_model(cfg["paths"]["xgb_model"])
    return model
