"""Main orchestration CLI for the F1 strategy ML repository."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer

from data.circuit_fingerprint import compute_circuit_fingerprints, save_fingerprints
from data.dataset import (
    build_tft_dataset,
    build_xgb_dataset,
)
from data.features import run_full_feature_pipeline
from data.ingest import enable_cache, load_all_seasons, load_config, load_qualifying_sessions
from data.labels import apply_cleaning_masks, encode_pit_labels, filter_top4_constructors
from eval.ablations import run_all_ablations
from eval.counterfactual import (
    aggregate_counterfactual,
    aggregate_counterfactual_by_entity,
    counterfactual_delta_position,
)
from eval.loco_cv import run_loco_cv
from eval.metrics import compute_xgb_metrics, print_metric_report
from models.compound_rec.model import train_xgb
from models.compound_rec.shap_analysis import check_heuristic_recovery, compute_shap_values
from models.pit_classifier.model import PitWindowLightGBM
from models.pit_classifier.train import (
    evaluate_pit_lightgbm,
    load_pit_model,
    train_pit_lightgbm,
)
from models.tft.evaluate import evaluate_tft
from models.tft.train import run_lr_finder, train_tft
from viz.degradation_curves import plot_degradation_curves
from viz.delta_position_plot import plot_delta_position_distribution
from viz.shap_waterfall import plot_shap_waterfall
from viz.strategy_overlay import plot_strategy_overlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def _assert_season_loaded(df: pd.DataFrame, season_name: str) -> None:
    """Fail fast when a season was not fully loaded.

    Args:
        df: Loaded season dataframe.
        season_name: Human-readable season label for error messages.

    Returns:
        None.

    Raises:
        RuntimeError: If the season is empty or was interrupted by rate limiting.
    """
    status = df.attrs.get("load_status", {})
    if df.empty:
        reason = "FastF1/Ergast rate limiting or upstream session failures" if status.get("rate_limited") else "session loading returned no rows"
        raise RuntimeError(
            f"{season_name} data is empty. Likely cause: {reason}. "
            "Wait for the Ergast rate-limit window to reset, then rerun `python pipeline.py --mode data`."
        )
    if status.get("rate_limited"):
        raise RuntimeError(
            f"{season_name} data loaded only partially before hitting the Ergast rate limit "
            f"(loaded_sessions={status.get('loaded_sessions', 0)}). "
            "Stop here, let the cache cool down / rate window reset, and rerun `python pipeline.py --mode data`."
        )


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist a dataframe using parquet when possible, CSV otherwise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    persist_df = df.copy()
    persist_df.attrs = {}
    if path.suffix == ".parquet":
        persist_df.to_parquet(path)
    else:
        persist_df.to_csv(path, index=False)


def _load_dataframe(path: Path) -> pd.DataFrame:
    """Load a dataframe from parquet or CSV."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_cached_raw_splits(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Load previously saved raw train/val/test splits when all are available."""
    paths = {
        "train": output_dir / "train_raw.parquet",
        "val": output_dir / "val_raw.parquet",
        "test": output_dir / "test_raw.parquet",
    }
    if not all(path.exists() for path in paths.values()):
        return None
    LOGGER.info("Loading cached raw splits from %s", output_dir.resolve())
    return tuple(_load_dataframe(paths[name]) for name in ["train", "val", "test"])


def _load_cached_feature_splits(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Load previously saved feature train/val/test splits when all are available."""
    paths = {
        "train": output_dir / "train_features.parquet",
        "val": output_dir / "val_features.parquet",
        "test": output_dir / "test_features.parquet",
    }
    if not all(path.exists() for path in paths.values()):
        return None
    LOGGER.info("Loading cached feature splits from %s", output_dir.resolve())
    return tuple(_load_dataframe(paths[name]) for name in ["train", "val", "test"])


def _load_cached_qualifying(path: Path) -> pd.DataFrame | None:
    """Load cached qualifying sessions when available."""
    if not path.exists():
        return None
    LOGGER.info("Loading cached qualifying sessions from %s", path.resolve())
    return _load_dataframe(path)


def _prepare_features(df: pd.DataFrame, cfg: dict, historical_reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run full feature engineering with a training-only historical SC reference."""
    work = df.copy()
    if historical_reference is not None:
        work.attrs["historical_sc_reference"] = historical_reference
        if historical_reference.attrs.get("encoders") is not None:
            work.attrs["reference_encoders"] = historical_reference.attrs["encoders"]
        work.attrs["train_seasons"] = cfg["data"]["train_seasons"]
    else:
        work.attrs["train_seasons"] = sorted(df["season"].unique().tolist())
    return run_full_feature_pipeline(work, cfg)


def _prepare_labeled_training(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply label creation and training-time-only filters."""
    labeled = encode_pit_labels(df)
    cleaned = apply_cleaning_masks(labeled, cfg)
    filtered = filter_top4_constructors(cleaned, cfg)
    return filtered


def _load_pit_checkpoint(cfg: dict) -> PitWindowLightGBM:
    """Load the persisted LightGBM pit-window model for inference."""
    return load_pit_model(cfg)


def _predict_pit_labels(model: PitWindowLightGBM, df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Predict pit labels aligned to the original dataframe index."""
    del cfg
    predictions = pd.Series(0, index=df.index, dtype=int)
    labels = model.predict_labels(df)
    predictions.loc[df.index] = labels.astype(int)
    return predictions


def _load_tft_checkpoint(cfg: dict) -> TemporalFusionTransformer:
    """Load the persisted TFT checkpoint for inference."""
    checkpoint_path = Path(cfg["paths"]["tft_checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"TFT checkpoint not found at {checkpoint_path}")
    return TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path))


def _predict_tft_quantiles(
    model: TemporalFusionTransformer,
    train_ds,
    test_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Predict TFT quantiles keyed by driver stint id and lap number."""
    test_ds = build_tft_dataset(test_df, cfg, reference_dataset=train_ds)
    raw_predictions = model.predict(
        test_ds,
        mode="raw",
        return_x=True,
        return_index=True,
        return_decoder_lengths=True,
        batch_size=cfg["tft"]["batch_size"],
        trainer_kwargs={"logger": False, "enable_progress_bar": False},
    )
    decoder_lengths = raw_predictions.decoder_lengths.detach().cpu().numpy()
    decoder_time_idx = raw_predictions.x["decoder_time_idx"].detach().cpu().numpy()
    quantile_predictions = raw_predictions.output.prediction.detach().cpu().numpy()
    prediction_index = raw_predictions.index.reset_index(drop=True)

    records: list[dict[str, object]] = []
    for row_idx, length in enumerate(decoder_lengths):
        driver_stint_id = str(prediction_index.loc[row_idx, "driver_stint_id"])
        for step in range(int(length)):
            records.append(
                {
                    "driver_stint_id": driver_stint_id,
                    "lap_number_global": int(decoder_time_idx[row_idx, step]),
                    "p10": float(quantile_predictions[row_idx, step, 0]),
                    "p50": float(quantile_predictions[row_idx, step, 1]),
                    "p90": float(quantile_predictions[row_idx, step, 2]),
                }
            )
    if not records:
        return pd.DataFrame(columns=["driver_stint_id", "LapNumber", "event_name", "p10", "p50", "p90"])

    pred_df = pd.DataFrame(records)
    session_keys = ["season", "event_name", "session_type"]
    session_offsets = {
        key: idx * 1000 for idx, key in enumerate(test_df[session_keys].drop_duplicates().itertuples(index=False, name=None))
    }
    lap_lookup = test_df.copy()
    lap_lookup["lap_number_global"] = lap_lookup.apply(
        lambda row: int(row["LapNumber"]) + session_offsets[(row["season"], row["event_name"], row["session_type"])],
        axis=1,
    )
    lap_lookup = lap_lookup[["driver_stint_id", "lap_number_global", "LapNumber", "event_name"]].drop_duplicates()
    pred_df = pred_df.merge(lap_lookup, on=["driver_stint_id", "lap_number_global"], how="left")
    pred_df = pred_df.dropna(subset=["LapNumber"])
    aggregated = (
        pred_df.groupby(["driver_stint_id", "event_name", "LapNumber"], as_index=False)[["p10", "p50", "p90"]]
        .mean()
        .sort_values(["driver_stint_id", "LapNumber"])
    )
    return aggregated


def _tft_p50_curve_lookup(prediction_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build a per-stint p50 lookup from TFT predictions."""
    if prediction_df.empty:
        return {}
    return {
        stint_id: group.set_index("LapNumber")["p50"]
        for stint_id, group in prediction_df.groupby("driver_stint_id")
    }


def _compute_per_race_deltas(
    labeled_test: pd.DataFrame,
    lstm_pred_labels: pd.Series,
    tft_prediction_df: pd.DataFrame,
    cfg: dict,
) -> list[dict[str, object]]:
    """Compute per-race counterfactual deltas from real model outputs."""
    tft_p50_curves = _tft_p50_curve_lookup(tft_prediction_df)
    per_race_deltas: list[dict[str, object]] = []
    for race, race_df in labeled_test.groupby("event_name"):
        preds = lstm_pred_labels.loc[race_df.index]
        race_curves = {
            stint_id: tft_p50_curves.get(stint_id)
            for stint_id in race_df["driver_stint_id"].dropna().unique()
        }
        delta = counterfactual_delta_position(race, preds, race_curves, race_df, cfg)
        per_race_deltas.append(
            {
                "race_id": race,
                "delta_pos": delta,
                "is_wet": bool(race_df["Compound"].isin(cfg["data"]["wet_compounds"]).any()),
                "is_street": race in cfg["eval"]["street_circuits"],
            }
        )
    return per_race_deltas


def _compute_per_driver_deltas(
    labeled_test: pd.DataFrame,
    lstm_pred_labels: pd.Series,
    tft_prediction_df: pd.DataFrame,
    cfg: dict,
) -> list[dict[str, object]]:
    """Compute per-driver counterfactual deltas from real model outputs."""
    tft_p50_curves = _tft_p50_curve_lookup(tft_prediction_df)
    per_driver_deltas: list[dict[str, object]] = []
    group_keys = ["event_name", "Driver"]
    for (race, driver), driver_df in labeled_test.groupby(group_keys):
        preds = lstm_pred_labels.loc[driver_df.index]
        driver_curves = {
            stint_id: tft_p50_curves.get(stint_id)
            for stint_id in driver_df["driver_stint_id"].dropna().unique()
        }
        delta = counterfactual_delta_position(f"{race}_{driver}", preds, driver_curves, driver_df, cfg)
        team_name = str(driver_df["Team"].dropna().iloc[0]) if driver_df["Team"].notna().any() else "unknown"
        per_driver_deltas.append(
            {
                "driver_id": f"{race}_{driver}",
                "race_id": race,
                "driver": driver,
                "team": team_name,
                "delta_pos": delta,
                "is_wet": bool(driver_df["Compound"].isin(cfg["data"]["wet_compounds"]).any()),
                "is_street": race in cfg["eval"]["street_circuits"],
            }
        )
    return per_driver_deltas


def _compute_per_team_deltas(per_driver_deltas: list[dict[str, object]]) -> list[dict[str, object]]:
    """Roll per-driver deltas up to per-team, per-race summaries."""
    if not per_driver_deltas:
        return []
    df = pd.DataFrame(per_driver_deltas)
    team_records: list[dict[str, object]] = []
    for (race, team), team_df in df.groupby(["race_id", "team"], dropna=False):
        team_records.append(
            {
                "team_id": f"{race}_{team}",
                "race_id": str(race),
                "team": str(team),
                "delta_pos": float(team_df["delta_pos"].sum()),
                "is_wet": bool(team_df["is_wet"].iloc[0]),
                "is_street": bool(team_df["is_street"].iloc[0]),
            }
        )
    return team_records


def _team_matches_focus(team_name: str, focus_team: str | None) -> bool:
    """Match a team name against the configured focus team using substring checks."""
    if not focus_team:
        return False
    team_norm = str(team_name).strip().lower()
    focus_norm = str(focus_team).strip().lower()
    return bool(team_norm) and (team_norm in focus_norm or focus_norm in team_norm)


def _filter_records_to_focus_team(records: list[dict[str, object]], focus_team: str | None) -> list[dict[str, object]]:
    """Keep only records that belong to the configured focus team."""
    if not focus_team:
        return []
    return [record for record in records if _team_matches_focus(record.get("team", ""), focus_team)]


def _flatten_counterfactual_sections(results_summary: dict[str, object]) -> dict[str, object]:
    """Flatten nested counterfactual summaries for console printing."""
    flattened: dict[str, object] = {}
    for key, section in results_summary.items():
        if not isinstance(section, dict):
            continue
        prefix = f"{key}_" if key.startswith("counterfactual") else ""
        for metric_key, metric_value in section.items():
            if isinstance(metric_value, (int, float, str)):
                flattened[f"{prefix}{metric_key}"] = metric_value
    return flattened


def _generate_visualizations(
    test_df: pd.DataFrame,
    cfg: dict,
    tft_prediction_df: pd.DataFrame | None = None,
    lstm_predictions: pd.Series | None = None,
    per_race_deltas: list[dict[str, object]] | None = None,
    pit_model: PitWindowLightGBM | None = None,
    shap_values: np.ndarray | None = None,
    xgb_features: pd.DataFrame | None = None,
) -> None:
    """Generate the requested visualization artifacts where inputs are available."""
    viz_dir = Path(cfg["paths"]["viz_output"])
    viz_dir.mkdir(parents=True, exist_ok=True)

    prediction_df = tft_prediction_df if tft_prediction_df is not None else pd.DataFrame()
    for circuit in cfg["viz"]["degradation_circuits"]:
        circuit_name = "Britain" if circuit == "Britain" else circuit
        circuit_predictions = prediction_df[prediction_df["event_name"] == circuit_name] if not prediction_df.empty else prediction_df
        plot_degradation_curves(circuit_name, test_df, circuit_predictions, cfg)

    for race, race_df in test_df.groupby("event_name"):
        divergence_df = race_df[race_df["pit_label"] == 2][["Driver", "LapNumber"]].copy()
        divergence_df["annotation_text"] = "Actual pit"
        if lstm_predictions is not None:
            predicted_rows = race_df.loc[lstm_predictions.loc[race_df.index] == 2, ["Driver", "LapNumber"]].copy()
            if not predicted_rows.empty:
                predicted_rows["annotation_text"] = "Model pit"
                divergence_df = pd.concat([divergence_df, predicted_rows], ignore_index=True, sort=False).drop_duplicates()
        plot_strategy_overlay(race, race_df, divergence_df, cfg)

    if pit_model is not None:
        LOGGER.info("Skipping attention heatmaps: module 2 is now LightGBM and has no LSTM attention weights.")

    if shap_values is not None and xgb_features is not None and not xgb_features.empty:
        for circuit in cfg["viz"]["shap_circuits"]:
            idx = xgb_features.index[0]
            plot_shap_waterfall(
                circuit,
                shap_values[idx],
                xgb_features.iloc[idx].to_numpy(),
                xgb_features.columns.tolist(),
                cfg,
            )

    delta_df = pd.DataFrame(per_race_deltas if per_race_deltas is not None else [])
    if delta_df.empty:
        delta_df = pd.DataFrame(
            {
                "race_id": sorted(test_df["event_name"].unique()),
                "delta_pos": np.zeros(len(test_df["event_name"].unique())),
                "is_wet": False,
                "is_street": [race in cfg["eval"]["street_circuits"] for race in sorted(test_df["event_name"].unique())],
            }
        )
    plot_delta_position_distribution(delta_df, cfg)


def main() -> None:
    """Parse CLI args and run the requested pipeline stage(s)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["full", "data", "train-tft", "train-lstm", "train-pit", "train-xgb", "eval", "viz"])
    parser.add_argument("--config", default="./config.yaml")
    parser.add_argument("--find-lr", action="store_true")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--loco", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    cfg = load_config(args.config)
    enable_cache(cfg["data"]["cache_dir"])

    results_summary: dict[str, object] = {}
    train_raw = val_raw = test_raw = pd.DataFrame()
    train_features = val_features = test_features = pd.DataFrame()
    tft_model = None
    pit_model = None
    output_dir = Path(cfg["data"]["output_dir"])

    if args.mode in {"full", "data", "train-tft", "train-lstm", "train-pit", "train-xgb", "eval", "viz"}:
        cached_raw = _load_cached_raw_splits(output_dir)
        if cached_raw is not None:
            train_raw, val_raw, test_raw = cached_raw
        else:
            train_raw, val_raw, test_raw = load_all_seasons(cfg)
            _assert_season_loaded(train_raw, "Training")
            _assert_season_loaded(val_raw, "Validation")
            _assert_season_loaded(test_raw, "Test")
            _save_dataframe(train_raw, output_dir / "train_raw.parquet")
            _save_dataframe(val_raw, output_dir / "val_raw.parquet")
            _save_dataframe(test_raw, output_dir / "test_raw.parquet")

        _assert_season_loaded(train_raw, "Training")
        _assert_season_loaded(val_raw, "Validation")
        _assert_season_loaded(test_raw, "Test")
        combined_raw = pd.concat(
            [
                train_raw.assign(split="train"),
                val_raw.assign(split="val"),
                test_raw.assign(split="test"),
            ],
            ignore_index=True,
            sort=False,
        )
        _save_dataframe(combined_raw, output_dir / "all_races.parquet")
        cached_features = None if args.mode == "data" else _load_cached_feature_splits(output_dir)
        if cached_features is not None:
            train_features, val_features, test_features = cached_features
        else:
            train_features = _prepare_features(train_raw, cfg)
            reference = train_features.copy()
            val_features = _prepare_features(val_raw, cfg, historical_reference=reference)
            test_features = _prepare_features(test_raw, cfg, historical_reference=reference)
            _save_dataframe(train_features, output_dir / "train_features.parquet")
            _save_dataframe(val_features, output_dir / "val_features.parquet")
            _save_dataframe(test_features, output_dir / "test_features.parquet")

    if args.mode == "data":
        LOGGER.info("Data mode completed in %.2fs", time.time() - start_time)
        return

    cleaned_train = apply_cleaning_masks(train_features, cfg)
    cleaned_val = apply_cleaning_masks(val_features, cfg)
    cleaned_test = apply_cleaning_masks(test_features, cfg)

    labeled_train = encode_pit_labels(cleaned_train)
    labeled_val = encode_pit_labels(cleaned_val)
    labeled_test = encode_pit_labels(cleaned_test)

    fp_df = compute_circuit_fingerprints(labeled_train, cfg)
    save_fingerprints(fp_df, cfg["paths"]["fingerprint_table"])

    if args.mode in {"full", "train-tft"}:
        train_tft_ds = build_tft_dataset(cleaned_train, cfg)
        val_tft_ds = build_tft_dataset(cleaned_val, cfg, reference_dataset=train_tft_ds)
        tft_model = train_tft(train_tft_ds, val_tft_ds, cfg)
        if args.find_lr:
            run_lr_finder(tft_model, train_tft_ds.to_dataloader(train=True, batch_size=cfg["tft"]["batch_size"]), cfg)
        test_loader = build_tft_dataset(cleaned_test, cfg, reference_dataset=train_tft_ds).to_dataloader(
            train=False,
            batch_size=cfg["tft"]["batch_size"],
        )
        results_summary["tft"] = evaluate_tft(tft_model, test_loader, cfg)

    if args.mode in {"full", "train-lstm", "train-pit"}:
        pit_model = train_pit_lightgbm(labeled_train, labeled_val, cfg)
        results_summary["pit_window"] = evaluate_pit_lightgbm(pit_model, labeled_val, cfg)

    xgb_features = pd.DataFrame()
    shap_values = None
    if args.mode in {"full", "train-xgb"}:
        qualifying_cache_path = output_dir / "qualifying_raw.parquet"
        qualifying_df = _load_cached_qualifying(qualifying_cache_path)
        if qualifying_df is None:
            qualifying_df = load_qualifying_sessions(
                cfg["data"]["train_seasons"] + [cfg["data"]["val_season"], cfg["data"]["test_season"]],
                cfg,
            )
            if not qualifying_df.empty:
                _save_dataframe(qualifying_df, qualifying_cache_path)
        if not qualifying_df.empty:
            xgb_features, xgb_target, xgb_meta = build_xgb_dataset(qualifying_df, fp_df, cfg, return_metadata=True)
            train_mask = xgb_meta["season"].isin(cfg["data"]["train_seasons"])
            val_mask = xgb_meta["season"] == cfg["data"]["val_season"]
            test_mask = xgb_meta["season"] == cfg["data"]["test_season"]
            if not train_mask.any() or not val_mask.any() or not test_mask.any():
                raise ValueError("XGBoost season-based split produced an empty train/val/test partition.")
            xgb_model = train_xgb(
                xgb_features.loc[train_mask],
                xgb_target.loc[train_mask],
                xgb_features.loc[val_mask],
                xgb_target.loc[val_mask],
                cfg,
            )
            test_features_xgb = xgb_features.loc[test_mask].reset_index(drop=True)
            test_target_xgb = xgb_target.loc[test_mask].reset_index(drop=True)
            y_pred = xgb_model.predict(test_features_xgb)
            shap_values = compute_shap_values(xgb_model, test_features_xgb.to_numpy(), xgb_features.columns.tolist())
            circuit_indices = {
                name: test_features_xgb.index[xgb_meta.loc[test_mask, "event_name"].reset_index(drop=True) == name].tolist()
                for name in cfg["viz"]["shap_circuits"]
            }
            heuristics = check_heuristic_recovery(
                shap_values,
                test_features_xgb.to_numpy(),
                circuit_indices,
                xgb_features.columns.tolist(),
                cfg,
            )
            results_summary["xgboost"] = compute_xgb_metrics(test_target_xgb, y_pred, heuristics)

    pit_pred_labels = None
    tft_prediction_df = pd.DataFrame()
    per_race_deltas: list[dict[str, object]] | None = None
    if args.mode in {"full", "eval"}:
        if pit_model is None:
            pit_model = _load_pit_checkpoint(cfg)
        if tft_model is None:
            tft_model = _load_tft_checkpoint(cfg)
        train_tft_ds = build_tft_dataset(cleaned_train, cfg)
        pit_pred_labels = _predict_pit_labels(pit_model, labeled_test, cfg)
        tft_prediction_df = _predict_tft_quantiles(tft_model, train_tft_ds, cleaned_test, cfg)
        per_race_deltas = _compute_per_race_deltas(labeled_test, pit_pred_labels, tft_prediction_df, cfg)
        per_driver_deltas = _compute_per_driver_deltas(labeled_test, pit_pred_labels, tft_prediction_df, cfg)
        per_team_deltas = _compute_per_team_deltas(per_driver_deltas)
        focus_team = cfg["eval"].get("focus_team")
        focus_driver_deltas = _filter_records_to_focus_team(per_driver_deltas, focus_team)
        focus_team_deltas = _filter_records_to_focus_team(per_team_deltas, focus_team)
        results_summary["counterfactual"] = aggregate_counterfactual(per_race_deltas, cfg)
        results_summary["counterfactual_driver"] = aggregate_counterfactual_by_entity(per_driver_deltas, entity_name="driver")
        results_summary["counterfactual_team"] = aggregate_counterfactual_by_entity(per_team_deltas, entity_name="team")
        if focus_team:
            results_summary["counterfactual_focus_driver"] = {
                "focus_team": focus_team,
                **aggregate_counterfactual_by_entity(focus_driver_deltas, entity_name="driver"),
            }
            results_summary["counterfactual_focus_team"] = {
                "focus_team": focus_team,
                **aggregate_counterfactual_by_entity(focus_team_deltas, entity_name="team"),
            }
        flattened_metrics = _flatten_counterfactual_sections(results_summary)
        print_metric_report(flattened_metrics, cfg)
        if args.ablations:
            results_summary["ablations"] = run_all_ablations(labeled_train, labeled_val, labeled_test, cfg).to_dict(orient="records")
        if args.loco:
            loco_df = run_loco_cv(pd.concat([train_features, val_features, test_features], ignore_index=True), fp_df, cfg)
            results_summary["loco"] = {
                "rows": loco_df.to_dict(orient="records"),
                "pearson_r": loco_df.attrs.get("pearson_r", 0.0),
            }

    if args.mode in {"full", "viz"}:
        if pit_model is None:
            pit_model = _load_pit_checkpoint(cfg)
        if tft_model is None:
            tft_model = _load_tft_checkpoint(cfg)
        if tft_prediction_df.empty:
            train_tft_ds = build_tft_dataset(cleaned_train, cfg)
            tft_prediction_df = _predict_tft_quantiles(tft_model, train_tft_ds, cleaned_test, cfg)
        if pit_pred_labels is None:
            pit_pred_labels = _predict_pit_labels(pit_model, labeled_test, cfg)
        if per_race_deltas is None:
            per_race_deltas = _compute_per_race_deltas(labeled_test, pit_pred_labels, tft_prediction_df, cfg)
        _generate_visualizations(
            labeled_test,
            cfg,
            tft_prediction_df=tft_prediction_df,
            lstm_predictions=pit_pred_labels,
            per_race_deltas=per_race_deltas,
            pit_model=pit_model,
            shap_values=shap_values,
            xgb_features=xgb_features,
        )

    runtime_s = time.time() - start_time
    results_summary["runtime_s"] = runtime_s
    Path("results_summary.json").write_text(json.dumps(results_summary, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Pipeline completed in %.2fs", runtime_s)


if __name__ == "__main__":
    main()
