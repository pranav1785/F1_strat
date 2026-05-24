"""Unit tests for counterfactual strategy analysis."""

from __future__ import annotations

import pandas as pd

from eval.counterfactual import aggregate_counterfactual_by_entity, counterfactual_delta_position


def _cfg() -> dict:
    return {
        "eval": {
            "pit_stop_time_loss_s": 22.0,
            "strategy_policy": "conservative_advisor",
            "min_time_gain_s": 0.0,
            "allow_late_divergence": False,
        }
    }


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_stint_id": ["s1", "s1", "s1", "s1"],
            "LapNumber": [1, 2, 3, 4],
            "pit_label": [0, 0, 2, 0],
            "lap_time_delta_s": [0.1, 0.2, 1.5, 0.3],
            "sc_or_vsc_flag": [0, 0, 0, 0],
        }
    )


def test_zero_delta_when_model_agrees_with_team() -> None:
    df = _base_df()
    preds = pd.Series([0, 0, 2, 0])
    curves = {"s1": pd.Series([0.1, 0.2, 1.5, 0.3], index=[1, 2, 3, 4])}
    assert counterfactual_delta_position("race", preds, curves, df, _cfg()) == 0.0


def test_positive_delta_when_model_pits_earlier() -> None:
    df = _base_df()
    preds = pd.Series([0, 2, 0, 0])
    curves = {"s1": pd.Series([0.1, 0.1, 0.8, 0.3], index=[1, 2, 3, 4])}
    assert counterfactual_delta_position("race", preds, curves, df, _cfg()) > 0.0


def test_conservative_policy_falls_back_to_zero_when_model_pits_later() -> None:
    df = _base_df()
    preds = pd.Series([0, 0, 0, 0])
    curves = {"s1": pd.Series([0.1, 0.2, 0.5, 0.3], index=[1, 2, 3, 4])}
    assert counterfactual_delta_position("race", preds, curves, df, _cfg()) == 0.0


def test_aggressive_policy_can_still_go_negative() -> None:
    df = _base_df()
    preds = pd.Series([0, 0, 0, 0])
    curves = {"s1": pd.Series([0.1, 0.2, 0.5, 0.3], index=[1, 2, 3, 4])}
    cfg = _cfg()
    cfg["eval"]["strategy_policy"] = "aggressive"
    cfg["eval"]["allow_late_divergence"] = True
    assert counterfactual_delta_position("race", preds, curves, df, cfg) < 0.0


def test_delta_capped_at_five_positions() -> None:
    df = _base_df()
    df["lap_time_delta_s"] = [0.1, 200.0, 200.0, 0.3]
    preds = pd.Series([0, 2, 0, 0])
    curves = {"s1": pd.Series([0.1, 0.0, 0.2, 0.3], index=[1, 2, 3, 4])}
    assert counterfactual_delta_position("race", preds, curves, df, _cfg()) == 5.0


def test_sc_laps_excluded_from_divergence_counting() -> None:
    df = _base_df()
    df.loc[df["LapNumber"] == 2, "sc_or_vsc_flag"] = 1
    preds = pd.Series([0, 2, 0, 0])
    curves = {"s1": pd.Series([0.1, 0.0, 0.2, 0.3], index=[1, 2, 3, 4])}
    assert counterfactual_delta_position("race", preds, curves, df, _cfg()) == 0.0


def test_aggregate_counterfactual_by_entity_reports_driver_counts() -> None:
    records = [
        {"driver_id": "RaceA_VER", "delta_pos": 0.2, "is_wet": False, "is_street": False},
        {"driver_id": "RaceA_PER", "delta_pos": -0.3, "is_wet": False, "is_street": False},
        {"driver_id": "RaceB_LEC", "delta_pos": 0.01, "is_wet": True, "is_street": True},
    ]
    out = aggregate_counterfactual_by_entity(records, entity_name="driver")
    assert out["drivers_positive"] == 1
    assert out["drivers_negative"] == 1
    assert out["drivers_neutral"] == 1
    assert out["per_driver_deltas"]["RaceA_VER"] == 0.2
