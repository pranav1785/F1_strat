"""Unit tests for pit label encoding."""

from __future__ import annotations

import pandas as pd

from data.labels import apply_cleaning_masks, encode_pit_labels


def _make_driver_laps() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Driver": ["A"] * 20,
            "LapNumber": list(range(1, 21)),
            "PitInTime": [pd.NaT] * 20,
            "sc_or_vsc_flag": [0] * 20,
        }
    )
    df.loc[df["LapNumber"] == 10, "PitInTime"] = pd.Timestamp("2023-01-01")
    df.loc[df["LapNumber"] == 15, "PitInTime"] = pd.Timestamp("2023-01-02")
    return df


def test_pit_now_labeled_exactly_on_pit_lap() -> None:
    out = encode_pit_labels(_make_driver_laps())
    assert out.loc[out["LapNumber"] == 10, "pit_label"].iloc[0] == 2


def test_pit_soon_labeled_for_three_prior_laps() -> None:
    out = encode_pit_labels(_make_driver_laps())
    expected = out[out["LapNumber"].isin([7, 8, 9])]["pit_label"].tolist()
    assert expected == [1, 1, 1]


def test_stay_out_not_overwritten_when_pit_now_adjacent() -> None:
    df = _make_driver_laps()
    df.loc[df["LapNumber"] == 11, "PitInTime"] = pd.Timestamp("2023-01-03")
    out = encode_pit_labels(df)
    assert out.loc[out["LapNumber"] == 10, "pit_label"].iloc[0] == 2


def test_sc_laps_get_zero_loss_weight() -> None:
    df = _make_driver_laps()
    df.loc[df["LapNumber"] == 9, "sc_or_vsc_flag"] = 1
    out = encode_pit_labels(df)
    assert out.loc[out["LapNumber"] == 9, "loss_weight"].iloc[0] == 0.0


def test_class_distribution_matches_reasonable_split() -> None:
    dfs = []
    for driver in ["A", "B", "C", "D"]:
        df = _make_driver_laps()
        df["Driver"] = driver
        dfs.append(df)
    out = encode_pit_labels(pd.concat(dfs, ignore_index=True))
    distribution = out["pit_label"].value_counts(normalize=True).sort_index()
    assert 0.55 < distribution.get(0, 0) < 0.8
    assert 0.15 < distribution.get(1, 0) < 0.35
    assert 0.05 < distribution.get(2, 0) < 0.15


def test_apply_cleaning_masks_keeps_pit_out_laps_and_flags_them() -> None:
    df = pd.DataFrame(
        {
            "LapNumber": [2, 3, 4],
            "Compound": ["SOFT", "SOFT", "SOFT"],
            "driver_stint_id": ["stint_a"] * 3,
            "LapTime_s": [91.0, 140.0, 91.2],
            "PitOutTime": [pd.NaT, pd.Timestamp("2023-01-01"), pd.NaT],
        }
    )
    cfg = {"data": {"wet_compounds": ["INTERMEDIATE", "WET"], "outlier_threshold": 1.2}}
    out = apply_cleaning_masks(df, cfg)
    assert out["LapNumber"].tolist() == [2, 3, 4]
    assert out["is_pit_out_lap"].tolist() == [0, 1, 0]
