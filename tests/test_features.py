"""Unit tests for feature engineering."""

from __future__ import annotations

import pandas as pd

from data.features import (
    compute_degradation_derivatives,
    compute_fuel_load,
    compute_gap_features,
    compute_sc_flags,
    compute_lap_time_delta,
    compute_undercut_threat,
    create_stint_id,
    encode_categoricals,
    run_full_feature_pipeline,
)


def _feature_cfg() -> dict:
    return {
        "data": {
            "fuel_kg_start": 110.0,
            "fuel_consumption_per_lap": 2.3,
            "fuel_lap_time_effect": 0.1495,
            "track_evolution_fuel_effect": 0.1495,
            "undercut_threat_gap_s": 3.0,
            "undercut_threat_laps": 2,
        },
        "circuits": {
            "abrasion": {"Alpha": 3, "Beta": 2},
        },
    }


def test_compute_lap_time_delta_removes_car_pace_bias() -> None:
    df = pd.DataFrame(
        {
            "season": [2023] * 10,
            "event_name": ["Test"] * 10,
            "Driver": ["A"] * 5 + ["B"] * 5,
            "Stint": [1] * 10,
            "LapNumber": [1, 2, 3, 4, 5] * 2,
            "LapTime": pd.to_timedelta([90.0, 90.2, 90.4, 90.6, 90.8, 92.0, 92.2, 92.4, 92.6, 92.8], unit="s"),
        }
    )
    df = create_stint_id(df)
    out = compute_lap_time_delta(df, _feature_cfg())
    curve_a = out[out["Driver"] == "A"]["lap_time_delta_s"].round(6).tolist()
    curve_b = out[out["Driver"] == "B"]["lap_time_delta_s"].round(6).tolist()
    assert curve_a == curve_b


def test_compute_lap_time_delta_handles_short_stints_without_fixed_window_bias() -> None:
    df = pd.DataFrame(
        {
            "season": [2023] * 6,
            "event_name": ["Test"] * 6,
            "Driver": ["A"] * 3 + ["B"] * 3,
            "Stint": [2] * 6,
            "LapNumber": [18, 19, 20] * 2,
            "LapTime": pd.to_timedelta([92.2, 91.0, 91.4, 94.2, 93.0, 93.4], unit="s"),
        }
    )
    df = create_stint_id(df)
    out = compute_lap_time_delta(df, _feature_cfg())
    curve_a = out[out["Driver"] == "A"]["lap_time_delta_s"].round(6).tolist()
    curve_b = out[out["Driver"] == "B"]["lap_time_delta_s"].round(6).tolist()
    assert curve_a == curve_b


def test_compute_undercut_threat_flags_recent_close_rival() -> None:
    df = pd.DataFrame(
        {
            "LapNumber": [4, 4, 5, 5],
            "Driver": ["LEC", "VER", "LEC", "VER"],
            "GapToLeader": [10.0, 12.5, 10.2, 12.7],
            "PitInTime": [pd.Timestamp("2023-01-01"), pd.NaT, pd.NaT, pd.NaT],
        }
    )
    out = compute_undercut_threat(df, _feature_cfg())
    assert int(out.loc[(out["LapNumber"] == 5) & (out["Driver"] == "VER"), "undercut_threat"].iloc[0]) == 1


def test_compute_undercut_threat_ignores_non_threats() -> None:
    df = pd.DataFrame(
        {
            "LapNumber": [4, 4, 5, 5],
            "Driver": ["LEC", "VER", "LEC", "VER"],
            "GapToLeader": [10.0, 15.5, 10.2, 15.7],
            "PitInTime": [pd.Timestamp("2023-01-01"), pd.NaT, pd.NaT, pd.NaT],
        }
    )
    out = compute_undercut_threat(df, _feature_cfg())
    assert int(out.loc[(out["LapNumber"] == 5) & (out["Driver"] == "VER"), "undercut_threat"].iloc[0]) == 0


def test_compute_gap_features_derives_gaps_from_time() -> None:
    df = pd.DataFrame(
        {
            "season": [2023] * 4,
            "event_name": ["Test"] * 4,
            "session_type": ["R"] * 4,
            "LapNumber": [5, 5, 6, 6],
            "Driver": ["LEC", "VER", "LEC", "VER"],
            "Position": [1, 2, 1, 2],
            "Time": pd.to_timedelta([100.0, 102.5, 190.0, 193.0], unit="s"),
            "driver_stint_id": ["test_LEC", "test_VER", "test_LEC", "test_VER"],
        }
    )
    out = compute_gap_features(df)
    assert out["gap_ahead_s"].round(6).tolist() == [0.0, 2.5, 0.0, 3.0]
    assert out["gap_ahead_trend"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_compute_undercut_threat_uses_time_when_gap_column_missing() -> None:
    df = pd.DataFrame(
        {
            "LapNumber": [4, 4, 5, 5],
            "Driver": ["LEC", "VER", "LEC", "VER"],
            "Time": pd.to_timedelta([100.0, 102.5, 190.2, 192.7], unit="s"),
            "PitInTime": [pd.Timestamp("2023-01-01"), pd.NaT, pd.NaT, pd.NaT],
        }
    )
    out = compute_undercut_threat(df, _feature_cfg())
    assert int(out.loc[(out["LapNumber"] == 5) & (out["Driver"] == "VER"), "undercut_threat"].iloc[0]) == 1


def test_compute_undercut_threat_respects_session_boundaries() -> None:
    df = pd.DataFrame(
        {
            "season": [2023] * 8,
            "event_name": ["Alpha"] * 4 + ["Beta"] * 4,
            "session_type": ["R"] * 8,
            "LapNumber": [4, 4, 5, 5, 4, 4, 5, 5],
            "Driver": ["LEC", "VER", "LEC", "VER", "LEC", "VER", "LEC", "VER"],
            "GapToLeader": [10.0, 12.5, 10.2, 12.7, 20.0, 26.0, 20.2, 26.2],
            "PitInTime": [pd.Timestamp("2023-01-01"), pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        }
    )
    out = compute_undercut_threat(df, _feature_cfg())
    assert int(out.loc[(out["event_name"] == "Alpha") & (out["LapNumber"] == 5) & (out["Driver"] == "VER"), "undercut_threat"].iloc[0]) == 1
    assert int(out.loc[(out["event_name"] == "Beta") & (out["LapNumber"] == 5) & (out["Driver"] == "VER"), "undercut_threat"].iloc[0]) == 0


def test_run_full_feature_pipeline_handles_rich_attrs_across_session_concat() -> None:
    df = pd.DataFrame(
        {
            "season": [2023] * 4,
            "event_name": ["Alpha", "Alpha", "Beta", "Beta"],
            "session_type": ["R"] * 4,
            "Driver": ["LEC", "VER", "HAM", "NOR"],
            "Stint": [1, 1, 1, 1],
            "LapNumber": [3, 3, 3, 3],
            "LapTime": pd.to_timedelta([90.0, 91.0, 92.0, 93.0], unit="s"),
            "Time": pd.to_timedelta([300.0, 302.0, 400.0, 403.0], unit="s"),
            "TyreLife": [3, 3, 3, 3],
            "Position": [1, 2, 1, 2],
            "TrackStatus": ["1", "1", "1", "1"],
            "Compound": ["SOFT", "MEDIUM", "SOFT", "MEDIUM"],
            "Team": ["Ferrari", "Red Bull Racing", "Mercedes", "McLaren"],
            "PitInTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        }
    )
    reference = df.copy()
    reference["safety_car_flag"] = [0, 0, 1, 1]
    df.attrs["historical_sc_reference"] = reference
    df.attrs["train_seasons"] = [2023]

    out = run_full_feature_pipeline(df, _feature_cfg())
    assert len(out) == 4
    assert "historical_sc_prob" in out.columns


def test_deg_acceleration_zero_on_first_lap_of_stint() -> None:
    df = pd.DataFrame({"driver_stint_id": ["a", "a", "b"], "lap_time_delta_s": [0.0, 0.1, 0.0]})
    out = compute_degradation_derivatives(df)
    assert out.iloc[0]["deg_acceleration"] == 0.0
    assert out.iloc[2]["deg_acceleration"] == 0.0


def test_fuel_load_clips_at_zero() -> None:
    df = pd.DataFrame({"LapNumber": [50]})
    out = compute_fuel_load(df, _feature_cfg())
    assert out.loc[0, "fuel_load_kg"] == 0.0


def test_compute_sc_flags_parses_multicode_track_status() -> None:
    df = pd.DataFrame({"TrackStatus": ["1", "2", "12", "14", "16", "17"]})
    cfg = {"data": {"sc_track_statuses": ["4", "5", "6", "7"]}}
    out = compute_sc_flags(df, cfg)
    assert out["sc_or_vsc_flag"].tolist() == [0, 1, 1, 1, 1, 1]
    assert out["safety_car_flag"].tolist() == [0, 0, 0, 1, 0, 0]
    assert out["vsc_flag"].tolist() == [0, 0, 0, 0, 1, 1]


def test_encode_categoricals_reuses_reference_encoders() -> None:
    train_df = pd.DataFrame(
        {
            "event_name": ["Bahrain", "Monaco"],
            "Compound": ["SOFT", "MEDIUM"],
            "Team": ["Ferrari", "Mercedes"],
            "TrackTemp": [40.0, 35.0],
            "GridPosition": [1, 2],
        }
    )
    train_encoded, encoders = encode_categoricals(train_df)

    val_df = pd.DataFrame(
        {
            "event_name": ["Monaco", "New GP"],
            "Compound": ["SOFT", "HARD"],
            "Team": ["Mercedes", "Brand New Team"],
            "TrackTemp": [33.0, 28.0],
            "GridPosition": [3, 4],
        }
    )
    val_encoded, _ = encode_categoricals(val_df, reference_encoders=encoders)

    monaco_train = int(train_encoded.loc[train_df["event_name"] == "Monaco", "circuit_id_enc"].iloc[0])
    monaco_val = int(val_encoded.loc[val_df["event_name"] == "Monaco", "circuit_id_enc"].iloc[0])
    mercedes_train = int(train_encoded.loc[train_df["Team"] == "Mercedes", "team_id_enc"].iloc[0])
    mercedes_val = int(val_encoded.loc[val_df["Team"] == "Mercedes", "team_id_enc"].iloc[0])
    assert monaco_train == monaco_val
    assert mercedes_train == mercedes_val
    assert int(val_encoded.loc[val_df["event_name"] == "New GP", "circuit_id_enc"].iloc[0]) == -1
    assert int(val_encoded.loc[val_df["Team"] == "Brand New Team", "team_id_enc"].iloc[0]) == -1
