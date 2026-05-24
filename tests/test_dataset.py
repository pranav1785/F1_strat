"""Unit tests for dataset builders."""

from __future__ import annotations

import pandas as pd

from data.dataset import build_lstm_windows, build_tft_dataset, build_xgb_dataset


def _tft_cfg() -> dict:
    return {
        "tft": {
            "target_col": "lap_time_delta_s",
            "encoder_length": 5,
            "prediction_length": 2,
            "static_categoricals": ["circuit_id", "compound_type", "team_id"],
            "static_reals": ["circuit_softness", "historical_sc_prob"],
            "known_time_varying_reals": ["lap_norm", "fuel_load_kg", "track_evolution_delta", "air_temp", "humidity"],
            "unknown_time_varying_reals": [
                "lap_time_delta_s",
                "deg_velocity",
                "deg_acceleration",
                "tire_age_laps",
                "tire_age_sq",
                "gap_ahead_s",
                "gap_ahead_trend",
                "undercut_threat",
            ],
        }
    }


def test_build_tft_dataset_adapts_numeric_categories_and_missing_reals() -> None:
    rows = []
    for lap in range(1, 11):
        rows.append(
            {
                "season": 2023,
                "event_name": "Test",
                "session_type": "R",
                "driver_stint_id": "2023_Test_LEC_1",
                "LapNumber": lap,
                "circuit_id": 1,
                "compound_type": 2,
                "team_id": 3,
                "circuit_softness": 0.5,
                "historical_sc_prob": 0.25,
                "lap_norm": lap / 10.0,
                "fuel_load_kg": 100.0 - lap,
                "track_evolution_delta": None if lap == 1 else 0.1,
                "AirTemp": 30.0,
                "Humidity": 55.0,
                "lap_time_delta_s": float(lap) * 0.1,
                "deg_velocity": 0.01,
                "deg_acceleration": 0.0,
                "tire_age_laps": None if lap == 2 else float(lap),
                "tire_age_sq": None if lap == 2 else float(lap * lap),
                "gap_ahead_s": 1.0,
                "gap_ahead_trend": 0.0,
                "undercut_threat": 0,
            }
        )
    df = pd.DataFrame(rows)

    ds = build_tft_dataset(df, _tft_cfg())

    assert len(ds) > 0
    assert "air_temp" in ds.reals
    assert "humidity" in ds.reals
    assert ds.categoricals == ["circuit_id", "compound_type", "team_id"]


def test_build_tft_dataset_reuses_train_encoders_for_unseen_categories() -> None:
    train_rows = []
    val_rows = []
    for lap in range(1, 11):
        train_rows.append(
            {
                "season": 2023,
                "event_name": "Train GP",
                "session_type": "R",
                "driver_stint_id": "2023_Train_A_1",
                "LapNumber": lap,
                "Compound": "SOFT",
                "Team": "Ferrari",
                "circuit_softness": 0.5,
                "historical_sc_prob": 0.2,
                "lap_norm": lap / 10.0,
                "fuel_load_kg": 100.0 - lap,
                "track_evolution_delta": 0.1,
                "AirTemp": 30.0,
                "Humidity": 50.0,
                "lap_time_delta_s": float(lap) * 0.1,
                "deg_velocity": 0.01,
                "deg_acceleration": 0.0,
                "tire_age_laps": float(lap),
                "tire_age_sq": float(lap * lap),
                "gap_ahead_s": 1.0,
                "gap_ahead_trend": 0.0,
                "undercut_threat": 0,
            }
        )
        val_rows.append(
            {
                "season": 2024,
                "event_name": "New GP",
                "session_type": "R",
                "driver_stint_id": "2024_New_B_1",
                "LapNumber": lap,
                "Compound": "MEDIUM",
                "Team": "Brand New Team",
                "circuit_softness": 0.4,
                "historical_sc_prob": 0.1,
                "lap_norm": lap / 10.0,
                "fuel_load_kg": 98.0 - lap,
                "track_evolution_delta": 0.2,
                "AirTemp": 28.0,
                "Humidity": 60.0,
                "lap_time_delta_s": float(lap) * 0.2,
                "deg_velocity": 0.02,
                "deg_acceleration": 0.0,
                "tire_age_laps": float(lap),
                "tire_age_sq": float(lap * lap),
                "gap_ahead_s": 2.0,
                "gap_ahead_trend": 0.0,
                "undercut_threat": 0,
            }
        )

    train_ds = build_tft_dataset(pd.DataFrame(train_rows), _tft_cfg())
    val_ds = build_tft_dataset(pd.DataFrame(val_rows), _tft_cfg(), reference_dataset=train_ds)

    assert len(train_ds) > 0
    assert len(val_ds) > 0


def test_build_xgb_dataset_uses_event_name_merge_and_compound_fallback() -> None:
    qualifying_df = pd.DataFrame(
        {
            "season": [2023, 2023],
            "event_name": ["Monaco", "Monaco"],
            "circuit_id": [99, 99],
            "LapTime": pd.to_timedelta([70.0, 71.0], unit="s"),
            "AirTemp": [30.0, 30.0],
            "Humidity": [55.0, 55.0],
            "TrackTemp": [40.0, 40.0],
            "FreshTyre": [True, True],
            "TyreLife": [1, 1],
            "Compound": ["SOFT", "MEDIUM"],
        }
    )
    fingerprint_df = pd.DataFrame(
        {
            "event_name": ["Monaco"],
            "avg_deg_rate": [0.1],
            "compound_variance": [0.2],
            "track_temp_percentile": [0.3],
            "high_speed_fraction": [0.05],
            "hist_soft_stint_len": [12.0],
            "sc_vsc_frequency": [0.4],
        },
        index=pd.Index([1], name="circuit_id"),
    )
    cfg = {"xgboost": {"target_col": "compound_chosen"}}

    X, y = build_xgb_dataset(qualifying_df, fingerprint_df, cfg)

    assert len(X) == 2
    assert "avg_deg_rate" in X.columns
    assert "qualifying_pace_delta_s" in X.columns
    assert y.tolist() == [2, 1]


def test_build_lstm_windows_imputes_missing_sequence_values() -> None:
    cfg = {
        "lstm": {
            "seq_len": 2,
            "features": ["tire_age_laps", "tire_age_sq", "lap_norm", "Position"],
            "static_features": ["circuit_id_enc", "compound_enc"],
        }
    }
    df = pd.DataFrame(
        {
            "driver_stint_id": ["s1", "s1", "s1", "s1"],
            "LapNumber": [2, 3, 4, 5],
            "tire_age_laps": [1.0, None, 3.0, 4.0],
            "tire_age_sq": [1.0, None, 9.0, 16.0],
            "lap_norm": [0.2, 0.3, 0.4, 0.5],
            "Position": [10, 9, 8, 8],
            "circuit_id_enc": [1, 1, 1, 1],
            "compound_enc": [0, 0, 0, 0],
            "pit_label": [0, 1, 2, 0],
            "loss_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )

    X_seq, X_static, y, w = build_lstm_windows(df, cfg)

    assert X_seq.shape == (2, 2, 4)
    assert X_static.shape == (2, 2)
    assert y.tolist() == [2, 0]
    assert w.tolist() == [1.0, 1.0]
    assert not pd.isna(X_seq).any()
