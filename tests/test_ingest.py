"""Unit tests for ingestion helpers."""

from __future__ import annotations

import pandas as pd

import data.ingest as ingest
from data.ingest import _canonical_event_name, _is_selected_circuit, _rate_limit_retry_settings, load_qualifying_sessions


def test_canonical_event_name_maps_british_gp_to_britain() -> None:
    assert _canonical_event_name("British Grand Prix") == "Britain"


def test_selected_circuit_accepts_all_when_config_list_is_empty() -> None:
    cfg = {"data": {"selected_circuits": []}}
    assert _is_selected_circuit("Japanese Grand Prix", cfg)


def test_retry_settings_use_config_overrides() -> None:
    cfg = {"data": {"rate_limit_max_retries": 5, "rate_limit_retry_seconds": 120}}
    assert _rate_limit_retry_settings(cfg) == (5, 120)


def test_load_qualifying_sessions_handles_missing_session_part_columns(monkeypatch) -> None:
    fake_qualifying = pd.DataFrame(
        {
            "season": [2023],
            "event_name": ["Test GP"],
            "Driver": ["LEC"],
            "Compound": ["SOFT"],
        }
    )

    def _fake_load_season(year, cfg, include_sessions):
        del year, cfg, include_sessions
        return fake_qualifying.copy()

    monkeypatch.setattr(ingest, "load_season", _fake_load_season)
    out = load_qualifying_sessions([2023], {"data": {}})
    assert "q2_compound" in out.columns
    assert out["q2_compound"].isna().all()
