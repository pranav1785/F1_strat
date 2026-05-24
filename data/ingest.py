"""FastF1 ingestion helpers."""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Optional

import fastf1
import numpy as np
import pandas as pd
import yaml
from fastf1.req import RateLimitExceededError

LOGGER = logging.getLogger(__name__)

EVENT_ALIASES = {
    "silverstone": "Britain",
    "great britain": "Britain",
    "british": "Britain",
    "britain": "Britain",
    "barcelona": "Spain",
    "spanish": "Spain",
    "spain": "Spain",
    "monza": "Italy",
    "italian": "Italy",
    "italy": "Italy",
    "bahrain": "Bahrain",
    "monaco": "Monaco",
    "singapore": "Singapore",
}


def enable_cache(cache_dir: str) -> None:
    """Enable the FastF1 cache directory.

    Args:
        cache_dir: Filesystem path where FastF1 should cache downloaded data.

    Returns:
        None.
    """
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(path))
    LOGGER.info("FastF1 cache enabled at %s", path.resolve())


def _rate_limit_retry_settings(cfg: dict | None = None) -> tuple[int, int]:
    """Return bounded retry settings for FastF1/Ergast rate limits."""
    data_cfg = cfg.get("data", {}) if cfg is not None else {}
    max_retries = int(data_cfg.get("rate_limit_max_retries", 3))
    retry_seconds = int(data_cfg.get("rate_limit_retry_seconds", 900))
    return max(max_retries, 0), max(retry_seconds, 1)


def load_race_session(
    year: int,
    event_name: str,
    session_type: str = "R",
    cfg: dict | None = None,
) -> Optional[fastf1.core.Session]:
    """Load a FastF1 session and return it if successful.

    Args:
        year: Championship season.
        event_name: Grand Prix event name understood by FastF1.
        session_type: One of 'R', 'Q', or 'FP2'.
        cfg: Optional project configuration for retry/backoff behavior.

    Returns:
        The loaded session, or ``None`` when the load fails.
    """
    max_retries, retry_seconds = _rate_limit_retry_settings(cfg)
    for attempt in range(max_retries + 1):
        try:
            session = fastf1.get_session(year, event_name, session_type)
            session.load(telemetry=False, weather=True, messages=False)
            return session
        except RateLimitExceededError as exc:  # pragma: no cover - external API behavior
            if attempt == max_retries:
                LOGGER.warning(
                    "FastF1/Ergast rate limit persisted while loading year=%s event=%s session=%s "
                    "after %s attempts: %s",
                    year,
                    event_name,
                    session_type,
                    attempt + 1,
                    exc,
                )
                return None
            wait_seconds = retry_seconds * (attempt + 1)
            LOGGER.warning(
                "FastF1/Ergast rate limit reached while loading year=%s event=%s session=%s. "
                "Waiting %ss before retry %s/%s.",
                year,
                event_name,
                session_type,
                wait_seconds,
                attempt + 1,
                max_retries,
            )
            time.sleep(wait_seconds)
        except Exception as exc:  # pragma: no cover - external API behavior
            LOGGER.error(
                "Failed to load session year=%s event=%s session=%s: %s",
                year,
                event_name,
                session_type,
                exc,
            )
            return None
    return None


def _load_event_schedule(year: int, cfg: dict) -> Optional[pd.DataFrame]:
    """Load a season schedule with bounded retry/backoff for rate limits."""
    max_retries, retry_seconds = _rate_limit_retry_settings(cfg)
    for attempt in range(max_retries + 1):
        try:
            return fastf1.get_event_schedule(year, include_testing=False)
        except RateLimitExceededError as exc:  # pragma: no cover - external API behavior
            if attempt == max_retries:
                LOGGER.warning(
                    "Rate limit persisted while fetching schedule for %s after %s attempts: %s",
                    year,
                    attempt + 1,
                    exc,
                )
                return None
            wait_seconds = retry_seconds * (attempt + 1)
            LOGGER.warning(
                "Rate limit reached while fetching schedule for %s. Waiting %ss before retry %s/%s.",
                year,
                wait_seconds,
                attempt + 1,
                max_retries,
            )
            time.sleep(wait_seconds)
        except Exception as exc:  # pragma: no cover - external API behavior
            LOGGER.error("Failed to fetch schedule for %s: %s", year, exc)
            return None
    return None


def _event_name_from_row(event_row: pd.Series) -> str:
    """Extract the canonical event name from a FastF1 schedule row."""
    for candidate in ("EventName", "OfficialEventName", "Country", "Location"):
        if candidate in event_row and pd.notna(event_row[candidate]):
            return str(event_row[candidate])
    raise KeyError("No event name-like column found in event schedule row")


def _canonical_event_name(event_name: str) -> str:
    """Canonicalize circuit names to the config's naming convention.

    Args:
        event_name: Raw event name from FastF1.

    Returns:
        Canonical event name when an alias is recognized.
    """
    normalized = str(event_name).strip().lower()
    for alias, canonical in EVENT_ALIASES.items():
        if alias in normalized:
            return canonical
    return str(event_name)


def _is_selected_circuit(event_name: str, cfg: dict) -> bool:
    """Check whether an event is inside the configured circuit subset.

    Args:
        event_name: Raw event name from FastF1.
        cfg: Project configuration dictionary.

    Returns:
        ``True`` when the event should be loaded.
    """
    selected = cfg.get("data", {}).get("selected_circuits", [])
    if not selected:
        return True
    return _canonical_event_name(event_name) in set(selected)


def _merge_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Merge weather data onto laps using nearest timestamp matching."""
    if laps.empty:
        return laps.copy()
    laps_sorted = laps.sort_values("Time").copy()
    weather_sorted = weather.sort_values("Time").copy()
    merged = pd.merge_asof(
        laps_sorted,
        weather_sorted,
        on="Time",
        direction="nearest",
    )
    return merged


def _prepare_session_laps(
    session: fastf1.core.Session,
    year: int,
    event_name: str,
    session_type: str,
) -> pd.DataFrame:
    """Extract clean lap data and weather for a loaded session."""
    laps = session.laps.pick_accurate().copy()
    if laps.empty:
        return laps
    weather = session.weather_data.copy()
    if not weather.empty and "Time" in weather.columns and "Time" in laps.columns:
        laps = _merge_weather(laps, weather)
    laps["season"] = year
    laps["event_name"] = event_name
    laps["session_type"] = session_type
    return laps


def load_season(
    year: int,
    cfg: dict,
    include_sessions: list[str] = ["R"],
) -> pd.DataFrame:
    """Load all requested sessions for a season.

    Args:
        year: Championship season to load.
        cfg: Project configuration dictionary.
        include_sessions: Session codes to request from FastF1.

    Returns:
        Concatenated lap dataframe for the season.
    """
    schedule = _load_event_schedule(year, cfg)
    if schedule is None:
        empty_df = pd.DataFrame()
        empty_df.attrs["load_status"] = {
            "year": year,
            "failed_sessions": 0,
            "loaded_sessions": 0,
            "rate_limited": True,
            "complete": False,
        }
        return empty_df
    frames: list[pd.DataFrame] = []
    failed_sessions = 0
    loaded_sessions = 0
    rate_limited = False
    for _, event_row in schedule.iterrows():
        event_name = _event_name_from_row(event_row)
        if not _is_selected_circuit(event_name, cfg):
            continue
        event_name = _canonical_event_name(event_name)
        for session_type in include_sessions:
            session = load_race_session(year, event_name, session_type=session_type, cfg=cfg)
            if session is None:
                failed_sessions += 1
                rate_limited = True
                continue
            try:
                laps = _prepare_session_laps(session, year, event_name, session_type)
                if not laps.empty:
                    frames.append(laps)
                loaded_sessions += 1
            finally:
                del session
                gc.collect()
    if not frames:
        empty_df = pd.DataFrame()
        empty_df.attrs["load_status"] = {
            "year": year,
            "failed_sessions": failed_sessions,
            "loaded_sessions": loaded_sessions,
            "rate_limited": rate_limited,
            "complete": False,
        }
        return empty_df
    season_df = pd.concat(frames, ignore_index=True, sort=False)
    event_codes = {
        name: idx for idx, name in enumerate(sorted(season_df["event_name"].dropna().unique()))
    }
    season_df["circuit_id"] = season_df["event_name"].map(event_codes).astype("Int64")
    season_df.attrs["load_status"] = {
        "year": year,
        "failed_sessions": failed_sessions,
        "loaded_sessions": loaded_sessions,
        "rate_limited": rate_limited,
        "complete": not rate_limited and failed_sessions == 0,
    }
    return season_df


def load_all_seasons(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test season data.

    Args:
        cfg: Project configuration dictionary.

    Returns:
        Tuple of train, validation, and test dataframes.
    """
    train_frames = [load_season(year, cfg, include_sessions=["R"]) for year in cfg["data"]["train_seasons"]]
    df_train = pd.concat(train_frames, ignore_index=True, sort=False) if train_frames else pd.DataFrame()
    df_val = load_season(cfg["data"]["val_season"], cfg, include_sessions=["R"])
    df_test = load_season(cfg["data"]["test_season"], cfg, include_sessions=["R"])
    LOGGER.info("Train shape: %s", df_train.shape)
    LOGGER.info("Val shape: %s", df_val.shape)
    LOGGER.info("Test shape: %s", df_test.shape)
    return df_train, df_val, df_test


def load_qualifying_sessions(years: list[int], cfg: dict) -> pd.DataFrame:
    """Load Q2/Q3 qualifying laps and annotate Q2 starting compounds.

    Args:
        years: List of seasons to load.
        cfg: Project configuration dictionary.

    Returns:
        Qualifying lap dataframe with ``q2_compound`` annotations.
    """
    frames: list[pd.DataFrame] = []
    for year in years:
        season_df = load_season(year, cfg, include_sessions=["Q"])
        if season_df.empty:
            continue
        qualifying = season_df.copy()
        session_part = pd.Series("", index=qualifying.index, dtype=object)
        if "SessionPart" in qualifying.columns:
            session_part = qualifying["SessionPart"].fillna("").astype(str)
            qualifying = qualifying[qualifying["SessionPart"].isin(["Q2", "Q3"])]
        elif "SessionName" in qualifying.columns:
            session_part = qualifying["SessionName"].fillna("").astype(str)
            qualifying = qualifying[qualifying["SessionName"].isin(["Q2", "Q3"])]
        session_part = session_part.reindex(qualifying.index, fill_value="")
        q2_mask = session_part.eq("Q2")
        q2_laps = qualifying[q2_mask].copy()
        if not q2_laps.empty:
            q2_laps["LapTime_s"] = q2_laps["LapTime"].dt.total_seconds()
            fastest_q2 = q2_laps.sort_values("LapTime_s").groupby(
                ["season", "event_name", "Driver"], as_index=False
            ).first()
            fastest_q2 = fastest_q2[
                ["season", "event_name", "Driver", "Compound"]
            ].rename(columns={"Compound": "q2_compound"})
            qualifying = qualifying.merge(
                fastest_q2,
                on=["season", "event_name", "Driver"],
                how="left",
            )
        else:
            qualifying["q2_compound"] = np.nan
        frames.append(qualifying)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def load_config(path: str = "config.yaml") -> dict:
    """Load the YAML configuration file.

    Args:
        path: Path to the YAML configuration.

    Returns:
        Parsed configuration dictionary.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
