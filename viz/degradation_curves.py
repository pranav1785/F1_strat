"""Static degradation curve plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_degradation_curves(circuit: str, actual_df: pd.DataFrame, prediction_df: pd.DataFrame, cfg: dict) -> Path:
    """Plot actual and predicted degradation curves for one circuit."""
    colors = {"SOFT": "#FF4444", "MEDIUM": "#FFD700", "HARD": "#CCCCCC"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0F0F0F")
    axes = axes.ravel()
    for ax in axes:
        ax.set_facecolor("#1A1A1A")
        ax.tick_params(colors="white")
    compounds = ["SOFT", "MEDIUM", "HARD", "combined"]
    for ax, compound in zip(axes, compounds):
        if compound == "combined":
            subset = actual_df[actual_df["event_name"] == circuit]
        else:
            subset = actual_df[(actual_df["event_name"] == circuit) & (actual_df["Compound"] == compound)]
        if subset.empty:
            continue
        ax.scatter(subset["LapNumber"], subset["lap_time_delta_s"], s=16, color=colors.get(compound, "#E10600"), alpha=0.7)
        pred_subset = prediction_df[prediction_df["event_name"] == circuit]
        if not pred_subset.empty:
            pred_subset = pred_subset.groupby("LapNumber", as_index=False)[["p10", "p50", "p90"]].mean().sort_values("LapNumber")
            ax.plot(pred_subset["LapNumber"], pred_subset["p50"], color="#E10600", linewidth=2)
            ax.fill_between(pred_subset["LapNumber"], pred_subset["p10"], pred_subset["p90"], color="#E10600", alpha=0.2)
            if len(pred_subset) >= 4:
                poly = np.poly1d(np.polyfit(pred_subset["LapNumber"], pred_subset["p50"], 3))
                ax.plot(pred_subset["LapNumber"], poly(pred_subset["LapNumber"]), "--", color="#888888")
        if "pit_label" in subset.columns:
            pit_laps = subset[subset["pit_label"] == 2]["LapNumber"].tolist()
        else:
            pit_laps = subset[subset["PitInTime"].notna()]["LapNumber"].tolist()
        for pit_lap in pit_laps:
            ax.axvline(pit_lap, color="#FF4444", linestyle="--")
        if not pred_subset.empty:
            over_threshold = pred_subset[pred_subset["p90"] > cfg["tft"]["cliff_threshold_s"]]
            if not over_threshold.empty:
                ax.axvline(over_threshold["LapNumber"].iloc[0], color="#FFA500", linestyle="--")
        ax.set_title(f"{circuit} - {compound}", color="white")
    output = Path(cfg["paths"]["viz_output"]) / f"degradation_curves_{circuit}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output
