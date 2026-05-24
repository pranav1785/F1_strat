"""Counterfactual delta-position distribution plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_delta_position_distribution(delta_df: pd.DataFrame, cfg: dict) -> Path:
    """Plot box plots for all, dry, and street-circuit race deltas."""
    subsets = [
        (f"All {len(delta_df)} Races", delta_df),
        ("Dry Races", delta_df[delta_df["is_wet"] == False]),
        ("Street Circuits", delta_df[delta_df["is_street"] == True]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0F0F0F")
    for ax, (title, subset) in zip(axes, subsets):
        ax.set_facecolor("#1A1A1A")
        values = subset["delta_pos"].tolist() if not subset.empty else [0.0]
        ax.boxplot(values, patch_artist=True)
        scatter_colors = ["green" if val > 0 else "red" for val in values]
        ax.scatter(np.ones(len(values)), values, c=scatter_colors, zorder=3)
        ax.axhline(0.0, color="#FF4444", linestyle="--")
        ax.set_title(f"{title}\n{np.mean(values):.2f} +/- {np.std(values):.2f}", color="white")
        ax.tick_params(colors="white")
    output = Path(cfg["paths"]["viz_output"]) / "delta_position_distribution.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output
