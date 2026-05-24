"""Attention visualization for the pit classifier."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_attention_heatmap(
    race: str,
    driver: str,
    pitstop: int,
    feature_matrix: np.ndarray,
    attention_weights: np.ndarray,
    feature_names: list[str],
    cfg: dict,
) -> Path:
    """Render feature heatmap and attention weights for one pit stop example."""
    z = (feature_matrix - feature_matrix.mean(axis=0, keepdims=True)) / np.clip(feature_matrix.std(axis=0, keepdims=True), 1e-6, None)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0F0F0F")
    for ax in axes:
        ax.set_facecolor("#1A1A1A")
        ax.tick_params(colors="white")
    im = axes[0].imshow(z.T, aspect="auto", cmap="coolwarm")
    axes[0].set_yticks(np.arange(len(feature_names)))
    axes[0].set_yticklabels(feature_names, color="white")
    axes[0].set_xticks(np.arange(feature_matrix.shape[0]))
    axes[0].set_xticklabels([f"n-{feature_matrix.shape[0] - 1 - i}" for i in range(feature_matrix.shape[0])], color="white")
    axes[0].set_title("Feature Heatmap", color="white")
    axes[1].barh(np.arange(len(attention_weights)), attention_weights, color="#E10600")
    axes[1].set_title("Attention Weights", color="white")
    axes[1].set_yticks(np.arange(len(attention_weights)))
    axes[1].set_yticklabels([f"Lap {i + 1}" for i in range(len(attention_weights))], color="white")
    fig.colorbar(im, ax=axes[0])
    output = Path(cfg["paths"]["viz_output"]) / f"attention_heatmap_{race}_{driver}_{pitstop}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output
