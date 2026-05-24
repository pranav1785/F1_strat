"""SHAP waterfall rendering."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap


def plot_shap_waterfall(circuit: str, shap_values: np.ndarray, feature_values: np.ndarray, feature_names: list[str], cfg: dict) -> Path:
    """Render one SHAP waterfall plot for the SOFT class."""
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 6), facecolor="#0F0F0F")
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[:, 2],
            base_values=0.0,
            data=feature_values,
            feature_names=feature_names,
        ),
        show=False,
    )
    output = Path(cfg["paths"]["viz_output"]) / f"shap_waterfall_{circuit}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().set_facecolor("#0F0F0F")
    plt.savefig(output, dpi=150, facecolor="#0F0F0F", bbox_inches="tight")
    plt.close(fig)
    return output
