"""Interactive race strategy overlays."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def plot_strategy_overlay(race: str, race_df: pd.DataFrame, divergence_df: pd.DataFrame, cfg: dict) -> Path:
    """Create an interactive strategy overlay chart for one race."""
    palette = {"SOFT": "#FF4444", "MEDIUM": "#FFD700", "HARD": "#CCCCCC"}
    fig = go.Figure()
    drivers = list(race_df["Driver"].dropna().unique())
    for y_idx, driver in enumerate(drivers):
        driver_df = race_df[race_df["Driver"] == driver].sort_values("LapNumber")
        for _, stint_df in driver_df.groupby("Stint"):
            fig.add_trace(
                go.Bar(
                    x=[int(stint_df["LapNumber"].max() - stint_df["LapNumber"].min() + 1)],
                    y=[driver],
                    base=[int(stint_df["LapNumber"].min())],
                    orientation="h",
                    marker=dict(color=palette.get(stint_df["Compound"].iloc[0], "#E10600")),
                    hovertext=stint_df.apply(
                        lambda row: f"Lap {row['LapNumber']}<br>{row['Compound']}<br>TireAge {row['TyreLife']}<br>Delta {row.get('lap_time_delta_s', 0):.3f}",
                        axis=1,
                    ),
                    name=driver if y_idx == 0 else None,
                    showlegend=False,
                )
            )
        for _, row in divergence_df[divergence_df["Driver"] == driver].iterrows():
            fig.add_vline(x=row["LapNumber"], line_color="#E10600")
            annotation_text = row.get("annotation_text")
            if pd.isna(annotation_text) or annotation_text is None:
                annotation_text = f"Delta {row['delta_pos']:.2f}"
            fig.add_annotation(x=row["LapNumber"], y=driver, text=str(annotation_text), showarrow=True)
    fig.update_layout(
        template="plotly_white",
        title=f"Strategy Overlay - {race}",
        barmode="overlay",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
    )
    output = Path(cfg["paths"]["viz_output"]) / f"strategy_overlay_{race}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output))
    return output
