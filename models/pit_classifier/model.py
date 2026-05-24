"""LightGBM pit-window classifier wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PitWindowLightGBM:
    """Persistable binary pit-window model with calibration metadata."""

    estimator: object
    calibrator: object | None
    feature_names: list[str]
    circuit_thresholds: dict[str, float]
    global_threshold: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return calibrated pit-window probabilities for a feature frame."""
        X = frame.reindex(columns=self.feature_names, fill_value=0.0).fillna(0.0)
        raw = self.estimator.predict_proba(X)[:, 1]
        if self.calibrator is None:
            return raw
        return self.calibrator.transform(raw)

    def predict_labels(self, frame: pd.DataFrame, circuit_col: str = "event_name") -> np.ndarray:
        """Return pipeline-compatible labels: 2 means pit signal, 0 means stay out."""
        probabilities = self.predict_proba(frame)
        circuits = frame[circuit_col].astype(str).to_numpy() if circuit_col in frame.columns else np.asarray([""] * len(frame))
        thresholds = np.asarray(
            [self.circuit_thresholds.get(circuit, self.global_threshold) for circuit in circuits],
            dtype=np.float64,
        )
        return np.where(probabilities >= thresholds, 2, 0).astype(int)
