"""Temporal Fusion Transformer builder."""

from __future__ import annotations

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


def build_tft(training_dataset, cfg: dict) -> TemporalFusionTransformer:
    """Instantiate a TFT from a training dataset.

    Args:
        training_dataset: ``pytorch_forecasting.TimeSeriesDataSet`` instance.
        cfg: Project configuration.

    Returns:
        Configured TFT model.
    """
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=cfg["tft"]["hidden_size"],
        attention_head_size=cfg["tft"]["attention_head_size"],
        lstm_layers=cfg["tft"]["lstm_layers"],
        dropout=cfg["tft"]["dropout"],
        output_size=len(cfg["tft"]["quantiles"]),
        loss=QuantileLoss(cfg["tft"]["quantiles"]),
        learning_rate=cfg["tft"]["learning_rate"],
        reduce_on_plateau_patience=5,
        log_interval=10,
        log_val_interval=1,
    )
