"""Training helpers for the Temporal Fusion Transformer."""

from __future__ import annotations

import logging
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.tuner import Tuner

from models.tft.model import build_tft

LOGGER = logging.getLogger(__name__)


def train_tft(train_ds, val_ds, cfg: dict):
    """Train the TFT with Lightning callbacks and checkpointing.

    Args:
        train_ds: Training ``TimeSeriesDataSet``.
        val_ds: Validation ``TimeSeriesDataSet``.
        cfg: Project configuration.

    Returns:
        Trained TFT model.
    """
    pl.seed_everything(cfg["tft"]["seed"])
    train_loader = train_ds.to_dataloader(train=True, batch_size=cfg["tft"]["batch_size"])
    val_loader = val_ds.to_dataloader(train=False, batch_size=cfg["tft"]["batch_size"])
    model = build_tft(train_ds, cfg)

    checkpoint_path = Path(cfg["paths"]["tft_checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg["tft"]["patience"],
            min_delta=cfg["tft"].get("early_stopping_min_delta", 0.0),
            mode="min",
        ),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=str(checkpoint_path.parent),
            filename=checkpoint_path.stem,
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(
        max_epochs=cfg["tft"]["max_epochs"],
        gradient_clip_val=cfg["tft"]["gradient_clip_val"],
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_loader)
    best_path = callbacks[1].best_model_path
    LOGGER.info("Best TFT checkpoint: %s", best_path)
    if best_path:
        checkpoint = model.__class__.load_from_checkpoint(best_path)
        return checkpoint
    return model


def run_lr_finder(model, train_loader, cfg: dict) -> float:
    """Run Lightning's LR finder and return the suggested learning rate.

    Args:
        model: Lightning module to tune.
        train_loader: Training dataloader.
        cfg: Project configuration.

    Returns:
        Suggested learning rate.
    """
    trainer = pl.Trainer(max_epochs=cfg["tft"]["max_epochs"], logger=False, enable_progress_bar=False)
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_dataloaders=train_loader)
    suggestion = lr_finder.suggestion()
    LOGGER.info("Suggested TFT learning rate: %s", suggestion)
    return float(suggestion)
