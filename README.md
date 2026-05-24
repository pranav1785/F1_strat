# F1 Strategy ML

This repository builds an end-to-end machine learning pipeline for Formula 1 race strategy modeling. The current v1 scope focuses on six circuits: Bahrain, Britain (Silverstone), Spain (Barcelona), Italy (Monza), Monaco, and Singapore. The project combines FastF1 race data, a Temporal Fusion Transformer for tire degradation forecasting, a calibrated LightGBM model for pit-window classification, and an XGBoost model for opening-compound recommendation. The codebase is organized so that data ingestion, feature engineering, modeling, evaluation, and visualization remain separated and reproducible.

The implementation is designed around strict temporal hygiene. Training, validation, and test seasons are isolated by year; the pit classifier uses only point-in-time lap features; and safety-car handling, circuit fingerprinting, and the default constructor subset of Red Bull, Mercedes, Ferrari, and McLaren are explicitly constrained to avoid leakage or unrealistic offline advantages. The included tests use only synthetic data, so core logic can be validated without internet access or GPU hardware.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

FastF1 cache is configured through `config.yaml` and enabled automatically by `pipeline.py`. By default it writes into `./f1_cache`.

## Quick Start

```bash
python pipeline.py --mode full
```

To train only module 2, use:

```bash
python pipeline.py --mode train-pit
```

`train-lstm` remains accepted as a compatibility alias and now runs the LightGBM pit-window module.

## Modules

- `data/ingest.py`: loads FastF1 race and qualifying sessions with resilient error handling.
- `data/features.py`: builds all stint-aware engineered features used by the models.
- `data/labels.py`: encodes pit-stop labels and applies training-time cleaning/filtering rules.
- `data/circuit_fingerprint.py`: computes six-dimensional circuit embeddings from training data only.
- `data/dataset.py`: builds TFT datasets and XGBoost inputs.
- `models/tft/`: defines, trains, and evaluates the degradation forecaster.
- `models/pit_classifier/`: trains the v3 LightGBM pit-window classifier with isotonic calibration and validation-tuned per-circuit thresholds.
- `models/compound_rec/`: trains the XGBoost recommender and runs SHAP analysis.
- `eval/`: centralizes metrics, counterfactual analysis, ablations, and LOCO evaluation.
- `viz/`: generates degradation, strategy, attention, SHAP, and delta-position artifacts.
- `pipeline.py`: orchestrates end-to-end execution from raw data to final reports.

## Expected Runtime

For the current 6-circuit v1 subset, runtime should be materially lower than the original full-calendar target. The exact duration still depends on FastF1 cache warmness, network conditions, and whether training runs on CPU or GPU.

## Expected Outputs

- Checkpoints in `./checkpoints/`
- Processed parquet files in `./processed/`
- Circuit fingerprints in `./processed/circuit_fingerprints.parquet`
- Visualization artifacts in `./visualizations/`
- `results_summary.json` in the repository root

## Metric Targets

| Metric | Target |
|---|---:|
| TFT P50 MAE | 0.15 |
| TFT P90 Coverage | 0.90 |
| TFT Cliff Lead Time | 2.0 |
| LightGBM Pit AP | 0.72 |
| LightGBM Pit Recall | 0.68 |
| LightGBM Pit Precision | 0.55 |
| LightGBM Pit F1 | 0.55 |
| XGBoost Accuracy | 0.70 |
| LOCO Permanent MAE | 0.22 |
| LOCO Street MAE | 0.40 |

## Known Limitations

- The pit classifier is a point-in-time LightGBM model; attention heatmaps from the old LSTM module are no longer generated.
- The default v1 constructor subset is Red Bull, Mercedes, Ferrari, and McLaren; this improves iteration speed but narrows generalization.
- The 2022 regulation break creates a distribution shift between the earlier training seasons and later validation/test seasons.
