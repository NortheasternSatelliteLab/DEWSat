# Run report: teacher

- Data: **synthetic** (`synthetic_soil_store_percentile`)
- Task: lead **3** month(s), 12-month input window
- Split mode: **temporal**, counts {'train': 1957, 'val': 314, 'test': 619}
- Model: 3,832,065 parameters (3x256 bi-LSTM, pool `final_masked_mean`)
- Trained in 18.5s on `mps` (1 member(s))

## What this run does and does not show

| Question | Answer from this run |
|---|---|
| Does the pipeline run end to end? | Yes: 2890 windows, reload difference 1.19e-07 |
| Better than carrying the last value forward? | RMSE skill 0.404 vs persistence |
| Better than seasonal climatology? | RMSE skill 0.281 |
| Better than a linear model on the same window? | RMSE skill -0.020 vs ridge |
| Drought accuracy on real observations? | No: synthetic data |
| Jetson latency, energy or INT8 accuracy? | No: run `dewsat bench` on the device |

_Synthetic data: these numbers test the software, not drought accuracy. Forecast at 3-month lead using only prior months. Temporal holdout only; every test location was seen in training._

## Held-out test metrics

| Predictor | RMSE | MAE | R2 | Spearman | Dry RMSE | POD | FAR | CSI | HSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Bi-LSTM** | 0.17456 | 0.13562 | 0.5377 | 0.7544 | 0.12687 | 0.766 | 0.344 | 0.546 | 0.560 |
| persistence | 0.29297 | 0.23802 | -0.3022 | 0.3594 | 0.30000 | 0.469 | 0.511 | 0.315 | 0.251 |
| ridge | 0.17108 | 0.13253 | 0.5560 | 0.7604 | 0.15269 | 0.583 | 0.296 | 0.469 | 0.497 |
| seasonal_climatology | 0.24263 | 0.20081 | 0.1069 | 0.5674 | 0.26599 | 0.260 | 0.468 | 0.212 | 0.183 |
| train_mean | 0.27441 | 0.23432 | -0.1424 | n/a | 0.39202 | 0.000 | n/a | 0.000 | 0.000 |

Dry event threshold: target < 0.2, base rate 0.310. POD is hit rate, FAR is false alarm ratio, CSI is critical success index, HSS is Heidke skill score.

## Data admission

- Rows: 5499 over 64 cells, 2015-01..2024-12
- Observed fraction of the within-cell month span: 0.7202
- Interior missing months: 2136 (imputed with the observed flag set to 0)
- Rows dropped by min_valid_frac: 0
- Windows rejected: {'target_month_unavailable': 1263, 'anchor_month_missing': 1247, 'too_few_observed_months': 1531}
- Mean observed months per window: 9.342 of 12
- Scaler (zscore) fitted on 3405 monthly rows ending 2021-09

## Selection

| Member | Seed | Epochs run | Best epoch | Validation RMSE |
|---:|---:|---:|---:|---:|
| 0 (kept) | 42 | 24 | 12 | 0.15961 |

## Provenance

- Command: `.venv/bin/dewsat train --data data/synthetic_monthly.csv --config configs/teacher.json --out runs/teacher`
- Input: `data/synthetic_monthly.csv`
- SHA256: `831376ffcea2551e7f7d2cad370d84c04b0e6905fc6a923900b70a6ea034dc87`
- dewsat 2.0.0, Python 3.14.4, torch 2.14.0, numpy 2.5.3
- Platform: macOS-26.6.2-arm64-arm-64bit-Mach-O (arm64)
- Checkpoint: `best.pt`, 15,339,293 bytes
