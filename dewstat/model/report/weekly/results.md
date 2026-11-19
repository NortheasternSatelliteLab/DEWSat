# Daily vs weekly vs monthly

Answer key: NLDAS-2 Noah soil moisture 10-40 cm, daily (12:00 UTC), at the same 2,625 cells as the model. Scored on 2024-2025; normal maps from 2019-2023. Persistence = last step's value (needs ground data).

## 22 test regions

| Step | Persistence R² | Normal map R² | Persistence RMSE | Normal map RMSE | Persistence ÷ normal (MSE) | Anomaly already known one step earlier (R²) | Typical change per step |
|---|---|---|---|---|---|---|---|
| Daily | 0.982 | 0.820 | 0.0102 | 0.0322 | 0.10× | 0.900 | 0.0102 |
| Weekly | 0.939 | 0.826 | 0.0186 | 0.0313 | 0.35× | 0.630 | 0.0186 |
| Monthly | 0.770 | 0.865 | 0.0352 | 0.0269 | 1.70× | 0.154 | 0.0351 |

## all 102 regions

| Step | Persistence R² | Normal map R² | Persistence RMSE | Normal map RMSE | Persistence ÷ normal (MSE) | Anomaly already known one step earlier (R²) | Typical change per step |
|---|---|---|---|---|---|---|---|
| Daily | 0.981 | 0.797 | 0.0099 | 0.0324 | 0.09× | 0.907 | 0.0099 |
| Weekly | 0.935 | 0.803 | 0.0181 | 0.0316 | 0.33× | 0.645 | 0.0181 |
| Monthly | 0.791 | 0.842 | 0.0318 | 0.0276 | 1.32× | 0.169 | 0.0318 |

## Clear photos (≥ 30% of the cell cloud-free), 2021-2025, average over 2,625 cells

| Satellites | Days with a photo | Weeks with a photo | Months with a photo |
|---|---|---|---|
| Landsat only (2 satellites) | 6% | 38% | 78% |
| all HLS (4-5 satellites) | 17% | 67% | 92% |

For reference, on FLDAS (monthly, 22 test regions, 2024+): model V2 R² 0.79, persistence 0.78, normal map 0.74.
