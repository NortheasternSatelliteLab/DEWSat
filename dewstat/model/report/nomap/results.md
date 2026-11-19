# With vs without the normal map

Without the map, the model predicts soil moisture itself from the same satellite inputs (it still gets season and location, so it can learn the usual pattern on its own). With the map (V2), it predicts wetter or drier than the cell's normal, and the map is added back. HLS 2020+ data, 22 test regions; FLDAS 2024-01..2026-08, NLDAS 2024-2025. Each run picked by validation loss, never by test.

| Answer key, step | With map R² | **Without map R²** | Normal map alone R² | Persistence R² | With map RMSE | Without map RMSE | Without ÷ persistence (MSE) | Anomaly correlation with / without | Without map better in |
|---|---|---|---|---|---|---|---|---|---|
| FLDAS, monthly | 0.793 | **0.582** | 0.741 | 0.781 | 0.0363 | 0.0516 | 1.91× | 0.30 / 0.19 | 2/22 regions |
| NLDAS, monthly | 0.870 | **0.395** | 0.869 | 0.761 | 0.0259 | 0.0557 | 2.53× | 0.15 / -0.00 | 0/22 regions |
| NLDAS, weekly | 0.848 | **0.372** | 0.831 | 0.933 | 0.0272 | 0.0596 | 10.90× | 0.25 / 0.06 | 0/22 regions |

Picked runs: FLDAS, monthly: `anom_h16_win12` (with), `sat_h32_win6` (without); NLDAS, monthly: `anom_h16_win3` (with), `sat_h16_win3` (without); NLDAS, weekly: `anom_h16_win4` (with), `sat_h16_win52` (without)

## Every setting without the map (picked = lowest validation loss; test for reference only)

| Answer key | Setting | Val loss | Test RMSE | Test R² |
|---|---|---|---|---|
| FLDAS, monthly | `sat_h32_win6` (picked) | 0.3495 | 0.0516 | 0.582 |
| FLDAS, monthly | `sat_h16_win6` | 0.3498 | 0.0518 | 0.578 |
| FLDAS, monthly | `sat_h16_win3` | 0.3518 | 0.0519 | 0.557 |
| FLDAS, monthly | `sat_h16_win12` | 0.3525 | 0.0517 | 0.580 |
| FLDAS, monthly | `sat_h16_win6_noloc` | 0.4241 | 0.0615 | 0.405 |
| NLDAS, monthly | `sat_h16_win3` (picked) | 0.5831 | 0.0557 | 0.395 |
| NLDAS, monthly | `sat_h32_win6` | 0.6009 | 0.0560 | 0.417 |
| NLDAS, monthly | `sat_h16_win6` | 0.6027 | 0.0566 | 0.404 |
| NLDAS, monthly | `sat_h16_win12` | 0.6335 | 0.0555 | 0.428 |
| NLDAS, monthly | `sat_h16_win6_noloc` | 0.7539 | 0.0603 | 0.325 |
| NLDAS, weekly | `sat_h16_win52` (picked) | 0.6145 | 0.0596 | 0.372 |
| NLDAS, weekly | `sat_h16_win13` | 0.6429 | 0.0560 | 0.421 |
| NLDAS, weekly | `sat_h16_win4` | 0.6553 | 0.0553 | 0.374 |
| NLDAS, weekly | `sat_h32_win13` | 0.6617 | 0.0565 | 0.411 |
| NLDAS, weekly | `sat_h16_win26` | 0.6811 | 0.0574 | 0.416 |
