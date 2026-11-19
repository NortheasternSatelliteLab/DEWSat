# Validation years: all years vs 2023 only

First runs: validation = the 13 val regions in **every** year, 2020-2025, so it shared years with training (2020-2022) and with test (2024-2025), though never places. Re-run: validation = the val regions in **2023 only**, a year used by neither training nor test. Everything else is the same. 22 test regions, 2024+.

| Model | Validation | Picked setting | Test R² | Test RMSE | Normal map R² | Persistence R² | Beats persistence in |
|---|---|---|---|---|---|---|---|
| FLDAS monthly, with map (V2) | all years | `anom_h16_win12` | **0.793** | 0.0363 | 0.741 | 0.781 | 11/22 |
| FLDAS monthly, with map (V2) | 2023 only | `anom_h16_win12` | **0.800** | 0.0357 | 0.741 | 0.781 | 11/22 |
| FLDAS monthly, no map | all years | `sat_h32_win6` | **0.582** | 0.0516 | 0.741 | 0.781 | 2/22 |
| FLDAS monthly, no map | 2023 only | `sat_h16_win12` | **0.580** | 0.0517 | 0.741 | 0.781 | 3/22 |
| NLDAS weekly, with map (final) | all years | `anom_h16_win4` | **0.848** | 0.0272 | 0.831 | 0.933 | 2/22 |
| NLDAS weekly, with map (final) | 2023 only | `anom_h16_win4` | **0.848** | 0.0272 | 0.831 | 0.933 | 2/22 |
| NLDAS weekly, no map | all years | `sat_h16_win52` | **0.372** | 0.0596 | 0.844 | 0.938 | 0/22 |
| NLDAS weekly, no map | 2023 only | `sat_h16_win52` | **0.365** | 0.0599 | 0.844 | 0.938 | 0/22 |

## Every setting (picked = lowest validation loss; test R² for reference only)

| Model | Validation | Setting | Val loss | Test R² |
|---|---|---|---|---|
| FLDAS monthly, with map (V2) | all years | `anom_h16_win12` (picked) | 0.7635 | 0.793 |
| FLDAS monthly, with map (V2) | all years | `anom_h16_win3` | 0.8507 | 0.771 |
| FLDAS monthly, with map (V2) | all years | `anom_h16_win6` | 0.8632 | 0.792 |
| FLDAS monthly, with map (V2) | all years | `anom_h32_win6` | 0.8655 | 0.792 |
| FLDAS monthly, with map (V2) | all years | `anom_h16_win6_noloc` | 0.8859 | 0.788 |
| FLDAS monthly, with map (V2) | 2023 only | `anom_h16_win12` (picked) | 0.9238 | 0.800 |
| FLDAS monthly, with map (V2) | 2023 only | `anom_h16_win3` | 0.9584 | 0.777 |
| FLDAS monthly, with map (V2) | 2023 only | `anom_h32_win6` | 0.9975 | 0.793 |
| FLDAS monthly, with map (V2) | 2023 only | `anom_h16_win6` | 1.0207 | 0.792 |
| FLDAS monthly, with map (V2) | 2023 only | `anom_h16_win6_noloc` | 1.0231 | 0.789 |
| FLDAS monthly, no map | all years | `sat_h32_win6` (picked) | 0.3495 | 0.582 |
| FLDAS monthly, no map | all years | `sat_h16_win6` | 0.3498 | 0.578 |
| FLDAS monthly, no map | all years | `sat_h16_win3` | 0.3518 | 0.557 |
| FLDAS monthly, no map | all years | `sat_h16_win12` | 0.3525 | 0.580 |
| FLDAS monthly, no map | all years | `sat_h16_win6_noloc` | 0.4241 | 0.405 |
| FLDAS monthly, no map | 2023 only | `sat_h16_win12` (picked) | 0.4167 | 0.580 |
| FLDAS monthly, no map | 2023 only | `sat_h16_win3` | 0.4258 | 0.568 |
| FLDAS monthly, no map | 2023 only | `sat_h16_win6` | 0.4312 | 0.578 |
| FLDAS monthly, no map | 2023 only | `sat_h32_win6` | 0.4367 | 0.579 |
| FLDAS monthly, no map | 2023 only | `sat_h16_win6_noloc` | 0.5413 | 0.424 |
| NLDAS weekly, with map (final) | all years | `anom_h16_win4` (picked) | 0.9032 | 0.848 |
| NLDAS weekly, with map (final) | all years | `anom_h16_win13` | 0.9514 | 0.834 |
| NLDAS weekly, with map (final) | all years | `anom_h16_win26` | 0.9701 | 0.834 |
| NLDAS weekly, with map (final) | all years | `anom_h32_win13` | 0.9750 | 0.828 |
| NLDAS weekly, with map (final) | all years | `anom_h16_win52` | 0.9987 | 0.847 |
| NLDAS weekly, with map (final) | 2023 only | `anom_h16_win4` (picked) | 0.8916 | 0.848 |
| NLDAS weekly, with map (final) | 2023 only | `anom_h16_win13` | 0.9876 | 0.834 |
| NLDAS weekly, with map (final) | 2023 only | `anom_h32_win13` | 1.0028 | 0.832 |
| NLDAS weekly, with map (final) | 2023 only | `anom_h16_win26` | 1.1261 | 0.835 |
| NLDAS weekly, with map (final) | 2023 only | `anom_h16_win52` | 1.1843 | 0.842 |
| NLDAS weekly, no map | all years | `sat_h16_win52` (picked) | 0.6145 | 0.372 |
| NLDAS weekly, no map | all years | `sat_h16_win13` | 0.6429 | 0.421 |
| NLDAS weekly, no map | all years | `sat_h16_win4` | 0.6553 | 0.374 |
| NLDAS weekly, no map | all years | `sat_h32_win13` | 0.6617 | 0.411 |
| NLDAS weekly, no map | all years | `sat_h16_win26` | 0.6811 | 0.416 |
| NLDAS weekly, no map | 2023 only | `sat_h16_win52` (picked) | 0.6661 | 0.365 |
| NLDAS weekly, no map | 2023 only | `sat_h16_win4` | 0.6744 | 0.382 |
| NLDAS weekly, no map | 2023 only | `sat_h16_win13` | 0.6947 | 0.421 |
| NLDAS weekly, no map | 2023 only | `sat_h32_win13` | 0.7272 | 0.406 |
| NLDAS weekly, no map | 2023 only | `sat_h16_win26` | 0.7580 | 0.411 |
