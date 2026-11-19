# Weekly vs monthly (NLDAS-2 answer key)

Same V2 model and inputs (NDVI, NDVI anomaly, VCI, has_photo, season, location), same 102 regions and splits. Each step's setting picked by validation loss. Test = 22 test regions; FLDAS 2024-01..2026-08, NLDAS 2024-2025. Persistence needs ground data the satellite does not have.

| Answer key, step | Picked setting | Model R² | Normal map R² | Persistence R² | Model RMSE | Persistence RMSE | Model ÷ persistence (MSE) | Anomaly correlation | Beats persistence | Beats normal map |
|---|---|---|---|---|---|---|---|---|---|---|
| FLDAS, monthly (current model) | `v2_satellite_anomaly` | **0.793** | 0.741 | 0.781 | 0.0363 | 0.0373 | 0.95× | 0.30 | 11/22 | n/a |
| NLDAS, monthly | `anom_h16_win3` | **0.870** | 0.869 | 0.761 | 0.0259 | 0.0350 | 0.55× | 0.15 | 16/22 | 13/22 |
| NLDAS, weekly | `anom_h16_win4` | **0.848** | 0.831 | 0.933 | 0.0272 | 0.0180 | 2.28× | 0.25 | 2/22 | 16/22 |

## Every setting (picked = lowest validation loss; test shown for reference only)

| Step | Setting | Val loss | Test RMSE | Test R² |
|---|---|---|---|---|
| month | `anom_h16_win3` (picked) | 0.8406 | 0.0259 | 0.870 |
| month | `anom_h16_win6_noloc` | 0.8507 | 0.0253 | 0.881 |
| month | `anom_h16_win6` | 0.8689 | 0.0260 | 0.874 |
| month | `anom_h32_win6` | 0.8780 | 0.0273 | 0.861 |
| month | `anom_h16_win12` | 0.9051 | 0.0262 | 0.872 |
| week | `anom_h16_win4` (picked) | 0.9032 | 0.0272 | 0.848 |
| week | `anom_h16_win13` | 0.9514 | 0.0300 | 0.834 |
| week | `anom_h16_win26` | 0.9701 | 0.0306 | 0.834 |
| week | `anom_h32_win13` | 0.9750 | 0.0305 | 0.828 |
| week | `anom_h16_win52` | 0.9987 | 0.0294 | 0.847 |
