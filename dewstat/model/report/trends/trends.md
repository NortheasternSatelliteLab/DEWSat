# What the weekly model learned (run `runs_valgap_week/anom_h16_win4`, picked by validation loss)

22 test regions, 2024-2025: places and years the model never trained on. Validation: val regions in 2023 only.

- Test cell-weeks: 52,019. R² model 0.848, normal map 0.831.
- Wetter/drier than normal: model vs actual correlation 0.25.

## 1. Greener than usual → wetter than usual

Correlation of the latest NDVI anomaly with the actual soil-moisture anomaly: 0.18; with the model's predicted anomaly: 0.64.

| NDVI anomaly (group median) | Actual SM − normal | Model SM − normal | Cell-weeks |
|---|---|---|---|
| -0.091 | -0.0185 | -0.0228 | 5,211 |
| -0.040 | -0.0163 | -0.0138 | 5,214 |
| -0.021 | -0.0121 | -0.0106 | 5,215 |
| -0.009 | -0.0082 | -0.0080 | 5,227 |
| +0.000 | -0.0066 | -0.0054 | 5,170 |
| +0.009 | -0.0058 | -0.0037 | 5,189 |
| +0.019 | -0.0040 | -0.0022 | 5,203 |
| +0.033 | -0.0013 | -0.0005 | 5,189 |
| +0.054 | -0.0002 | +0.0013 | 5,211 |
| +0.098 | +0.0006 | +0.0047 | 5,190 |

## 2. When in the year the satellite helps

| Month | Error cut vs normal map | Anomaly correlation | Weeks with a clear photo |
|---|---|---|---|
| Jan | -9% | 0.06 | 61% |
| Feb | -8% | 0.02 | 59% |
| Mar | +16% | 0.34 | 70% |
| Apr | +8% | 0.26 | 75% |
| May | +8% | 0.35 | 71% |
| Jun | +8% | 0.29 | 76% |
| Jul | +3% | 0.24 | 75% |
| Aug | +2% | 0.24 | 76% |
| Sep | +13% | 0.20 | 82% |
| Oct | +31% | 0.35 | 84% |
| Nov | +12% | 0.18 | 66% |
| Dec | +15% | 0.21 | 57% |

## 3. Across all test regions, week by week

Correlation of the weekly average anomaly (model vs actual): 0.66. See 3_test_average.png.

## 4. Big vs small departures from normal

| Actual departure from normal | Right direction (wetter/drier) | Median share of the departure predicted | Cell-weeks |
|---|---|---|---|
| < 0.01 m³/m³ | 52% | 9% | 16,481 |
| 0.01-0.02 m³/m³ | 58% | 12% | 13,558 |
| 0.02-0.04 m³/m³ | 67% | 14% | 14,188 |
| 0.04-0.08 m³/m³ | 77% | 15% | 6,989 |
| > 0.08 m³/m³ | 72% | 13% | 798 |

## 5. By region

Regions where the model beats the map: 16 of 22. Across regions, the error cut correlates 0.08 with longitude (east +) and -0.17 with the region's usual soil moisture.

| Region | Error cut vs map | Model anomaly correlation | NDVI anomaly correlation | Usual SM |
|---|---|---|---|---|
| nebraska_panhandle | +39% | 0.58 | 0.60 | 0.186 |
| texas_south_brush | +33% | 0.59 | 0.51 | 0.191 |
| texas_blackland | +30% | 0.56 | 0.45 | 0.228 |
| indiana_corn | +29% | 0.43 | 0.18 | 0.296 |
| michigan_northern_lp | +26% | 0.30 | 0.10 | 0.169 |
| louisiana_rice | +24% | 0.58 | 0.48 | 0.285 |
| pennsylvania_lancaster | +24% | 0.43 | 0.25 | 0.283 |
| kentucky_bluegrass | +23% | 0.49 | 0.47 | 0.278 |
| oregon_willamette | +19% | 0.35 | 0.26 | 0.275 |
| idaho_upper_snake | +14% | 0.27 | 0.11 | 0.180 |
| montana_east | +14% | 0.14 | 0.23 | 0.239 |
| alabama_wiregrass | +14% | 0.42 | 0.26 | 0.202 |
| eastern_washington | +11% | 0.16 | 0.03 | 0.252 |
| georgia_coastal_plain | +8% | 0.40 | 0.38 | 0.182 |
| nevada_great_basin | +6% | 0.25 | 0.20 | 0.204 |
| south_carolina_sandhills | +3% | 0.18 | 0.18 | 0.160 |
| maine_aroostook | -7% | -0.07 | 0.03 | 0.287 |
| arizona_southeast | -10% | 0.39 | 0.24 | 0.175 |
| california_sacramento | -17% | -0.16 | -0.24 | 0.184 |
| north_dakota_east | -26% | 0.07 | 0.08 | 0.244 |
| kansas_flint_hills | -42% | 0.13 | 0.27 | 0.302 |
| minnesota_south | -55% | 0.28 | 0.05 | 0.266 |
