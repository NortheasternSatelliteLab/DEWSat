# DEWSat drought estimation: data card (DEWSat_BiLSTM_v2 dataset)

For Prof. Fatemeh Ghoreishi · 2026-11-09 · code `9745865` in `code`. Every number comes from a file listed under Sources; "not recorded" means no file records it.

**Credits.** As in the paper run: the original 102 regions, the HLS product and Fmask choices, the 30% clear-cell rule, the ~120 m pixel and the pull design are Austin Schlather's (`dewstat-northeastern@7adb8d6`); the 98° W East/West split is Ryan's (`RyanBoomer30/Atlas@227042b`). The pull code is the paper run's (`f17e700`) with SWIR1 and per-scene values added.

## 1. Inputs and target

| Step | What the code does |
|---|---|
| Imagery | HLS v2.0 S30 and L30: Red (B04), NIR (B8A for S30, B05 for L30), SWIR1 (B11 for S30, B06 for L30, what-if only) and Fmask. 2016–2019 from NASA LP DAAC (CMR version 2.0), 2020-01 to 2026-09 from Microsoft Planetary Computer (`hls2-s30`, `hls2-l30`); one reader for both. Granules not ending in `.v2.0` are skipped. Scenes with more than 80% cloud are not searched. |
| 30 m to 120 m | Each band from the file's 4× overview (~120 m), warped with nearest neighbour onto a fixed EPSG:4326 grid per region with a whole number of pixels per 0.1° FLDAS cell. |
| Clear pixel | Fmask not fill; bits 1–5 all 0 (cloud, adjacent cloud/shadow, shadow, snow/ice, water); aerosol not high; Red and NIR > 0. A scene counts for a cell's monthly composite only if at least 30% of the cell is clear. |
| Monthly composite | Per pixel NDVI; same-date granules averaged, then the median over the month's distinct dates (the paper's rule, reproduced bit for bit on probe months). NDMI = (NIR − SWIR1)/(NIR + SWIR1) is composited the same way. |
| Per-view values (new) | Per (sensor, MGRS tile, date) view, granules merged pixel by pixel; per cell: covered and clear fraction (no 30% rule applied), NDVI mean, population SD, 10th percentile, NDMI mean. |
| VCI | Per pixel and calendar month, min/max over the reference months of each fold (development months 2016–2023 outside the held-out block ± 12 months), leave-one-year-out, ≥ 3 years, range ≥ 0.05; clipped 0–100. Fresh test: all development months 2016–2023. |

**Features** (per cell-month, over usable pixels): NDVI_mean, NDVI_std, VCI_mean, VCI_p10, valid_frac, month sin/cos, observed (usable ≥ 5% of the cell) and months since the last observation. Z-scores of the first four are fit on each model's own training windows.

**Label.** FLDAS_NOAH01_C_GL_M v001 SoilMoi10_40cm_tavg, 2001-01 to 2026-09 (309 monthly granules, one CONUS hyperslab each through Earthdata OPeNDAP). Drought = below the numpy-linear 20th percentile of the cell's 2001–2022 values for that calendar month; cells with fewer than 20 reference values in any month are dropped.

The new pull reproduces the paper run's FLDAS values for the 102 regions: 810,816 values, max absolute difference 1.5e-08 m³/m³.

## 2. The dataset

**Regions.** New regions are 5 × 5 FLDAS cells (0.5°) across the contiguous U.S., ≥ 75% vegetated in ESA CCI Land Cover 2020 (300 m), all 25 cell centres inside CONUS (Census 2023 boundaries). Development regions are ≥ 0.25° from each other and the existing 102; fresh-test regions ≥ 0.5° from every other region. 75,103 box positions qualified; the layout holds 561 development and 164 fresh-test sites, of which 600 form 12 balanced batches of 50 (25 West / 25 East; 12 fresh test).

| | Development: original 102 | Development: new | Fresh test |
|---|---|---|---|
| Regions West / East | 48 / 54 | 228 / 228 | 72 / 72 |
| Cells | 13,977 (both development columns; 48 dropped for label history) | | 3,600 |

**HLS granules searched** (unique ids): Planetary Computer S30 992,473, L30 545,687; NASA S30 365,685, L30 188,888; all v2.0: 2,092,733.

**Per-view record**: 1,726,243 views (L30 610,990, S30 1,115,253), 43,156,075 view × cell rows, 12,129,773 of them ≥ 30% clear.

**Drought base rate**, development 2016–2023: 0.2212 (West 0.2340, East 0.2086). Share of labelled development cell-months observed (all-development VCI table): 0.834. Distinct clear dates per region-month: median 6 (range 0–19).

**Windows per job and fold** (development):

| Job, fold | Outer train | Outer validation | Inner train | Inner validation |
|---|---|---|---|---|
| job1_fold1 | 508,428 | 65,750 | 201,891 | 34,353 |
| job1_fold2 | 350,186 | 67,014 | 115,516 | 29,453 |
| job1_fold3 | 323,298 | 67,160 | 164,182 | 25,857 |
| job1_fold4 | 480,333 | 67,488 | 188,565 | 33,208 |
| job2_fold1 | 462,184 | 6,215 | 182,843 | 4,511 |
| job2_fold2 | 234,082 | 8,666 | 63,479 | 4,827 |
| job2_fold3 | 191,239 | 8,350 | 81,907 | 6,837 |
| job2_fold4 | 342,312 | 8,525 | 117,785 | 5,039 |
| job3_fold1 | 466,492 | 65,750 | 174,112 | 25,625 |
| job3_fold2 | 295,227 | 67,014 | 79,904 | 22,040 |
| job3_fold3 | 272,212 | 67,160 | 150,578 | 15,927 |
| job3_fold4 | 455,742 | 58,816 | 172,787 | 25,058 |

**Size on disk**: new regions 40.9 GB, re-extracted originals 0.8 GB, copy of the paper's composites 3.8 GB, feature cache 1.3 GB.

### VCI image tiles (input of the CNN models added after the development results, amendment 4)

Per cell and month, the pixel VCI behind VCI_mean averaged onto a 23 x 21 grid of near-equal pixel bins (uint8 0-100; an empty bin, or a month the cell is unobserved, is marked empty). One array per VCI table: development 13,977 cells x 96 months for the 4 fold tables and the all-development table; fresh test 3,600 cells x 129 months (all-development table). Built 2026-11-08 15:15:45 in 9.9 minutes; check: each cell's mean VCI recomputed from the same pixels matches VCI_mean within 3.8e-06 VCI units, observed flags agree (0 mismatches). Files `cache/tiles/` (`tiles.json`); code `code/dewsat2/tiles.py`. No labels are read.

### Other drought labels (Phase 6, fetched only after the fresh test was scored)

Two labels from other sources, used to ask whether the fresh-test result is specific to FLDAS. No model, setting or input was chosen with them; the one choice made with SPEI (its timescale) used development cells only.

| | NLDAS-2 soil moisture | gridMET SPEI |
|---|---|---|
| Product | NLDAS_NOAH0125_M v2.0, `SoilM_10_40cm` (Noah model run with NLDAS-2 forcing) | gridMET SPEI (Abatzoglou), 8 timescales 14 days to 5 years |
| Measures | Soil moisture in the same 10–40 cm layer as the label, from a second land-surface run with different forcing | Climatic water balance (precipitation minus reference evapotranspiration), standardized |
| Sampling | 0.125° CONUS grid, 309 monthly granules 2001-01 to 2026-09; each 0.1° cell takes the nearest NLDAS cell | One point series per region at its centre, 2016-01 to 2026-09; a month takes the last value on or before its last day; fetched values −2.09 to +2.09 |
| Drought | Below the cell's calendar-month 20th percentile of 2001–2022 (the FLDAS rule) | SPEI ≤ −0.84 (the standard-normal 20th percentile) |
| Coverage | Every development and fresh-test cell | Timescale choice: 35 West + 35 East development regions (seed 20261109, 1,735 cells) × all timescales; scoring: every fresh-test region at 180d and the two runners-up; 992 series in all |
| Agreement with FLDAS, development 2016–2023 | Flags agree 80.8%, Cohen's kappa 0.458, AUC of the NLDAS percentile for the FLDAS flag 0.824 (1,338,912 cell-months) | AUC of −SPEI for the FLDAS flag: 180d 0.804, 90d 0.796, 270d 0.796, 1y 0.776 (chosen: 180d) |
| Agreement with FLDAS, fresh test | Flags agree 70.8%, kappa 0.364, AUC 0.740 (75,096 cell-months) | See the next line |

On the 64,995 job-1 fresh-test windows where every label exists (FLDAS drought rate 0.395), each label's call agrees with FLDAS's: SPEI-180d 66.0% (drought rate 0.355), SPEI-90d 66.1% (drought rate 0.345), SPEI-270d 64.8% (drought rate 0.381), NLDAS-2 71.3% (drought rate 0.314). The scores against these labels are in report.md section 9a.

## 3. Training, validation and results

Four outer folds = four region groups (balanced East/West) × the year blocks 2016–17, 2018–19, 2020–21, 2022–23, 12-month buffers (15 for forecasts). Inner validation holds out the next group and one year. 30-configuration random searches per network family, fold and job; 10 seeds of the chosen setting. Primary metric: within-fold AUC with 2,000 whole-region resamples. Rules: `code/design/DESIGN.md`, `gate.json`.

**Development** (gate rows):

- Estimation now (month T observed): GRU − Boosted trees (monotone in VCI) −0.033 [−0.047, −0.020] → worse.
- Missing month (T unobserved, >= 8 of the other 11 observed): GRU − Boosted trees (monotone in VCI) −0.075 [−0.112, −0.038] → worse.
- Forecast T+1..T+3, onsets only (not in drought at T): GRU − LR + trend features −0.036 [−0.053, −0.019] → worse.
- Estimation now where VCI changed most (top quartile of 1- or 3-month change): GRU − Boosted trees (monotone in VCI) −0.034 [−0.051, −0.017] → worse.

**Fresh test** (the fresh-test regions, 2025-01 to 2026-09, scored once, 2026-11-09 09:36:30; the pre-registered rows and the rule of report.md section 9):

- Job 1, estimation now, gate: GRU − Boosted trees (monotone in VCI) +0.036 [+0.019, +0.054] → passed.
- Job 1, estimation now, best Bi-LSTM variant: Residual Bi-LSTM − Boosted trees (monotone in VCI) +0.039 [+0.020, +0.057] → passed.
- Job 1, estimation now, against LR on VCI(T): GRU − LR on VCI(T) +0.035 [+0.019, +0.054] → passed.
- Job 1, estimation now, against LR on VCI(T): Residual Bi-LSTM − LR on VCI(T) +0.038 [+0.020, +0.057] → passed.
- Job 2, missing month, gate: GRU − Boosted trees (monotone in VCI) +0.002 [−0.038, +0.042] → no detectable difference.
- Job 2, missing month, best Bi-LSTM variant: Residual Bi-LSTM − Boosted trees (monotone in VCI) +0.013 [−0.024, +0.048] → no detectable difference.
- Job 3, forecast (mean onset AUC, leads 1-3), gate: GRU − LR + trend features +0.024 [−0.007, +0.054] → no detectable difference.
- Job 3, forecast (mean onset AUC, leads 1-3), best Bi-LSTM variant: Bi-LSTM − LR + trend features +0.038 [+0.007, +0.069] → passed.
- Job 4, where VCI changed most, gate: GRU − Boosted trees (monotone in VCI) +0.051 [+0.027, +0.075] → passed.
- Job 4, where VCI changed most, best Bi-LSTM variant: Residual Bi-LSTM − Boosted trees (monotone in VCI) +0.045 [+0.019, +0.070] → passed.

## 4. Mission scope

Unchanged from the 2026-11-05 card (`~/Desktop/DEWSat_Fatemeh_package/data_card.md`, section 4); this dataset changes no mission assumption. SWIR is used only in a labelled what-if, because DEWSat's camera has Red and NIR only.

## Sources

- `data/plan/plan.json`, `data/plan/regions_plan.csv`, `data/plan/batches.json`: candidate count, layout, batches.
- `data/pull_state.json`, `data/FROZEN.json`: pull timing, batches, frozen regions.
- `results/scores/data_counts.json`: regions, cells, granules, views, windows, base rates, sizes.
- `cache/dataset.json`: assembled cells and coverage; `logs/pull/fldas_check.json`: FLDAS agreement.
- `results/scores/scores.json`, `results/scores/phase5.json`: results; `data/locked/FRESH_TEST_SCORED.lock`: when the fresh test was scored.
- `results/scores/phase6.json`: NLDAS-2 and SPEI agreement with FLDAS, the SPEI timescale choice; `results/scores/fresh_labels_posthoc.json`: the fresh-test predictions scored against those labels.
- Land cover: ESA CCI Land Cover 2020 v2.1.1 via Planetary Computer `esa-cci-lc` (`data/landcover/*.json`).
- Boundaries: U.S. Census Bureau `cb_2023_us_state_20m` (`data/boundaries/`).
- HLS v2.0: NASA LP DAAC (HLSS30/HLSL30 v2.0, CMR) and Microsoft Planetary Computer (`hls2-s30`, `hls2-l30`).
- FLDAS: FLDAS_NOAH01_C_GL_M v001 (NASA GES DISC, Earthdata OPeNDAP).
- NLDAS-2: NLDAS_NOAH0125_M v2.0 (NASA GES DISC, Earthdata OPeNDAP), `data/phase6/nldas/`.
- SPEI: gridMET drought indices (Abatzoglou), Northwest Knowledge Network THREDDS, NetCDF Subset Service (`MET/spei/spei<timescale>.nc`), `data/phase6/spei/`.

