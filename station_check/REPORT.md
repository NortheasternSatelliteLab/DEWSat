# Station check: report

Run 2026-11-11, following `PLAN.md`, which was written before any result. Amendment 1 was added after the pilot and
before any full-run metric. Every number below is in `results/pilot.json` or `results/full.json`. The scripts are in
`scripts/`. Nothing was trained, no label was changed, and no input folder was modified.

## Answers

1. **FLDAS is a good-enough label; switching to NLDAS-2 is not supported.**
   - Against measured soil moisture at 230 stations, FLDAS and NLDAS-2 agree about equally well. Their anomaly
     correlation with the probes is about 0.5.
   - NLDAS-2 is slightly better on drought calls (kappa +0.045), and FLDAS is slightly better on anomalies.
   - Under the plan's rule, neither is closer to measured soil.
2. **The satellite's ceiling is not caused by the label.**
   - VCI ranks *measured* drought less well than *FLDAS* drought: AUC 0.660 against 0.704 in the growing season,
     difference −0.045 [−0.067, −0.023].
   - Making the satellite's footprint match the probe (1 km instead of 11 km) did not change agreement with the
     probes.
   - A better label would not unlock a stronger satellite signal.
3. **The 10-year VCI reference does not help.** It clips fewer pixels, but its AUC is slightly lower: −0.006
   [−0.010, −0.001] against station drought. The paper's 7-year reference stays.
4. **The paper gains a ground-truth number.** Against measured soil-moisture drought at 230 stations
   (2016–2024, April–October), 1 km VCI ranks drought months with AUC **0.660 [0.640, 0.679]**. That is 0.680 in the
   West and 0.640 in the East.

## Data checks (pilot, PLAN.md 6)

| Check | Needed | Result |
|---|---|---|
| (a) Stations and growing-season months with all three labels and observed VCI | ≥ 30 and ≥ 1,000 | 44 and 2,105: pass |
| (b) Stations whose anomalies correlate positively with NLDAS or FLDAS | ≥ 2/3 | 44 of 44: pass |

## 1. How well each label matches measured soil (no satellite involved)

Growing season (April–October), 2016–2024: 230 stations, 11,579 station-months. Intervals are 95%, from 2,000
station resamples.

| Label vs measured | Anomaly r | Kappa (drought calls) | Drought calls agree |
|---|---|---|---|
| FLDAS | 0.535 [0.503, 0.566] | 0.323 [0.292, 0.354] | 77.2% |
| NLDAS-2 | 0.508 [0.472, 0.546] | 0.368 [0.339, 0.398] | 78.3% |
| NLDAS-2 − FLDAS | −0.027 [−0.053, 0.001] | +0.045 [0.018, 0.071] | |

- Over all months: anomaly r −0.038 [−0.067, −0.009] (FLDAS ahead), kappa +0.045 [0.021, 0.070] (NLDAS ahead).
- FLDAS and NLDAS agree with each other on 82.3% of drought calls.
- Both model labels match the probes only moderately: about a quarter of the measured anomaly variance.

## 2. Satellite VCI against each label

1 km chip VCI with the 7-year reference. Growing season: 230 stations, 10,913 station-months. LR on VCI ranks
exactly as VCI does, so these are also its AUCs.

| Label | AUC of VCI | VCI–anomaly r |
|---|---|---|
| Measured (station) | 0.660 [0.640, 0.679] | 0.274 [0.246, 0.300] |
| NLDAS-2 | 0.690 [0.671, 0.709] | 0.344 [0.305, 0.381] |
| FLDAS (matched years) | 0.704 [0.683, 0.725] | 0.352 [0.319, 0.385] |
| FLDAS as trained on (2001–2022) | 0.712 [0.692, 0.732] | |
| Measured − FLDAS | −0.045 [−0.067, −0.023] | −0.078 [−0.109, −0.048] |

- **West** (104 stations): measured 0.680, FLDAS 0.743, difference −0.062 [−0.095, −0.030].
- **East** (126 stations): measured 0.640, FLDAS 0.662, difference −0.022 [−0.050, 0.006], so no difference detected.
- **All months:** measured 0.639, FLDAS 0.681, difference −0.042 [−0.062, −0.022].

## 3. VCI reference and clipping

| Comparison, growing season | AUC vs measured | AUC vs FLDAS |
|---|---|---|
| 10-year (2013–2022) − 7-year (2016–2022) | −0.006 [−0.010, −0.001] | −0.006 [−0.010, −0.002] |
| Clip −50…150 − clip 0…100 (secondary) | +0.004 [0.002, 0.006] | +0.002 [0.001, 0.004] |

In 2024, pixels clipped below 0 / above 100 were 11.6% / 15.0% with the 7-year reference and 9.4% / 13.0% with the
10-year one (`results/chip_clipping.json`). Fewer pixels were clipped, but the score did not improve. 2013–2015 is
mostly Landsat 8 alone, so it adds few clear views per pixel.

## 4. Scale: 1 km chip vs 11 km cell (amendment 1)

44 pilot stations, growing season, 2,084 months with both observed.

| Label | 1 km chip | 11 km cell | Chip − cell |
|---|---|---|---|
| Measured | 0.664 | 0.650 | +0.014 [−0.013, 0.041] |
| FLDAS | 0.716 | 0.759 | −0.044 [−0.073, −0.017] |
| NLDAS-2 | 0.708 | 0.730 | −0.023 [−0.050, 0.004] |

- Matching the probe's scale did not make VCI agree detectably better with the probe.
- The 11 km cell agrees better with FLDAS, itself an 11 km product.

## 5. v2's held-out models at the pilot stations

Each fold's validation block. Growing season: 43 stations, 490 windows.

| Model | AUC vs measured | AUC vs FLDAS | Measured − FLDAS |
|---|---|---|---|
| LR on VCI(T) | 0.602 | 0.689 | −0.087 [−0.175, 0.004] |
| Boosted trees | 0.614 | 0.690 | −0.076 [−0.167, 0.017] |
| GRU | 0.595 | 0.680 | −0.084 [−0.177, 0.004] |
| Residual Bi-LSTM | 0.621 | 0.666 | −0.045 [−0.137, 0.043] |
| Bi-LSTM | 0.601 | 0.665 | −0.063 [−0.150, 0.019] |

The models follow the same pattern as VCI: lower against measured drought. The sample is small, so the intervals
are wide.

## What this means

- **Keep FLDAS.** It is as close to measured soil as NLDAS-2, and the satellite agrees with it better than with the
  probes. Retraining on another label is not justified by this check.
- **The limit is the vegetation signal, not the label.** Even at 1 km and against measurements, VCI reaches only
  about AUC 0.66, with r 0.27. The levers that remain are a water-sensitive band (thermal) and more years, as before.
- **For the paper,** add a short ground-truth validation. Against measured soil moisture at 230 USCRN/SCAN stations,
  VCI ranks drought months with AUC 0.66 [0.64, 0.68], against 0.70 [0.68, 0.73] for FLDAS drought at the same sites
  and months. FLDAS and NLDAS-2 themselves match the probes at r ≈ 0.5.

## Limits

- A probe measures one spot. Every area-based estimate, the satellite's included, is penalised for that. The 1 km
  test shows the gap is not simply footprint size, but sub-kilometre soil variation remains.
- Most stations sit on grass or open ground, so crop fields and forests are under-represented.
- Station drought thresholds use 8–22 reference years, compared with FLDAS's 22 as trained. FLDAS and NLDAS were
  given the same years as each station for the comparison.
- The data run to 2024 only, so the fresh test (2025–2026) is not covered. Station-months are 2016–2024, the same
  years v2 developed on, so this is a validation of the signal, not a held-out model test.
- Ryan's chips are Earth Engine 30 m pixels, not the paper's 120 m pipeline. The cloud rule and VCI recipe are the
  paper's.

## Files

- `PLAN.md`: the plan and amendment 1.
- `results/labels_monthly.csv`: station, NLDAS-2 and FLDAS monthly values, drought flags and anomalies.
- `results/sites_check.csv`: stations, FLDAS cells and v2 regions.
- `results/chip_vci.csv`: 1 km VCI (7-year, 10-year and wide clip) per station-month.
- `results/chip_clipping.json`: clipping by year.
- `results/pilot.json`, `results/full.json`: every metric above, with intervals.
- `scripts/`: `labels.py`, `pilot.py`, `chip_vci.py`, `full.py`, `scoring.py`, `common.py`.
