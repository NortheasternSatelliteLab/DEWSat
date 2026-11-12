# Station check: plan

Written 2026-11-11, before any result was computed. Changes after this point go at the end as dated amendments.

**Question.** Is FLDAS a good enough drought label? Measured soil moisture at Ryan's 278 USCRN/SCAN stations decides
between FLDAS and NLDAS-2, and shows whether the satellite's drought signal is limited by the label. A second question:
does a 10-year VCI reference (2013–2022) beat the paper's 7-year one (2016–2022)?

**Scope.** A check only. No training, no label change, no new downloads and no fresh-test scoring; the station data
stops at 2024-12, before the fresh test. Read-only: `~/Desktop/DEWSat_BiLSTM_v2`, `~/Desktop/DEWSat_Paper_BiLSTM`,
`~/Desktop/stations`, `~/Desktop/hls Ryan`, `~/Desktop/nldas Ryan`. Everything written goes in this folder.

## 1. Data

| Item | Source |
|---|---|
| Measured soil moisture, 10–40 cm, daily | `~/Desktop/stations` (`sites.csv`, one file per station), 2000–2024 |
| NLDAS-2 Noah 10–40 cm at each station, daily | `~/Desktop/nldas Ryan`, 2000–2024 |
| FLDAS `SoilMoi10_40cm_tavg`, monthly, 0.1° | `DEWSat_BiLSTM_v2/data/locked/fldas_conus.npz`, the cell containing each station |
| Satellite, 1 km around each station | `~/Desktop/hls Ryan`: HLS L30 + S30 Red, NIR and Fmask per 30 m pixel, 2013-04 to 2024-12 |
| Satellite, 11 km cell (pilot) | `DEWSat_BiLSTM_v2/cache/dev.npz`, VCI_mean |
| v2 held-out predictions, job 1 | `DEWSat_BiLSTM_v2/runs/simple/job1`, `runs/final/job1` |

## 2. Monthly values and labels

- **Station QC.** Keep daily values with 0 < sm_10_40 < 0.7 m³/m³. A month needs at least 20 valid days; its value is
  their mean. NLDAS months are formed the same way. FLDAS months are used as stored.
- **Reference years ("matched").** For each station and calendar month, the years 2001–2022 in which the station has
  a monthly value. At least 8 are needed. FLDAS and NLDAS use the same years at that station and calendar month, so
  differences come from the data source, not the period.
- **Drought.** Value below the numpy-linear 20th percentile of the reference values (the paper's rule).
  **Anomaly.** Value minus the reference mean.
- **Secondary label.** FLDAS drought exactly as trained on: its own 2001–2022 reference (22 years).
- **Months scored.** 2016-01 to 2024-12. Primary season April–October; all months is secondary.

## 3. Satellite at station scale (Ryan's 1 km chips, the paper's rules on 30 m pixels)

- **Clear pixel.** Fmask not 255, bits 1–5 zero, aerosol not high (bits 6–7 not both set), Red > 0, NIR > 0.
- **Composite.** A date counts if at least 30% of the chip's pixels are clear. Images from the same date are
  averaged per pixel first. Monthly NDVI is the per-pixel median over distinct dates.
- **VCI.** 100 · (NDVI − min) / (max − min), with min/max per pixel and calendar month over the reference years.
  It needs at least 3 years and a range of at least 0.05. Months inside the reference period use leave-one-year-out
  min/max. The result is clipped to 0–100.
- **References.** R7 = 2016–2022 (the paper's). R10 = 2013–2022, which adds 2013–2015 (Landsat 8 only before
  2015-11).
- **Chip VCI_mean** is the mean over usable pixels. A month counts as observed if usable pixels make up at least 5%
  of the chip.
- **Secondary.** R7 with the clip range widened to −50…150.

## 4. Satellite at cell scale (pilot)

- **Stations.** Those inside v2 development regions.
- **VCI_mean** comes from the station's cell. Months 2016–2023 use the VCI table of the outer fold whose block holds
  that month (f1–f4), so each month's VCI excludes its own years. 2024 uses the all-development table, which is built
  from 2016–2023.

## 5. Metrics

1. **Label agreement, with no satellite involved.** FLDAS vs station and NLDAS vs station:
   - Pearson r of the anomalies;
   - Cohen's kappa and percent agreement of the drought flags.

   This covers every scored month where all three labels exist.
2. **Satellite against each label.**
   - AUC of −VCI_mean(T) for the drought flag;
   - Pearson r of VCI_mean(T) with the anomaly.

   This covers months where all three labels exist and VCI is observed. LR on VCI(T) ranks exactly as VCI does, so
   these are also its AUCs.
3. **R10 − R7.** Paired AUC difference against station drought (primary) and FLDAS drought, on months where both are
   observed. Clipping rates are reported for both.
4. **Networks (pilot only).** v2's held-out probabilities at the cells of stations inside development regions, for
   each fold's validation block:
   - AUC against station drought vs FLDAS drought, on the same windows;
   - models: lr_vci_t, gbt_mono, gru, resbilstm, bilstm (the 10-seed mean of probabilities for the networks).

**Intervals.** 95%, from 2,000 resamples of whole stations (seed 20261111). Differences are paired on the same
resamples.

## 6. Pilot → full run (data checks only)

The pilot is the development-region stations, using v2's cell-scale VCI. The full run follows if both checks pass,
whatever the pilot's VCI results show:

- (a) at least 30 stations and at least 1,000 April–October station-months with all three labels and an observed
  VCI;
- (b) station anomalies correlate positively (r > 0) with NLDAS or FLDAS anomalies at two-thirds or more of the pilot
  stations.

If either check fails, the run stops and the report says why.

## 7. How results will be read (fixed now)

- **A label is "closer to measured soil"** if its anomaly r with the stations is higher and the paired 95% interval of
  the difference excludes zero.
- **The satellite signal is "label-limited"** if VCI's AUC against station drought is higher than its AUC against
  FLDAS drought on the same months, with the paired interval excluding zero. If the interval includes zero, the report
  says no difference was detected.
- **R10 is "better"** if its AUC against station drought is higher than R7's, with the paired interval above zero.
- **Point-vs-area mismatch.** A probe measures one spot, which pulls station agreement down for any area-based
  estimate. The report states this; it is not corrected for.

## Amendment 1 (2026-11-11, after the pilot, before any full-run metric)

The pilot passed both data checks (`results/pilot.json`). It also showed the 11 km cell VCI agreeing less with
station drought than with FLDAS drought. One comparison is added to measure how much of that is scale. At the pilot
stations, on months where both are observed, the 1 km chip VCI (R7) and the 11 km cell VCI are compared on their
AUC against station drought and against FLDAS drought. Paired station resamples are used, and the difference is
reported as chip − cell. It is read with the rule in section 7: a difference counts only if its interval excludes
zero. Nothing else in the plan changes.
