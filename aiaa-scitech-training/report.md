# DEWSat_BiLSTM_v2: report

_Generated 2026-11-09 16:04:02 by `code/dewsat2/report.py` at commit `9745865`. Every number is read from a logged run._

## 0. Answer

- Development, job 1, GRU − Boosted trees (monotone in VCI): within-fold ΔAUC −0.033 [−0.047, −0.020] → **worse**.
- Development, job 1, Residual Bi-LSTM − LR on VCI(T): within-fold ΔAUC −0.050 [−0.064, −0.035] → **worse**.
- **Fresh test (pre-registered), job 1, gate:** GRU − Boosted trees (monotone in VCI) ΔAUC +0.036 [+0.019, +0.054] (West +0.022 [+0.001, +0.044], East +0.009 [−0.014, +0.030]) → **passed**.
- **Fresh test (pre-registered), job 1, best Bi-LSTM variant:** Residual Bi-LSTM − Boosted trees (monotone in VCI) ΔAUC +0.039 [+0.020, +0.057] (West +0.025 [+0.003, +0.048], East +0.011 [−0.015, +0.035]) → **passed**.
- **Fresh test (pre-registered), job 1, against LR on VCI(T):** GRU − LR on VCI(T) ΔAUC +0.035 [+0.019, +0.054] (West +0.016 [−0.004, +0.036], East +0.026 [+0.001, +0.049]) → **passed**.
- **Fresh test (pre-registered), job 1, against LR on VCI(T):** Residual Bi-LSTM − LR on VCI(T) ΔAUC +0.038 [+0.020, +0.057] (West +0.019 [−0.001, +0.041], East +0.028 [+0.000, +0.054]) → **passed**.
- Fresh test, job 1, Bi-LSTM − LR on VCI(T) (added after the development results; not in the gate): ΔAUC +0.042 [+0.024, +0.060] (West +0.013 [−0.008, +0.035], East +0.037 [+0.010, +0.061]).
- Fresh test, job 1, VCI-only Bi-LSTM − LR on VCI(T) (added after the development results; not in the gate): ΔAUC +0.023 [+0.013, +0.033] (West +0.003 [−0.010, +0.017], East +0.015 [+0.002, +0.028]).
- Fresh test, job 1, CNN + LR − LR on VCI(T) (added after the development results; not in the gate): ΔAUC +0.016 [+0.008, +0.024] (West +0.003 [−0.007, +0.014], East +0.005 [−0.000, +0.010]).
- Fresh test, job 1, CNN + Bi-LSTM − LR on VCI(T) (added after the development results; not in the gate): ΔAUC +0.015 [+0.004, +0.025] (West −0.002 [−0.017, +0.013], East +0.012 [−0.004, +0.027]).

Development and the fresh test disagree; section 8a shows why (the networks' gap to LR follows the training-set size) and section 10 lists the limits of a single test period.

## 1. Phase 0: folders, teammates' repositories, audit of the last run

- Folders: see `FOLDERS.md` (nothing was changed in them).
- Teammates' repositories (cloned in `external/`): see `DECISIONS.md` #11-#12. No teammate data fits this setup (none ships data in git; their pipelines use station or NLDAS labels, Earth Engine chips, or FLDAS climatology as an input).

- Audit (details `audit/phase0_audit.md`): the old networks learn (training AUC well above LR) but overfit the training years; the inner set shared every month with training, so it could not see it. Positive control with the old code: LR 0.797, Bi-LSTM 0.9999. Order test: shuffling the earlier months changed the old Bi-LSTM's within-fold AUC by +0.006 [−0.013, +0.026].

## 2. Data (Phase 1)

- Plan: 600 new regions in 12 balanced batches of 50 (25 West / 25 East, each side 6 fresh-test + 19 development), from 75,103 candidate box positions (ESA CCI 2020 vegetated >= 75%, all cell centres in CONUS); 125 further planned regions could not form another balanced batch.

- Pull started 2026-11-07 03:03:45; 24.0 h so far; batches: 1: complete, 2: complete, 3: complete, 4: complete, 5: complete, 6: complete, 7: complete, 8: complete, 9: complete, 10: complete, 11: complete, 12: complete.

- Frozen: {"new_dev": 456, "new_test": 144, "new_dev_west": 228, "new_dev_east": 228, "new_test_west": 72, "new_test_east": 72}.

- Development: 558 regions, 13,977 cells (48 dropped for missing label history); cells with SWIR/NDMI 13,977, with per-scene values 13,977.

- Fresh test: 144 regions, 3,600 cells.

- FLDAS labels reproduce the paper run's for the 102 regions: 810,816 values, max abs difference 1.5e-08.

- Scenes and views: {"regions": 702, "granules_searched": 2092733, "views": 1726243, "view_cell_rows": 43156075}.

## 3. Design (Phase 2, committed before training)

See `code/design/DESIGN.md` and `code/design/gate.json` (commit `f5e92aa` and any dated amendments). Four outer folds = four region groups (balanced East/West) x the year blocks 2016-17, 2018-19, 2020-21, 2022-23, 12-month buffers (15 for forecasts); VCI tables and scalers per fold from training data only; inner validation holds out the next region group and one year, with the same buffers; within-fold AUC primary.

## 4. Development results (outer folds)

Primary metric: within-fold AUC (only pairs from the same outer fold count), 95% intervals from 2,000 resamples of whole regions; paired differences use the same resamples. Networks are the mean of 10 seeds of the configuration each fold's inner split chose.

### Gate (rules fixed in `code/design/gate.json`)

| Job | Comparison | ΔAUC within-fold, all | West | East | Result |
|---|---|---|---|---|---|
| Estimation now (month T observed) | best network − best simple (the gate): GRU − Boosted trees (monotone in VCI) | −0.033 [−0.047, −0.020] | −0.014 [−0.028, +0.001] | −0.021 [−0.033, −0.007] | **worse** |
| Estimation now (month T observed) | best Bi-LSTM variant − best simple: Residual Bi-LSTM − Boosted trees (monotone in VCI) | −0.050 [−0.064, −0.036] | −0.020 [−0.035, −0.004] | −0.045 [−0.064, −0.027] | **worse** |
| Estimation now (month T observed) | best network − LR on VCI(T): GRU − LR on VCI(T) | −0.033 [−0.049, −0.019] | −0.020 [−0.037, −0.004] | −0.012 [−0.028, +0.004] | **worse** |
| Estimation now (month T observed) | best Bi-LSTM variant − LR on VCI(T): Residual Bi-LSTM − LR on VCI(T) | −0.050 [−0.064, −0.035] | −0.026 [−0.043, −0.010] | −0.036 [−0.057, −0.016] | **worse** |
| Missing month (T unobserved, >= 8 of the other 11 observed) | best network − best simple (the gate): GRU − Boosted trees (monotone in VCI) | −0.075 [−0.112, −0.038] | −0.054 [−0.092, −0.019] | −0.039 [−0.108, +0.030] | **worse** |
| Missing month (T unobserved, >= 8 of the other 11 observed) | best Bi-LSTM variant − best simple: Residual Bi-LSTM − Boosted trees (monotone in VCI) | −0.079 [−0.116, −0.040] | −0.052 [−0.091, −0.015] | −0.029 [−0.100, +0.040] | **worse** |
| Forecast T+1..T+3, onsets only (not in drought at T) | best network − best simple (the gate): GRU − LR + trend features | −0.036 [−0.053, −0.019] | −0.012 [−0.035, +0.012] | −0.054 [−0.075, −0.033] | **worse** |
| Forecast T+1..T+3, onsets only (not in drought at T) | best Bi-LSTM variant − best simple: Bi-LSTM − LR + trend features | −0.038 [−0.054, −0.020] | −0.010 [−0.032, +0.013] | −0.045 [−0.066, −0.025] | **worse** |
| Estimation now where VCI changed most (top quartile of 1- or 3-month change) | best network − best simple (the gate): GRU − Boosted trees (monotone in VCI) | −0.034 [−0.051, −0.017] | −0.027 [−0.043, −0.011] | −0.023 [−0.041, −0.004] | **worse** |
| Estimation now where VCI changed most (top quartile of 1- or 3-month change) | best Bi-LSTM variant − best simple: Residual Bi-LSTM − Boosted trees (monotone in VCI) | −0.054 [−0.071, −0.037] | −0.031 [−0.048, −0.014] | −0.048 [−0.071, −0.024] | **worse** |

- job1: best network **GRU**, best simple **Boosted trees (monotone in VCI)**, best Bi-LSTM variant **Residual Bi-LSTM** (development within-fold AUC: Boosted trees (monotone in VCI) 0.686, LR on VCI(T) 0.686, LR on latest VCI + age 0.686, Smoothed-VCI state model 0.685, LR + trend features 0.682, GRU 0.653, Temporal CNN 0.643, One-direction LSTM 0.641, Residual Bi-LSTM 0.636, Bi-LSTM 0.627, Climatology 0.500)
- job2: best network **GRU**, best simple **Boosted trees (monotone in VCI)**, best Bi-LSTM variant **Residual Bi-LSTM** (development within-fold AUC: Boosted trees (monotone in VCI) 0.654, Smoothed-VCI state model 0.653, LR on latest VCI + age 0.646, LR + trend features 0.644, GRU 0.579, Residual Bi-LSTM 0.576, Bi-LSTM 0.575, Temporal CNN 0.572, One-direction LSTM 0.570, Climatology 0.500)
- job3: best network **GRU**, best simple **LR + trend features**, best Bi-LSTM variant **Bi-LSTM** (development within-fold AUC: LR + trend features 0.555, Boosted trees (monotone in VCI) 0.548, LR on VCI(T) 0.531, LR on latest VCI + age 0.531, Smoothed-VCI state model 0.531, GRU 0.519, Bi-LSTM 0.518, One-direction LSTM 0.517, Residual Bi-LSTM 0.510, Temporal CNN 0.505, Climatology 0.500, Persistence (uses ground data) 0.500)
- job4: best network **GRU**, best simple **Boosted trees (monotone in VCI)**, best Bi-LSTM variant **Residual Bi-LSTM** (development within-fold AUC: Boosted trees (monotone in VCI) 0.688, LR on VCI(T) 0.685, LR on latest VCI + age 0.685, LR + trend features 0.680, Smoothed-VCI state model 0.678, GRU 0.654, Temporal CNN 0.643, One-direction LSTM 0.639, Residual Bi-LSTM 0.635, Bi-LSTM 0.625, Climatology 0.500)

### Job 1: estimation now (month T observed)

267,412 windows, 61,618 in drought, 558 regions (West 276 / 129,736 windows; East 282 / 137,676).

| Model | Within-fold AUC, all | West | East | Pooled AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| Climatology | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.389 [0.370, 0.408] | 0.1867 [0.1776, 0.1951] | 0.118 |
| LR on VCI(T) | 0.686 [0.670, 0.703] | 0.716 [0.694, 0.738] | 0.651 [0.632, 0.670] | 0.642 [0.626, 0.658] | 0.1721 [0.1639, 0.1796] | 0.031 |
| LR on latest VCI + age | 0.686 [0.670, 0.703] | 0.716 [0.694, 0.738] | 0.651 [0.632, 0.670] | 0.642 [0.626, 0.658] | 0.1721 [0.1639, 0.1796] | 0.031 |
| LR + trend features | 0.682 [0.665, 0.700] | 0.712 [0.690, 0.735] | 0.650 [0.631, 0.670] | 0.638 [0.621, 0.655] | 0.1730 [0.1648, 0.1807] | 0.037 |
| Boosted trees (monotone in VCI) | 0.686 [0.670, 0.703] | 0.710 [0.687, 0.732] | 0.660 [0.641, 0.678] | 0.627 [0.610, 0.645] | 0.1728 [0.1646, 0.1805] | 0.031 |
| Smoothed-VCI state model | 0.685 [0.667, 0.703] | 0.714 [0.691, 0.737] | 0.653 [0.633, 0.673] | 0.643 [0.626, 0.660] | 0.1714 [0.1632, 0.1790] | 0.029 |
| Bi-LSTM | 0.627 [0.611, 0.644] | 0.686 [0.662, 0.710] | 0.603 [0.582, 0.624] | 0.575 [0.557, 0.594] | 0.1949 [0.1850, 0.2044] | 0.117 |
| Residual Bi-LSTM | 0.636 [0.620, 0.653] | 0.690 [0.666, 0.713] | 0.615 [0.595, 0.635] | 0.593 [0.576, 0.611] | 0.1909 [0.1814, 0.1999] | 0.107 |
| GRU | 0.653 [0.636, 0.669] | 0.696 [0.672, 0.719] | 0.639 [0.621, 0.657] | 0.612 [0.595, 0.629] | 0.1855 [0.1765, 0.1939] | 0.094 |
| One-direction LSTM | 0.641 [0.626, 0.658] | 0.694 [0.670, 0.718] | 0.620 [0.601, 0.638] | 0.594 [0.576, 0.612] | 0.1887 [0.1794, 0.1975] | 0.100 |
| Temporal CNN | 0.643 [0.628, 0.660] | 0.694 [0.670, 0.717] | 0.634 [0.616, 0.652] | 0.610 [0.594, 0.627] | 0.1886 [0.1796, 0.1972] | 0.104 |

Paired differences against LR on VCI(T):

| Model − LR on VCI(T) | ΔAUC within-fold, all | West | East | ΔBrier |
|---|---|---|---|---|
| Climatology | −0.186 [−0.203, −0.170] | −0.216 [−0.238, −0.194] | −0.151 [−0.170, −0.132] | +0.0147 [+0.0119, +0.0175] |
| LR on latest VCI + age | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.0000 [+0.0000, +0.0000] |
| LR + trend features | −0.004 [−0.008, −0.000] | −0.004 [−0.010, +0.002] | −0.001 [−0.008, +0.006] | +0.0009 [+0.0003, +0.0015] |
| Boosted trees (monotone in VCI) | +0.000 [−0.004, +0.004] | −0.006 [−0.013, +0.000] | +0.009 [+0.002, +0.016] | +0.0008 [−0.0000, +0.0016] |
| Smoothed-VCI state model | −0.001 [−0.004, +0.002] | −0.002 [−0.005, +0.001] | +0.002 [−0.003, +0.007] | −0.0007 [−0.0011, −0.0002] |
| Bi-LSTM | −0.059 [−0.075, −0.043] | −0.030 [−0.048, −0.012] | −0.048 [−0.071, −0.025] | +0.0228 [+0.0192, +0.0264] |
| Residual Bi-LSTM | −0.050 [−0.064, −0.035] | −0.026 [−0.043, −0.010] | −0.036 [−0.057, −0.016] | +0.0189 [+0.0158, +0.0221] |
| GRU | −0.033 [−0.049, −0.019] | −0.020 [−0.037, −0.004] | −0.012 [−0.028, +0.004] | +0.0135 [+0.0106, +0.0164] |
| One-direction LSTM | −0.045 [−0.059, −0.030] | −0.022 [−0.039, −0.005] | −0.031 [−0.050, −0.012] | +0.0166 [+0.0135, +0.0199] |
| Temporal CNN | −0.043 [−0.059, −0.028] | −0.022 [−0.040, −0.006] | −0.017 [−0.034, +0.001] | +0.0166 [+0.0134, +0.0199] |

Paired differences against the best simple model (Boosted trees (monotone in VCI)):

| Model − Boosted trees (monotone in VCI) | ΔAUC within-fold, all | West | East | ΔBrier |
|---|---|---|---|---|
| Climatology | −0.186 [−0.203, −0.170] | −0.210 [−0.232, −0.187] | −0.160 [−0.178, −0.141] | +0.0139 [+0.0112, +0.0166] |
| LR on VCI(T) | −0.000 [−0.004, +0.004] | +0.006 [−0.000, +0.013] | −0.009 [−0.016, −0.002] | −0.0008 [−0.0016, +0.0000] |
| LR on latest VCI + age | −0.000 [−0.004, +0.004] | +0.006 [−0.000, +0.013] | −0.009 [−0.016, −0.002] | −0.0008 [−0.0016, +0.0000] |
| LR + trend features | −0.004 [−0.008, −0.001] | +0.002 [−0.003, +0.008] | −0.010 [−0.016, −0.005] | +0.0002 [−0.0005, +0.0007] |
| Smoothed-VCI state model | −0.001 [−0.006, +0.004] | +0.004 [−0.003, +0.012] | −0.007 [−0.016, +0.002] | −0.0014 [−0.0023, −0.0005] |
| Bi-LSTM | −0.059 [−0.075, −0.044] | −0.024 [−0.041, −0.007] | −0.057 [−0.079, −0.035] | +0.0220 [+0.0185, +0.0255] |
| Residual Bi-LSTM | −0.050 [−0.064, −0.036] | −0.020 [−0.035, −0.004] | −0.045 [−0.064, −0.027] | +0.0181 [+0.0151, +0.0212] |
| GRU | −0.033 [−0.047, −0.020] | −0.014 [−0.028, +0.001] | −0.021 [−0.033, −0.007] | +0.0127 [+0.0100, +0.0156] |
| One-direction LSTM | −0.045 [−0.058, −0.031] | −0.016 [−0.032, +0.000] | −0.040 [−0.056, −0.023] | +0.0158 [+0.0128, +0.0189] |
| Temporal CNN | −0.043 [−0.058, −0.029] | −0.016 [−0.031, −0.000] | −0.026 [−0.040, −0.010] | +0.0158 [+0.0128, +0.0190] |

AUC within each fold (point estimates):

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|
| Climatology | 0.500 | 0.500 | 0.500 | 0.500 |
| LR on VCI(T) | 0.641 | 0.717 | 0.733 | 0.654 |
| LR on latest VCI + age | 0.641 | 0.717 | 0.733 | 0.654 |
| LR + trend features | 0.637 | 0.716 | 0.732 | 0.645 |
| Boosted trees (monotone in VCI) | 0.638 | 0.712 | 0.733 | 0.659 |
| Smoothed-VCI state model | 0.647 | 0.720 | 0.739 | 0.640 |
| Bi-LSTM | 0.652 | 0.666 | 0.619 | 0.597 |
| Residual Bi-LSTM | 0.648 | 0.659 | 0.637 | 0.615 |
| GRU | 0.642 | 0.666 | 0.641 | 0.665 |
| One-direction LSTM | 0.650 | 0.666 | 0.633 | 0.631 |
| Temporal CNN | 0.640 | 0.664 | 0.626 | 0.652 |

### Job 2: missing month (T unobserved, ≥ 8 of the other 11 observed)

31,756 windows, 7,442 in drought, 497 regions (West 248 / 17,494 windows; East 249 / 14,262).

| Model | Within-fold AUC, all | West | East | Pooled AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| Climatology | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.389 [0.355, 0.425] | 0.1912 [0.1777, 0.2048] | 0.099 |
| LR on latest VCI + age | 0.646 [0.611, 0.681] | 0.656 [0.615, 0.698] | 0.613 [0.558, 0.667] | 0.561 [0.525, 0.598] | 0.1866 [0.1745, 0.1994] | 0.063 |
| LR + trend features | 0.644 [0.608, 0.679] | 0.650 [0.609, 0.691] | 0.618 [0.564, 0.671] | 0.574 [0.538, 0.609] | 0.1866 [0.1755, 0.1981] | 0.053 |
| Boosted trees (monotone in VCI) | 0.654 [0.617, 0.692] | 0.668 [0.624, 0.711] | 0.621 [0.563, 0.676] | 0.569 [0.530, 0.606] | 0.1814 [0.1698, 0.1934] | 0.052 |
| Smoothed-VCI state model | 0.653 [0.615, 0.690] | 0.665 [0.621, 0.707] | 0.619 [0.560, 0.674] | 0.568 [0.530, 0.606] | 0.1844 [0.1724, 0.1970] | 0.054 |
| Bi-LSTM | 0.575 [0.541, 0.611] | 0.608 [0.558, 0.659] | 0.597 [0.542, 0.650] | 0.544 [0.511, 0.578] | 0.2236 [0.2099, 0.2384] | 0.145 |
| Residual Bi-LSTM | 0.576 [0.542, 0.611] | 0.616 [0.568, 0.666] | 0.592 [0.536, 0.647] | 0.539 [0.504, 0.577] | 0.2118 [0.1986, 0.2250] | 0.138 |
| GRU | 0.579 [0.545, 0.615] | 0.614 [0.565, 0.665] | 0.582 [0.529, 0.635] | 0.546 [0.511, 0.583] | 0.2122 [0.1988, 0.2263] | 0.137 |
| One-direction LSTM | 0.570 [0.535, 0.606] | 0.617 [0.566, 0.672] | 0.576 [0.517, 0.633] | 0.531 [0.494, 0.570] | 0.2126 [0.1986, 0.2255] | 0.143 |
| Temporal CNN | 0.572 [0.537, 0.607] | 0.613 [0.564, 0.664] | 0.582 [0.526, 0.638] | 0.530 [0.494, 0.568] | 0.2220 [0.2069, 0.2372] | 0.163 |

Paired differences against LR on latest VCI + age:

| Model − LR on latest VCI + age | ΔAUC within-fold, all | West | East | ΔBrier |
|---|---|---|---|---|
| Climatology | −0.146 [−0.181, −0.111] | −0.156 [−0.198, −0.115] | −0.113 [−0.167, −0.058] | +0.0046 [−0.0001, +0.0095] |
| LR + trend features | −0.003 [−0.011, +0.005] | −0.006 [−0.015, +0.003] | +0.006 [−0.009, +0.020] | +0.0000 [−0.0017, +0.0018] |
| Boosted trees (monotone in VCI) | +0.008 [−0.004, +0.021] | +0.011 [−0.005, +0.030] | +0.008 [−0.009, +0.027] | −0.0052 [−0.0075, −0.0030] |
| Smoothed-VCI state model | +0.006 [−0.002, +0.014] | +0.009 [+0.001, +0.018] | +0.006 [−0.010, +0.022] | −0.0022 [−0.0033, −0.0012] |
| Bi-LSTM | −0.072 [−0.110, −0.033] | −0.049 [−0.096, −0.002] | −0.016 [−0.078, +0.045] | +0.0370 [+0.0264, +0.0477] |
| Residual Bi-LSTM | −0.071 [−0.108, −0.032] | −0.040 [−0.085, +0.003] | −0.021 [−0.089, +0.041] | +0.0252 [+0.0163, +0.0338] |
| GRU | −0.068 [−0.105, −0.030] | −0.043 [−0.086, +0.000] | −0.030 [−0.097, +0.032] | +0.0257 [+0.0163, +0.0350] |
| One-direction LSTM | −0.077 [−0.117, −0.036] | −0.040 [−0.090, +0.009] | −0.037 [−0.107, +0.027] | +0.0260 [+0.0171, +0.0344] |
| Temporal CNN | −0.074 [−0.113, −0.035] | −0.044 [−0.092, +0.005] | −0.030 [−0.098, +0.033] | +0.0354 [+0.0246, +0.0456] |

Paired differences against the best simple model (Boosted trees (monotone in VCI)):

| Model − Boosted trees (monotone in VCI) | ΔAUC within-fold, all | West | East | ΔBrier |
|---|---|---|---|---|
| Climatology | −0.154 [−0.192, −0.117] | −0.168 [−0.211, −0.124] | −0.121 [−0.176, −0.063] | +0.0098 [+0.0056, +0.0142] |
| LR on latest VCI + age | −0.008 [−0.021, +0.004] | −0.011 [−0.030, +0.005] | −0.008 [−0.027, +0.009] | +0.0052 [+0.0030, +0.0075] |
| LR + trend features | −0.010 [−0.021, −0.000] | −0.018 [−0.033, −0.004] | −0.003 [−0.018, +0.013] | +0.0052 [+0.0025, +0.0080] |
| Smoothed-VCI state model | −0.002 [−0.014, +0.010] | −0.002 [−0.019, +0.013] | −0.002 [−0.019, +0.015] | +0.0030 [+0.0008, +0.0052] |
| Bi-LSTM | −0.079 [−0.116, −0.040] | −0.060 [−0.101, −0.021] | −0.024 [−0.092, +0.043] | +0.0422 [+0.0319, +0.0529] |
| Residual Bi-LSTM | −0.079 [−0.116, −0.040] | −0.052 [−0.091, −0.015] | −0.029 [−0.100, +0.040] | +0.0304 [+0.0219, +0.0391] |
| GRU | −0.075 [−0.112, −0.038] | −0.054 [−0.092, −0.019] | −0.039 [−0.108, +0.030] | +0.0309 [+0.0221, +0.0400] |
| One-direction LSTM | −0.084 [−0.123, −0.044] | −0.051 [−0.095, −0.010] | −0.045 [−0.117, +0.021] | +0.0312 [+0.0224, +0.0392] |
| Temporal CNN | −0.082 [−0.122, −0.043] | −0.055 [−0.097, −0.014] | −0.039 [−0.109, +0.026] | +0.0406 [+0.0301, +0.0509] |

AUC within each fold (point estimates):

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|
| Climatology | 0.500 | 0.500 | 0.500 | 0.500 |
| LR on latest VCI + age | 0.658 | 0.620 | 0.695 | 0.605 |
| LR + trend features | 0.627 | 0.634 | 0.704 | 0.592 |
| Boosted trees (monotone in VCI) | 0.649 | 0.655 | 0.719 | 0.586 |
| Smoothed-VCI state model | 0.658 | 0.647 | 0.710 | 0.592 |
| Bi-LSTM | 0.553 | 0.529 | 0.617 | 0.564 |
| Residual Bi-LSTM | 0.566 | 0.541 | 0.606 | 0.566 |
| GRU | 0.567 | 0.538 | 0.618 | 0.564 |
| One-direction LSTM | 0.538 | 0.540 | 0.611 | 0.555 |
| Temporal CNN | 0.544 | 0.551 | 0.603 | 0.562 |

### Job 4: estimation now where VCI changed most (top quartile of the 1- or 3-month change)

95,119 windows, 21,691 in drought, 558 regions (West 276 / 43,860 windows; East 282 / 51,259).

| Model | Within-fold AUC, all | West | East | Pooled AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| Climatology | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.390 [0.367, 0.414] | 0.1854 [0.1763, 0.1946] | 0.111 |
| LR on VCI(T) | 0.685 [0.667, 0.702] | 0.742 [0.719, 0.763] | 0.641 [0.618, 0.663] | 0.633 [0.613, 0.650] | 0.1736 [0.1653, 0.1822] | 0.045 |
| LR on latest VCI + age | 0.685 [0.667, 0.702] | 0.742 [0.719, 0.763] | 0.641 [0.618, 0.663] | 0.633 [0.613, 0.650] | 0.1736 [0.1653, 0.1822] | 0.045 |
| LR + trend features | 0.680 [0.662, 0.698] | 0.740 [0.716, 0.762] | 0.641 [0.617, 0.665] | 0.630 [0.610, 0.648] | 0.1732 [0.1649, 0.1816] | 0.039 |
| Boosted trees (monotone in VCI) | 0.688 [0.671, 0.705] | 0.739 [0.715, 0.760] | 0.654 [0.632, 0.677] | 0.622 [0.602, 0.640] | 0.1732 [0.1648, 0.1817] | 0.041 |
| Smoothed-VCI state model | 0.678 [0.659, 0.696] | 0.738 [0.714, 0.759] | 0.637 [0.614, 0.661] | 0.624 [0.602, 0.643] | 0.1728 [0.1642, 0.1816] | 0.039 |
| Bi-LSTM | 0.625 [0.605, 0.643] | 0.710 [0.687, 0.733] | 0.590 [0.566, 0.616] | 0.557 [0.534, 0.580] | 0.2007 [0.1901, 0.2117] | 0.133 |
| Residual Bi-LSTM | 0.635 [0.616, 0.653] | 0.707 [0.684, 0.731] | 0.606 [0.583, 0.630] | 0.579 [0.557, 0.600] | 0.1954 [0.1854, 0.2057] | 0.119 |
| GRU | 0.654 [0.636, 0.672] | 0.712 [0.689, 0.734] | 0.631 [0.609, 0.654] | 0.600 [0.578, 0.619] | 0.1892 [0.1796, 0.1989] | 0.104 |
| One-direction LSTM | 0.639 [0.620, 0.657] | 0.714 [0.692, 0.737] | 0.606 [0.583, 0.629] | 0.578 [0.555, 0.598] | 0.1931 [0.1832, 0.2035] | 0.114 |
| Temporal CNN | 0.643 [0.624, 0.662] | 0.711 [0.688, 0.734] | 0.621 [0.599, 0.644] | 0.594 [0.573, 0.613] | 0.1933 [0.1839, 0.2031] | 0.114 |

Paired differences against LR on VCI(T):

| Model − LR on VCI(T) | ΔAUC within-fold, all | West | East | ΔBrier |
|---|---|---|---|---|
| Climatology | −0.185 [−0.202, −0.167] | −0.242 [−0.263, −0.219] | −0.141 [−0.163, −0.118] | +0.0118 [+0.0089, +0.0146] |
| LR on latest VCI + age | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.0000 [+0.0000, +0.0000] |
| LR + trend features | −0.004 [−0.010, +0.001] | −0.003 [−0.011, +0.005] | +0.001 [−0.008, +0.009] | −0.0004 [−0.0013, +0.0006] |
| Boosted trees (monotone in VCI) | +0.004 [−0.002, +0.010] | −0.004 [−0.012, +0.004] | +0.014 [+0.004, +0.023] | −0.0004 [−0.0015, +0.0006] |
| Smoothed-VCI state model | −0.007 [−0.012, −0.001] | −0.005 [−0.011, +0.001] | −0.003 [−0.012, +0.004] | −0.0008 [−0.0016, −0.0000] |
| Bi-LSTM | −0.060 [−0.080, −0.041] | −0.033 [−0.052, −0.015] | −0.050 [−0.079, −0.020] | +0.0271 [+0.0224, +0.0321] |
| Residual Bi-LSTM | −0.050 [−0.069, −0.031] | −0.035 [−0.053, −0.017] | −0.034 [−0.059, −0.009] | +0.0218 [+0.0177, +0.0261] |
| GRU | −0.030 [−0.048, −0.012] | −0.031 [−0.048, −0.013] | −0.010 [−0.031, +0.013] | +0.0155 [+0.0118, +0.0194] |
| One-direction LSTM | −0.046 [−0.064, −0.028] | −0.028 [−0.048, −0.010] | −0.035 [−0.060, −0.009] | +0.0195 [+0.0156, +0.0237] |
| Temporal CNN | −0.042 [−0.061, −0.023] | −0.031 [−0.050, −0.014] | −0.020 [−0.044, +0.004] | +0.0197 [+0.0155, +0.0242] |

Paired differences against the best simple model (Boosted trees (monotone in VCI)):

| Model − Boosted trees (monotone in VCI) | ΔAUC within-fold, all | West | East | ΔBrier |
|---|---|---|---|---|
| Climatology | −0.188 [−0.205, −0.171] | −0.239 [−0.260, −0.215] | −0.154 [−0.177, −0.132] | +0.0122 [+0.0097, +0.0148] |
| LR on VCI(T) | −0.004 [−0.010, +0.002] | +0.004 [−0.004, +0.012] | −0.014 [−0.023, −0.004] | +0.0004 [−0.0006, +0.0015] |
| LR on latest VCI + age | −0.004 [−0.010, +0.002] | +0.004 [−0.004, +0.012] | −0.014 [−0.023, −0.004] | +0.0004 [−0.0006, +0.0015] |
| LR + trend features | −0.008 [−0.013, −0.003] | +0.001 [−0.004, +0.007] | −0.013 [−0.021, −0.005] | +0.0001 [−0.0007, +0.0008] |
| Smoothed-VCI state model | −0.010 [−0.018, −0.003] | −0.001 [−0.009, +0.008] | −0.017 [−0.030, −0.005] | −0.0004 [−0.0014, +0.0007] |
| Bi-LSTM | −0.064 [−0.083, −0.046] | −0.029 [−0.048, −0.012] | −0.064 [−0.092, −0.037] | +0.0276 [+0.0229, +0.0324] |
| Residual Bi-LSTM | −0.054 [−0.071, −0.037] | −0.031 [−0.048, −0.014] | −0.048 [−0.071, −0.024] | +0.0222 [+0.0182, +0.0264] |
| GRU | −0.034 [−0.051, −0.017] | −0.027 [−0.043, −0.011] | −0.023 [−0.041, −0.004] | +0.0160 [+0.0124, +0.0196] |
| One-direction LSTM | −0.049 [−0.067, −0.032] | −0.025 [−0.043, −0.007] | −0.049 [−0.071, −0.026] | +0.0200 [+0.0162, +0.0239] |
| Temporal CNN | −0.045 [−0.063, −0.028] | −0.028 [−0.045, −0.011] | −0.033 [−0.054, −0.013] | +0.0202 [+0.0161, +0.0244] |

AUC within each fold (point estimates):

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|
| Climatology | 0.500 | 0.500 | 0.500 | 0.500 |
| LR on VCI(T) | 0.653 | 0.707 | 0.709 | 0.669 |
| LR on latest VCI + age | 0.653 | 0.707 | 0.709 | 0.669 |
| LR + trend features | 0.654 | 0.709 | 0.711 | 0.655 |
| Boosted trees (monotone in VCI) | 0.656 | 0.699 | 0.717 | 0.675 |
| Smoothed-VCI state model | 0.661 | 0.715 | 0.715 | 0.639 |
| Bi-LSTM | 0.669 | 0.644 | 0.602 | 0.613 |
| Residual Bi-LSTM | 0.661 | 0.635 | 0.611 | 0.641 |
| GRU | 0.664 | 0.645 | 0.613 | 0.686 |
| One-direction LSTM | 0.670 | 0.641 | 0.601 | 0.653 |
| Temporal CNN | 0.659 | 0.637 | 0.599 | 0.673 |

### Job 3: forecast T+1, T+2, T+3

Within-fold AUC by lead (persistence uses the FLDAS flag at T, i.e. ground data):

| Lead | Subset | Climatology | LR on VCI(T) | LR on latest VCI + age | LR + trend features | Boosted trees (monotone in VCI) | Smoothed-VCI state model | Bi-LSTM | Residual Bi-LSTM | GRU | One-direction LSTM | Temporal CNN | Persistence (uses ground data) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | all months (n 258,740) | 0.500 [0.500, 0.500] | 0.640 [0.623, 0.657] | 0.640 [0.623, 0.657] | 0.639 [0.622, 0.656] | 0.628 [0.613, 0.643] | 0.640 [0.623, 0.657] | 0.542 [0.527, 0.558] | 0.538 [0.521, 0.555] | 0.549 [0.534, 0.566] | 0.539 [0.524, 0.556] | 0.527 [0.510, 0.545] | 0.754 [0.743, 0.766] |
| 1 | onsets (n 199,808) | 0.500 [0.500, 0.500] | 0.562 [0.544, 0.579] | 0.562 [0.544, 0.579] | 0.578 [0.562, 0.595] | 0.573 [0.557, 0.589] | 0.562 [0.544, 0.579] | 0.544 [0.529, 0.560] | 0.537 [0.521, 0.553] | 0.541 [0.524, 0.557] | 0.541 [0.525, 0.557] | 0.531 [0.515, 0.547] | 0.500 [0.500, 0.500] |
| 2 | all months (n 258,740) | 0.500 [0.500, 0.500] | 0.592 [0.574, 0.610] | 0.592 [0.574, 0.610] | 0.588 [0.572, 0.605] | 0.576 [0.560, 0.591] | 0.592 [0.574, 0.610] | 0.504 [0.489, 0.521] | 0.498 [0.482, 0.515] | 0.513 [0.497, 0.531] | 0.507 [0.491, 0.524] | 0.496 [0.480, 0.512] | 0.676 [0.663, 0.687] |
| 2 | onsets (n 199,808) | 0.500 [0.500, 0.500] | 0.524 [0.508, 0.541] | 0.524 [0.508, 0.541] | 0.546 [0.528, 0.564] | 0.535 [0.518, 0.552] | 0.524 [0.508, 0.541] | 0.511 [0.495, 0.528] | 0.504 [0.489, 0.520] | 0.515 [0.499, 0.532] | 0.514 [0.498, 0.530] | 0.503 [0.487, 0.520] | 0.500 [0.500, 0.500] |
| 3 | all months (n 258,740) | 0.500 [0.500, 0.500] | 0.566 [0.548, 0.585] | 0.566 [0.548, 0.585] | 0.560 [0.545, 0.575] | 0.557 [0.542, 0.572] | 0.566 [0.548, 0.585] | 0.492 [0.476, 0.508] | 0.486 [0.470, 0.503] | 0.499 [0.483, 0.516] | 0.493 [0.477, 0.511] | 0.485 [0.470, 0.501] | 0.634 [0.622, 0.646] |
| 3 | onsets (n 199,808) | 0.500 [0.500, 0.500] | 0.507 [0.490, 0.525] | 0.507 [0.490, 0.525] | 0.541 [0.524, 0.560] | 0.535 [0.516, 0.553] | 0.507 [0.490, 0.525] | 0.497 [0.479, 0.516] | 0.488 [0.470, 0.507] | 0.501 [0.482, 0.521] | 0.496 [0.477, 0.516] | 0.483 [0.465, 0.501] | 0.500 [0.500, 0.500] |

Mean onset within-fold AUC over leads 1-3 (the job-3 gate metric):

| Model | All | West | East |
|---|---|---|---|
| Climatology | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| LR on VCI(T) | 0.531 [0.516, 0.547] | 0.539 [0.518, 0.562] | 0.525 [0.506, 0.542] |
| LR on latest VCI + age | 0.531 [0.516, 0.547] | 0.539 [0.518, 0.562] | 0.525 [0.506, 0.542] |
| LR + trend features | 0.555 [0.540, 0.571] | 0.546 [0.522, 0.571] | 0.590 [0.573, 0.607] |
| Boosted trees (monotone in VCI) | 0.548 [0.532, 0.562] | 0.546 [0.525, 0.568] | 0.574 [0.555, 0.593] |
| Smoothed-VCI state model | 0.531 [0.516, 0.547] | 0.539 [0.518, 0.562] | 0.525 [0.506, 0.542] |
| Bi-LSTM | 0.518 [0.502, 0.533] | 0.536 [0.512, 0.561] | 0.545 [0.522, 0.567] |
| Residual Bi-LSTM | 0.510 [0.495, 0.526] | 0.540 [0.516, 0.566] | 0.525 [0.503, 0.548] |
| GRU | 0.519 [0.504, 0.535] | 0.534 [0.508, 0.563] | 0.536 [0.513, 0.557] |
| One-direction LSTM | 0.517 [0.501, 0.533] | 0.537 [0.511, 0.566] | 0.531 [0.510, 0.553] |
| Temporal CNN | 0.505 [0.490, 0.522] | 0.535 [0.513, 0.558] | 0.528 [0.506, 0.552] |
| Persistence (uses ground data) | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |

Paired, mean onset metric, against LR + trend features:

| Model | Δ all | Δ West | Δ East |
|---|---|---|---|
| Climatology | −0.055 [−0.071, −0.040] | −0.046 [−0.071, −0.022] | −0.090 [−0.107, −0.073] |
| LR on VCI(T) | −0.024 [−0.036, −0.013] | −0.006 [−0.020, +0.007] | −0.065 [−0.084, −0.048] |
| LR on latest VCI + age | −0.024 [−0.036, −0.013] | −0.006 [−0.020, +0.007] | −0.065 [−0.084, −0.048] |
| Boosted trees (monotone in VCI) | −0.008 [−0.017, +0.001] | −0.000 [−0.012, +0.013] | −0.015 [−0.029, −0.003] |
| Smoothed-VCI state model | −0.024 [−0.036, −0.013] | −0.006 [−0.020, +0.007] | −0.065 [−0.084, −0.048] |
| Bi-LSTM | −0.038 [−0.054, −0.020] | −0.010 [−0.032, +0.013] | −0.045 [−0.066, −0.025] |
| Residual Bi-LSTM | −0.045 [−0.063, −0.028] | −0.006 [−0.030, +0.018] | −0.065 [−0.086, −0.044] |
| GRU | −0.036 [−0.053, −0.019] | −0.012 [−0.035, +0.012] | −0.054 [−0.075, −0.033] |
| One-direction LSTM | −0.038 [−0.055, −0.021] | −0.009 [−0.034, +0.015] | −0.058 [−0.079, −0.038] |
| Temporal CNN | −0.050 [−0.067, −0.032] | −0.010 [−0.032, +0.013] | −0.062 [−0.082, −0.041] |
| Persistence (uses ground data) | −0.055 [−0.071, −0.040] | −0.046 [−0.071, −0.022] | −0.090 [−0.107, −0.073] |

Job 4 strata (within-fold AUC, all regions):

| Stratum | n | LR on VCI(T) | Bi-LSTM | Residual Bi-LSTM | GRU | One-direction LSTM | Temporal CNN |
|---|---|---|---|---|---|---|---|
| top_d1 | 60,067 | 0.653 [0.634, 0.671] | 0.599 [0.580, 0.617] | 0.606 [0.587, 0.625] | 0.630 [0.611, 0.649] | 0.612 [0.594, 0.629] | 0.620 [0.601, 0.640] |
| top_d3 | 53,584 | 0.719 [0.699, 0.738] | 0.650 [0.628, 0.674] | 0.663 [0.641, 0.685] | 0.682 [0.662, 0.704] | 0.667 [0.646, 0.688] | 0.669 [0.648, 0.691] |
| neither | 172,293 | 0.688 [0.667, 0.708] | 0.628 [0.609, 0.650] | 0.638 [0.619, 0.658] | 0.652 [0.634, 0.672] | 0.643 [0.624, 0.664] | 0.644 [0.626, 0.665] |

Thresholds (75th percentiles over pooled job-1 windows): |ΔVCI| over 1 month 20.62, over 3 months 30.34.

## 4b. Models added after the development results (amendments 4 and 4b; development)

Added after section 4's results, with their designs fixed before they were trained (DESIGN.md amendments 4 and 4b; not part of the gate):

- **VCI-only Bi-LSTM**: each job's Bi-LSTM search, with NDVI, the valid fraction and the season channels zeroed, so only VCI_mean, VCI_p10, the observed flag and the months since a view carry information.
- **CNN + LR** (job 1): a small CNN on month T's 23x21 VCI tile of the cell, then one linear layer.
- **CNN + Bi-LSTM** (job 1): the same encoder on each of the 12 monthly tiles, then a Bi-LSTM.

Same folds, inner selection, 10 seeds, pooling and whole-region resamples as section 4.

**The 2x2 table** (job 1, development, within-fold AUC, all regions):

| History | Cell average (VCI) | Cell image (23x21 VCI tiles) | Image − average |
|---|---|---|---|
| Month T only | LR on VCI(T): 0.686 [0.670, 0.703] | CNN + LR: 0.679 [0.665, 0.695] | −0.007 [−0.011, −0.003] |
| 12 months | VCI-only Bi-LSTM: 0.658 [0.643, 0.673] | CNN + Bi-LSTM: 0.658 [0.643, 0.672] | +0.000 [−0.005, +0.005] |
| 12 months − month T | −0.028 [−0.037, −0.019] | −0.021 [−0.029, −0.012] | |

### Estimation now (month T observed) (within-fold AUC)

267,412 windows, 61,618 in drought.

| Model | All | West | East | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|---|---|
| LR on VCI(T) | 0.686 [0.670, 0.703] | 0.716 [0.694, 0.738] | 0.651 [0.632, 0.670] | 0.641 | 0.717 | 0.733 | 0.654 |
| Boosted trees (monotone in VCI) | 0.686 [0.670, 0.703] | 0.710 [0.687, 0.732] | 0.660 [0.641, 0.678] | 0.638 | 0.712 | 0.733 | 0.659 |
| GRU | 0.653 [0.636, 0.669] | 0.696 [0.672, 0.719] | 0.639 [0.621, 0.657] | 0.642 | 0.666 | 0.641 | 0.665 |
| Residual Bi-LSTM | 0.636 [0.620, 0.653] | 0.690 [0.666, 0.713] | 0.615 [0.595, 0.635] | 0.648 | 0.659 | 0.637 | 0.615 |
| Bi-LSTM | 0.627 [0.611, 0.644] | 0.686 [0.662, 0.710] | 0.603 [0.582, 0.624] | 0.652 | 0.666 | 0.619 | 0.597 |
| VCI-only Bi-LSTM | 0.658 [0.643, 0.673] | 0.697 [0.677, 0.719] | 0.629 [0.610, 0.647] | 0.619 | 0.696 | 0.680 | 0.644 |
| CNN + LR | 0.679 [0.665, 0.695] | 0.719 [0.699, 0.740] | 0.650 [0.631, 0.668] | 0.644 | 0.720 | 0.709 | 0.653 |
| CNN + Bi-LSTM | 0.658 [0.643, 0.672] | 0.688 [0.666, 0.712] | 0.635 [0.617, 0.653] | 0.622 | 0.688 | 0.678 | 0.648 |

| Comparison | Δ all | Δ West | Δ East | Result |
|---|---|---|---|---|
| VCI-only Bi-LSTM − LR on VCI(T) | −0.028 [−0.037, −0.019] | −0.019 [−0.031, −0.006] | −0.022 [−0.033, −0.011] | worse |
| VCI-only Bi-LSTM − Boosted trees (monotone in VCI) | −0.028 [−0.036, −0.019] | −0.012 [−0.024, −0.001] | −0.031 [−0.042, −0.021] | worse |
| CNN + LR − LR on VCI(T) | −0.007 [−0.011, −0.003] | +0.003 [−0.002, +0.009] | −0.001 [−0.005, +0.002] | worse |
| CNN + LR − Boosted trees (monotone in VCI) | −0.007 [−0.013, −0.001] | +0.010 [+0.002, +0.018] | −0.010 [−0.019, −0.003] | worse |
| CNN + Bi-LSTM − LR on VCI(T) | −0.028 [−0.038, −0.018] | −0.028 [−0.041, −0.015] | −0.016 [−0.028, −0.004] | worse |
| CNN + Bi-LSTM − Boosted trees (monotone in VCI) | −0.028 [−0.038, −0.018] | −0.022 [−0.034, −0.010] | −0.025 [−0.037, −0.013] | worse |
| VCI-only Bi-LSTM − Bi-LSTM | +0.032 [+0.018, +0.044] | +0.011 [−0.005, +0.027] | +0.026 [+0.006, +0.044] | passed |
| CNN + Bi-LSTM − VCI-only Bi-LSTM | +0.000 [−0.005, +0.005] | −0.009 [−0.017, −0.002] | +0.006 [−0.001, +0.014] | no detectable difference |
| CNN + Bi-LSTM − CNN + LR | −0.021 [−0.029, −0.012] | −0.031 [−0.044, −0.019] | −0.015 [−0.027, −0.003] | worse |

### Missing month (T unobserved, >= 8 of the other 11 observed) (within-fold AUC)

31,756 windows, 7,442 in drought.

| Model | All | West | East | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|---|---|
| LR on latest VCI + age | 0.646 [0.611, 0.681] | 0.656 [0.615, 0.698] | 0.613 [0.558, 0.667] | 0.658 | 0.620 | 0.695 | 0.605 |
| Boosted trees (monotone in VCI) | 0.654 [0.617, 0.692] | 0.668 [0.624, 0.711] | 0.621 [0.563, 0.676] | 0.649 | 0.655 | 0.719 | 0.586 |
| GRU | 0.579 [0.545, 0.615] | 0.614 [0.565, 0.665] | 0.582 [0.529, 0.635] | 0.567 | 0.538 | 0.618 | 0.564 |
| Residual Bi-LSTM | 0.576 [0.542, 0.611] | 0.616 [0.568, 0.666] | 0.592 [0.536, 0.647] | 0.566 | 0.541 | 0.606 | 0.566 |
| Bi-LSTM | 0.575 [0.541, 0.611] | 0.608 [0.558, 0.659] | 0.597 [0.542, 0.650] | 0.553 | 0.529 | 0.617 | 0.564 |
| VCI-only Bi-LSTM | 0.599 [0.571, 0.628] | 0.616 [0.580, 0.652] | 0.570 [0.520, 0.620] | 0.558 | 0.601 | 0.602 | 0.611 |

| Comparison | Δ all | Δ West | Δ East | Result |
|---|---|---|---|---|
| VCI-only Bi-LSTM − LR on latest VCI + age | −0.047 [−0.074, −0.019] | −0.040 [−0.070, −0.009] | −0.042 [−0.088, +0.001] | worse |
| VCI-only Bi-LSTM − Boosted trees (monotone in VCI) | −0.055 [−0.085, −0.023] | −0.052 [−0.082, −0.021] | −0.051 [−0.099, −0.005] | worse |
| VCI-only Bi-LSTM − Bi-LSTM | +0.024 [−0.007, +0.056] | +0.009 [−0.033, +0.050] | −0.027 [−0.082, +0.032] | no detectable difference |

### Forecast T+1..T+3, onsets only (not in drought at T) (mean onset within-fold AUC over leads 1-3)

| Model | All | West | East |
|---|---|---|---|
| LR on VCI(T) | 0.531 [0.516, 0.547] | 0.539 [0.518, 0.562] | 0.525 [0.506, 0.542] |
| LR + trend features | 0.555 [0.540, 0.571] | 0.546 [0.522, 0.571] | 0.590 [0.573, 0.607] |
| GRU | 0.519 [0.504, 0.535] | 0.534 [0.508, 0.563] | 0.536 [0.513, 0.557] |
| Bi-LSTM | 0.518 [0.502, 0.533] | 0.536 [0.512, 0.561] | 0.545 [0.522, 0.567] |
| VCI-only Bi-LSTM | 0.512 [0.499, 0.525] | 0.538 [0.519, 0.556] | 0.516 [0.498, 0.533] |

| Comparison | Δ all | Δ West | Δ East | Result |
|---|---|---|---|---|
| VCI-only Bi-LSTM − LR on VCI(T) | −0.019 [−0.036, −0.003] | −0.002 [−0.023, +0.018] | −0.009 [−0.030, +0.012] | worse |
| VCI-only Bi-LSTM − LR + trend features | −0.044 [−0.061, −0.027] | −0.008 [−0.032, +0.015] | −0.074 [−0.094, −0.056] | worse |
| VCI-only Bi-LSTM − Bi-LSTM | −0.006 [−0.020, +0.008] | +0.002 [−0.019, +0.021] | −0.029 [−0.050, −0.008] | no detectable difference |

### Estimation now where VCI changed most (top quartile of 1- or 3-month change) (within-fold AUC)

95,119 windows, 21,691 in drought.

| Model | All | West | East | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|---|---|
| LR on VCI(T) | 0.685 [0.667, 0.702] | 0.742 [0.719, 0.763] | 0.641 [0.618, 0.663] | 0.653 | 0.707 | 0.709 | 0.669 |
| Boosted trees (monotone in VCI) | 0.688 [0.671, 0.705] | 0.739 [0.715, 0.760] | 0.654 [0.632, 0.677] | 0.656 | 0.699 | 0.717 | 0.675 |
| GRU | 0.654 [0.636, 0.672] | 0.712 [0.689, 0.734] | 0.631 [0.609, 0.654] | 0.664 | 0.645 | 0.613 | 0.686 |
| Residual Bi-LSTM | 0.635 [0.616, 0.653] | 0.707 [0.684, 0.731] | 0.606 [0.583, 0.630] | 0.661 | 0.635 | 0.611 | 0.641 |
| Bi-LSTM | 0.625 [0.605, 0.643] | 0.710 [0.687, 0.733] | 0.590 [0.566, 0.616] | 0.669 | 0.644 | 0.602 | 0.613 |
| VCI-only Bi-LSTM | 0.655 [0.639, 0.671] | 0.712 [0.690, 0.733] | 0.619 [0.597, 0.642] | 0.640 | 0.689 | 0.654 | 0.647 |
| CNN + LR | 0.680 [0.663, 0.696] | 0.746 [0.724, 0.767] | 0.641 [0.619, 0.663] | 0.659 | 0.710 | 0.686 | 0.669 |
| CNN + Bi-LSTM | 0.658 [0.642, 0.674] | 0.709 [0.686, 0.731] | 0.625 [0.603, 0.648] | 0.639 | 0.682 | 0.661 | 0.653 |

| Comparison | Δ all | Δ West | Δ East | Result |
|---|---|---|---|---|
| VCI-only Bi-LSTM − LR on VCI(T) | −0.029 [−0.041, −0.017] | −0.030 [−0.046, −0.015] | −0.021 [−0.036, −0.006] | worse |
| VCI-only Bi-LSTM − Boosted trees (monotone in VCI) | −0.033 [−0.045, −0.020] | −0.027 [−0.041, −0.011] | −0.035 [−0.050, −0.019] | worse |
| CNN + LR − LR on VCI(T) | −0.005 [−0.010, −0.000] | +0.004 [−0.002, +0.009] | +0.000 [−0.004, +0.005] | worse |
| CNN + LR − Boosted trees (monotone in VCI) | −0.009 [−0.016, −0.002] | +0.008 [−0.002, +0.017] | −0.014 [−0.024, −0.003] | worse |
| CNN + Bi-LSTM − LR on VCI(T) | −0.026 [−0.038, −0.014] | −0.034 [−0.049, −0.019] | −0.015 [−0.030, +0.001] | worse |
| CNN + Bi-LSTM − Boosted trees (monotone in VCI) | −0.030 [−0.041, −0.018] | −0.030 [−0.045, −0.014] | −0.029 [−0.045, −0.012] | worse |
| VCI-only Bi-LSTM − Bi-LSTM | +0.031 [+0.017, +0.045] | +0.002 [−0.014, +0.018] | +0.029 [+0.006, +0.051] | passed |
| CNN + Bi-LSTM − VCI-only Bi-LSTM | +0.003 [−0.003, +0.009] | −0.003 [−0.013, +0.006] | +0.006 [−0.003, +0.015] | no detectable difference |
| CNN + Bi-LSTM − CNN + LR | −0.021 [−0.031, −0.011] | −0.037 [−0.051, −0.024] | −0.016 [−0.029, −0.001] | worse |

## 5. Checks on the new models (job 1)

**Order test** (months T-11..T-1 shuffled within each window, 5 permutations, final 10-seed models):

| Family | Original | Shuffled (mean of 5) | Shuffled − original (permutation 1) | Mean |Δp| |
|---|---|---|---|---|
| Bi-LSTM | 0.627 [0.610, 0.644] | 0.648 | +0.022 [+0.010, +0.035] | 0.104 |
| Residual Bi-LSTM | 0.636 [0.621, 0.653] | 0.634 | −0.003 [−0.013, +0.008] | 0.089 |
| GRU | 0.653 [0.637, 0.668] | 0.663 | +0.010 [−0.001, +0.022] | 0.086 |
| One-direction LSTM | 0.641 [0.624, 0.657] | 0.665 | +0.023 [+0.014, +0.033] | 0.081 |
| Temporal CNN | 0.643 [0.627, 0.659] | 0.670 | +0.026 [+0.019, +0.034] | 0.073 |

**Positive control** (fold 1; label: VCI_mean(T-3) - VCI_mean(T) > 20, both observed; 424,527 training and 47,212 validation windows, rate 0.236):

| Model | AUC | Δ vs LR on VCI(T) |
|---|---|---|
| LR on VCI(T) | 0.820 [0.802, 0.836] | — |
| Bi-LSTM | 0.982 [0.980, 0.984] | +0.163 [+0.146, +0.180] |
| Residual Bi-LSTM | 0.985 [0.983, 0.987] | +0.166 [+0.149, +0.183] |
| GRU | 1.000 [1.000, 1.000] | +0.180 [+0.164, +0.198] |
| One-direction LSTM | 1.000 [0.999, 1.000] | +0.180 [+0.163, +0.197] |
| Temporal CNN | 0.999 [0.999, 0.999] | +0.180 [+0.163, +0.197] |

## 6. What-ifs (secondary, job 1)

### NDMI added (SWIR, **not DEWSat's camera**)

267,410 windows in 558 regions (same windows for every row).

| Model | Within-fold AUC, all | West | East | Pooled AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| GRU (ndmi off) | 0.654 [0.640, 0.670] | 0.697 [0.672, 0.719] | 0.640 [0.621, 0.658] | 0.615 [0.599, 0.632] | 0.1852 [0.1765, 0.1942] | 0.093 |
| GRU (ndmi on) | 0.656 [0.642, 0.671] | 0.698 [0.674, 0.720] | 0.643 [0.623, 0.661] | 0.623 [0.609, 0.639] | 0.1838 [0.1755, 0.1922] | 0.090 |
| LR on VCI(T) | 0.686 [0.669, 0.703] | 0.716 [0.694, 0.737] | 0.651 [0.632, 0.670] | 0.642 [0.626, 0.659] | 0.1721 [0.1641, 0.1803] | 0.031 |
| lr_vci_t_ndmi | 0.683 [0.668, 0.699] | 0.720 [0.698, 0.740] | 0.644 [0.625, 0.661] | 0.643 [0.628, 0.658] | 0.1721 [0.1641, 0.1804] | 0.033 |

| Comparison | ΔAUC within-fold, all | West | East |
|---|---|---|---|
| GRU (ndmi on) − GRU (ndmi off) | +0.002 [−0.005, +0.009] | +0.001 [−0.008, +0.010] | +0.003 [−0.008, +0.013] |
| lr_vci_t_ndmi − LR on VCI(T) | −0.003 [−0.008, +0.003] | +0.004 [−0.001, +0.009] | −0.008 [−0.015, −0.000] |
| GRU (ndmi off) − LR on VCI(T) | −0.032 [−0.046, −0.017] | −0.019 [−0.037, −0.004] | −0.011 [−0.027, +0.004] |
| GRU (ndmi on) − lr_vci_t_ndmi | −0.027 [−0.040, −0.013] | −0.022 [−0.039, −0.007] | −0.001 [−0.020, +0.017] |

### VCI per cell vs per pixel

150,160 windows in 557 regions (same windows for every row).

| Model | Within-fold AUC, all | West | East | Pooled AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| GRU (pixel) | 0.658 [0.639, 0.678] | 0.723 [0.696, 0.749] | 0.616 [0.592, 0.641] | 0.612 [0.591, 0.632] | 0.1870 [0.1779, 0.1966] | 0.085 |
| GRU (cell) | 0.651 [0.632, 0.669] | 0.707 [0.682, 0.731] | 0.613 [0.591, 0.636] | 0.606 [0.585, 0.625] | 0.1878 [0.1788, 0.1968] | 0.083 |
| LR on VCI(T) (pixel) | 0.689 [0.671, 0.707] | 0.739 [0.715, 0.762] | 0.639 [0.617, 0.661] | 0.640 [0.619, 0.659] | 0.1761 [0.1674, 0.1852] | 0.041 |
| LR on VCI(T) (cell) | 0.679 [0.662, 0.695] | 0.723 [0.699, 0.746] | 0.628 [0.608, 0.649] | 0.630 [0.612, 0.647] | 0.1782 [0.1696, 0.1872] | 0.052 |

| Comparison | ΔAUC within-fold, all | West | East |
|---|---|---|---|
| GRU (cell) − GRU (pixel) | −0.007 [−0.015, +0.000] | −0.015 [−0.028, −0.005] | −0.003 [−0.013, +0.008] |
| LR on VCI(T) (cell) − LR on VCI(T) (pixel) | −0.010 [−0.017, −0.004] | −0.017 [−0.029, −0.006] | −0.011 [−0.019, −0.003] |
| GRU (pixel) − LR on VCI(T) (pixel) | −0.031 [−0.050, −0.014] | −0.017 [−0.038, +0.003] | −0.023 [−0.040, −0.004] |
| GRU (cell) − LR on VCI(T) (cell) | −0.028 [−0.046, −0.012] | −0.016 [−0.037, +0.005] | −0.015 [−0.029, +0.000] |

## 7. Weekly-input experiment (secondary; amendment 3)

52 weekly steps ending on the label month's last day, from single clear views (>= 30% of the cell clear); 327,213 validation cell-months with at least one clear view in the 52 weeks, 558 regions. Training windows capped at 300,000 per fold.

| Model | Within-fold AUC, all | West | East | Pooled AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| LR on latest VCI + age (weekly) | 0.649 [0.634, 0.664] | 0.678 [0.656, 0.697] | 0.609 [0.590, 0.626] | 0.580 [0.564, 0.597] | 0.1766 [0.1687, 0.1848] | 0.048 |
| GRU (weekly) | 0.631 [0.616, 0.646] | 0.669 [0.648, 0.691] | 0.641 [0.624, 0.658] | 0.589 [0.574, 0.606] | 0.1819 [0.1738, 0.1904] | 0.075 |

- GRU (weekly) − LR on latest VCI + age (weekly): −0.018 [−0.033, −0.003] (West −0.009 [−0.028, +0.009], East +0.032 [+0.014, +0.051]).

| Fold | Validation windows | With a clear view | Share of weeks with a new view |
|---|---|---|---|
| 1 | 85,464 | 78,926 | 0.285 |
| 2 | 83,640 | 82,554 | 0.455 |
| 3 | 83,184 | 83,160 | 0.484 |
| 4 | 83,160 | 82,573 | 0.536 |

## 7b. Learning curve (amendment 5; secondary, no pass/fail)

Does more training data close the gap to LR? Networks Bi-LSTM, CNN + Bi-LSTM, VCI-only Bi-LSTM, each fold's chosen setting retrained on slices of the fold's training windows (seeds 42, 43, 44); LR on VCI(T) refit on every slice. Job 1, development folds only, within-fold AUC, whole-region resamples shared by every row of an axis.

### More years (folds 1, 4; 133,238 scored windows in 279 regions)

| Training data | Mean training windows | LR on VCI(T) | Bi-LSTM | CNN + Bi-LSTM | VCI-only Bi-LSTM | Bi-LSTM − LR | CNN + Bi-LSTM − LR | VCI-only Bi-LSTM − LR |
|---|---|---|---|---|---|---|---|---|
| 2 of 5 years (mean of 3 subsets) | 199,264 | 0.649 [0.629, 0.670] | 0.582 [0.563, 0.604] | 0.614 [0.594, 0.633] | 0.611 [0.591, 0.630] | −0.067 [−0.084, −0.049] | −0.035 [−0.046, −0.024] | −0.038 [−0.052, −0.025] |
| 3 of 5 years (mean of 3 subsets) | 296,884 | 0.649 [0.629, 0.670] | 0.616 [0.597, 0.635] | 0.631 [0.612, 0.650] | 0.629 [0.609, 0.648] | −0.033 [−0.052, −0.013] | −0.018 [−0.028, −0.009] | −0.020 [−0.030, −0.011] |
| 4 of 5 years (mean of 3 subsets) | 395,297 | 0.649 [0.629, 0.670] | 0.612 [0.591, 0.635] | 0.635 [0.615, 0.654] | 0.630 [0.611, 0.648] | −0.037 [−0.057, −0.017] | −0.014 [−0.024, −0.004] | −0.019 [−0.029, −0.009] |
| 5 of 5 years (full) | 494,380 | 0.649 [0.629, 0.670] | 0.621 [0.599, 0.643] | 0.640 [0.621, 0.660] | 0.633 [0.613, 0.652] | −0.028 [−0.047, −0.009] | −0.009 [−0.019, +0.001] | −0.016 [−0.026, −0.006] |

- Bi-LSTM: change in (network − LR) from the smallest slice to the full data +0.038 [+0.028, +0.048]; from the second-largest to the full data +0.008 [+0.004, +0.014]: still rising at the full data, so more of this kind of data would likely help (reading rule fixed in amendment 5).
- CNN + Bi-LSTM: change in (network − LR) from the smallest slice to the full data +0.026 [+0.020, +0.033]; from the second-largest to the full data +0.006 [+0.004, +0.008]: still rising at the full data, so more of this kind of data would likely help (reading rule fixed in amendment 5).
- VCI-only Bi-LSTM: change in (network − LR) from the smallest slice to the full data +0.022 [+0.015, +0.030]; from the second-largest to the full data +0.003 [+0.001, +0.005]: still rising at the full data, so more of this kind of data would likely help (reading rule fixed in amendment 5).

### More regions (folds 1, 2, 3, 4; 267,412 scored windows in 558 regions)

| Training data | Mean training windows | LR on VCI(T) | Bi-LSTM | CNN + Bi-LSTM | VCI-only Bi-LSTM | Bi-LSTM − LR | CNN + Bi-LSTM − LR | VCI-only Bi-LSTM − LR |
|---|---|---|---|---|---|---|---|---|
| 1/8 of the regions | 51,250 | 0.686 [0.670, 0.702] | 0.635 [0.618, 0.652] | 0.659 [0.643, 0.676] | 0.649 [0.634, 0.664] | −0.051 [−0.066, −0.035] | −0.027 [−0.036, −0.018] | −0.037 [−0.048, −0.027] |
| 1/4 of the regions | 105,297 | 0.686 [0.670, 0.702] | 0.644 [0.628, 0.661] | 0.670 [0.654, 0.685] | 0.663 [0.648, 0.677] | −0.042 [−0.057, −0.026] | −0.016 [−0.023, −0.009] | −0.023 [−0.032, −0.015] |
| 1/2 of the regions | 208,413 | 0.686 [0.670, 0.702] | 0.633 [0.618, 0.650] | 0.663 [0.648, 0.678] | 0.662 [0.647, 0.677] | −0.053 [−0.068, −0.035] | −0.023 [−0.032, −0.014] | −0.024 [−0.034, −0.015] |
| all regions (full) | 415,561 | 0.686 [0.670, 0.702] | 0.628 [0.612, 0.645] | 0.660 [0.645, 0.674] | 0.658 [0.644, 0.672] | −0.057 [−0.073, −0.041] | −0.026 [−0.036, −0.016] | −0.028 [−0.037, −0.019] |

- Bi-LSTM: change in (network − LR) from the smallest slice to the full data −0.007 [−0.019, +0.006]; from the second-largest to the full data −0.005 [−0.012, +0.002]: flat, so more of this kind is not expected to close the gap (reading rule fixed in amendment 5).
- CNN + Bi-LSTM: change in (network − LR) from the smallest slice to the full data +0.000 [−0.009, +0.009]; from the second-largest to the full data −0.003 [−0.007, +0.000]: flat, so more of this kind is not expected to close the gap (reading rule fixed in amendment 5).
- VCI-only Bi-LSTM: change in (network − LR) from the smallest slice to the full data +0.009 [+0.001, +0.018]; from the second-largest to the full data −0.004 [−0.007, −0.001]: rose, then fell slightly before the full data, so more of this kind is not expected to close the gap (reading rule fixed in amendment 5).

**Across the two axes:** 3 of 3 networks are still rising at the full data when years are added (training windows × 2.5), 0 of 3 when regions are added (× 8.1). More windows from more places did not narrow the gap; more years did. What these networks lack in development is years of record, not training windows.

## 7c. ReLU Bi-LSTM (amendment 7; secondary, no pass/fail)

Added after every result, at the user's request: the plain Bi-LSTM with ReLU in place of tanh inside the LSTM cell (candidate memory and output squash; the gates stay sigmoid), written step by step (`nets.HandLSTM`). Each fold's chosen `bilstm` setting, the 10 final seeds, and the same initial weights and data order as the original runs. Development folds only: the fresh test is locked and was not scored again.

Checks before the runs: with tanh, the hand-written cell matches `nn.LSTM` to within 4.2e-07 on cpu and mps (passed); retrained as final runs on fold 1, it reproduced the original runs of seeds 42, 43, 46 (AUC difference at most 1e-08; passed); pointed at the original runs, the scoring path returns the score stage's own AUCs (passed).

### Estimation now (month T observed) (267,412 windows, 61,618 in drought)

| Model | Within-fold AUC | West | East | Per-fold AUC |
|---|---|---|---|---|
| ReLU Bi-LSTM | 0.628 [0.612, 0.645] | 0.685 [0.660, 0.709] | 0.607 [0.587, 0.627] | 0.651, 0.667, 0.615, 0.603 |
| Bi-LSTM | 0.627 [0.611, 0.644] | 0.686 [0.662, 0.710] | 0.603 [0.582, 0.624] | 0.652, 0.666, 0.619, 0.597 |
| LR on VCI(T) | 0.686 [0.670, 0.703] | 0.716 [0.694, 0.738] | 0.651 [0.632, 0.670] | 0.641, 0.717, 0.733, 0.654 |
| Boosted trees (monotone in VCI) | 0.686 [0.670, 0.703] | 0.710 [0.687, 0.732] | 0.660 [0.641, 0.678] | 0.638, 0.712, 0.733, 0.659 |

| Comparison | All regions | West | East | Reading |
|---|---|---|---|---|
| ReLU Bi-LSTM − Bi-LSTM | +0.001 [−0.003, +0.005] | −0.001 [−0.006, +0.003] | +0.004 [−0.002, +0.010] | no detectable difference (the primary comparison; reading fixed in amendment 7) |
| ReLU Bi-LSTM − LR on VCI(T) | −0.058 [−0.074, −0.042] | −0.031 [−0.049, −0.014] | −0.044 [−0.065, −0.022] | worse (gate wording) |
| ReLU Bi-LSTM − Boosted trees (monotone in VCI) | −0.058 [−0.074, −0.043] | −0.025 [−0.043, −0.008] | −0.053 [−0.073, −0.032] | worse (gate wording) |
| Bi-LSTM − LR on VCI(T) | −0.059 [−0.075, −0.043] | −0.030 [−0.048, −0.012] | −0.048 [−0.071, −0.025] | worse (for reference) |

### Estimation now where VCI changed most (top quartile of 1- or 3-month change) (95,119 windows, 21,691 in drought)

| Model | Within-fold AUC | West | East | Per-fold AUC |
|---|---|---|---|---|
| ReLU Bi-LSTM | 0.626 [0.607, 0.644] | 0.707 [0.684, 0.730] | 0.594 [0.571, 0.619] | 0.667, 0.643, 0.596, 0.621 |
| Bi-LSTM | 0.625 [0.605, 0.643] | 0.710 [0.687, 0.733] | 0.590 [0.566, 0.616] | 0.669, 0.644, 0.602, 0.613 |
| LR on VCI(T) | 0.685 [0.667, 0.702] | 0.742 [0.719, 0.763] | 0.641 [0.618, 0.663] | 0.653, 0.707, 0.709, 0.669 |
| Boosted trees (monotone in VCI) | 0.688 [0.671, 0.705] | 0.739 [0.715, 0.760] | 0.654 [0.632, 0.677] | 0.656, 0.699, 0.717, 0.675 |

| Comparison | All regions | West | East | Reading |
|---|---|---|---|---|
| ReLU Bi-LSTM − Bi-LSTM | +0.001 [−0.004, +0.006] | −0.003 [−0.009, +0.003] | +0.004 [−0.004, +0.012] | no detectable difference (the primary comparison; reading fixed in amendment 7) |
| ReLU Bi-LSTM − LR on VCI(T) | −0.059 [−0.079, −0.039] | −0.035 [−0.056, −0.016] | −0.047 [−0.075, −0.018] | worse (gate wording) |
| ReLU Bi-LSTM − Boosted trees (monotone in VCI) | −0.063 [−0.081, −0.044] | −0.032 [−0.051, −0.013] | −0.060 [−0.086, −0.034] | worse (gate wording) |
| Bi-LSTM − LR on VCI(T) | −0.060 [−0.080, −0.041] | −0.033 [−0.052, −0.015] | −0.050 [−0.079, −0.020] | worse (for reference) |

Runs: 40 (10 seeds x 4 folds). Every training loss fell: yes; all finite: yes. Saved seed-42 models reproduce their predictions to within 5.7e-06. Mean seconds per run 318, against 265 for the original runs (the hand-written cell gives up PyTorch's fused LSTM kernel). Parameters equal the Bi-LSTM's (fold 1 11,617, fold 2 11,617, fold 3 274,945, fold 4 670,209).

## 8. Fresh test (Phase 5, scored once)

**Pre-registered verdicts** (the section 9 rule: passed = the all-regions interval above zero and neither side's interval entirely below zero; AUC differences, 2,000 whole-region resamples):

| Job | Row | Comparison | All | West | East | Verdict |
|---|---|---|---|---|---|---|
| Job 1, estimation now | gate | GRU − Boosted trees (monotone in VCI) | +0.036 [+0.019, +0.054] | +0.022 [+0.001, +0.044] | +0.009 [−0.014, +0.030] | **passed** |
| Job 1, estimation now | best Bi-LSTM variant | Residual Bi-LSTM − Boosted trees (monotone in VCI) | +0.039 [+0.020, +0.057] | +0.025 [+0.003, +0.048] | +0.011 [−0.015, +0.035] | **passed** |
| Job 1, estimation now | against LR on VCI(T) | GRU − LR on VCI(T) | +0.035 [+0.019, +0.054] | +0.016 [−0.004, +0.036] | +0.026 [+0.001, +0.049] | **passed** |
| Job 1, estimation now | against LR on VCI(T) | Residual Bi-LSTM − LR on VCI(T) | +0.038 [+0.020, +0.057] | +0.019 [−0.001, +0.041] | +0.028 [+0.000, +0.054] | **passed** |
| Job 2, missing month | gate | GRU − Boosted trees (monotone in VCI) | +0.002 [−0.038, +0.042] | +0.047 [+0.006, +0.095] | −0.035 [−0.091, +0.020] | **no detectable difference** |
| Job 2, missing month | best Bi-LSTM variant | Residual Bi-LSTM − Boosted trees (monotone in VCI) | +0.013 [−0.024, +0.048] | +0.041 [+0.011, +0.073] | −0.016 [−0.067, +0.038] | **no detectable difference** |
| Job 3, forecast (mean onset AUC, leads 1-3) | gate | GRU − LR + trend features | +0.024 [−0.007, +0.054] | +0.062 [+0.013, +0.114] | −0.036 [−0.070, −0.001] | **no detectable difference** |
| Job 3, forecast (mean onset AUC, leads 1-3) | best Bi-LSTM variant | Bi-LSTM − LR + trend features | +0.038 [+0.007, +0.069] | +0.062 [+0.013, +0.113] | −0.025 [−0.059, +0.009] | **passed** |
| Job 4, where VCI changed most | gate | GRU − Boosted trees (monotone in VCI) | +0.051 [+0.027, +0.075] | +0.021 [+0.002, +0.041] | +0.028 [−0.011, +0.063] | **passed** |
| Job 4, where VCI changed most | best Bi-LSTM variant | Residual Bi-LSTM − Boosted trees (monotone in VCI) | +0.045 [+0.019, +0.070] | +0.020 [−0.002, +0.045] | +0.020 [−0.023, +0.058] | **passed** |

Rows added after the development results (DESIGN.md amendment 6; not part of the gate): job1: Bi-LSTM, VCI-only Bi-LSTM, CNN + LR, CNN + Bi-LSTM; job2: Bi-LSTM, VCI-only Bi-LSTM; job3: VCI-only Bi-LSTM; job4: Bi-LSTM, VCI-only Bi-LSTM, CNN + LR, CNN + Bi-LSTM. The pre-registered rows are each job's best network and best Bi-LSTM variant (section 4).

**The 2x2 table** (job 1, fresh test, AUC, all regions):

| History | Cell average (VCI) | Cell image (23x21 VCI tiles) | Image − average |
|---|---|---|---|
| Month T only | LR on VCI(T): 0.621 [0.590, 0.652] | CNN + LR: 0.638 [0.609, 0.665] | +0.016 [+0.008, +0.024] |
| 12 months | VCI-only Bi-LSTM: 0.644 [0.615, 0.672] | CNN + Bi-LSTM: 0.636 [0.608, 0.664] | −0.008 [−0.016, +0.000] |
| 12 months − month T | +0.023 [+0.013, +0.033] | −0.001 [−0.011, +0.008] | |

### Job 1

65,170 windows, 25,734 in drought, 144 regions.

| Model | AUC, all | West | East | Pooled AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| Climatology | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.2671 [0.2476, 0.2883] | 0.168 |
| LR on VCI(T) | 0.621 [0.590, 0.652] | 0.715 [0.674, 0.751] | 0.588 [0.559, 0.619] | 0.621 [0.590, 0.652] | 0.2559 [0.2382, 0.2742] | 0.161 |
| LR on latest VCI + age | 0.621 [0.590, 0.652] | 0.715 [0.674, 0.751] | 0.588 [0.559, 0.619] | 0.621 [0.590, 0.652] | 0.2559 [0.2382, 0.2742] | 0.161 |
| LR + trend features | 0.613 [0.581, 0.645] | 0.705 [0.659, 0.743] | 0.590 [0.560, 0.622] | 0.613 [0.581, 0.645] | 0.2580 [0.2403, 0.2764] | 0.161 |
| Boosted trees (monotone in VCI) | 0.621 [0.589, 0.652] | 0.709 [0.664, 0.747] | 0.606 [0.575, 0.637] | 0.621 [0.589, 0.652] | 0.2560 [0.2387, 0.2742] | 0.159 |
| Smoothed-VCI state model | 0.620 [0.589, 0.651] | 0.713 [0.670, 0.750] | 0.592 [0.562, 0.624] | 0.620 [0.589, 0.651] | 0.2564 [0.2387, 0.2744] | 0.160 |
| GRU | 0.657 [0.631, 0.683] | 0.731 [0.699, 0.761] | 0.614 [0.581, 0.648] | 0.657 [0.631, 0.683] | 0.2472 [0.2305, 0.2642] | 0.149 |
| Residual Bi-LSTM | 0.659 [0.633, 0.686] | 0.734 [0.701, 0.765] | 0.617 [0.581, 0.654] | 0.659 [0.633, 0.686] | 0.2466 [0.2300, 0.2642] | 0.150 |
| Bi-LSTM | 0.663 [0.639, 0.688] | 0.727 [0.695, 0.758] | 0.625 [0.593, 0.658] | 0.663 [0.639, 0.688] | 0.2456 [0.2293, 0.2626] | 0.149 |
| VCI-only Bi-LSTM | 0.644 [0.615, 0.672] | 0.718 [0.679, 0.752] | 0.604 [0.574, 0.632] | 0.644 [0.615, 0.672] | 0.2514 [0.2345, 0.2688] | 0.154 |
| CNN + LR | 0.638 [0.609, 0.665] | 0.718 [0.678, 0.755] | 0.594 [0.564, 0.623] | 0.638 [0.609, 0.665] | 0.2532 [0.2351, 0.2717] | 0.162 |
| CNN + Bi-LSTM | 0.636 [0.608, 0.664] | 0.712 [0.673, 0.748] | 0.600 [0.570, 0.632] | 0.636 [0.608, 0.664] | 0.2542 [0.2366, 0.2721] | 0.157 |

| Comparison | ΔAUC, all | West | East |
|---|---|---|---|
| Climatology − Boosted trees (monotone in VCI) | −0.121 [−0.152, −0.089] | −0.209 [−0.247, −0.164] | −0.106 [−0.137, −0.075] |
| LR on VCI(T) − Boosted trees (monotone in VCI) | +0.001 [−0.004, +0.005] | +0.006 [+0.001, +0.011] | −0.017 [−0.024, −0.011] |
| LR on latest VCI + age − Boosted trees (monotone in VCI) | +0.001 [−0.004, +0.005] | +0.006 [+0.001, +0.011] | −0.017 [−0.024, −0.011] |
| LR + trend features − Boosted trees (monotone in VCI) | −0.007 [−0.010, −0.004] | −0.004 [−0.007, −0.001] | −0.015 [−0.020, −0.010] |
| Smoothed-VCI state model − Boosted trees (monotone in VCI) | −0.001 [−0.005, +0.003] | +0.004 [−0.000, +0.009] | −0.013 [−0.020, −0.007] |
| GRU − Boosted trees (monotone in VCI) | +0.036 [+0.019, +0.054] | +0.022 [+0.001, +0.044] | +0.009 [−0.014, +0.030] |
| Residual Bi-LSTM − Boosted trees (monotone in VCI) | +0.039 [+0.020, +0.057] | +0.025 [+0.003, +0.048] | +0.011 [−0.015, +0.035] |
| Bi-LSTM − Boosted trees (monotone in VCI) | +0.042 [+0.024, +0.061] | +0.019 [−0.004, +0.043] | +0.020 [−0.005, +0.041] |
| VCI-only Bi-LSTM − Boosted trees (monotone in VCI) | +0.024 [+0.012, +0.035] | +0.009 [−0.006, +0.025] | −0.002 [−0.016, +0.011] |
| CNN + LR − Boosted trees (monotone in VCI) | +0.017 [+0.007, +0.026] | +0.009 [−0.003, +0.023] | −0.012 [−0.021, −0.003] |
| CNN + Bi-LSTM − Boosted trees (monotone in VCI) | +0.015 [+0.004, +0.027] | +0.003 [−0.013, +0.021] | −0.005 [−0.019, +0.008] |
| Climatology − LR on VCI(T) | −0.121 [−0.152, −0.090] | −0.215 [−0.251, −0.174] | −0.088 [−0.119, −0.059] |
| LR on latest VCI + age − LR on VCI(T) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| LR + trend features − LR on VCI(T) | −0.008 [−0.012, −0.004] | −0.010 [−0.017, −0.004] | +0.002 [−0.004, +0.008] |
| Boosted trees (monotone in VCI) − LR on VCI(T) | −0.001 [−0.005, +0.004] | −0.006 [−0.011, −0.001] | +0.017 [+0.011, +0.024] |
| Smoothed-VCI state model − LR on VCI(T) | −0.001 [−0.004, +0.001] | −0.002 [−0.005, +0.001] | +0.004 [−0.000, +0.008] |
| GRU − LR on VCI(T) | +0.035 [+0.019, +0.054] | +0.016 [−0.004, +0.036] | +0.026 [+0.001, +0.049] |
| Residual Bi-LSTM − LR on VCI(T) | +0.038 [+0.020, +0.057] | +0.019 [−0.001, +0.041] | +0.028 [+0.000, +0.054] |
| Bi-LSTM − LR on VCI(T) | +0.042 [+0.024, +0.060] | +0.013 [−0.008, +0.035] | +0.037 [+0.010, +0.061] |
| VCI-only Bi-LSTM − LR on VCI(T) | +0.023 [+0.013, +0.033] | +0.003 [−0.010, +0.017] | +0.015 [+0.002, +0.028] |
| CNN + LR − LR on VCI(T) | +0.016 [+0.008, +0.024] | +0.003 [−0.007, +0.014] | +0.005 [−0.000, +0.010] |
| CNN + Bi-LSTM − LR on VCI(T) | +0.015 [+0.004, +0.025] | −0.002 [−0.017, +0.013] | +0.012 [−0.004, +0.027] |
| VCI-only Bi-LSTM − Bi-LSTM | −0.019 [−0.032, −0.006] | −0.010 [−0.025, +0.005] | −0.022 [−0.042, −0.001] |
| CNN + Bi-LSTM − VCI-only Bi-LSTM | −0.008 [−0.016, +0.000] | −0.005 [−0.019, +0.008] | −0.003 [−0.015, +0.008] |
| CNN + Bi-LSTM − CNN + LR | −0.001 [−0.011, +0.008] | −0.006 [−0.020, +0.009] | +0.007 [−0.009, +0.023] |

### Job 2

7,894 windows, 3,439 in drought, 130 regions.

| Model | AUC, all | West | East | Pooled AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| Climatology | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.2834 [0.2521, 0.3167] | 0.194 |
| LR on latest VCI + age | 0.602 [0.535, 0.667] | 0.709 [0.614, 0.789] | 0.552 [0.465, 0.636] | 0.602 [0.535, 0.667] | 0.2847 [0.2488, 0.3252] | 0.214 |
| LR + trend features | 0.589 [0.524, 0.653] | 0.700 [0.604, 0.782] | 0.542 [0.461, 0.625] | 0.589 [0.524, 0.653] | 0.2838 [0.2484, 0.3235] | 0.206 |
| Boosted trees (monotone in VCI) | 0.603 [0.537, 0.666] | 0.710 [0.615, 0.791] | 0.554 [0.468, 0.638] | 0.603 [0.537, 0.666] | 0.2797 [0.2465, 0.3165] | 0.200 |
| Smoothed-VCI state model | 0.593 [0.524, 0.659] | 0.719 [0.626, 0.797] | 0.529 [0.442, 0.615] | 0.593 [0.524, 0.659] | 0.2865 [0.2501, 0.3276] | 0.215 |
| GRU | 0.605 [0.534, 0.675] | 0.758 [0.676, 0.830] | 0.518 [0.438, 0.607] | 0.605 [0.534, 0.675] | 0.2774 [0.2384, 0.3193] | 0.193 |
| Residual Bi-LSTM | 0.616 [0.547, 0.682] | 0.751 [0.662, 0.828] | 0.538 [0.449, 0.629] | 0.616 [0.547, 0.682] | 0.2707 [0.2339, 0.3105] | 0.181 |
| Bi-LSTM | 0.618 [0.546, 0.687] | 0.762 [0.675, 0.836] | 0.499 [0.413, 0.590] | 0.618 [0.546, 0.687] | 0.2719 [0.2352, 0.3120] | 0.190 |
| VCI-only Bi-LSTM | 0.605 [0.541, 0.665] | 0.711 [0.618, 0.790] | 0.538 [0.458, 0.618] | 0.605 [0.541, 0.665] | 0.2837 [0.2480, 0.3223] | 0.212 |

| Comparison | ΔAUC, all | West | East |
|---|---|---|---|
| Climatology − Boosted trees (monotone in VCI) | −0.103 [−0.166, −0.037] | −0.210 [−0.291, −0.115] | −0.054 [−0.138, +0.032] |
| LR on latest VCI + age − Boosted trees (monotone in VCI) | −0.001 [−0.010, +0.008] | −0.002 [−0.010, +0.006] | −0.002 [−0.017, +0.012] |
| LR + trend features − Boosted trees (monotone in VCI) | −0.014 [−0.026, −0.002] | −0.010 [−0.021, +0.000] | −0.012 [−0.030, +0.007] |
| Smoothed-VCI state model − Boosted trees (monotone in VCI) | −0.010 [−0.025, +0.005] | +0.008 [−0.002, +0.020] | −0.025 [−0.048, −0.003] |
| GRU − Boosted trees (monotone in VCI) | +0.002 [−0.038, +0.042] | +0.047 [+0.006, +0.095] | −0.035 [−0.091, +0.020] |
| Residual Bi-LSTM − Boosted trees (monotone in VCI) | +0.013 [−0.024, +0.048] | +0.041 [+0.011, +0.073] | −0.016 [−0.067, +0.038] |
| Bi-LSTM − Boosted trees (monotone in VCI) | +0.016 [−0.027, +0.055] | +0.051 [+0.019, +0.085] | −0.055 [−0.110, +0.008] |
| VCI-only Bi-LSTM − Boosted trees (monotone in VCI) | +0.002 [−0.031, +0.033] | +0.000 [−0.023, +0.022] | −0.016 [−0.071, +0.035] |
| Climatology − LR on latest VCI + age | −0.102 [−0.167, −0.035] | −0.209 [−0.289, −0.114] | −0.052 [−0.136, +0.035] |
| LR + trend features − LR on latest VCI + age | −0.013 [−0.023, −0.003] | −0.008 [−0.021, +0.004] | −0.010 [−0.024, +0.006] |
| Boosted trees (monotone in VCI) − LR on latest VCI + age | +0.001 [−0.008, +0.010] | +0.002 [−0.006, +0.010] | +0.002 [−0.012, +0.017] |
| Smoothed-VCI state model − LR on latest VCI + age | −0.009 [−0.020, +0.002] | +0.010 [+0.002, +0.020] | −0.023 [−0.040, −0.007] |
| GRU − LR on latest VCI + age | +0.003 [−0.038, +0.043] | +0.049 [+0.007, +0.096] | −0.034 [−0.089, +0.020] |
| Residual Bi-LSTM − LR on latest VCI + age | +0.015 [−0.024, +0.050] | +0.043 [+0.010, +0.075] | −0.014 [−0.064, +0.038] |
| Bi-LSTM − LR on latest VCI + age | +0.017 [−0.027, +0.057] | +0.053 [+0.020, +0.087] | −0.053 [−0.107, +0.009] |
| VCI-only Bi-LSTM − LR on latest VCI + age | +0.003 [−0.032, +0.037] | +0.002 [−0.026, +0.027] | −0.014 [−0.068, +0.034] |
| VCI-only Bi-LSTM − Bi-LSTM | −0.014 [−0.040, +0.014] | −0.051 [−0.080, −0.022] | +0.039 [−0.009, +0.085] |

### Job 4

21,095 windows, 7,717 in drought, 144 regions.

| Model | AUC, all | West | East | Pooled AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| Climatology | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.2512 [0.2303, 0.2717] | 0.139 |
| LR on VCI(T) | 0.633 [0.598, 0.667] | 0.725 [0.688, 0.760] | 0.590 [0.544, 0.633] | 0.633 [0.598, 0.667] | 0.2405 [0.2202, 0.2616] | 0.138 |
| LR on latest VCI + age | 0.633 [0.598, 0.667] | 0.725 [0.688, 0.760] | 0.590 [0.544, 0.633] | 0.633 [0.598, 0.667] | 0.2405 [0.2202, 0.2616] | 0.138 |
| LR + trend features | 0.627 [0.593, 0.660] | 0.715 [0.680, 0.749] | 0.598 [0.554, 0.642] | 0.627 [0.593, 0.660] | 0.2417 [0.2216, 0.2625] | 0.141 |
| Boosted trees (monotone in VCI) | 0.643 [0.609, 0.675] | 0.718 [0.682, 0.753] | 0.626 [0.582, 0.669] | 0.643 [0.609, 0.675] | 0.2409 [0.2212, 0.2617] | 0.145 |
| Smoothed-VCI state model | 0.633 [0.600, 0.667] | 0.720 [0.684, 0.755] | 0.602 [0.557, 0.646] | 0.633 [0.600, 0.667] | 0.2417 [0.2216, 0.2626] | 0.142 |
| GRU | 0.694 [0.663, 0.723] | 0.739 [0.700, 0.776] | 0.654 [0.610, 0.698] | 0.694 [0.663, 0.723] | 0.2268 [0.2085, 0.2456] | 0.128 |
| Residual Bi-LSTM | 0.688 [0.655, 0.718] | 0.738 [0.696, 0.779] | 0.646 [0.599, 0.693] | 0.688 [0.655, 0.718] | 0.2279 [0.2088, 0.2473] | 0.129 |
| Bi-LSTM | 0.697 [0.666, 0.725] | 0.734 [0.692, 0.774] | 0.661 [0.619, 0.703] | 0.697 [0.666, 0.725] | 0.2261 [0.2076, 0.2448] | 0.129 |
| VCI-only Bi-LSTM | 0.670 [0.638, 0.703] | 0.731 [0.688, 0.772] | 0.626 [0.585, 0.667] | 0.670 [0.638, 0.703] | 0.2335 [0.2138, 0.2536] | 0.135 |
| CNN + LR | 0.650 [0.616, 0.685] | 0.729 [0.685, 0.770] | 0.598 [0.552, 0.642] | 0.650 [0.616, 0.685] | 0.2386 [0.2175, 0.2607] | 0.144 |
| CNN + Bi-LSTM | 0.653 [0.620, 0.686] | 0.712 [0.662, 0.758] | 0.614 [0.571, 0.655] | 0.653 [0.620, 0.686] | 0.2383 [0.2180, 0.2595] | 0.139 |

| Comparison | ΔAUC, all | West | East |
|---|---|---|---|
| Climatology − Boosted trees (monotone in VCI) | −0.143 [−0.175, −0.109] | −0.218 [−0.253, −0.182] | −0.126 [−0.169, −0.082] |
| LR on VCI(T) − Boosted trees (monotone in VCI) | −0.010 [−0.016, −0.004] | +0.006 [+0.000, +0.012] | −0.036 [−0.046, −0.027] |
| LR on latest VCI + age − Boosted trees (monotone in VCI) | −0.010 [−0.016, −0.004] | +0.006 [+0.000, +0.012] | −0.036 [−0.046, −0.027] |
| LR + trend features − Boosted trees (monotone in VCI) | −0.016 [−0.020, −0.011] | −0.003 [−0.009, +0.001] | −0.027 [−0.033, −0.021] |
| Smoothed-VCI state model − Boosted trees (monotone in VCI) | −0.009 [−0.014, −0.004] | +0.001 [−0.004, +0.007] | −0.024 [−0.032, −0.016] |
| GRU − Boosted trees (monotone in VCI) | +0.051 [+0.027, +0.075] | +0.021 [+0.002, +0.041] | +0.028 [−0.011, +0.063] |
| Residual Bi-LSTM − Boosted trees (monotone in VCI) | +0.045 [+0.019, +0.070] | +0.020 [−0.002, +0.045] | +0.020 [−0.023, +0.058] |
| Bi-LSTM − Boosted trees (monotone in VCI) | +0.054 [+0.028, +0.080] | +0.015 [−0.007, +0.041] | +0.035 [−0.003, +0.071] |
| VCI-only Bi-LSTM − Boosted trees (monotone in VCI) | +0.027 [+0.013, +0.041] | +0.012 [−0.004, +0.029] | +0.001 [−0.015, +0.015] |
| CNN + LR − Boosted trees (monotone in VCI) | +0.008 [−0.004, +0.019] | +0.011 [−0.004, +0.026] | −0.028 [−0.040, −0.018] |
| CNN + Bi-LSTM − Boosted trees (monotone in VCI) | +0.010 [−0.008, +0.027] | −0.006 [−0.029, +0.016] | −0.011 [−0.033, +0.010] |
| Climatology − LR on VCI(T) | −0.133 [−0.167, −0.098] | −0.225 [−0.260, −0.188] | −0.090 [−0.133, −0.044] |
| LR on latest VCI + age − LR on VCI(T) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| LR + trend features − LR on VCI(T) | −0.006 [−0.011, −0.000] | −0.010 [−0.018, −0.002] | +0.009 [+0.001, +0.017] |
| Boosted trees (monotone in VCI) − LR on VCI(T) | +0.010 [+0.004, +0.016] | −0.006 [−0.012, −0.000] | +0.036 [+0.027, +0.046] |
| Smoothed-VCI state model − LR on VCI(T) | +0.001 [−0.004, +0.005] | −0.005 [−0.011, +0.001] | +0.012 [+0.006, +0.018] |
| GRU − LR on VCI(T) | +0.061 [+0.034, +0.086] | +0.014 [−0.003, +0.033] | +0.064 [+0.023, +0.102] |
| Residual Bi-LSTM − LR on VCI(T) | +0.055 [+0.027, +0.081] | +0.014 [−0.006, +0.035] | +0.056 [+0.010, +0.096] |
| Bi-LSTM − LR on VCI(T) | +0.064 [+0.036, +0.091] | +0.009 [−0.012, +0.032] | +0.072 [+0.029, +0.109] |
| VCI-only Bi-LSTM − LR on VCI(T) | +0.037 [+0.022, +0.052] | +0.006 [−0.009, +0.022] | +0.037 [+0.021, +0.053] |
| CNN + LR − LR on VCI(T) | +0.018 [+0.007, +0.027] | +0.005 [−0.008, +0.017] | +0.008 [+0.003, +0.013] |
| CNN + Bi-LSTM − LR on VCI(T) | +0.020 [+0.001, +0.038] | −0.013 [−0.035, +0.008] | +0.025 [+0.000, +0.049] |
| VCI-only Bi-LSTM − Bi-LSTM | −0.027 [−0.045, −0.007] | −0.003 [−0.018, +0.011] | −0.035 [−0.064, +0.001] |
| CNN + Bi-LSTM − VCI-only Bi-LSTM | −0.017 [−0.028, −0.006] | −0.019 [−0.034, −0.004] | −0.012 [−0.032, +0.007] |
| CNN + Bi-LSTM − CNN + LR | +0.002 [−0.013, +0.018] | −0.017 [−0.034, +0.000] | +0.017 [−0.008, +0.042] |

### Job 3 (mean onset AUC over leads 1-3)

| Model | All | West | East |
|---|---|---|---|
| Bi-LSTM | 0.538 [0.513, 0.565] | 0.551 [0.505, 0.595] | 0.548 [0.513, 0.587] |
| VCI-only Bi-LSTM | 0.527 [0.503, 0.550] | 0.523 [0.485, 0.559] | 0.552 [0.525, 0.579] |
| Climatology | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| Boosted trees (monotone in VCI) | 0.525 [0.501, 0.551] | 0.502 [0.460, 0.546] | 0.598 [0.567, 0.629] |
| GRU | 0.524 [0.499, 0.550] | 0.550 [0.505, 0.592] | 0.538 [0.502, 0.576] |
| LR on latest VCI + age | 0.493 [0.469, 0.519] | 0.521 [0.479, 0.562] | 0.524 [0.495, 0.554] |
| LR + trend features | 0.500 [0.473, 0.529] | 0.489 [0.441, 0.535] | 0.573 [0.539, 0.607] |
| LR on VCI(T) | 0.493 [0.469, 0.519] | 0.521 [0.479, 0.562] | 0.524 [0.495, 0.554] |
| Persistence (uses ground data) | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| Smoothed-VCI state model | 0.493 [0.469, 0.519] | 0.521 [0.479, 0.562] | 0.524 [0.495, 0.554] |

| Comparison | Δ all | Δ West | Δ East |
|---|---|---|---|
| Bi-LSTM − LR + trend features | +0.038 [+0.007, +0.069] | +0.062 [+0.013, +0.113] | −0.025 [−0.059, +0.009] |
| VCI-only Bi-LSTM − LR + trend features | +0.027 [+0.001, +0.052] | +0.034 [−0.006, +0.079] | −0.021 [−0.053, +0.012] |
| Climatology − LR + trend features | −0.000 [−0.029, +0.027] | +0.011 [−0.035, +0.059] | −0.073 [−0.107, −0.039] |
| Boosted trees (monotone in VCI) − LR + trend features | +0.025 [+0.013, +0.037] | +0.014 [−0.004, +0.032] | +0.025 [+0.008, +0.042] |
| GRU − LR + trend features | +0.024 [−0.007, +0.054] | +0.062 [+0.013, +0.114] | −0.036 [−0.070, −0.001] |
| LR on latest VCI + age − LR + trend features | −0.007 [−0.022, +0.008] | +0.032 [+0.008, +0.056] | −0.049 [−0.067, −0.030] |
| LR on VCI(T) − LR + trend features | −0.007 [−0.022, +0.008] | +0.032 [+0.008, +0.056] | −0.049 [−0.067, −0.030] |
| Persistence (uses ground data) − LR + trend features | −0.000 [−0.029, +0.027] | +0.011 [−0.035, +0.059] | −0.073 [−0.107, −0.039] |
| Smoothed-VCI state model − LR + trend features | −0.007 [−0.022, +0.008] | +0.032 [+0.008, +0.056] | −0.049 [−0.067, −0.030] |
| Bi-LSTM − LR on VCI(T) | +0.045 [+0.016, +0.076] | +0.030 [−0.016, +0.074] | +0.024 [−0.013, +0.063] |
| VCI-only Bi-LSTM − LR on VCI(T) | +0.034 [+0.012, +0.056] | +0.002 [−0.028, +0.035] | +0.028 [−0.004, +0.062] |
| Climatology − LR on VCI(T) | +0.007 [−0.019, +0.031] | −0.021 [−0.062, +0.021] | −0.024 [−0.054, +0.005] |
| Boosted trees (monotone in VCI) − LR on VCI(T) | +0.032 [+0.014, +0.050] | −0.019 [−0.043, +0.006] | +0.074 [+0.046, +0.101] |
| GRU − LR on VCI(T) | +0.031 [+0.001, +0.061] | +0.029 [−0.015, +0.071] | +0.013 [−0.024, +0.054] |
| LR on latest VCI + age − LR on VCI(T) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| LR + trend features − LR on VCI(T) | +0.007 [−0.008, +0.022] | −0.032 [−0.056, −0.008] | +0.049 [+0.030, +0.067] |
| Persistence (uses ground data) − LR on VCI(T) | +0.007 [−0.019, +0.031] | −0.021 [−0.062, +0.021] | −0.024 [−0.054, +0.005] |
| Smoothed-VCI state model − LR on VCI(T) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| VCI-only Bi-LSTM − Bi-LSTM | −0.011 [−0.035, +0.012] | −0.028 [−0.059, +0.007] | +0.004 [−0.030, +0.035] |

### Realistic sampling (fresh test, per-scene values)

Share of fresh-test cell-months (2025-01..2026-09, labelled; 75,306) with a fresh clear view: full HLS record (monthly composite) 86.5%; one pass every 7 days 47.9% (starts: 48.6%, 47.5%, 47.7%); one pass every 16 days 27.3% (starts: 27.4%, 27.8%, 26.8%); one pass every 32 days 15.2% (starts: 15.9%, 14.8%, 14.8%).

**job1, models trained on composites** (AUC per start; windows pooled over starts below):

| Set | n | Climatology | LR on VCI(T) | LR on latest VCI + age | LR + trend features | Boosted trees (monotone in VCI) | Smoothed-VCI state model | GRU | Residual Bi-LSTM | Bi-LSTM | VCI-only Bi-LSTM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| thin7_s0 | 36,633 | 0.500 [0.500, 0.500] | 0.627 [0.595, 0.659] | 0.627 [0.595, 0.659] | 0.623 [0.590, 0.656] | 0.622 [0.589, 0.656] | 0.631 [0.597, 0.664] | 0.648 [0.619, 0.677] | 0.648 [0.620, 0.677] | 0.640 [0.611, 0.670] | 0.611 [0.579, 0.642] |
| thin7_s1 | 35,745 | 0.500 [0.500, 0.500] | 0.627 [0.596, 0.658] | 0.627 [0.596, 0.658] | 0.624 [0.593, 0.656] | 0.624 [0.592, 0.656] | 0.633 [0.601, 0.665] | 0.652 [0.624, 0.679] | 0.654 [0.627, 0.681] | 0.646 [0.617, 0.674] | 0.628 [0.597, 0.658] |
| thin7_s2 | 35,899 | 0.500 [0.500, 0.500] | 0.620 [0.590, 0.651] | 0.620 [0.590, 0.651] | 0.612 [0.580, 0.645] | 0.616 [0.583, 0.648] | 0.623 [0.591, 0.656] | 0.645 [0.615, 0.675] | 0.646 [0.617, 0.676] | 0.640 [0.610, 0.670] | 0.624 [0.593, 0.655] |
| thin16_s0 | 20,668 | 0.500 [0.500, 0.500] | 0.649 [0.610, 0.689] | 0.649 [0.610, 0.689] | 0.641 [0.599, 0.683] | 0.633 [0.592, 0.674] | 0.651 [0.609, 0.693] | 0.654 [0.620, 0.688] | 0.662 [0.626, 0.696] | 0.644 [0.609, 0.681] | 0.605 [0.563, 0.645] |
| thin16_s1 | 20,946 | 0.500 [0.500, 0.500] | 0.632 [0.583, 0.679] | 0.632 [0.583, 0.679] | 0.630 [0.578, 0.680] | 0.628 [0.574, 0.681] | 0.635 [0.583, 0.686] | 0.653 [0.609, 0.697] | 0.654 [0.612, 0.695] | 0.653 [0.606, 0.699] | 0.627 [0.576, 0.677] |
| thin16_s2 | 20,158 | 0.500 [0.500, 0.500] | 0.638 [0.597, 0.676] | 0.638 [0.597, 0.676] | 0.635 [0.592, 0.678] | 0.631 [0.587, 0.673] | 0.643 [0.600, 0.683] | 0.661 [0.628, 0.696] | 0.664 [0.630, 0.698] | 0.647 [0.613, 0.682] | 0.614 [0.573, 0.654] |
| thin32_s0 | 11,977 | 0.500 [0.500, 0.500] | 0.640 [0.588, 0.691] | 0.640 [0.588, 0.691] | 0.637 [0.581, 0.690] | 0.626 [0.568, 0.682] | 0.644 [0.590, 0.697] | 0.632 [0.575, 0.688] | 0.644 [0.588, 0.697] | 0.638 [0.581, 0.694] | 0.571 [0.508, 0.634] |
| thin32_s1 | 11,170 | 0.500 [0.500, 0.500] | 0.629 [0.583, 0.671] | 0.629 [0.583, 0.671] | 0.627 [0.579, 0.672] | 0.616 [0.565, 0.663] | 0.630 [0.581, 0.674] | 0.649 [0.605, 0.688] | 0.642 [0.598, 0.682] | 0.648 [0.602, 0.692] | 0.600 [0.544, 0.647] |
| thin32_s2 | 11,133 | 0.500 [0.500, 0.500] | 0.653 [0.608, 0.691] | 0.653 [0.608, 0.691] | 0.644 [0.595, 0.684] | 0.629 [0.577, 0.673] | 0.648 [0.600, 0.688] | 0.639 [0.596, 0.678] | 0.650 [0.608, 0.686] | 0.621 [0.572, 0.666] | 0.563 [0.499, 0.619] |

**job1, models trained on development regions thinned the same way** (AUC per start; windows pooled over starts below):

| Set | n | Climatology | LR on VCI(T) | LR on latest VCI + age | LR + trend features | Boosted trees (monotone in VCI) | Smoothed-VCI state model | GRU |
|---|---|---|---|---|---|---|---|---|
| thin7_s0 | 36,633 | 0.500 [0.500, 0.500] | 0.627 [0.595, 0.660] | 0.627 [0.595, 0.660] | 0.622 [0.587, 0.657] | 0.630 [0.595, 0.664] | 0.631 [0.596, 0.664] | 0.646 [0.616, 0.675] |
| thin7_s1 | 35,745 | 0.500 [0.500, 0.500] | 0.627 [0.595, 0.657] | 0.627 [0.595, 0.657] | 0.625 [0.590, 0.658] | 0.633 [0.600, 0.664] | 0.633 [0.600, 0.665] | 0.647 [0.617, 0.675] |
| thin7_s2 | 35,899 | 0.500 [0.500, 0.500] | 0.620 [0.586, 0.652] | 0.620 [0.586, 0.652] | 0.615 [0.578, 0.651] | 0.624 [0.590, 0.659] | 0.623 [0.588, 0.657] | 0.642 [0.611, 0.673] |
| thin16_s0 | 20,668 | 0.500 [0.500, 0.500] | 0.649 [0.608, 0.686] | 0.649 [0.608, 0.686] | 0.648 [0.604, 0.690] | 0.649 [0.605, 0.691] | 0.651 [0.608, 0.691] | 0.665 [0.629, 0.701] |
| thin16_s1 | 20,946 | 0.500 [0.500, 0.500] | 0.632 [0.583, 0.681] | 0.632 [0.583, 0.681] | 0.636 [0.582, 0.687] | 0.640 [0.588, 0.690] | 0.635 [0.583, 0.688] | 0.651 [0.613, 0.684] |
| thin16_s2 | 20,158 | 0.500 [0.500, 0.500] | 0.638 [0.598, 0.677] | 0.638 [0.598, 0.677] | 0.643 [0.599, 0.686] | 0.649 [0.606, 0.690] | 0.643 [0.600, 0.683] | 0.668 [0.633, 0.700] |
| thin32_s0 | 11,977 | 0.500 [0.500, 0.500] | 0.640 [0.589, 0.689] | 0.640 [0.589, 0.689] | 0.643 [0.590, 0.693] | 0.642 [0.589, 0.691] | 0.644 [0.590, 0.696] | 0.645 [0.595, 0.691] |
| thin32_s1 | 11,170 | 0.500 [0.500, 0.500] | 0.629 [0.582, 0.673] | 0.629 [0.582, 0.673] | 0.630 [0.580, 0.675] | 0.632 [0.583, 0.676] | 0.630 [0.581, 0.675] | 0.643 [0.599, 0.684] |
| thin32_s2 | 11,133 | 0.500 [0.500, 0.500] | 0.653 [0.610, 0.691] | 0.653 [0.610, 0.691] | 0.653 [0.610, 0.690] | 0.652 [0.610, 0.691] | 0.648 [0.601, 0.688] | 0.654 [0.614, 0.691] |

**job2, models trained on composites** (AUC per start; windows pooled over starts below):

| Set | n | Climatology | LR on latest VCI + age | LR + trend features | Boosted trees (monotone in VCI) | Smoothed-VCI state model | GRU | Residual Bi-LSTM | Bi-LSTM | VCI-only Bi-LSTM |
|---|---|---|---|---|---|---|---|---|---|---|
| thin7_s0 | 5,652 | 0.500 [0.500, 0.500] | 0.634 [0.569, 0.690] | 0.635 [0.569, 0.689] | 0.645 [0.580, 0.700] | 0.644 [0.578, 0.699] | 0.615 [0.550, 0.674] | 0.627 [0.566, 0.681] | 0.640 [0.582, 0.691] | 0.640 [0.579, 0.691] |
| thin7_s1 | 5,508 | 0.500 [0.500, 0.500] | 0.698 [0.650, 0.745] | 0.691 [0.641, 0.737] | 0.703 [0.654, 0.749] | 0.706 [0.658, 0.752] | 0.656 [0.594, 0.713] | 0.673 [0.614, 0.729] | 0.661 [0.602, 0.716] | 0.687 [0.634, 0.737] |
| thin7_s2 | 5,529 | 0.500 [0.500, 0.500] | 0.669 [0.617, 0.715] | 0.668 [0.617, 0.714] | 0.672 [0.620, 0.718] | 0.672 [0.620, 0.721] | 0.645 [0.581, 0.702] | 0.659 [0.600, 0.715] | 0.651 [0.594, 0.703] | 0.662 [0.602, 0.715] |
| thin16_s0 | 1,635 | 0.500 [0.500, 0.500] | 0.620 [0.521, 0.716] | 0.610 [0.505, 0.710] | 0.620 [0.522, 0.715] | 0.617 [0.510, 0.720] | 0.651 [0.551, 0.739] | 0.652 [0.548, 0.750] | 0.633 [0.522, 0.735] | 0.632 [0.523, 0.727] |
| thin16_s1 | 1,814 | 0.500 [0.500, 0.500] | 0.631 [0.551, 0.706] | 0.634 [0.555, 0.708] | 0.646 [0.566, 0.721] | 0.631 [0.546, 0.710] | 0.584 [0.487, 0.683] | 0.557 [0.462, 0.652] | 0.576 [0.478, 0.670] | 0.602 [0.506, 0.689] |
| thin16_s2 | 1,354 | 0.500 [0.500, 0.500] | 0.638 [0.516, 0.737] | 0.636 [0.514, 0.740] | 0.645 [0.529, 0.741] | 0.640 [0.518, 0.742] | 0.573 [0.456, 0.676] | 0.601 [0.488, 0.708] | 0.582 [0.467, 0.685] | 0.597 [0.485, 0.697] |
| thin32_s0 | 554 | 0.500 [0.500, 0.500] | 0.672 [0.578, 0.790] | 0.664 [0.562, 0.779] | 0.683 [0.588, 0.816] | 0.679 [0.582, 0.797] | 0.700 [0.525, 0.837] | 0.712 [0.586, 0.822] | 0.706 [0.566, 0.835] | 0.679 [0.579, 0.814] |
| thin32_s1 | 524 | 0.500 [0.500, 0.500] | 0.615 [0.446, 0.771] | 0.598 [0.424, 0.771] | 0.636 [0.468, 0.808] | 0.622 [0.425, 0.791] | 0.604 [0.400, 0.731] | 0.694 [0.510, 0.825] | 0.631 [0.353, 0.803] | 0.521 [0.282, 0.699] |
| thin32_s2 | 447 | 0.500 [0.500, 0.500] | 0.687 [0.585, 0.845] | 0.681 [0.570, 0.815] | 0.686 [0.580, 0.827] | 0.697 [0.599, 0.861] | 0.605 [0.421, 0.772] | 0.645 [0.444, 0.802] | 0.652 [0.476, 0.826] | 0.721 [0.628, 0.869] |

**job2, models trained on development regions thinned the same way** (AUC per start; windows pooled over starts below):

| Set | n | Climatology | LR on latest VCI + age | LR + trend features | Boosted trees (monotone in VCI) | Smoothed-VCI state model | GRU |
|---|---|---|---|---|---|---|---|
| thin7_s0 | 5,652 | 0.500 [0.500, 0.500] | 0.628 [0.567, 0.682] | 0.652 [0.591, 0.704] | 0.656 [0.593, 0.709] | 0.637 [0.574, 0.691] | 0.622 [0.563, 0.670] |
| thin7_s1 | 5,508 | 0.500 [0.500, 0.500] | 0.690 [0.640, 0.738] | 0.701 [0.652, 0.748] | 0.709 [0.658, 0.756] | 0.699 [0.648, 0.746] | 0.629 [0.573, 0.681] |
| thin7_s2 | 5,529 | 0.500 [0.500, 0.500] | 0.666 [0.616, 0.712] | 0.670 [0.620, 0.717] | 0.678 [0.626, 0.724] | 0.670 [0.618, 0.718] | 0.630 [0.571, 0.681] |
| thin16_s0 | 1,635 | 0.500 [0.500, 0.500] | 0.619 [0.522, 0.717] | 0.608 [0.505, 0.715] | 0.627 [0.529, 0.722] | 0.616 [0.511, 0.722] | 0.632 [0.520, 0.730] |
| thin16_s1 | 1,814 | 0.500 [0.500, 0.500] | 0.632 [0.553, 0.707] | 0.636 [0.551, 0.715] | 0.656 [0.577, 0.733] | 0.631 [0.545, 0.710] | 0.577 [0.482, 0.678] |
| thin16_s2 | 1,354 | 0.500 [0.500, 0.500] | 0.642 [0.525, 0.738] | 0.641 [0.520, 0.740] | 0.645 [0.534, 0.736] | 0.644 [0.525, 0.740] | 0.645 [0.556, 0.731] |
| thin32_s0 | 554 | 0.500 [0.500, 0.500] | 0.661 [0.559, 0.774] | 0.717 [0.596, 0.824] | 0.722 [0.622, 0.836] | 0.668 [0.571, 0.783] | 0.688 [0.519, 0.826] |
| thin32_s1 | 524 | 0.500 [0.500, 0.500] | 0.604 [0.437, 0.750] | 0.658 [0.470, 0.805] | 0.653 [0.444, 0.831] | 0.612 [0.422, 0.773] | 0.667 [0.403, 0.861] |
| thin32_s2 | 447 | 0.500 [0.500, 0.500] | 0.692 [0.589, 0.859] | 0.710 [0.564, 0.868] | 0.687 [0.566, 0.897] | 0.703 [0.603, 0.876] | 0.593 [0.318, 0.789] |

## 8a. Fresh test: post-hoc diagnostics (after the lock; not a second test)

The same saved fresh-test predictions as section 8 (job 1), sliced to explain the result (`results/scores/fresh_posthoc.json`, `code/dewsat2/fresh_posthoc.py`). Nothing here was used to choose or change a model, and the fresh test was not scored again.

- **Validity:** the 13 saved Phase 5 networks, reloaded from `model.pt`, reproduce their scored fresh-test logits (largest difference 2.9e-06; the CPU-trained ones exactly). Parameter counts match the metadata: yes. Every training loss fell: yes.

**The networks against each other** (job 1 and job 4; paired AUC differences with the scorer's resamples):

| Comparison | Job 1, all | West | East | Job 4, all |
|---|---|---|---|---|
| Bi-LSTM − Residual Bi-LSTM | +0.004 [−0.001, +0.009] | −0.006 [−0.011, −0.001] | +0.008 [+0.001, +0.016] | +0.009 [+0.002, +0.015] |
| Bi-LSTM − GRU | +0.006 [+0.002, +0.011] | −0.003 [−0.008, +0.002] | +0.011 [+0.005, +0.017] | +0.003 [−0.003, +0.009] |
| Residual Bi-LSTM − GRU | +0.003 [−0.002, +0.007] | +0.003 [−0.002, +0.008] | +0.003 [−0.004, +0.008] | −0.006 [−0.011, −0.000] |
| Bi-LSTM − VCI-only Bi-LSTM | +0.019 [+0.006, +0.032] | +0.010 [−0.005, +0.025] | +0.022 [+0.001, +0.042] | +0.027 [+0.007, +0.045] |
| Residual Bi-LSTM − VCI-only Bi-LSTM | +0.015 [+0.002, +0.029] | +0.016 [+0.001, +0.031] | +0.013 [−0.011, +0.036] | +0.018 [−0.002, +0.036] |
| GRU − VCI-only Bi-LSTM | +0.012 [−0.000, +0.026] | +0.013 [−0.002, +0.028] | +0.011 [−0.011, +0.031] | +0.024 [+0.005, +0.041] |
| Bi-LSTM − CNN + LR | +0.026 [+0.010, +0.041] | +0.009 [−0.007, +0.027] | +0.031 [+0.005, +0.056] | +0.046 [+0.023, +0.069] |
| Bi-LSTM − CNN + Bi-LSTM | +0.027 [+0.013, +0.041] | +0.015 [−0.006, +0.035] | +0.025 [+0.007, +0.041] | +0.044 [+0.027, +0.062] |
| VCI-only Bi-LSTM − CNN + Bi-LSTM | +0.008 [−0.000, +0.016] | +0.005 [−0.008, +0.019] | +0.003 [−0.008, +0.015] | +0.017 [+0.006, +0.028] |

The three full-input recurrent networks (Bi-LSTM, residual Bi-LSTM, GRU) are within 0.006 AUC of each other; each is about 0.04 above LR.

**By year and side** (pooled AUC; point estimates):

| Year | Side | Windows | Drought rate | LR on VCI(T) | Bi-LSTM | Residual Bi-LSTM | GRU | VCI-only Bi-LSTM | CNN + Bi-LSTM | Bi-LSTM − LR |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025 | all | 37,749 | 0.342 | 0.692 | 0.730 | 0.732 | 0.731 | 0.704 | 0.710 | +0.037 |
| 2026 (Jan-Sep) | all | 27,421 | 0.467 | 0.558 | 0.606 | 0.598 | 0.593 | 0.589 | 0.575 | +0.047 |
| 2025 | west | 18,658 | 0.373 | 0.747 | 0.770 | 0.778 | 0.775 | 0.754 | 0.752 | +0.024 |
| 2026 (Jan-Sep) | west | 13,699 | 0.310 | 0.671 | 0.663 | 0.668 | 0.663 | 0.663 | 0.653 | -0.008 |
| 2025 | east | 19,091 | 0.312 | 0.629 | 0.677 | 0.675 | 0.674 | 0.636 | 0.657 | +0.048 |
| 2026 (Jan-Sep) | east | 13,722 | 0.624 | 0.632 | 0.642 | 0.637 | 0.630 | 0.632 | 0.631 | +0.011 |

**Over time and across places.** Over time: AUC from pairs of windows in the same region, so only ranking months within a region counts. Across places: pairs in the same month, so only ranking cells within a month counts. Network − LR, with whole-region intervals (1,000 resamples) for the over-time slice:

| Slice | Side | LR | Boosted trees (monotone in VCI) | Bi-LSTM | Residual Bi-LSTM | GRU | VCI-only Bi-LSTM | CNN + LR | CNN + Bi-LSTM |
|---|---|---|---|---|---|---|---|---|---|
| over time | all | 0.640 | 0.643 (+0.003 [−0.003, +0.009]) | 0.671 (+0.031 [+0.016, +0.045]) | 0.668 (+0.028 [+0.012, +0.043]) | 0.665 (+0.025 [+0.010, +0.039]) | 0.643 (+0.003 [−0.006, +0.012]) | 0.639 (−0.000 [−0.004, +0.004]) | 0.651 (+0.011 [+0.002, +0.020]) |
| over time | west | 0.691 | 0.679 (−0.013 [−0.020, −0.006]) | 0.714 (+0.023 [+0.006, +0.041]) | 0.721 (+0.030 [+0.012, +0.049]) | 0.712 (+0.021 [+0.002, +0.039]) | 0.698 (+0.007 [−0.008, +0.022]) | 0.692 (+0.000 [−0.005, +0.007]) | 0.704 (+0.013 [+0.001, +0.027]) |
| over time | east | 0.605 | 0.619 (+0.014 [+0.007, +0.021]) | 0.642 (+0.037 [+0.017, +0.056]) | 0.632 (+0.027 [+0.005, +0.049]) | 0.633 (+0.028 [+0.007, +0.047]) | 0.606 (+0.001 [−0.011, +0.013]) | 0.604 (−0.001 [−0.006, +0.004]) | 0.615 (+0.010 [−0.004, +0.023]) |
| across places | all | 0.628 | 0.628 (+0.001) | 0.675 (+0.048) | 0.670 (+0.043) | 0.668 (+0.040) | 0.652 (+0.025) | 0.643 (+0.016) | 0.646 (+0.019) |
| across places | west | 0.708 | 0.707 (-0.001) | 0.717 (+0.009) | 0.722 (+0.014) | 0.720 (+0.011) | 0.705 (-0.003) | 0.713 (+0.005) | 0.699 (-0.009) |
| across places | east | 0.587 | 0.596 (+0.008) | 0.619 (+0.032) | 0.612 (+0.025) | 0.608 (+0.021) | 0.604 (+0.017) | 0.598 (+0.011) | 0.595 (+0.007) |

The networks' gain shows in both slices and both years, so it is not one artifact. The VCI-only Bi-LSTM gains nothing over time, so on the fresh test the extra skill comes from the NDVI level and spread, clear fraction and season inputs, not from the VCI history alone.

**Why development said the opposite.** Each development fold's training-set size, its validation drought rate and the job-1 AUCs, beside Phase 5 (trained on all development data) on the fresh test:

| Data | Validation | Training windows | Drought rate | LR on VCI(T) | Bi-LSTM | Residual Bi-LSTM | GRU | VCI-only Bi-LSTM | CNN + LR | CNN + Bi-LSTM | Bi-LSTM − LR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fold 3 | 2020-2021 | 323,298 | 0.306 | 0.733 | 0.619 | 0.637 | 0.641 | 0.680 | 0.709 | 0.678 | -0.114 |
| fold 2 | 2018-2019 | 350,186 | 0.125 | 0.717 | 0.666 | 0.659 | 0.666 | 0.696 | 0.720 | 0.688 | -0.051 |
| fold 4 | 2022-2023 | 480,333 | 0.297 | 0.654 | 0.597 | 0.615 | 0.665 | 0.644 | 0.653 | 0.648 | -0.058 |
| fold 1 | 2016-2017 | 508,428 | 0.192 | 0.641 | 0.652 | 0.648 | 0.642 | 0.619 | 0.644 | 0.622 | +0.011 |
| Phase 5 | fresh test 2025-01..2026-09 | 1,118,986 | 0.395 | 0.621 | 0.663 | 0.659 | 0.657 | 0.644 | 0.638 | 0.636 | +0.042 |

The Bi-LSTM's gap to LR follows the training-set size, not the dryness: the driest validation block (2020-21) was its worst, and the fresh test, drier still, was its best, with 2.2 times the largest fold's training windows. With the learning curve (section 7b: more years help, more regions do not), the best-supported reading is that the networks need more training data than a development fold gave them. Caveats: five points; training size is confounded with which years and regions each fold trained on; the fresh test is one unusually dry 21-month period.

## 8b. Model sizes for the Jetson

One prediction is one cell-month window (12 monthly steps; a tile CNN reads one 23x21 tile per step). Multiply-adds are counted layer by layer (recurrent gates, convolutions, linear layers; norms and activations left out). Weights: `model.pt` (PyTorch state dict, float32) of the seed-42 run; the 10 seeds are the same size. A float32 model needs about 4 bytes per parameter.

| Job | Model | Setting | Parameters | Multiply-adds per prediction | model.pt | Folder (10 seeds) |
|---|---|---|---|---|---|---|
| job1 | Bi-LSTM | cfg05: hidden 24, layers 1 | 11,617 | 80,688 | 49.5 KiB | `runs/phase5/composite/job1/bilstm/` |
| job1 | VCI-only Bi-LSTM | cfg05: hidden 24, layers 1 | 11,617 | 69,168 | 49.5 KiB | `runs/phase5/composite/job1/bilstm_vci/` |
| job1 | CNN + Bi-LSTM | cfg01: hidden 16, layers 1, width 16, depth 3 | 23,217 | 13,030,048 | 95.9 KiB | `runs/phase5/composite/job1/cnn_bilstm/` |
| job1 | CNN + LR | cfg09: width 32, depth 3 | 56,098 | 4,038,401 | 222.1 KiB | `runs/phase5/composite/job1/cnn_lr/` |
| job1 | GRU | cfg16: hidden 32, layers 1 | 16,833 | 102,720 | 69.8 KiB | `runs/phase5/composite/job1/gru/` |
| job1 | Residual Bi-LSTM | cfg00: hidden 24, layers 1 | 11,617 | 80,688 | 49.5 KiB | `runs/phase5/composite/job1/resbilstm/` |
| job2 | Bi-LSTM | cfg17: hidden 16, layers 2 | 12,097 | 114,208 | 52.9 KiB | `runs/phase5/composite/job2/bilstm/` |
| job2 | VCI-only Bi-LSTM | cfg17: hidden 16, layers 2 | 12,097 | 106,528 | 52.9 KiB | `runs/phase5/composite/job2/bilstm_vci/` |
| job2 | GRU | cfg09: hidden 16, layers 1 | 4,833 | 30,880 | 23.0 KiB | `runs/phase5/composite/job2/gru/` |
| job2 | Residual Bi-LSTM | cfg17: hidden 32, layers 2 | 44,673 | 429,120 | 180.2 KiB | `runs/phase5/composite/job2/resbilstm/` |
| job3 | Bi-LSTM | cfg04: hidden 24, layers 1 | 11,715 | 80,784 | 49.8 KiB | `runs/phase5/composite/job3/bilstm/` |
| job3 | VCI-only Bi-LSTM | cfg21: hidden 96, layers 1 | 157,443 | 995,904 | 619.1 KiB | `runs/phase5/composite/job3/bilstm_vci/` |
| job3 | GRU | cfg09: hidden 64, layers 1 | 31,555 | 184,960 | 126.5 KiB | `runs/phase5/composite/job3/gru/` |

## 9. Extras (Phase 6)

### Label check: NLDAS-2 Noah 10-40 cm vs FLDAS

Development cells 2016-2023: 1,338,912 cell-months; Spearman correlation of percentiles 0.640; drought flags agree 80.8% of the time, Cohen's kappa 0.458; AUC of the NLDAS percentile for the FLDAS flag 0.824; drought rate FLDAS 0.221, NLDAS 0.238.
Fresh-test cells 2025-01..2026-09: 75,096 cell-months; Spearman correlation of percentiles 0.510; drought flags agree 70.8% of the time, Cohen's kappa 0.364; AUC of the NLDAS percentile for the FLDAS flag 0.740; drought rate FLDAS 0.394, NLDAS 0.310.

### Benchmark: gridMET SPEI

AUC of -SPEI for the FLDAS drought flag, development cells 2016-2023 of 70 sampled development regions (1,735 cells): 180d 0.804, 90d 0.796, 270d 0.796, 1y 0.776, 30d 0.705, 2y 0.628, 14d 0.611, 5y 0.573; chosen: 180d.
job1 fresh test (65,170 windows; SPEI-180d drought rate 0.354, FLDAS 0.395, the two calls agree on 66.0%): climatology: AUC vs SPEI drought 0.500, vs FLDAS 0.500; lr_vci_t: AUC vs SPEI drought 0.688, vs FLDAS 0.621; lr_latest_age: AUC vs SPEI drought 0.688, vs FLDAS 0.621; lr_trend: AUC vs SPEI drought 0.676, vs FLDAS 0.613; gbt_mono: AUC vs SPEI drought 0.686, vs FLDAS 0.621; state_ewma: AUC vs SPEI drought 0.690, vs FLDAS 0.620; gru: AUC vs SPEI drought 0.652, vs FLDAS 0.657; resbilstm: AUC vs SPEI drought 0.658, vs FLDAS 0.659; bilstm: AUC vs SPEI drought 0.656, vs FLDAS 0.663; bilstm_vci: AUC vs SPEI drought 0.671, vs FLDAS 0.644; cnn_lr: AUC vs SPEI drought 0.680, vs FLDAS 0.638; cnn_bilstm: AUC vs SPEI drought 0.667, vs FLDAS 0.636.
job2 fresh test (7,894 windows; SPEI-180d drought rate 0.529, FLDAS 0.436, the two calls agree on 62.9%): climatology: AUC vs SPEI drought 0.500, vs FLDAS 0.500; lr_latest_age: AUC vs SPEI drought 0.603, vs FLDAS 0.602; lr_trend: AUC vs SPEI drought 0.600, vs FLDAS 0.589; gbt_mono: AUC vs SPEI drought 0.610, vs FLDAS 0.603; state_ewma: AUC vs SPEI drought 0.608, vs FLDAS 0.593; gru: AUC vs SPEI drought 0.582, vs FLDAS 0.605; resbilstm: AUC vs SPEI drought 0.586, vs FLDAS 0.616; bilstm: AUC vs SPEI drought 0.598, vs FLDAS 0.618; bilstm_vci: AUC vs SPEI drought 0.592, vs FLDAS 0.605.

## 9a. Fresh-test predictions against other drought labels (post-hoc, after Phase 6)

Every model learned the FLDAS label. To see whether their fresh-test ranking holds for drought defined another way, the same saved predictions are scored against SPEI drought (SPEI <= -0.84) at the timescale Phase 6 chose on development cells and at the two runners-up within 0.01 AUC of it, and against the NLDAS-2 drought flag (the same 20th-percentile rule on another land model's 10-40 cm soil moisture). All labels use the windows where every label exists; paired differences use 2,000 whole-region resamples (`results/scores/fresh_labels_posthoc.json`). Post-hoc: nothing was chosen or changed on these results.

**Job 1, estimation now** (64,995 of 65,170 windows have every label):

| Label | Drought rate | Agrees with FLDAS | LR on VCI(T) AUC | Residual Bi-LSTM − LR | GRU − LR | Bi-LSTM − LR | VCI-only Bi-LSTM − LR | CNN + LR − LR | CNN + Bi-LSTM − LR |
|---|---|---|---|---|---|---|---|---|---|
| FLDAS (training label) | 0.395 | 100.0% | 0.621 [0.590, 0.652] | +0.038 [+0.020, +0.058] | +0.036 [+0.019, +0.054] | +0.042 [+0.024, +0.061] | +0.023 [+0.013, +0.033] | +0.016 [+0.008, +0.024] | +0.015 [+0.004, +0.026] |
| SPEI-180d (Phase 6's choice) | 0.355 | 66.0% | 0.688 [0.660, 0.714] | −0.030 [−0.047, −0.012] | −0.036 [−0.053, −0.017] | −0.032 [−0.049, −0.013] | −0.017 [−0.028, −0.006] | −0.008 [−0.015, −0.001] | −0.020 [−0.032, −0.010] |
| SPEI-90d (runner-up) | 0.345 | 66.1% | 0.629 [0.604, 0.654] | −0.003 [−0.020, +0.014] | −0.002 [−0.019, +0.016] | +0.002 [−0.016, +0.021] | −0.008 [−0.018, +0.002] | +0.003 [−0.004, +0.009] | −0.005 [−0.016, +0.005] |
| SPEI-270d (runner-up) | 0.381 | 64.8% | 0.683 [0.651, 0.713] | −0.021 [−0.038, −0.003] | −0.025 [−0.042, −0.008] | −0.024 [−0.042, −0.005] | −0.000 [−0.012, +0.013] | −0.009 [−0.017, −0.002] | −0.008 [−0.019, +0.003] |
| NLDAS-2 soil moisture | 0.314 | 71.3% | 0.614 [0.584, 0.641] | +0.027 [+0.013, +0.041] | +0.023 [+0.010, +0.037] | +0.031 [+0.017, +0.044] | +0.014 [+0.005, +0.022] | +0.017 [+0.011, +0.024] | +0.011 [+0.002, +0.020] |

**Job 2, missing month** (7,878 of 7,894 windows have every label):

| Label | Drought rate | Agrees with FLDAS | LR on latest VCI + age AUC | Residual Bi-LSTM − LR | GRU − LR | Bi-LSTM − LR | VCI-only Bi-LSTM − LR |
|---|---|---|---|---|---|---|---|
| FLDAS (training label) | 0.435 | 100.0% | 0.602 [0.536, 0.667] | +0.014 [−0.024, +0.050] | +0.003 [−0.038, +0.042] | +0.017 [−0.027, +0.057] | +0.003 [−0.032, +0.037] |
| SPEI-180d (Phase 6's choice) | 0.529 | 62.9% | 0.603 [0.544, 0.656] | −0.017 [−0.058, +0.020] | −0.020 [−0.062, +0.020] | −0.004 [−0.047, +0.037] | −0.010 [−0.040, +0.016] |
| SPEI-90d (runner-up) | 0.362 | 52.2% | 0.535 [0.484, 0.585] | −0.023 [−0.057, +0.008] | −0.008 [−0.044, +0.027] | −0.049 [−0.081, −0.017] | −0.026 [−0.057, +0.004] |
| SPEI-270d (runner-up) | 0.406 | 56.4% | 0.578 [0.510, 0.640] | −0.053 [−0.097, −0.011] | −0.013 [−0.057, +0.030] | −0.057 [−0.107, −0.009] | −0.017 [−0.048, +0.012] |
| NLDAS-2 soil moisture | 0.296 | 65.8% | 0.587 [0.534, 0.636] | +0.018 [−0.018, +0.055] | +0.013 [−0.028, +0.053] | −0.001 [−0.041, +0.039] | −0.026 [−0.052, +0.001] |

## 10. Problems and limitations

- **One test period.** The fresh test covers one 21-month period (2025-01 to 2026-09), and it was unusually dry: 39.5% of job-1 windows in drought, against 22% in development. Its intervals come from resampling regions, so they do not show how much another period would differ.
- **Development and the fresh test disagree on job 1.** The training-size explanation (section 8a) rests on five points, and training size is confounded with which years and regions each fold trained on.
- **The label is a model, not ground truth.** It is FLDAS Noah 10-40 cm soil moisture, not ground observations. FLDAS and NLDAS-2 agree on 81% of drought flags in development and 71% in the fresh period. The networks' gain holds against NLDAS-2 but reverses against climatic drought (SPEI over 180 and 270 days; section 9a), so the result is about soil-moisture drought only.
- **Some rows were added after development.** The plain Bi-LSTM, the VCI-only Bi-LSTM and the two CNN models were added after the development results (DESIGN.md amendments 4-6) and are reported in their own rows. Only the GRU and the residual Bi-LSTM were pre-registered for the fresh test; both passed.
- **Realistic sampling shrinks the gain.** With single clear views (one pass every 7, 16 or 32 days), the networks' gain over LR shrinks:
  - about +0.02 AUC with a pass every 7 days for the residual Bi-LSTM and the GRU (intervals above zero for all three start dates);
  - mostly inside the intervals at 16 days;
  - none at 32 days.
  The plain Bi-LSTM is less robust: its intervals include zero even at 7 days.
- **The camera is not modelled.** The inputs are HLS (Landsat 8/9 and Sentinel-2) Red/NIR composites standing in for DEWSat's camera; its resolution, noise and calibration are not modelled.
- **The SPEI timescale was chosen on a sample.** Each point request took about 22 s, so the timescale was chosen on a seeded sample of 70 development regions. The top three timescales were within 0.008 AUC; LR leads the networks at two of them, and they tie at the third. SPEI is read at region centres, which is coarser than the cells.
- **Transfer beyond CONUS is untested.** Every fresh-test region is at least 0.5 degrees from every development region, but all lie in the same CONUS climates.
- **Jobs 2 and 3 do not support a network.** The missing-month job shows no detectable difference overall (West better, East worse). Onset forecasting is near chance for every model (AUC at most 0.54).
- **Memory and reproducibility.** Training ran on one 18 GB Mac. Side jobs were paused to keep memory from swap thrashing, which changed only timing. GPU-trained runs reproduce to float tolerance (about 1e-6); CPU runs reproduce exactly.
- **Phase 6 needed a retry.** It first failed on two server changes: the NLDAS-2 cloud OPeNDAP server's bracket encoding, and the SPEI catalog's new path, variable name and packing. Both were fixed and the retry succeeded (DECISIONS.md #77).

## 11. Sources of the numbers

`results/scores/scores.json` (development), `results/scores/checks.json`, `results/scores/whatif.json`, `results/scores/weekly.json`, `results/scores/posthoc.json` (post-hoc models, development), `results/scores/lcurve.json` (learning curve), `results/scores/phase5.json` and `results/scores/phase5_settings.json` (fresh test), `results/scores/fresh_posthoc.json` (post-hoc slices of the fresh test), `results/scores/fresh_labels_posthoc.json` (fresh test against other labels), `results/scores/phase6.json`, `audit/phase0_results.json`, `data/plan/plan.json`, `data/pull_state.json`, `data/FROZEN.json`, `cache/dataset.json`, `logs/pull/fldas_check.json`, `results/scores/data_counts.json`; per-run metadata under `runs/` (configuration, seed, device, commit, timings).

