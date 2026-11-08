# DEWSat_BiLSTM_v2: design, fixed before any training

Every rule here was committed to git before any model was trained in this folder. `gate.json` is the
machine-readable copy of section 9 and is what the scoring code reads. Departures found necessary later are added
as dated amendments at the end, never edited in place.

## 1. Data

- **Regions.** Development = the paper run's 102 regions (`kansas_ref` is 10 x 10 cells, the rest 5 x 5) plus every
  new development region in the frozen dataset (`data/FROZEN.json`). Fresh test = the new fresh-test regions in the
  frozen dataset. Roles, spacing and batches: `DECISIONS.md` #13-#24 and `data/plan/`.
- **Development data**: development regions, months 2016-01..2023-12 only. Nothing from 2024 onward enters
  development (inputs or labels), so 2024 is a full-year buffer before the fresh test.
- **Fresh test**: fresh-test regions, target months 2025-01..2026-09 (inputs reach back to 2024-02). Locked (section 11).
- **Inputs** per cell-month, from the per-pixel monthly composites (paper rules, unchanged): NDVI_mean, NDVI_std,
  VCI_mean, VCI_p10 over usable pixels, valid_frac; month sin/cos; observed (usable pixels >= 5% of the cell); and,
  new, **months since the last observation** (0 when observed; counts up through unobserved months; months before
  the window's first observation count from the window start, capped at 12). Composites for the 102 are the paper
  run's (hashed copy); for new regions they come from this pull, which reproduces the paper's composites bit for
  bit on probe months (`DECISIONS.md` #18).
- **VCI table per fold, from training years only.** Per pixel and calendar month, NDVI min/max over the fold's
  reference months = development months outside [block start - 12, block end + 12] (section 2). Reference months
  get leave-one-year-out min/max (their own year excluded), as in the paper's primary table; other months use all
  reference months. Valid with >= 3 years and range >= 0.05. VCI = 100 (NDVI - min)/(max - min), clipped to 0-100.
  Inner-split models use their outer fold's table: it contains no outer-validation year, and with
  leave-one-year-out no month's own NDVI enters its own VCI. The Phase 5 table uses all development months
  2016-01..2023-12. Fresh-test regions get their table from their own pixels in development years only.
- **Labels**: FLDAS_NOAH01_C_GL_M v001 SoilMoi10_40cm_tavg; drought = value below the numpy-linear 20th percentile
  of the cell's 2001-2022 values for that calendar month (the paper's rule, unchanged; needs 20 reference years).
  Labels are never model inputs; the threshold uses FLDAS only.
- **Scalers**: z-scores of NDVI_mean, NDVI_std, VCI_mean, VCI_p10 over observed cell-months inside each model's own
  training windows only (inner-train for search runs, outer-train for final runs, all development for Phase 5).

## 2. Outer folds

- **Region groups**: development regions split into 4 groups balanced East/West of 98 W: within each side,
  regions sorted by name, shuffled with `numpy.random.default_rng(20261108)`, dealt round-robin; West is dealt from
  group 1, East continues where West stopped.
- **Year blocks** (target month T): B1 2016-01..2017-12, B2 2018-01..2019-12, B3 2020-01..2021-12,
  B4 2022-01..2023-12. Windows are 12 months T-11..T; months before 2016-01 are unobserved, so T may start at 2016-01.
- **Outer fold k** validates group k in block k: every group and every block is held out once.
- **Training** for fold k: the other three groups, T outside [start(B_k) - 12, end(B_k) + 12] for nowcast jobs; for
  the forecast job the origin T outside [start(B_k) - 15, end(B_k) + 15] and every target month <= 2023-12.

| Fold | Validation T | Nowcast training T | Forecast training origins |
|---|---|---|---|
| 1 | 2016-01..2017-12 | 2019-01..2023-12 (60) | 2019-04..2023-09 (54) |
| 2 | 2018-01..2019-12 | 2016-01..2016-12 + 2021-01..2023-12 (48) | 2016-01..2016-09 + 2021-04..2023-09 (39) |
| 3 | 2020-01..2021-12 | 2016-01..2018-12 + 2023-01..2023-12 (48) | 2016-01..2018-09 + 2023-04..2023-09 (39) |
| 4 | 2022-01..2023-12 | 2016-01..2020-12 (60) | 2016-01..2020-09 (57) |

## 3. Inner validation (model selection only)

- For outer fold k, the inner validation holds out a region group and a year: group k+1 (4 wraps to 1) and one
  calendar year Y_k of fold k's training months. Inner training = the remaining two groups, fold k's training months
  outside [Y_k start - 12, Y_k end + 12] (15 for the forecast job). Y_k is the year that leaves the most inner-training
  months (ties: the later year); 2016 is never used because its windows have truncated history.

| Fold | Nowcast: inner year, inner-val T, inner-train T | Forecast: inner year, inner-val origins, inner-train origins |
|---|---|---|
| 1 | 2023; 2023-01..12; 2019-01..2021-12 (36) | 2023; 2023-01..09; 2019-04..2021-09 (30) |
| 2 | 2023; 2023-01..12; 2016-01..12 + 2021-01..12 (24) | 2023; 2023-01..09; 2016-01..09 + 2021-04..09 (15) |
| 3 | 2023; 2023-01..12; 2016-01..2018-12 (36) | 2023; 2023-04..09; 2016-01..2018-09 (33) |
| 4 | 2020; 2020-01..12; 2016-01..2018-12 (36) | 2020; 2020-01..09; 2016-01..2018-09 (33) |

- The inner split decides every tunable setting (network configurations and epochs, the boosted trees'
  configuration, the state model's smoothing). Nothing is decided on an outer fold except the section 9 choice of
  the best network family and the best simple model, which the gate defines and applies to both sides alike.

## 4. Jobs (one dedicated model per job)

Scored sets for fold k are group k with T in block k, labelled.
1. **Estimation now**: T observed. Trained on training windows with T observed. Masking augmentation, when on,
   hides months T-11..T-1 only.
2. **Missing month**: T unobserved and >= 8 of the other 11 months observed. Trained on training windows with >= 8 of
   T-11..T-1 observed, month T always hidden from the input (real job-2 windows plus T-observed windows with T
   masked), so every job-2 model, LR included, learns from the same windows. Masking augmentation, when on, hides
   further months among T-11..T-1.
3. **Forecast** T+1, T+2, T+3: origin T observed; one model with three heads (baselines fit per lead). Scored per
   lead on all windows and on onsets only (not in drought at T, FLDAS). Persistence (FLDAS flag at T) is a reference
   row labelled "uses ground data"; it is never a candidate. Job-3 metric = mean of the three leads' onset
   within-fold AUCs.
4. **Fast change**: job-1 windows (job-1 models) in the top quartile of |VCI(T) - VCI(T-1)| (T-1 observed) or of
   |VCI(T) - VCI(T-3)| (T-3 observed), quartiles over all pooled development job-1 windows. Primary stratum: either.
   Also reported: top 1-month change, top 3-month change, neither.

## 5. Models

Simple (fit on the same training windows as the networks of the same job):
- `climatology` (training drought rate), `lr_vci_t` (LR on VCI_mean(T); jobs 1, 3, 4), `lr_latest_age` (LR on the
  latest observed VCI_mean and the months since it), `lr_trend` (latest VCI, age, 1- and 3-month changes with
  availability flags, months below 35 among the last 6 observed, months observed of the last 6, season),
  `gbt_mono` (scikit-learn HistGradientBoosting on the trend inputs, monotone decreasing in the latest VCI),
  `state_ewma` (exponentially smoothed VCI + age, LR; smoothing from {0.2, 0.35, 0.5, 0.65, 0.8, 1.0} on inner
  validation). LR: standardized inputs, C = 1e6 (the paper's). "Months since the last observation" enters every
  model; for `lr_vci_t` it is 0 whenever the model applies (T observed), so it is constant there.
Temporal networks (inputs: the 9 monthly channels above with the months-since-last-observation channel; logits out):
- `bilstm`: the lab's DroughtBiLSTM layout (final states of both directions + masked mean of the outputs ->
  LayerNorm -> Linear -> GELU -> Dropout -> Linear; MLP width 2 x hidden, the paper's pairing).
- `resbilstm`: the same, plus the job's LR logit (jobs 1/3/4: `lr_vci_t` per head; job 2: `lr_latest_age`); the last
  layer starts at zero, so the model starts exactly at the LR.
- `gru`: same head; GRU encoder, one or two directions (searched).
- `lstm`: one-direction LSTM, same head.
- `tcn`: small temporal CNN: causal dilated convolutions (kernel 3, dilations 1, 2, 4 per block, residual, GELU,
  dropout), 1 or 2 blocks; last step + masked mean -> the same head.
Training: binary cross-entropy (masked per head), AdamW, cosine decay to 5% over the configuration's epochs, gradient
clip 1.0, batch 512, no early stopping (the epoch count is a searched setting). Ten seeds of the chosen setting; the
prediction is their mean probability.

## 6. Search (per job, per network family or tree model, per outer fold)

- 30 configurations per network family, drawn once per (job, family) with a fixed seed and evaluated on every outer
  fold's inner split: hidden size from {16, 24, 32, 48, 64, 96, 128}; layers (RNN layers, TCN blocks) from {1, 2};
  learning rate log-uniform 2e-4..3e-3; weight decay log-uniform 1e-6..1e-1; dropout uniform 0..0.5; masking
  augmentation off or on (each eligible month hidden with probability uniform 0.1..0.4); epochs from
  {6, 10, 15, 20, 30}; GRU directions from {1, 2}. One seed per configuration.
- `gbt_mono`: 30 configurations, the same way (learning rate log-uniform 0.02..0.3, max_iter from
  {50, 100, 200, 400, 800}, max_leaf_nodes from {7, 15, 31, 63}, min_samples_leaf from {20, 100, 400, 1600},
  l2_regularization log-uniform 1e-3..10).
- Selection: highest inner-validation AUC (job 3: mean onset AUC over leads 1-3). The chosen setting is refit on
  fold k's full training set (networks: 10 seeds, the chosen epochs) and scored on fold k's validation set.

## 7. Scoring

- Primary metric: **within-fold AUC**: only (drought, no-drought) pairs from the same outer fold count, pooled over
  the four folds. 95% percentile intervals from 2,000 resamples of whole regions (seed 20261108); paired
  differences use the same resamples for both models. Reported for all regions and West/East of 98 W (regions
  resampled within the side).
- Secondary: pooled AUC, Brier score, reliability (10 equal-width bins) and ECE, per fold AUC.

## 8. Checks on the new models

- Order test: each family's final job-1 model rescored with months T-11..T-1 shuffled within each window (5
  permutations); the drop in within-fold AUC with a paired interval.
- Positive control: synthetic label "VCI_mean(T-3) - VCI_mean(T) > 20" (both observed) on outer fold 1; LR on
  VCI(T) and each family with its fold-1 job-1 setting, trained with the same code. Each network should clearly win.

## 9. Gate (machine-readable copy: gate.json)

For each job (1, 2, 3 on onsets, 4):
- The **best temporal network** = the family with the highest development within-fold AUC among bilstm, resbilstm,
  gru, lstm, tcn (job 3: the mean onset metric).
- The **best simple model** = the model with the highest development within-fold AUC among climatology, lr_vci_t
  (not job 2), lr_latest_age, lr_trend, gbt_mono, state_ewma. Persistence is never a candidate.
- **Pass**: the paired within-fold AUC difference (network minus simple) has its 95% interval entirely above zero
  over all regions, and neither the West nor the East paired interval is entirely below zero.
- The **best Bi-LSTM variant** (bilstm or resbilstm, same rule) is compared with the best simple model separately,
  under the same pass rule, and reported beside the gate row. For job 1 both are also compared with `lr_vci_t`, the
  question this work was asked to answer.
- Wording: "passed" only under the pass rule. Otherwise: interval includes zero -> "no detectable difference";
  entirely below zero -> "worse"; above zero but one side entirely below zero -> "did not pass (better overall, worse
  in the West/East)".

## 10. Secondary experiments (no pass/fail)

- **NDMI what-if (not DEWSat's camera)**, job 1, on regions with SWIR: LR on VCI(T) vs LR on VCI(T) + NDMI(T)
  (cell mean of the per-pixel NDMI composite) + NDMI condition index(T) (per-pixel, built like VCI); the best job-1
  family with its per-fold settings with and without those two channels (3 seeds each), same windows.
- **VCI per cell vs per pixel**, job 1: VCI from the cell-mean NDVI with a per-cell table (same reference months,
  leave-one-year-out, >= 3 years, range >= 0.05) against the per-pixel VCI; LR on VCI(T) and the best job-1 family
  (3 seeds), same windows.
- **Weekly inputs** (amendment 3 of 2026-11-07), on regions with per-scene values, after jobs 1-4: 52 weekly steps
  ending on the label month's last day; each week holds the latest clear view (>= 30% of the cell clear) up to the
  week's end: its per-cell VCI (table from single views in the fold's reference months, leave-one-year-out) and
  NDVI, days since that view, and a new-view flag. Target: the month's FLDAS label. LR on the latest VCI + its age vs
  the best job-1 family (its fold-chosen setting, adapted to 52 steps; 3 seeds), same folds and metric, on
  cell-months with at least one clear view in the 52 weeks. It does not change the gate.

## 11. Fresh test (Phase 5, once)

- Lock: the fresh-test regions' labels live only in `data/locked/` (hashes in `LOCK.json`); no code reads them
  before Phase 5. The Phase 5 scorer verifies the hashes, refuses to run if `data/locked/FRESH_TEST_SCORED.lock`
  exists, and writes it when it has scored.
- Refit: for each job and model, the "development-chosen setting" is the configuration with the highest mean
  inner-validation AUC over the four outer folds (job 3: mean onset metric). It is trained on all development
  data (networks: 10 seeds) with the Phase 5 VCI table and scored once on the fresh test. Models scored per job:
  the gate's best network and best simple model, the best Bi-LSTM variant, and the job's LR (`lr_vci_t`; job 2
  `lr_latest_age`). Persistence for job 3 as the ground-data reference. AUC on the fresh test (one region set, one
  period, so within-fold = plain AUC) with 2,000 region resamples, paired differences, West/East.
- **Realistic sampling** on the fresh test from the per-scene values: one pass every 7, 16 and 32 days with 3 random
  start dates each (per region, seeded); a pass sees a cell only if a view of that cell that day has >= 30% of the
  cell clear. A cell-month is observed when at least one pass saw it; its features come from the latest such view
  that month: NDVI mean and spread and clear fraction from the view, VCI_mean from the view's NDVI mean and VCI_p10
  from its NDVI 10th percentile against a per-cell VCI table built from single clear views (>= 30% clear) in
  development years. Two versions: (a) models trained on composites (the Phase 5 models); (b) the same settings
  trained on development regions thinned the same way (3 starts pooled; regions with per-scene values; 3 seeds).
  Report AUC and the share of cell-months with a fresh view.

## 12. Running

One resumable pipeline: every run saves predictions, metadata (commit, device, seed, timings) and a DONE marker;
restarts skip finished runs; a failed run is logged and skipped. Every stage is smoke-tested in a sandbox on a
tiny subset first. STATUS.md is updated after every stage.

## Amendment 4 (2026-11-08): three job-1 models added after the development results

Added after the job-1 development results were seen (`DECISIONS.md` #50-#52), at the user's request, before the
fresh test is scored. They are reported in their own rows, always labelled "added after the development results",
and never enter the gate: section 9 and `gate.json` are unchanged. Their settings are fixed here, before any of them
is trained.

- **Why.** On job 1 the five networks trailed LR on VCI(T) on the outer folds, while on their own training windows
  they scored far above it: the inputs NDVI_mean, NDVI_std, valid_frac and the season let them learn where and when
  drought happened in the training years. The paper's architecture (a CNN on image tiles feeding a Bi-LSTM) was
  never tested: every network here sees per-cell summaries.
- **`bilstm_vci`** (VCI-only Bi-LSTM): `bilstm` exactly (layout, training, the same 30 job-1 search configurations,
  selection, 10 seeds), with NDVI_mean, NDVI_std, valid_frac and both season channels set to zero in every window, so
  only VCI_mean, VCI_p10, the observed flag and months-since-last-observation carry information.
- **Tiles.** For every cell and month: the per-pixel VCI and usable-pixel rule behind VCI_mean (section 1, same VCI
  table), averaged onto a 23 x 21 grid of near-equal pixel bins (a bin is usable if it holds at least one usable
  pixel; its value is the mean VCI of those pixels), stored as whole numbers 0-100 with 255 for unusable. Built per
  VCI table (f1-f4, all) from the same composites as the features (the paper's for the original 102). Model input:
  two channels, VCI/100 (0 where unusable) and the usable flag; a month whose cell is unobserved (section 1) gets an
  empty tile.
- **Tile encoder** (shared by both CNN models): conv 3x3 (2 -> w), GELU; conv 3x3 stride 2 (w -> 2w), GELU; with
  depth 3 a further conv 3x3 stride 2 (2w -> 2w), GELU; padding 1. Then each channel's mean over usable positions
  (usable bins pooled to the output grid; zeros when none is usable) and the tile's usable share: 2w + 1 numbers.
- **`cnn_lr`** (CNN + LR): the encoder on month T's tile, dropout, one linear layer to the logit; a logistic
  regression on learned features of one month. No other input.
- **`cnn_bilstm`** (CNN + Bi-LSTM): the encoder on each of the 12 monthly tiles (same weights); each month's 2w + 1
  numbers plus the observed flag and months-since-last-observation go into the `bilstm` encoder and head (final
  states + masked mean -> LayerNorm -> Linear -> GELU -> Dropout -> Linear).
- **Training** as section 5 (binary cross-entropy, AdamW, cosine decay to 5%, gradient clip 1.0, batch 512, the
  configuration's epochs, no early stopping); both CNN models are trained end to end. Masking augmentation for
  `cnn_bilstm` as section 6 (months T-11..T-1 only; a hidden month gets an empty tile, observed 0, months-since
  recomputed).
- **Search** (drawn once with a fixed seed, one seed per configuration, evaluated on every outer fold's inner split,
  selection by inner-validation AUC, then 10 seeds of the chosen setting on fold k's full training set):
  `bilstm_vci` the 30 configurations above; `cnn_lr` 20 configurations: w from {8, 16, 32}, depth from {2, 3},
  learning rate log-uniform 3e-4..3e-3, weight decay log-uniform 1e-6..1e-2, dropout uniform 0..0.5, epochs from
  {6, 10, 15, 20}; `cnn_bilstm` 12 configurations: w from {8, 16}, depth from {2, 3}, hidden from
  {16, 24, 32, 48, 64}, one layer, the same learning-rate, weight-decay and dropout ranges, masking off or on
  (probability uniform 0.1..0.4), epochs from {6, 10, 15}. Fewer than 30 because a CNN run costs several times more.
- **Scoring**: the same outer-fold windows, labels, within-fold AUC and region resamples as every job-1 model (the
  score stage's draws), paired against `lr_vci_t` and the gate's best simple model, all regions and West/East, with
  the gate's wording. Also reported: the 2 x 2 of cell average vs cell image and month T vs 12 months (`lr_vci_t`,
  `cnn_lr`, `bilstm_vci`, `cnn_bilstm`), per-fold AUC, parameter counts and multiply-adds per prediction.
- **Fresh test**: each at its development-chosen setting (section 11), trained on all development data (10 seeds)
  with the Phase 5 table, scored in the same single Phase 5 scoring and reported in separate rows. Not in the
  realistic-sampling part (per-scene records hold no per-pixel images). Phase 5 also saves every network it trains
  (`model.pt`); this changes no prediction.
- **Order**: tiles are built after the planned development stages finish; the fresh test waits (`STOP_CHAIN`) until
  these three are trained, scored on the development folds and added to Phase 5. Extending them to other jobs would
  be a further dated amendment, committed before that training.
