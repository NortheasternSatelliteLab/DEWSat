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

## Amendment 4b (2026-11-08, about 15:40): `bilstm_vci` also for jobs 2 and 3

Committed before any job-2 or job-3 outer-fold result was computed or seen: the score stage had not run, and only
job 1's outer folds had been looked at (`DECISIONS.md` #50-#51). The user asked to keep pursuing the Bi-LSTM, whose
case rests on the jobs that need the history: the missing month (job 2) and the forecast (job 3).

- **`bilstm_vci` for jobs 2 and 3**: that job's own `bilstm` (its 30 search configurations and seeds, selection, 10
  final seeds, job 2's hiding of month T, job 3's three heads and onset metric), with the same five channels zeroed as
  for job 1. Job 4 is scored with the job-1 models, as before.
- **Scoring**: the score stage's subsets and region draws for each job (job 2 against `lr_latest_age` and the gate's
  best simple model; job 3 on the mean onset AUC over leads 1-3 against `lr_vci_t` and the best simple model), with the
  gate's wording and the label "added after the development results".
- **Fresh test**: as amendment 4 (development-chosen setting, all development data, 10 seeds, the single Phase 5
  scoring).
- The CNN models stay job 1 only: each CNN run costs several times more, and adding them would move the fresh test
  past Wednesday.

## Amendment 5 (2026-11-09, about 00:40): learning curve for job 1 (secondary, no pass/fail)

Added after the development results (every gate row "worse", `DECISIONS.md` #61), at the user's request, to ask
whether more training data could close the gap. Development data only. It changes nothing in the gate, the selection
of any model or the fresh test.

- **Question**: as the labelled training data grow, does the Bi-LSTM's within-fold AUC difference to LR on VCI(T)
  (job 1) move toward zero, and does it matter whether the extra data are more regions or more years?
- **Models**:
  - `lr_vci_t`, refit on every slice.
  - `bilstm`, the lab's DroughtBiLSTM: the model the paper's question is about.
  - The better job-1 Bi-LSTM of amendment 4: `bilstm_vci` or `cnn_bilstm`, whichever has the higher development
    within-fold AUC in `scores/posthoc.json`.
  - Each network keeps its fold's chosen setting (no re-tuning) and trains with seeds 42, 43 and 44. Its prediction
    is the mean probability of the three seeds.
  - LR's AUC cannot change with the data: with one input its ranking is VCI(T)'s whenever its slope is negative. It is
    refit anyway, and its slopes are reported.
- **Slices of each fold's outer training windows**. The scaler comes from each slice's own windows. The fold's VCI
  table, the inputs and the scored windows are unchanged.
  - *Years*: folds 1 and 4, five training years each (2019-2023 and 2016-2020). Training targets T are restricted to k
    of the five years, k = 2, 3, 4. Three subsets per k are drawn without replacement from the C(5, k), with numpy
    `default_rng(20261109 + fold index)`. All training regions.
  - *Regions*: all four folds. A random 1/8, 1/4 or 1/2 of the fold's training regions, nested, drawn within West and
    within East separately (`default_rng(20261109 + 100 + fold index)`), at least one per side. All training years.
  - *Full data*: the existing final runs' seeds 42-44. Nothing is retrained.
- **Scoring**: the job-1 outer-fold scored windows of the axis's folds, within-fold AUC, and 2,000 whole-region
  resamples shared by every size and model of an axis. For each size:
  - each model's AUC, and the paired difference network minus LR, for all regions and for West/East;
  - on the years axis, the mean over the three subsets of each size (subset j of fold 1 pooled with subset j of
    fold 4).
  - Per network, the change in that difference from the smallest size to the full data, and from the second-largest
    size to the full data, with intervals.
- **Reading, stated in advance**:
  - If the difference rises toward zero as training years are added, and is still rising at the full five years (the
    last change's interval above zero), more years would likely help.
  - If it is flat, more years of this kind are not expected to close the gap.
  - The same reading applies to regions.
- **Runs**: 180 network runs: 2 networks x (2 folds x 9 year subsets + 4 folds x 3 region fractions) x 3 seeds. They
  are written to `runs/lcurve/`, `scores/lcurve.json` and `logs/lcurve/`, and run with the post-hoc runner
  (`--stages lcurve,lcurve_score`). They use spare capacity and never delay Phase 5.

## Amendment 6 (2026-11-09, about 03:40): the post-hoc models in the one fresh-test scoring

Written before Phase 5 has started and before any fresh-test label has been read. The pre-registered Phase 5 rows,
the gate and the selection of Section 11 are unchanged. These rows are added beside them and are not part of the gate.

- **Models added** (`phase5_stage.EXTRA`), each trained like the pre-registered Phase 5 networks: all development
  windows, the all-development VCI table, the 10 final seeds, and the mean probability as the prediction.
  - `bilstm_vci` (amendments 4 and 4b) for jobs 1, 2 and 3 (job 4 through job 1).
  - `cnn_lr` and `cnn_bilstm` (amendment 4) for job 1 (job 4 through job 1), on the all-development tiles
    (`cache/tiles/dev_all.npy`, `test_all.npy`).
  - The plain `bilstm` for jobs 1 and 2. Job 3 already scores it as its best Bi-LSTM variant. The paper's question
    is about this network, and the fresh test can be scored only once.
- **Settings**: per model and job, the search configuration with the highest mean inner-validation metric over the 4
  folds. This is the rule Section 11 uses for the pre-registered networks (`posthoc_settings()`, written to
  `scores/phase5_settings.json`).
- **Evaluation sets**:
  - All of them: the fresh test with composites.
  - `bilstm` and `bilstm_vci` for jobs 1-2: also the thinned fresh test (composite-trained, version (a) of
    DECISIONS.md #41).
  - The tile CNNs: no thinned version, since no thinned tiles exist.
  - None of them is retrained on thinned inputs; that stays with each job's best network.
- **Scoring**:
  - Every added model against the job's LR and best simple model, as for the pre-registered networks.
  - The 2x2 and input contrasts: `bilstm_vci - bilstm`, `cnn_bilstm - bilstm_vci`, `cnn_bilstm - cnn_lr`
    (`cnn_lr - lr_vci_t` and `bilstm_vci - lr_vci_t` are among the comparisons with LR).
  - Phase 6's SPEI benchmark also scores the added models.
- **Saved models**: every Phase 5 network run, pre-registered or added, saves `model.pt`, its parameter count and,
  for the tile CNNs and both Bi-LSTMs, multiply-adds per prediction (Jetson sizing).
- **Running**:
  - The composite stage uses 3 CPU workers instead of 4, plus 2 GPU workers. The GPU-only tile CNNs set the stage's
    length, and one fewer CPU process leaves memory headroom.
  - Failed runs are retried once before anything is scored.
  - A missing prediction stops the scoring before the lock is written, so no partial scoring can use up the fresh
    test.
  - Sandbox check: `tmp/phase5_sandbox` (the `smoke5` pseudo-test regions, a sandbox lock). Every added model trained,
    scored and saved; no real fresh-test label was read.

## Amendment 7 (2026-11-09, about 15:10): ReLU Bi-LSTM for job 1, development only (secondary, no pass/fail)

Added after every result, including the one fresh-test scoring, at the user's request: could the Bi-LSTM use ReLU
in place of tanh? Development folds only. The fresh test stays locked (Section 11) and is not scored again. Nothing
changes in the gate, the selection, the fresh-test rows or any earlier result.

- **Question**: with everything else fixed, does replacing tanh by ReLU inside the Bi-LSTM's cell change its job-1
  within-fold AUC, and its gap to LR on VCI(T)?
- **Model `bilstm_relu`**: `bilstm` exactly (layout, head, inputs, training, each fold's chosen setting, the 10 final
  seeds), except inside the LSTM cell:
  - The candidate memory and the output squash use ReLU instead of tanh.
  - The input, forget and output gates stay sigmoid. ReLU gates would be unbounded and could shut for good.
  - The cell is written step by step, since `nn.LSTM` fixes its activations. With two layers, dropout acts between
    them as in `nn.LSTM`.
  - Initial weights are copied from an `nn.LSTM` created at the same point with the same seed. So each seed starts
    from exactly the weights of the original `bilstm` run, and the data order is the same.
- **Checks before the 40 runs**:
  - With tanh, the hand-written cell reproduces `nn.LSTM`'s outputs given the same weights (largest difference below
    1e-5), for the three settings `bilstm` uses (hidden 24 x 1 layer, 128 x 1, 128 x 2). For the same seed, the whole
    model's initial weights and logits equal `bilstm`'s.
  - Pilot on fold 1 (CPU): the hand-written cell with tanh, trained as a final run, reproduces the original `bilstm`
    runs of seeds 42, 43 and 46 (trained on the CPU): each seed's outer-validation AUC within 0.001. Then one
    `bilstm_relu` run: its training loss falls and stays finite.
  - If a check fails, nothing more is trained until the cause is found, fixed and logged.
- **Scoring**: the score stage's job-1 subset (job 4 through job 1), within-fold AUC and its 2,000 whole-region
  resamples (the same draws as every job-1 model), all regions and West/East. Pairs:
  - `bilstm_relu - bilstm` (the primary comparison);
  - `bilstm_relu - lr_vci_t` and `bilstm_relu - gbt_mono` (the gate's best simple model), with the gate's wording;
  - `bilstm - lr_vci_t`, for reference.
- **Reading, stated in advance** (primary pair, all regions): interval entirely above zero -> "ReLU helps in
  development"; interval includes zero -> "no detectable difference"; entirely below zero -> "ReLU is worse in
  development". West/East are reported but do not change the reading. Also reported: per-fold AUC, whether every
  run's training loss fell, and seconds per run.
- **Limits**: the settings were chosen for tanh, and there is no search for ReLU. Development only, so it cannot
  change the fresh-test verdict.
- **Runs**: 40 (4 folds x 10 seeds) in `runs/relu/`, scores in `scores/relu.json`, logs in `logs/relu/`, with the
  post-hoc runner (`--stages relu,relu_score`).
