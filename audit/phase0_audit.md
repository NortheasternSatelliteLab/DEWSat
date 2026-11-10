# Phase 0 audit of the last run (DEWSat_Capability_Tests, commit e13e7a4)

Run on hashed copies only (`inputs/capability_run/`, `inputs/capability_code/`), with the audited code itself
(`code/audit/phase0_audit.py`). Every number below is in `audit/phase0_results.json`. "Within-fold AUC" counts only
pairs from the same outer fold.

## a. Loss by epoch (180 network runs: 6 configurations x 2 tasks x 5 folds x 3 seeds)

Early stopping on inner-validation log loss (patience 8, at most 60 epochs).

| Task / model | Best epoch, median (range) | Runs with best epoch <= 3 | Inner loss fell at all | Inner loss, epoch 1 -> best (mean) | Training loss, epoch 1 -> last (mean) |
|---|---|---|---|---|---|
| nowcast Bi-LSTM h32 | 3 (1-7) | 8/15 | 14/15 | 0.527 -> 0.518 | 0.528 -> 0.418 |
| nowcast residual Bi-LSTM h32 | 2 (1-8) | 11/15 | 9/15 | 0.524 -> 0.520 | 0.523 -> 0.417 |
| nowcast Bi-LSTM h16 | 4 (1-10) | 7/15 | 14/15 | 0.529 -> 0.518 | 0.534 -> 0.454 |
| nowcast one-direction LSTM h32 | 4 (1-13) | 6/15 | 13/15 | 0.527 -> 0.518 | 0.532 -> 0.439 |
| forecast Bi-LSTM h32 | 1 (1-3) | 15/15 | 5/15 | 0.560 -> 0.557 | 0.554 -> 0.452 |
| forecast residual Bi-LSTM h32 | 1 (1-3) | 15/15 | 5/15 | 0.559 -> 0.558 | 0.547 -> 0.448 |

Finding: the training loss falls steadily, while the inner-validation loss improves by about 0.01 at most and stops
improving within a few epochs (the forecast networks' best epoch is the first one in every run). The networks keep
fitting the training data after the point where it stops helping. The cosine schedule was set for 60 epochs, so with
runs ending after about 10 epochs the learning rate barely decayed.

## b. Training, inner-validation and outer AUC (windows with month T observed)

| Fold | LR on VCI(T): train / inner / outer | Bi-LSTM h32: train / inner / outer | Residual Bi-LSTM h32: train / inner / outer |
|---|---|---|---|
| 1 | 0.695 / 0.705 / 0.694 | 0.803 / 0.757 / 0.684 | 0.760 / 0.746 / 0.677 |
| 2 | 0.696 / 0.679 / 0.756 | 0.818 / 0.748 / 0.734 | 0.793 / 0.738 / 0.768 |
| 3 | 0.652 / 0.686 / 0.742 | 0.843 / 0.745 / 0.621 | 0.860 / 0.747 / 0.626 |
| 4 | 0.671 / 0.656 / 0.722 | 0.830 / 0.754 / 0.666 | 0.843 / 0.762 / 0.659 |
| 5 | 0.721 / 0.717 / 0.652 | 0.799 / 0.752 / 0.688 | 0.791 / 0.752 / 0.674 |

Finding: **overfitting, not failure to learn.** The networks learn a lot (training AUC 0.80-0.86 against LR's
0.65-0.72) and keep a clear lead on the inner set (other regions, same years: about 0.75 against 0.66-0.72), but
on the outer folds (other regions and other years) they fall to 0.62-0.73 and lose to LR in four of five folds. LR's
AUC hardly moves between the three sets. What the networks learn is specific to the training years. The inner set
could not reveal it, because it shared every one of its months with training (part f), so early stopping and the
inner loss rewarded it.

## c. Spot checks (5 windows per job, 20 in all)

All 20 windows span 12 months ending at T; observed flags match the source table; every label at T (and at T+1..T+3
for job 3) matches both `labels.csv` and an independent rebuild from `fldas.csv` (20th percentile of the cell's
2001-2022 values for the calendar month); job-4 windows meet the change thresholds. No problem found.

## d. Order test (job 1, 54,418 windows; mean of seeds 42-44)

| Model | Within-fold AUC, original | Months T-11..T-1 shuffled (5 permutations) | Shuffled minus original (permutation 1) | Reversed | Earlier months hidden |
|---|---|---|---|---|---|
| Bi-LSTM h32 | 0.676 | 0.681-0.683 | +0.006 [-0.013, 0.026] | 0.651 | 0.625 |
| Residual Bi-LSTM h32 | 0.672 | 0.681-0.683 | +0.010 [-0.006, 0.027] | 0.662 | 0.638 |
| One-direction LSTM h32 | 0.673 | 0.689-0.690 | +0.016 [-0.005, 0.042] | 0.658 | 0.593 |
| Bi-LSTM h16 | 0.673 | 0.674-0.677 | +0.003 [-0.017, 0.027] | 0.637 | 0.657 |

LR on VCI(T): 0.696. Finding: shuffling the earlier months changes the predicted probabilities by about 0.08-0.09 on
average, but it does not lower the AUC (it raises it slightly, within noise). Whatever the networks learned from the
order of the history does not help on new years. Hiding the history (out of distribution for these models) lowers
the AUC, so the networks do use the history's values.

## e. Positive control (fold 1, same code, label = VCI fell by more than 20 over the last 3 months)

LR on VCI(T): AUC 0.797. Bi-LSTM h32 (mean of 3 seeds): 0.9999. Paired difference +0.203 [0.151, 0.248]
(4,024 outer windows, region bootstrap). The networks needed 29-58 epochs for this. Finding: the architecture, the
inputs and the training code can learn a label that only the history explains. The real label's lack of gain is
not a code bug.

## f. Other problems in the code, data handling and evaluation

1. **Inner validation shared every month with training** (93/93, 70/70, 70/70, 70/70 and 73/73 months by fold). It
   held out regions only, so it could not detect year-specific overfitting (part b). Fixed: the inner split holds
   out a region group *and* a year, with the same buffers (DESIGN.md section 3).
2. **Pooled AUC as the primary metric.** Training-year drought rates run against the validation blocks' (fold 1:
   training 0.31, validation 0.14; fold 2: 0.35 vs 0.08; fold 4: 0.22 vs 0.41), so pooled AUC mixes base-rate shifts
   into ranking (climatology scored 0.32). Fixed: within-fold AUC is primary.
3. **No tuning**: one learning rate, weight decay, dropout and two hidden sizes, chosen in advance. Fixed: a random
   search of 30 configurations per family, fold and job, chosen on the new inner split.
4. **One nowcast model served jobs 1, 2 and 4**, trained on both T-observed and T-missing windows, with month T hidden
   20% of the time even for job 1. Fixed: a dedicated model per job; job-1 masking never hides T; job 2 trains with T
   always hidden.
5. **Early stopping on a 60-epoch cosine schedule**: runs stopped after about 10 epochs with the learning rate near
   its peak. Fixed: the number of epochs is a searched setting and the cosine schedule spans exactly those epochs.
6. **VCI reference included the later test period** (folds 1-4 used every month outside block +/- 12, including
   2024-2026). Fixed: the reference uses development months only (2016-2023) outside the held-out block +/- 12.
7. **Residual model drifted far from its LR start** (mean absolute logit change 0.48 on fold 1), with no
   regularisation toward it. It stays as specified (starts exactly at the LR; check: max logit difference 3e-7 at
   initialisation), and its weight decay and epochs are now tuned on the year-held-out inner split.
8. **No "months since last observation" input**: the networks had to infer it from the observed flags. Fixed: an
   explicit channel for every network, rebuilt after masking.
9. Checked and fine: no cell is in two of the training, inner and outer sets in any fold; training targets are at
   least 13 months from the held-out block; LR slopes on VCI are negative in every fold (-0.028 to -0.043 per VCI
   point); the window builder takes exactly T-11..T; labels and features rebuild the paper's tables.
