# Results

Tables and figures of the DEWSat Bi-LSTM study, generated from the score files in `scores/` by `code/dewsat2/export_results.py`. AUC differences carry 95% intervals from 2,000 whole-region resamples (columns `_lo`, `_hi`; `west_`, `east_` are the two sides of 98 degrees W). The full write-up is `../FINDINGS.md`; every table of the run is in `../report.md`.

| File | What it shows | Source in `scores/` |
|---|---|---|
| [`tables/fresh_test_verdicts.csv`](tables/fresh_test_verdicts.csv) | Fresh test (scored once), pre-registered rows for all four jobs | `phase5.json → fresh; selection in scores.json` |
| [`tables/fresh_test_auc.csv`](tables/fresh_test_auc.csv) | Fresh test, jobs 1, 2 and 4: AUC of every model and its difference to the job's LR | `phase5.json → fresh.<job>` |
| [`tables/development_gate.csv`](tables/development_gate.csv) | Development, pre-registered rows for all four jobs: AUC difference, verdict | `scores.json → gate` |
| [`tables/development_auc_job1.csv`](tables/development_auc_job1.csv) | Development, job 1: within-fold AUC of every model | `scores.json → subsets.job1.models; posthoc.json; relu.json` |
| [`tables/fresh_within_region.csv`](tables/fresh_within_region.csv) | Fresh test, job 1, pairs within the same region only (ranking over time) | `fresh_posthoc.json → within_region` |
| [`tables/fresh_network_pairs.csv`](tables/fresh_network_pairs.csv) | Fresh test, networks against each other | `fresh_posthoc.json → network_pairs` |
| [`tables/fresh_by_year.csv`](tables/fresh_by_year.csv) | Fresh test, job 1, by year and side: pooled AUC per model | `fresh_posthoc.json → by_year` |
| [`tables/training_size.csv`](tables/training_size.csv) | Bi-LSTM minus LR on VCI(T) against training windows: 4 development folds, fresh test | `fresh_posthoc.json → dev_folds` |
| [`tables/learning_curve.csv`](tables/learning_curve.csv) | Learning curve (development, job 1): network minus LR as training years or regions grow | `lcurve.json → axes.<axis>.sizes` |
| [`tables/learning_curve_changes.csv`](tables/learning_curve_changes.csv) | Learning curve: change in the gap, smallest to full and second-largest to full | `lcurve.json → axes.<axis>.change_in_difference` |
| [`tables/other_labels.csv`](tables/other_labels.csv) | Fresh-test predictions judged by other drought labels (FLDAS, NLDAS-2, SPEI 90/180/270 d) | `fresh_labels_posthoc.json → jobs.<job>.labels` |
| [`tables/revisit.csv`](tables/revisit.csv) | Realistic sampling, fresh test, job 1: single clear views every 7, 16 or 32 days | `phase5.json → realistic.job1 and coverage` |
| [`tables/relu.csv`](tables/relu.csv) | ReLU instead of tanh in the Bi-LSTM (development, jobs 1 and 4) | `relu.json → jobs.<job>.paired` |
| [`tables/relu_per_fold.csv`](tables/relu_per_fold.csv) | ReLU test per development fold: pooled AUC, tanh against ReLU | `relu.json` |
| [`tables/order_test.csv`](tables/order_test.csv) | Order test: earlier months shuffled within each window (development, job 1) | `checks.json → order_test` |
| [`tables/positive_control.csv`](tables/positive_control.csv) | Positive control: VCI_mean(T-3) - VCI_mean(T) > 20, both observed, fold 1 | `checks.json → positive_control` |
| [`tables/whatifs_and_weekly.csv`](tables/whatifs_and_weekly.csv) | What-ifs (NDMI, VCI per cell or pixel) and weekly inputs (development, job 1) | `whatif.json → ndmi, vci_cell_vs_pixel; weekly.json → scores` |
| [`tables/image_vs_cell_average.csv`](tables/image_vs_cell_average.csv) | Per-pixel VCI image tiles against cell averages, development and fresh test | `posthoc.json → jobs.job1.paired; phase5.json → fresh.job1.paired` |
| [`tables/model_sizes.csv`](tables/model_sizes.csv) | Fresh-test networks: parameters, multiply-adds per prediction, weight file size | `runs/phase5/composite/<job>/<model>/seed42/meta.json` |
| [`figures/fig1_gap_to_lr_dev_vs_fresh.png`](figures/fig1_gap_to_lr_dev_vs_fresh.png) | Gap to LR on VCI(T) per model, development against the fresh test | `scores.json, posthoc.json, phase5.json` |
| [`figures/fig2_training_size.png`](figures/fig2_training_size.png) | Bi-LSTM minus LR against training windows | `fresh_posthoc.json → dev_folds` |
| [`figures/fig3_learning_curve.png`](figures/fig3_learning_curve.png) | Learning curve: network minus LR as training years or regions grow | `lcurve.json → axes` |
| [`figures/fig4_other_labels.png`](figures/fig4_other_labels.png) | Fresh-test predictions judged by five drought labels | `fresh_labels_posthoc.json → jobs.job1.labels` |
| [`figures/fig5_revisit.png`](figures/fig5_revisit.png) | Network minus LR with single clear views every 7, 16 or 32 days | `phase5.json → realistic.job1` |
| [`figures/fig6_relu.png`](figures/fig6_relu.png) | ReLU against tanh and LR, development | `relu.json → jobs.job1.paired` |

The score files themselves (`scores/*.json`) hold every number of the run, including those not exported here.
