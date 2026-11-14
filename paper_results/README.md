# DEWSat paper results

Tables and figures for the paper from the v2 study (`~/Desktop/DEWSat_BiLSTM_v2`; published copy `~/Desktop/DEWSat_BiLSTM_github`): 558 development regions and a fresh test in 144 new regions, scored once. This is not the earlier 102-region run described in this folder's `report.md`.

The paper's model is the plain Bi-LSTM, labelled "Bi-LSTM" in the tables and figures. For job 1 it is not a pre-registered row: it was added to the single fresh-test scoring before any fresh-test label was read (`code/design/DESIGN.md`, amendment 6), so T1 marks it "added after development". Job 1's pre-registered networks are the Residual Bi-LSTM (best Bi-LSTM variant) and the GRU (best network); both passed on the fresh test, and the three are within 0.006 AUC of each other there. In job 3 the plain Bi-LSTM is the pre-registered best Bi-LSTM variant.

Every result is read from `results/scores/*.json` and rounded once, to 3 decimals. There is no prose for the paper here, only table headings and figure labels, so the text and captions can be the authors' own.

| File | What it holds |
|---|---|
| `tables/T1_fresh_test_job1.csv` | Main result: every model on the fresh test, job 1 (AUC, difference from LR on VCI(T), Brier score, ECE, mean predicted probability) |
| `tables/T2_development_job1.csv` | Every model in development, job 1 (4 outer folds) |
| `tables/T3_preregistered_verdicts.csv` | The pre-registered comparisons for all four jobs, development against the fresh test |
| `tables/T4_other_labels_job1.csv` | The fresh-test predictions judged by NLDAS-2 and by SPEI at 90, 180 and 270 days and 1 year (SPEI-12) |
| `tables/T5_revisit_job1.csv` | One clear view every 7, 16 or 32 days instead of monthly composites |
| `tables/T6_model_size_job1.csv` | Parameters, multiply-adds per prediction, weight size |
| `tables/T7_design_checks_job1.csv` | Design checks: image tiles, 12-month history, NDMI, weekly steps, ReLU, more years against more regions |
| `tables/T8_data_summary.csv` | Regions, cells, windows, drought rates, label checks |
| `tables.html` | All eight tables on one page: open it in a browser, select a table, copy it, paste it into Google Docs |
| `figures/` | The study's six figures, PNG at 300 dpi (for Google Docs) and PDF, without title text |
| `claims_vs_results.md` | Each results claim in the manuscript against what the study found |
| `build_paper_results.py` | Rebuilds this folder from the score files |

## Notes

- AUC: how well a model ranks drought windows above non-drought ones; 0.5 is chance. Differences are paired, with 95% intervals from 2,000 resamples of whole regions. West and East are the two sides of 98° W.
- Pre-registered: fixed in `code/design/DESIGN.md` before any training. Rows marked "added after development" were added after the development results and before the fresh test.
- T1 leaves out "LR on latest VCI + age": with month T observed it is the same model as LR on VCI(T).
- T4 uses the 64,995 windows that have every label, so its FLDAS row can differ from T1 in the third decimal.
- Figures 2, 3, 5 and 6 are redrawn from the same score files by the repository's own figure code. Figures 1 and 4 are drawn by `build_paper_results.py` in the same style with the Bi-LSTM first, since the repository's versions lead with the Residual Bi-LSTM. Figure 5 plots the Residual Bi-LSTM only; T5 has the Bi-LSTM's revisit numbers.
- The SPEI-12 results come from `~/Desktop/DEWSat_BiLSTM_v2/scores/fresh_spei12_posthoc.json` (generated 2026-11-12 01:27:00): the saved fresh-test predictions scored after the lock against gridMET's 1-year SPEI, with the same models and resamples as the other labels, on the same windows; every published row came out identical. The GitHub copy has it at `results/scores/fresh_spei12_posthoc.json`, with its script at `code/dewsat2/fresh_spei12_posthoc.py`.
