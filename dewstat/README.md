# DEWSat soil moisture (DewStat)

Estimates soil moisture at 10–40 cm (m³/m³) from satellite data only, so the DEWSat CubeSat can run it in orbit.
A small Bi-LSTM reads 12 months of what the satellite can measure (NDVI and VCI, or the SWIR moisture indices NDMI and
NDWI, plus season and location) and predicts how much wetter or drier than normal each 0.1° cell (about 9 × 11 km) is.
NASA's FLDAS land-surface model is the answer key for training. It is never a model input.

**Results:** open [`model/report/report.html`](model/report/report.html) (one file, all charts inside). The full
write-up, from the data to the final weekly model, is [`model/report/DewStat_report.docx`](model/report/DewStat_report.docx).

## Results in short

102 regions across US climate zones: 67 train, 13 validation, 22 test. Test regions are never seen in training and are
scored on 2024 onward.

| | Test RMSE (m³/m³) | Test R² | Error ÷ persistence (MSE) |
|---|---|---|---|
| Normal map alone (each cell's average for the month, no model) | 0.0406 | 0.74 | 1.18× |
| **V2: satellite + normal map** (chosen) | **0.0363** | **0.79** | **0.95×** |
| V3: V2 with SWIR instead of VCI | 0.0364 | 0.79 | 0.95× |
| Persistence (last month's value; needs ground data the satellite lacks) | 0.0373 | 0.78 | 1.00× |

How to read it:
- R² depends on which regions are tested: over random sets of 22 test regions it ranges 0.69–0.86, and on the 4
  original test regions it is 0.53. Quote it as "R² 0.79 on 22 held-out regions", next to the normal map's 0.74.
- Most of the R² comes from the normal map. The model's own part (wetter or drier than normal) is still weak:
  correlation 0.30 with the real anomaly.
- The map is needed: trained without it (predicting soil moisture directly), the model drops to R² 0.58, below the
  map alone. It can still stay on the ground; see [`model/normal_map/`](model/normal_map/).
- SWIR makes no detectable difference (V3 beats V2 in 11 of 22 test regions).
- The model beats persistence in 11 of 22 test regions, by the most where strong regular seasons make persistence weak
  (Willamette Valley, Sacramento, Eastern Washington), and loses by the most in Nevada, Minnesota, Georgia, Indiana,
  Alabama and Montana (about 1.8–2.5× persistence's error).

## Weekly instead of monthly?

FLDAS is monthly only in the US, so the weekly test uses NLDAS-2 Noah as the answer key (same 10–40 cm layer, 0.125°,
daily). Same V2 model, inputs and regions; settings picked on validation. Details: [`model/report/weekly/`](model/report/weekly/).

| Answer key, step (22 test regions) | Model R² | Normal map R² | Persistence R² | Model error ÷ persistence (MSE) |
|---|---|---|---|---|
| FLDAS, monthly (the chosen V2) | 0.79 | 0.74 | 0.78 | 0.95× |
| NLDAS, monthly | 0.87 | 0.87 | 0.76 | 0.55× |
| NLDAS, weekly | 0.85 | 0.83 | 0.93 | 2.28× (beats it in 2 of 22 regions) |

- Week to week, soil at 10–40 cm barely changes: last week's value explains R² 0.94, last day's 0.98. A satellite model
  does not come close at those steps.
- Photos are the other limit: a cell has a clear photo in 67% of weeks and 17% of days with all HLS satellites
  (38% and 6% with Landsat only). DEWSat, one satellite, will see less.
- The weekly model still beats the normal map slightly (16 of 22 regions), the yardstick for a satellite without ground
  data. Persistence needs ground data, which arrives late (NLDAS ~4 days, FLDAS ~1 month).

## Validation years check

Our first runs split validation by place only: the 13 validation regions were used in every year (2020–2025), so
validation shared years with training (2020–2022) and test (2024–2025), though never regions. Validation never trains
the weights, but it picks the stopping epoch and the setting. We re-ran every main sweep with validation on the
validation regions in **2023 only**, a year neither training nor test uses (`train.py --val-years gap`, `run_valgap.sh`).
Nothing meaningful changed. Details: [`model/report/validation_years/`](model/report/validation_years/results.md).

| Model (22 test regions) | Validation all years: R² | Validation 2023 only: R² |
|---|---|---|
| Weekly, with map (final) | 0.848 | **0.848** (same setting, 4-week window) |
| Monthly FLDAS, with map (V2) | 0.793 | **0.800** |
| Monthly FLDAS, no map | 0.582 | **0.580** |
| Weekly, no map | 0.372 | **0.365** |

The final weekly model's best weights come from the end of the first pass over the training data, so changing the
validation years could not change it. After that pass, training error keeps falling while validation error rises:
generalizing to new places is the limit, not training time.

## What the weekly model learned

Charts and tables in [`model/report/trends/`](model/report/trends/trends.md) (`make_trends.py`), on the 22 test regions, 2024–2025:

1. Greener than usual → wetter than usual: the model learned the right direction, smoothly.
2. It beats the normal map from March to December (most in October) and is slightly worse in January–February.
3. Averaged over all test regions it follows the real ups and downs, and catches part of the fall 2024 dry spell.
4. It gets wetter vs drier right 72–77% of the time for big departures from normal, but only 52% for tiny ones.
5. It beats the map in 16 of 22 test regions; it does worst in Minnesota South, Kansas Flint Hills and North Dakota East.

`make_trends.py` also writes the raw weekly table (every cell-week of all 102 regions: inputs, NLDAS answer, normal
map) and the test predictions to `model/report/trends/raw/` (~12 MB, not in git; ask for a copy).

## Layout

```
data-pipeline/practice/        download and prepare the data
  regions.csv, regions_new.csv   the 22 + 80 regions (0.5° boxes) and their split: train / val / test
  pull_regions.py                per region: fldas, bands, vci (+ nasa); resumable, retries built in
  build_table.py                 HLS scenes -> cloud-masked per-cell means -> monthly table (Planetary Computer)
  nasa_backfill.py               HLS 2016-2019 straight from NASA (Planetary Computer starts in 2020)
  merge_hls_years.py             joins NASA 2016-2019 and Planetary Computer 2020+ (only for the "full" setup)
  make_vci.py                    VCI from NDVI (each cell's own range, training years only)
  watch_pull.sh                  RAM and HTTP-error watcher for long pulls
  nldas_daily.py                 daily NLDAS-2 soil moisture at every region cell (answer key for the weekly model)
model/
  dew_stat_data.py               loads regions, builds 12-month windows, splits, scaling, normal map
  dew_stat_model_fast.py         the Bi-LSTM (PyTorch Lightning)
  train.py                       trains and scores one setting (3 seeds) against the baselines
                                 (--answer nldas_week / nldas_month: NLDAS-2 instead of FLDAS)
  dew_stat_nldas.py              weekly or monthly samples with NLDAS-2 as the answer key
  run_nldas.sh                   weekly vs monthly sweep, then make_results_nldas.py
  run_nomap.sh                   without the normal map (model predicts soil moisture itself), then make_results_nomap.py
  cadence_check.py               persistence by time step (day / week / month) and clear-photo frequency
  run_sweeps.sh                  V1, V2, V3 x 5 settings, then make_results.py
  make_results.py                picks each version's setting by validation loss; results tables
  compare_setups.py              compares data setups on validation (how HLS 2020+ only was chosen)
  plot_results.py                charts per run
  models/                        the two chosen trained models (V2, V3) with their scores
  normal_map/                    the normal map V2 adds back: CSV, 16-bit on-board table, pictures, explainer
  make_normal_map.py             exports normal_map/ from the FLDAS data
  report/                        the team report (report.html + PNGs); report/weekly/ = weekly vs monthly;
                                 report/nomap/ = with vs without the normal map
```

## Setup

Python 3.12 on Linux, macOS or Windows. You need a free [NASA Earthdata](https://urs.earthdata.nasa.gov/) account.

```bash
git clone <this repo> dewsat-soil-moisture && cd dewsat-soil-moisture
git clone https://github.com/AerospaceNU/Atlas.git data-pipeline/Atlas    # team tool for NASA downloads
echo "EARTHDATA_TOKEN=<your token>" > data-pipeline/Atlas/.env           # never commit this file
cd data-pipeline/Atlas && uv sync && cd ../..                              # Atlas environment (uv: docs.astral.sh/uv)
python -m venv model/.venv-train && model/.venv-train/bin/pip install -r model/requirements.txt
```

## Get the data (not in git)

```bash
cd data-pipeline/practice
uv run --project ../Atlas python -u pull_regions.py --regions regions.csv --steps fldas,bands,vci --parallel 6 --workers 6
uv run --project ../Atlas python -u pull_regions.py --regions regions_new.csv --steps fldas,bands,vci --parallel 6 --workers 6
uv run --project ../Atlas python pull_regions.py --regions regions_new.csv --status    # what is done / left
```

- Everything lands in `data-pipeline/practice/regions/<name>/`. A stopped pull resumes where it left off.
- About 9 hours for all 102 regions. Run it in `tmux` or `screen`, and watch it with `bash watch_pull.sh` in a second
  window. If memory gets low or HTTP 403/429 errors climb, stop it and restart with a lower `--parallel`.
- Pixels are read at 120 m instead of 30 m. This was checked against 30 m on two regions (correlation 0.9999), and it
  is much faster.
- Optional, for the HLS 2016+ setup only (it scored worse; see the report): add the `nasa` step (needs
  `requirements-nasa.txt`), then run `python merge_hls_years.py regions.csv regions_new.csv`.

## Train and evaluate

```bash
cd model
bash run_sweeps.sh                          # V1, V2, V3 x 5 settings x 3 seeds, ~45 min on 6 CPU cores
.venv-train/bin/python train.py --inputs satellite_anomaly --hidden 16 --window 12 --seeds 3   # just the chosen V2
```

Weekly vs monthly (NLDAS-2 answer key; the download takes ~1 hour and needs `curl`, training ~1 hour):

```bash
cd data-pipeline/practice
python -m venv .venv && .venv/bin/pip install -r requirements-nasa.txt     # rasterio, to read the NLDAS files
.venv/bin/python nldas_daily.py --start 2001-01-01 --end 2025-12-31
cd ../../model && bash run_nldas.sh                                        # writes report/weekly/weekly.md
```

Each run writes `runs_<setup>_<inputs>/<name>/` with `summary.txt` (scores against every baseline), `scores.json`,
`predictions_test.csv`, checkpoints and charts. `make_results.py` writes the tables to `results/results.md`.

The chosen models are in `model/models/` (`weekly_final/` is the final weekly model, picked with 2023-only validation). Their `config.json` holds the settings and the input scaling needed to
use them on new data.

## The versions

| | Inputs (all from the satellite) | What the model predicts |
|---|---|---|
| V1 | NDVI, NDVI anomaly, VCI, has-photo flag, season, latitude/longitude | soil moisture itself |
| V2 | same as V1 | soil moisture minus the cell's normal for the month; the stored normal map is added back after |
| V3 | NDVI, NDMI, NDWI and their anomalies (SWIR bands), has-photo, season, location | same as V2 |

The normal map is each cell's average FLDAS soil moisture per calendar month, 2001–2022. It is built on the ground,
shapes only the training labels, and is never a model input.

## How it is scored (please keep these rules)

- Test regions are never trained on, and test years (2024+) are never trained on. 2023 is a gap year.
- Every choice (settings, data setup) is made on the validation regions, never on test.
- Always report persistence, the seasonal average and the normal map alongside the model.
- FLDAS rain or temperature must never be an input: the satellite cannot measure them.

## Acknowledgments

Data: NASA FLDAS and HLS (LP DAAC), via Microsoft Planetary Computer and NASA Earthdata. Downloads use the Northeastern
Aerospace Atlas tool.

Parts of this code and analysis were developed with the help of an AI assistant (Anthropic's Claude). All results were
checked by the author.
