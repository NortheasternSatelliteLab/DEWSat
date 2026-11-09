"""data_card.md for the new dataset, in the layout of ~/Desktop/DEWSat_Fatemeh_package/data_card.md (amendment 4).
Every number is read from a run's files (sources listed at the end)."""

from __future__ import annotations

import time

from common import CACHE, DATA, LOCKED, SCORES, git_state, read_json
from report import ci, lab


def n(x):
    return "not recorded" if x is None else f"{x:,}"


def card(d: dict) -> str:
    gs = git_state()
    dc = read_json(SCORES / "data_counts.json", {}) or {}
    plan, ds, fz = d.get("plan") or {}, d.get("dataset") or {}, d.get("frozen") or d.get("frozen_regions") or {}
    reg = dc.get("regions", {})
    L = ["# DEWSat drought estimation: data card (DEWSat_BiLSTM_v2 dataset)", "",
         f"For Prof. Fatemeh Ghoreishi · {time.strftime('%Y-%m-%d')} · code `{gs['short']}` in "
         "`~/Desktop/DEWSat_BiLSTM_v2/code`. Every number comes from a file listed under Sources; \"not recorded\" means "
         "no file records it.", "",
         "**Credits.** As in the paper run: the original 102 regions, the HLS product and Fmask choices, the 30% clear-cell "
         "rule, the ~120 m pixel and the pull design are Austin Schlather's (`dewstat-northeastern@7adb8d6`); the 98° W "
         "East/West split is Ryan's (`RyanBoomer30/Atlas@227042b`). The pull code is the paper run's (`f17e700`) with SWIR1 "
         "and per-scene values added.", "",
         "## 1. Inputs and target", "",
         "| Step | What the code does |", "|---|---|",
         "| Imagery | HLS v2.0 S30 and L30: Red (B04), NIR (B8A for S30, B05 for L30), SWIR1 (B11 for S30, B06 for L30, what-if "
         "only) and Fmask. 2016–2019 from NASA LP DAAC (CMR version 2.0), 2020-01 to 2026-09 from Microsoft Planetary "
         "Computer (`hls2-s30`, `hls2-l30`); one reader for both. Granules not ending in `.v2.0` are skipped. Scenes with "
         "more than 80% cloud are not searched. |",
         "| 30 m to 120 m | Each band from the file's 4× overview (~120 m), warped with nearest neighbour onto a fixed "
         "EPSG:4326 grid per region with a whole number of pixels per 0.1° FLDAS cell. |",
         "| Clear pixel | Fmask not fill; bits 1–5 all 0 (cloud, adjacent cloud/shadow, shadow, snow/ice, water); aerosol "
         "not high; Red and NIR > 0. A scene counts for a cell's monthly composite only if at least 30% of the cell is "
         "clear. |",
         "| Monthly composite | Per pixel NDVI; same-date granules averaged, then the median over the month's distinct "
         "dates (the paper's rule, reproduced bit for bit on probe months). NDMI = (NIR − SWIR1)/(NIR + SWIR1) is "
         "composited the same way. |",
         "| Per-view values (new) | Per (sensor, MGRS tile, date) view, granules merged pixel by pixel; per cell: "
         "covered and clear fraction (no 30% rule applied), NDVI mean, population SD, 10th percentile, NDMI mean. |",
         "| VCI | Per pixel and calendar month, min/max over the reference months of each fold (development months "
         "2016–2023 outside the held-out block ± 12 months), leave-one-year-out, ≥ 3 years, range ≥ 0.05; clipped 0–100. "
         "Fresh test: all development months 2016–2023. |", "",
         "**Features** (per cell-month, over usable pixels): NDVI_mean, NDVI_std, VCI_mean, VCI_p10, valid_frac, month "
         "sin/cos, observed (usable ≥ 5% of the cell) and months since the last observation. Z-scores of the first four are "
         "fit on each model's own training windows.", "",
         "**Label.** FLDAS_NOAH01_C_GL_M v001 SoilMoi10_40cm_tavg, 2001-01 to 2026-09 (309 monthly granules, one CONUS "
         "hyperslab each through Earthdata OPeNDAP). Drought = below the numpy-linear 20th percentile of the cell's "
         "2001–2022 values for that calendar month; cells with fewer than 20 reference values in any month are dropped.", ""]
    fc = d.get("fldas_check")
    if fc:
        L += [f"The new pull reproduces the paper run's FLDAS values for the 102 regions: {fc['values']:,} values, max "
              f"absolute difference {fc['max_abs_diff']:.1e} m³/m³.", ""]
    L += ["## 2. The dataset", ""]
    if plan:
        L += [f"**Regions.** New regions are 5 × 5 FLDAS cells (0.5°) across the contiguous U.S., ≥ 75% vegetated in ESA CCI "
              f"Land Cover 2020 (300 m), all 25 cell centres inside CONUS (Census 2023 boundaries). Development regions are "
              f"≥ 0.25° from each other and the existing 102; fresh-test regions ≥ 0.5° from every other region. "
              f"{plan['candidate_corners']:,} box positions qualified; the layout holds {plan['n_dev']} development and "
              f"{plan['n_test']} fresh-test sites, of which {plan['in_batches']} form {plan['n_batches']} balanced batches "
              f"of 50 (25 West / 25 East; 12 fresh test).", ""]
    L += ["| | Development: original 102 | Development: new | Fresh test |", "|---|---|---|---|",
          f"| Regions West / East | {reg.get('dev_existing_west', 0)} / {reg.get('dev_existing_east', 0)} | "
          f"{reg.get('dev_new_west', 0)} / {reg.get('dev_new_east', 0)} | {reg.get('test_new_west', 0)} / {reg.get('test_new_east', 0)} |"]
    c = dc.get("cells", {})
    if c:
        L += [f"| Cells | {n(c.get('dev'))} (both development columns; {c.get('dev_dropped_no_label')} dropped for label history) | | {n(c.get('test'))} |"]
    L += [""]
    g = dc.get("granules", {})
    if g:
        L += [f"**HLS granules searched** (unique ids): Planetary Computer S30 {n(g.get('pc_S30'))}, L30 {n(g.get('pc_L30'))}; "
              f"NASA S30 {n(g.get('nasa_S30'))}, L30 {n(g.get('nasa_L30'))}; all v2.0: {n(g.get('v20'))}.", ""]
    v = dc.get("views", {})
    if v:
        L += [f"**Per-view record**: {n(v.get('views'))} views (L30 {n(v.get('views_L30'))}, S30 {n(v.get('views_S30'))}), "
              f"{n(v.get('view_cell_rows'))} view × cell rows, {n(v.get('view_cell_rows_clear30'))} of them ≥ 30% clear.", ""]
    if dc.get("drought_rate_dev"):
        r = dc["drought_rate_dev"]
        L += [f"**Drought base rate**, development 2016–2023: {r['all']:.4f} (West {r['west']:.4f}, East {r['east']:.4f}). "
              f"Share of labelled development cell-months observed (all-development VCI table): "
              f"{dc.get('observed_share_dev_all_table', 0):.3f}. Distinct clear dates per region-month: median "
              f"{dc['clear_dates_per_region_month']['median']:.0f} (range {dc['clear_dates_per_region_month']['min']}–"
              f"{dc['clear_dates_per_region_month']['max']}).", ""]
    if dc.get("windows"):
        L += ["**Windows per job and fold** (development):", "", "| Job, fold | Outer train | Outer validation | Inner train | Inner validation |",
              "|---|---|---|---|---|"]
        for k, w in dc["windows"].items():
            L.append(f"| {k} | {w['outer_train']:,} | {w['outer_val']:,} | {w['inner_train']:,} | {w['inner_val']:,} |")
        L += [""]
    dk = dc.get("disk", {})
    if dk:
        L += [f"**Size on disk**: new regions {dk.get('new_regions_bytes', 0) / 1e9:.1f} GB, re-extracted originals "
              f"{dk.get('reextract_bytes', 0) / 1e9:.1f} GB, copy of the paper's composites {dk.get('paper_regions_copy_bytes', 0) / 1e9:.1f} GB, "
              f"feature cache {dk.get('cache_bytes', 0) / 1e9:.1f} GB.", ""]
    tj = read_json(CACHE / "tiles" / "tiles.json")
    if tj:
        dv, ts = tj["parts"]["dev"], tj["parts"]["test"]
        L += ["### VCI image tiles (input of the CNN models added after the development results, amendment 4)", "",
              f"Per cell and month, the pixel VCI behind VCI_mean averaged onto a {tj['tile'][0]} x {tj['tile'][1]} grid of "
              "near-equal pixel bins (uint8 0-100; an empty bin, or a month the cell is unobserved, is marked empty). One "
              f"array per VCI table: development {dv['cells']:,} cells x {dv['months']} months for the 4 fold tables and the "
              f"all-development table; fresh test {ts['cells']:,} cells x {ts['months']} months (all-development table). "
              f"Built {tj['built']} in {tj['minutes']} minutes; check: each cell's mean VCI recomputed from the same pixels "
              f"matches VCI_mean within {tj['check']['worst_vci_mean_diff_this_build']:.1e} VCI units, observed flags agree "
              f"({tj['check']['observed_mismatches_this_build']} mismatches). Files `cache/tiles/` (`tiles.json`); code "
              "`code/dewsat2/tiles.py`. No labels are read.", ""]
    L += other_labels(d)
    L += ["## 3. Training, validation and results", "",
          "Four outer folds = four region groups (balanced East/West) × the year blocks 2016–17, 2018–19, 2020–21, 2022–23, "
          "12-month buffers (15 for forecasts). Inner validation holds out the next group and one year. 30-configuration "
          "random searches per network family, fold and job; 10 seeds of the chosen setting. Primary metric: within-fold AUC "
          "with 2,000 whole-region resamples. Rules: `code/design/DESIGN.md`, `gate.json`.", ""]
    sc = d.get("scores")
    if sc:
        L += ["**Development** (gate rows):", ""]
        for gg in sc["gate"]:
            for r in gg.get("rows", []):
                if r["kind"] == "gate":
                    L.append(f"- {gg['label']}: {lab(r['model'])} − {lab(r['comparator'])} "
                             f"{ci(r['dauc_within']['all'], signed=True)} → {r['verdict']}.")
        L += [""]
    if sc and d.get("phase5"):
        import report_extra as X
        rows = X.fresh_verdict_rows(d)
        lk = read_json(LOCKED / "FRESH_TEST_SCORED.lock") or {}
        if rows:
            L += [f"**Fresh test** (the fresh-test regions, 2025-01 to 2026-09, scored once, {lk.get('scored', 'date not recorded')}; "
                  "the pre-registered rows and the rule of report.md section 9):", ""]
            for r in rows:
                L.append(f"- {r['label']}, {r['kind']}: {lab(r['model'])} − {lab(r['comparator'])} "
                         f"{ci(r['diff']['all'], signed=True)} → {r['verdict']}.")
            L += [""]
    L += ["## 4. Mission scope", "", "Unchanged from the 2026-11-05 card (`~/Desktop/DEWSat_Fatemeh_package/data_card.md`, "
          "section 4); this dataset changes no mission assumption. SWIR is used only in a labelled what-if, because DEWSat's "
          "camera has Red and NIR only.", "",
          "## Sources", "",
          "- `data/plan/plan.json`, `data/plan/regions_plan.csv`, `data/plan/batches.json`: candidate count, layout, batches.",
          "- `data/pull_state.json`, `data/FROZEN.json`: pull timing, batches, frozen regions.",
          "- `scores/data_counts.json`: regions, cells, granules, views, windows, base rates, sizes.",
          "- `cache/dataset.json`: assembled cells and coverage; `logs/pull/fldas_check.json`: FLDAS agreement.",
          "- `scores/scores.json`, `scores/phase5.json`: results; `data/locked/FRESH_TEST_SCORED.lock`: when the fresh "
          "test was scored.",
          "- `scores/phase6.json`: NLDAS-2 and SPEI agreement with FLDAS, the SPEI timescale choice; "
          "`scores/fresh_labels_posthoc.json`: the fresh-test predictions scored against those labels.",
          "- Land cover: ESA CCI Land Cover 2020 v2.1.1 via Planetary Computer `esa-cci-lc` (`data/landcover/*.json`).",
          "- Boundaries: U.S. Census Bureau `cb_2023_us_state_20m` (`data/boundaries/`).",
          "- HLS v2.0: NASA LP DAAC (HLSS30/HLSL30 v2.0, CMR) and Microsoft Planetary Computer (`hls2-s30`, `hls2-l30`).",
          "- FLDAS: FLDAS_NOAH01_C_GL_M v001 (NASA GES DISC, Earthdata OPeNDAP).",
          "- NLDAS-2: NLDAS_NOAH0125_M v2.0 (NASA GES DISC, Earthdata OPeNDAP), `data/phase6/nldas/`.",
          "- SPEI: gridMET drought indices (Abatzoglou), Northwest Knowledge Network THREDDS, NetCDF Subset Service "
          "(`MET/spei/spei<timescale>.nc`), `data/phase6/spei/`.", ""]
    return "\n".join(L) + "\n"


def other_labels(d: dict) -> list[str]:
    """Section on the two other drought labels fetched after the fresh test was scored (Phase 6)."""
    p6 = d.get("phase6") or {}
    nl, sp = p6.get("nldas"), p6.get("spei")
    if not (nl and sp):
        return []
    fl = ((d.get("fresh_labels_posthoc") or {}).get("jobs") or {}).get("job1") or {}
    labs = fl.get("labels", {})
    ch = sp["chosen_timescale"]
    auc = sp["auc_minus_spei_for_fldas_flag_dev"]
    order = sorted(auc, key=lambda k: -auc[k])
    tc = sp["timescale_choice"]
    nv, nt = nl["dev"], nl["test"]
    n_series = len(list((DATA / "phase6" / "spei" / "series").glob("*.npy")))
    n_nldas = len(list((DATA / "phase6" / "nldas").glob("*.npy")))
    names = {"nldas": "NLDAS-2"}
    agree = ", ".join(f"{names.get(k, k.replace('spei_', 'SPEI-'))} {v['agree_with_fldas']:.1%} (drought rate "
                      f"{v['drought_rate']:.3f})" for k, v in labs.items() if k != "fldas")
    lo, hi = (f"{x:+.2f}".replace("-", "−") for x in sp["series"]["value_range"])
    L = ["### Other drought labels (Phase 6, fetched only after the fresh test was scored)", "",
         "Two labels from other sources, used to ask whether the fresh-test result is specific to FLDAS. No model, "
         "setting or input was chosen with them; the one choice made with SPEI (its timescale) used development cells only.", "",
         "| | NLDAS-2 soil moisture | gridMET SPEI |", "|---|---|---|",
         f"| Product | NLDAS_NOAH0125_M v2.0, `{nl['variable']}` (Noah model run with NLDAS-2 forcing) | gridMET SPEI "
         f"(Abatzoglou), {len(sp['datasets'])} timescales 14 days to 5 years |",
         "| Measures | Soil moisture in the same 10–40 cm layer as the label, from a second land-surface run with different "
         "forcing | Climatic water balance (precipitation minus reference evapotranspiration), standardized |",
         f"| Sampling | 0.125° CONUS grid, {n_nldas} monthly granules 2001-01 to 2026-09; each 0.1° cell takes the nearest "
         "NLDAS cell | One point series per region at its centre, 2016-01 to 2026-09; a month takes the last value on or "
         f"before its last day; fetched values {lo} to {hi} |",
         "| Drought | Below the cell's calendar-month 20th percentile of 2001–2022 (the FLDAS rule) | SPEI ≤ −0.84 (the "
         "standard-normal 20th percentile) |",
         f"| Coverage | Every development and fresh-test cell | Timescale choice: {tc['per_side']} West + {tc['per_side']} East "
         f"development regions (seed {tc['seed']}, {tc['dev_cells']:,} cells) × all timescales; scoring: every fresh-test "
         f"region at {ch} and the two runners-up; {n_series} series in all |",
         f"| Agreement with FLDAS, development 2016–2023 | Flags agree {nv['flag_agreement']:.1%}, Cohen's kappa "
         f"{nv['kappa_drought']:.3f}, AUC of the NLDAS percentile for the FLDAS flag {nv['auc_nldas_pct_for_fldas_flag']:.3f} "
         f"({nv['cell_months']:,} cell-months) | AUC of −SPEI for the FLDAS flag: "
         + ", ".join(f"{k} {auc[k]:.3f}" for k in order[:4]) + f" (chosen: {ch}) |",
         f"| Agreement with FLDAS, fresh test | Flags agree {nt['flag_agreement']:.1%}, kappa {nt['kappa_drought']:.3f}, AUC "
         f"{nt['auc_nldas_pct_for_fldas_flag']:.3f} ({nt['cell_months']:,} cell-months) | See the next line |", ""]
    if agree:
        L += [f"On the {fl.get('common_windows', 0):,} job-1 fresh-test windows where every label exists (FLDAS drought rate "
              f"{labs['fldas']['drought_rate']:.3f}), each label's call agrees with FLDAS's: {agree}. The scores against "
              "these labels are in report.md section 9a.", ""]
    return L
