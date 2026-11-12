"""audit.md, fatemeh_numbers.md and data_card.md, from the same logged files as report.md."""

from __future__ import annotations

import time

from common import OUT, SCORES, read_json, write_text
from report import ci, lab


def conclusion() -> dict:
    """Email-ready lines from the logged scores, plus the hand-written closing paragraph once it exists."""
    try:
        import conclusion as C
        return C.build()
    except Exception:  # noqa: BLE001
        return read_json(SCORES / "conclusion.json", {}) or {}


def write(d: dict) -> None:
    write_text(OUT / "audit.md", audit(d))
    write_text(OUT / "fatemeh_numbers.md", fatemeh(d))
    import data_card
    write_text(OUT / "data_card.md", data_card.card(d))


def gate_lines(d) -> list[str]:
    sc = d["scores"]
    out = []
    if not sc:
        return ["Development results not in yet."]
    for g in sc["gate"]:
        for r in g.get("rows", []):
            dd = r["dauc_within"]
            out.append(f"- {g['job']} [{r['kind']}] {lab(r['model'])} − {lab(r['comparator'])}: "
                       f"{ci(dd['all'], signed=True)} (W {ci(dd['west'], signed=True)}, E {ci(dd['east'], signed=True)}) "
                       f"→ {r['verdict']}")
    return out


def _key_models(sub, extra=()):
    from common import FAMILIES
    order = ["lr_vci_t", "lr_latest_age", "gbt_mono", "state_ewma", "lr_trend", "climatology"] + list(FAMILIES) + list(extra)
    return [m for m in order if m in sub.get("models", {})]


def _compact(sub, models, within=True) -> list[str]:
    key = "auc_within"
    rows = ["| Model | AUC all | West | East |", "|---|---|---|---|"]
    for m in models:
        r = sub["models"][m]
        rows.append(f"| {lab(m)} | {ci(r['all'][key])} | {ci(r['west'][key])} | {ci(r['east'][key])} |")
    return rows


def audit(d: dict) -> str:
    import json as _json
    sc, p5, dc = d["scores"], d["phase5"], read_json(SCORES / "data_counts.json", {}) or {}
    L = ["# DEWSat_BiLSTM_v2: self-contained summary", "",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M')} from the run's logged files (full detail: `report.md`)._", "",
         "Context: DEWSat is a Red/NIR CubeSat concept that must estimate each 0.1° cell's drought state on board. The label is "
         "FLDAS Noah 10-40 cm soil moisture below its 2001-2022 calendar-month 20th percentile. The question: does a Bi-LSTM "
         "over 12 months of HLS-derived features beat logistic regression (LR) on the current month's VCI?", ""]
    a = d["audit"]
    L += ["## Phase 0: audit of the last run (capability tests, e13e7a4)", ""]
    if a:
        L += ["- The old networks learned but overfit the training years: training AUC 0.80-0.86 vs LR 0.65-0.72; inner (other "
              "regions, same years) about 0.75; outer (other regions and years) 0.62-0.73, below LR in 4 of 5 folds.",
              "- Its inner validation shared every month with training, so early stopping could not see this; best epochs 1-7.",
              f"- Order test: shuffling the 11 earlier months changed the Bi-LSTM's within-fold AUC by "
              f"{ci(a['d']['bilstm_h32']['paired_shuffle0_minus_original_within']['diff'], signed=True)}.",
              f"- Positive control (label only history explains): LR {a['e']['auc_lr_vci_t']:.3f}, Bi-LSTM "
              f"{a['e']['auc_bilstm_h32_mean_of_seeds']:.4f}: the code can learn history.",
              "- 20 spot-checked windows correct. Other problems: pooled AUC as primary (base-rate shifts), no tuning, one shared "
              "nowcast model, VCI reference included test-period years, no months-since-observation input.", "",
              "**Fixes in this run:** year-held-out inner validation; within-fold AUC primary; 30-configuration searches per "
              "family, fold and job; dedicated models per job; searched epochs with a matching cosine schedule; development-only "
              "VCI references; explicit months-since-last-observation input; 5x more regions.", ""]
    L += ["## Data", ""]
    fz = d.get("frozen_regions") or {}
    if fz:
        L.append(f"- New regions (frozen {fz.get('frozen')}): {_json.dumps(fz.get('counts'))}; plus the paper's 102 (development).")
    if dc.get("cells"):
        c = dc["cells"]
        L.append(f"- Cells: development {c['dev']:,} ({c['dev_dropped_no_label']} dropped), fresh test {c['test']:,}; "
                 f"development cells with SWIR {c['dev_with_ndmi']:,}, with per-scene values {c['dev_with_views']:,}.")
    if dc.get("granules"):
        g = dc["granules"]
        L.append(f"- HLS granules searched: {dc['summary']['granules_searched']:,} (all v2.0: {g.get('v20', 0):,}); "
                 f"per-scene views {dc['summary']['views']:,}, view x cell rows {dc['summary']['view_cell_rows']:,}.")
    if dc.get("windows"):
        w = dc["windows"]
        L.append("- Windows (outer train / outer validation), summed over folds: " + ", ".join(
            f"{j} {sum(v['outer_train'] for k, v in w.items() if k.startswith(j)):,} / "
            f"{sum(v['outer_val'] for k, v in w.items() if k.startswith(j)):,}" for j in ("job1", "job2", "job3")) + ".")
    if dc.get("drought_rate_dev"):
        L.append(f"- Development drought rate 2016-2023: {dc['drought_rate_dev']['all']:.3f}.")
    L += ["", "## Design (committed before training)", "",
          "- 4 outer folds: region group k (balanced East/West of 98° W) x year block k (2016-17, 2018-19, 2020-21, 2022-23); "
          "12-month buffers (15 for forecasts); development data 2016-2023 only, 2024 is a buffer before the fresh test.",
          "- VCI tables and scalers per fold from training years only; inner validation = next group x one year, same buffers.",
          "- Jobs: 1 estimation now (T observed); 2 missing month (T unobserved, >= 8 of 11 observed); 3 forecast T+1..T+3 "
          "(gated on onsets); 4 = job 1 where VCI changed most.",
          "- Models: climatology, LR on VCI(T), LR latest VCI + age, LR + trend, monotone boosted trees, smoothed-VCI state; "
          "Bi-LSTM, residual Bi-LSTM (starts at the LR), GRU, LSTM, temporal CNN; 30-config searches, 10 seeds.",
          "- Primary metric: within-fold AUC, 95% intervals from 2,000 whole-region resamples. Gate: best network vs best simple, "
          "paired interval above zero and not significantly worse on either side of 98° W.", ""]
    if sc:
        L += ["## Gate outcomes (development)", ""] + gate_lines(d) + [""]
        L += ["## Development numbers (within-fold AUC)", ""]
        for key, title in (("job1", "Job 1"), ("job2", "Job 2"), ("job4_either", "Job 4")):
            sub = sc["subsets"].get(key)
            if sub:
                L += [f"**{title}** ({sub['n']:,} windows):", ""] + _compact(sub, _key_models(sub)) + [""]
        j3 = sc.get("job3_mean", {}).get("onset")
        if j3:
            L += ["**Job 3**, mean onset within-fold AUC over leads 1-3:", "", "| Model | All | West | East |", "|---|---|---|---|"]
            for m, v in j3["within"].items():
                L.append(f"| {lab(m)} | {ci(v['all'])} | {ci(v['west'])} | {ci(v['east'])} |")
            L += [""]
    ph = d.get("posthoc")
    if ph:
        L += ["## Models added after the development results (amendments 4 and 4b; development; not in the gate)", "",
              "VCI-only Bi-LSTM (jobs 1-3: only VCI, observed flag and months since a view as inputs); CNN + LR and "
              "CNN + Bi-LSTM on 23x21 VCI image tiles of each cell (job 1). Same folds, seeds and resamples.", ""]
        for job, e in ph["jobs"].items():
            L.append(f"- {e['label']}: " + "; ".join(
                f"{lab(k.split(' - ')[0])} − {lab(k.split(' - ')[1])} {ci(v['all'], signed=True)} ({e['verdicts'].get(k)})"
                for k, v in e["paired"].items() if k.split(" - ")[1] in ("lr_vci_t", "lr_latest_age", "bilstm",
                                                                         "bilstm_vci", "cnn_lr")) + ".")
        L += [""]
    if p5:
        L += ["## Fresh test (scored once; regions >= 0.5° from all development regions, 2025-01..2026-09)", ""]
        vl = conclusion().get("verdict_lines", [])
        if vl:
            L += ["**Pre-registered verdicts** (section 9 rule: interval above zero over all regions, neither side "
                  "entirely below zero):", ""] + vl + [""]
        for key, title in (("job1", "Job 1"), ("job2", "Job 2"), ("job4", "Job 4")):
            sub = p5["fresh"].get(key)
            if sub:
                L += [f"**{title}** ({sub['n']:,} windows, {sub['sides']['all']['regions']} regions):", ""] + \
                     _compact(sub, _key_models(sub, ("bilstm_vci", "cnn_lr", "cnn_bilstm"))) + [""]
                L += [f"- {lab(pp.split(' - ')[0])} − {lab(pp.split(' - ')[1])}: {ci(v['all']['dauc_within'], signed=True)} "
                      f"(West {ci(v['west']['dauc_within'], signed=True)}, East {ci(v['east']['dauc_within'], signed=True)})"
                      for pp, v in sub["paired"].items() if pp.split(" - ")[0] in ("bilstm", "resbilstm", "gru", "lstm", "tcn", "bilstm_vci", "cnn_lr", "cnn_bilstm")] + [""]
        m3 = p5["fresh"].get("job3_mean_onset")
        if m3:
            L += ["**Job 3** mean onset AUC: " + "; ".join(f"{lab(m)} {ci(v['all'])}" for m, v in m3["models"].items()), ""]
        L += _fresh_posthoc_lines(d.get("fresh_posthoc"))
        L += _fresh_labels_lines(d.get("fresh_labels_posthoc"))
        c = conclusion()
        L += ["## Realistic sampling", ""] + c.get("realistic_lines", []) + [""]
    w = d["whatif"]
    if w:
        L += ["## What-ifs", ""]
        for key, title in (("ndmi", "NDMI added (SWIR; not DEWSat's camera)"), ("vci_cell_vs_pixel", "VCI per cell vs per pixel")):
            s2 = w.get(key)
            if s2:
                L.append(f"- {title}, {s2['n']:,} windows: " + "; ".join(
                    f"{lab(pp.split(' - ')[0])} − {lab(pp.split(' - ')[1])} {ci(v['all']['dauc_within'], signed=True)}"
                    for pp, v in s2["paired"].items()) + ".")
        L += [""]
    wk = d["weekly"]
    if wk:
        s3 = wk["scores"]
        L += ["## Weekly inputs (secondary)", "", "- " + "; ".join(
            f"{lab(m)} {ci(v['all']['auc_within'])}" for m, v in s3["models"].items()) + "; difference " +
              "; ".join(ci(v['all']['dauc_within'], signed=True) for v in s3["paired"].values()) + ".", ""]
    lc = d.get("lcurve")
    if lc:
        L += ["## Learning curve (amendment 5; job 1, development; no pass/fail)", ""]
        import report_extra as X
        for axis, e in lc["axes"].items():
            for m in lc["networks"]:
                diffs = "; ".join(f"{r['size']}: {ci(r['network_minus_lr'][m]['all'], signed=True)}" for r in e["sizes"])
                ch_ = e["change_in_difference"][m]
                L.append(f"- {axis}, {lab(m)} − LR by training size ({diffs}); change second-largest to full "
                         f"{ci(ch_['second_largest_to_full'], signed=True)}: {X._lc_read(ch_).split(',')[0]}.")
        s = X.lcurve_summary(lc)
        L += ([f"- {s}"] if s else []) + [""]
    L += relu_lines(d)
    ch = d["checks"]
    if ch:
        L += ["## Checks on the new models", ""]
        ot = ch.get("order_test", {}).get("families", {})
        if ot:
            L.append("- Order test (shuffled minus original, permutation 1): " + "; ".join(
                f"{lab(f)} {ci(v['shuffled_minus_original'][0], signed=True)}" for f, v in ot.items()) + ".")
        pc = ch.get("positive_control")
        if pc:
            L.append("- Positive control AUC: " + "; ".join(f"{lab(m)} {ci(v)}" for m, v in pc["scores"].items()) + ".")
        L += [""]
    lim = read_json(SCORES / "limitations.json", {}) or {}
    if lim.get("items"):
        L += ["## Problems and limitations", ""] + [f"- {x}" for x in lim["items"]] + [""]
    c = conclusion()
    L += ["## Did the Bi-LSTM beat logistic regression?", "", c.get("paragraph", "Not decided yet: the run is not finished."), ""]
    return "\n".join(L) + "\n"


def _fresh_labels_lines(f, jobs=("job1", "job2")) -> list[str]:
    """One line per drought label: the fresh-test predictions scored against SPEI and NLDAS-2 drought (report 9a)."""
    if not f:
        return []
    import report_extra as X
    L = ["**Against other drought labels** (post-hoc, report section 9a; the same predictions scored against SPEI and "
         "NLDAS-2 drought on common windows):", ""]
    for job in jobs:
        for r in X.label_rows(f, job):
            parts = "; ".join(f"{lab(m)} − LR {ci(v['all'], signed=True)}" for m, v in r["minus_lr"].items()
                              if m in ("resbilstm", "gru", "bilstm"))
            L.append(f"- {job}, {r['label']} (drought rate {r['drought_rate']:.2f}, agrees with FLDAS {r['agree']:.0%}): "
                     f"LR AUC {ci(r['lr_auc'])}; {parts}.")
    return L + [""]


def _fresh_posthoc_lines(f) -> list[str]:
    """A compact version of report section 8a (post-hoc slices of the fresh test, after the lock)."""
    if not f:
        return []
    L = ["**Post-hoc slices of the fresh test** (after the lock; the same predictions, not a second test):", ""]
    rc = f.get("reload_check")
    if rc:
        L.append(f"- Validity: {rc['models']} saved networks reproduce their scored logits (max difference "
                 f"{rc['max_abs_logit_diff']:.1e}); parameter counts match; every training loss fell.")
    pp = (f.get("network_pairs") or {}).get("fresh_job1", {}).get("paired", {})
    if pp:
        L.append("- Networks against each other (job 1): " + "; ".join(
            f"{lab(k.split(' - ')[0])} − {lab(k.split(' - ')[1])} {ci(pp[k]['all'], signed=True)}"
            for k in ("bilstm - resbilstm", "bilstm - gru", "resbilstm - gru", "bilstm - bilstm_vci") if k in pp) + ".")
    wr, by = f.get("within_region"), f.get("by_year")
    if wr:
        mm = wr["all"]["models"]
        L.append("- Over time (pairs within a region), network − LR: " + "; ".join(
            f"{lab(m)} {ci(mm[m]['minus_lr'], signed=True)}" for m in ("bilstm", "resbilstm", "gru", "bilstm_vci") if m in mm) + ".")
    if by:
        L.append("- By year, Bi-LSTM − LR (pooled AUC): " + "; ".join(
            f"{k.replace('_', ' ')} {v['auc']['bilstm'] - v['auc']['lr_vci_t']:+.3f} (drought {v['drought_rate']:.2f})"
            for k, v in by.items()) + ".")
    dv = f.get("dev_folds")
    if dv:
        rows = sorted(dv["folds"], key=lambda r: r["training_windows"])
        L.append("- Bi-LSTM − LR by training-set size: " + "; ".join(
            f"fold {r['fold']} ({r['validation_years']}, {r['training_windows'] / 1e3:.0f}K windows) "
            f"{r['auc']['bilstm'] - r['auc']['lr_vci_t']:+.3f}" for r in rows) +
            f"; Phase 5 ({dv['phase5_training_windows'] / 1e3:.0f}K windows, fresh test) "
            f"{dv['fresh_auc']['bilstm'] - dv['fresh_auc']['lr_vci_t']:+.3f}.")
    return L + [""]


def relu_lines(d: dict, header: bool = True) -> list[str]:
    """The ReLU Bi-LSTM (DESIGN.md amendment 7; development only): the audit's section, or fatemeh_numbers' line."""
    import report_extra as X
    r = d.get("relu")
    if not r:
        return []
    L = ["## ReLU Bi-LSTM (amendment 7; job 1, development; no pass/fail)", ""] if header else []
    for job, e in r["jobs"].items():
        pr = e["paired"]
        a, b, c0 = pr["bilstm_relu - bilstm"], pr["bilstm_relu - lr_vci_t"], pr["bilstm - lr_vci_t"]
        L.append(f"- {job}: ReLU Bi-LSTM {ci(e['models']['bilstm_relu']['all'])} against Bi-LSTM "
                 f"{ci(e['models']['bilstm']['all'])}; ReLU − tanh {ci(a['all'], signed=True)} (West "
                 f"{ci(a['west'], signed=True)}, East {ci(a['east'], signed=True)}): "
                 f"{e['verdicts']['bilstm_relu - bilstm']}. ReLU − LR on VCI(T) {ci(b['all'], signed=True)}; "
                 f"Bi-LSTM − LR {ci(c0['all'], signed=True)}.")
    if header:
        s = X.relu_checks(d)
        L += ([f"- {s}"] if s else []) + [f"- Every training loss fell: {'yes' if r['all_losses_fell'] else 'no'}; "
                                          f"the fresh test was not scored (locked)."] + [""]
    return L


def fatemeh(d: dict) -> str:
    c = conclusion()
    L = ["# DEWSat: numbers for Prof. Ghoreishi", "", c.get("bottom_line", "Bottom line: pending (the run is still going)."), ""]
    if c.get("verdict_lines"):
        L += ["**Pre-registered verdicts on the fresh test:**", ""] + c["verdict_lines"] + [""]
    L += _fresh_labels_lines(d.get("fresh_labels_posthoc"), jobs=("job1",))
    p5 = d["phase5"]
    if p5:
        L += c.get("fresh_lines", []) + [""]
        L += c.get("realistic_lines", []) + [""]
    if c.get("dev_lines"):
        L += ["**Development** (4 outer folds, regions and years held out together; within-fold AUC, 95% intervals from "
              "2,000 whole-region resamples):"] + c["dev_lines"]
    rl = relu_lines(d, header=False)
    if rl:
        L += ["", "**ReLU instead of tanh in the Bi-LSTM** (amendment 7, added after every result; development only, the "
              "fresh test stays locked):"] + rl[:1]
    return "\n".join(L) + "\n"
