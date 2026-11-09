"""Write report.md, audit.md, fatemeh_numbers.md and data_card.md from logged files only.

Called after every pipeline stage; sections whose stage has not run yet say so. No number is typed by hand: each
comes from a JSON file written by a run (sources listed at the end of report.md and in data_card.md)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from common import FAMILIES, LABELS, LEADS, OUT, SCORES, SIMPLE, V2, git_state, read_json, write_text

GATE = None


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def sg(x, d=3):
    if x is None:
        return "—"
    s = f"{x:+.{d}f}"
    return s.replace("-", "−")


def ci(v, d=3, signed=False):
    if not v or v[0] is None:
        return "—"
    fmt = (lambda x: sg(x, d)) if signed else (lambda x: f"{x:.{d}f}".replace("-", "−"))
    if v[1] is None:
        return fmt(v[0])
    return f"{fmt(v[0])} [{fmt(v[1])}, {fmt(v[2])}]"


def lab(m):
    for suf in ("_weekly", "_ndmi_on", "_ndmi_off", "_cell", "_pixel"):
        if m.endswith(suf) and m[: -len(suf)] in LABELS:
            return f"{LABELS[m[: -len(suf)]]} ({suf[1:].replace('_', ' ')})"
    from report_extra import PH_LABELS
    return LABELS.get(m, PH_LABELS.get(m, m))


def load_all() -> dict:
    d = {"scores": read_json(SCORES / "scores.json"), "checks": read_json(SCORES / "checks.json"),
         "whatif": read_json(SCORES / "whatif.json"), "weekly": read_json(SCORES / "weekly.json"),
         "phase5": read_json(SCORES / "phase5.json"), "p5set": read_json(SCORES / "phase5_settings.json"),
         "phase6": read_json(SCORES / "phase6.json"), "audit": read_json(V2 / "audit" / "phase0_results.json"),
         "plan": read_json(V2 / "data" / "plan" / "plan.json"), "pull": read_json(V2 / "data" / "pull_state.json"),
         "frozen": read_json(V2 / "data" / "FROZEN.json"), "frozen_regions": read_json(V2 / "data" / "FROZEN_REGIONS.json"),
         "dataset": read_json(OUT / "cache" / "dataset.json"), "fldas_check": read_json(V2 / "logs" / "pull" / "fldas_check.json"),
         "datacount": read_json(SCORES / "data_counts.json"), "pipeline": read_json(OUT / "logs" / "status_pipeline.json"),
         "gate": json.loads((V2 / "code" / "design" / "gate.json").read_text()),
         "posthoc": read_json(SCORES / "posthoc.json"), "lcurve": read_json(SCORES / "lcurve.json"),
         "fresh_posthoc": read_json(SCORES / "fresh_posthoc.json"),
         "fresh_labels_posthoc": read_json(SCORES / "fresh_labels_posthoc.json")}
    return d


# ------------------------------------------------------------------------------------------- tables
def model_table(sub: dict, models: list[str]) -> list[str]:
    rows = ["| Model | Within-fold AUC, all | West | East | Pooled AUC | Brier | ECE |", "|---|---|---|---|---|---|---|"]
    for m in models:
        r = sub["models"].get(m)
        if not r:
            continue
        rows.append(f"| {lab(m)} | {ci(r['all']['auc_within'])} | {ci(r['west']['auc_within'])} | "
                    f"{ci(r['east']['auc_within'])} | {ci(r['all']['auc_pooled'])} | {ci(r['all']['brier'], 4)} | "
                    f"{f3(r['all']['ece'])} |")
    return rows


def paired_table(sub: dict, comp: str, models: list[str]) -> list[str]:
    rows = [f"| Model − {lab(comp)} | ΔAUC within-fold, all | West | East | ΔBrier |", "|---|---|---|---|---|"]
    for m in models:
        p = sub["paired"].get(f"{m} - {comp}")
        if not p:
            continue
        rows.append(f"| {lab(m)} | {ci(p['all']['dauc_within'], signed=True)} | {ci(p['west']['dauc_within'], signed=True)} | "
                    f"{ci(p['east']['dauc_within'], signed=True)} | {ci(p['all']['dbrier'], 4, signed=True)} |")
    return rows


def gate_rows(sc: dict) -> list[str]:
    rows = ["| Job | Comparison | ΔAUC within-fold, all | West | East | Result |", "|---|---|---|---|---|---|"]
    kinds = {"gate": "best network − best simple (the gate)", "best_bilstm": "best Bi-LSTM variant − best simple",
             "vs_lr_vci_t_network": "best network − LR on VCI(T)", "vs_lr_vci_t_bilstm": "best Bi-LSTM variant − LR on VCI(T)"}
    for g in sc["gate"]:
        for r in g.get("rows", []):
            d = r["dauc_within"]
            rows.append(f"| {g['label']} | {kinds[r['kind']]}: {lab(r['model'])} − {lab(r['comparator'])} | "
                        f"{ci(d['all'], signed=True)} | {ci(d['west'], signed=True)} | {ci(d['east'], signed=True)} | "
                        f"**{r['verdict']}** |")
    return rows


def all_models(sub):
    order = list(SIMPLE) + list(FAMILIES) + ["persistence"]
    return [m for m in order if m in sub["models"]] + [m for m in sub["models"] if m not in order]


# ------------------------------------------------------------------------------------------- sections
def sec_results(d: dict) -> list[str]:
    sc = d["scores"]
    L = ["## 4. Development results (outer folds)", ""]
    if not sc:
        return L + ["Not run yet.", ""]
    L += ["Primary metric: within-fold AUC (only pairs from the same outer fold count), 95% intervals from 2,000 "
          "resamples of whole regions; paired differences use the same resamples. Networks are the mean of 10 seeds "
          "of the configuration each fold's inner split chose.", ""]
    L += ["### Gate (rules fixed in `code/design/gate.json`)", ""] + gate_rows(sc) + [""]
    sel = sc.get("selection", {})
    for job in ("job1", "job2", "job3", "job4"):
        s = sel.get(job)
        if s:
            L.append(f"- {job}: best network **{lab(s['best_network'])}**, best simple **{lab(s['best_simple'])}**, best "
                     f"Bi-LSTM variant **{lab(s['best_bilstm'])}** (development within-fold AUC: " +
                     ", ".join(f"{lab(m)} {v:.3f}" for m, v in sorted(s["values"].items(), key=lambda kv: -(kv[1] or 0))
                               if v is not None) + ")")
    L.append("")
    names = [("job1", "Job 1: estimation now (month T observed)"), ("job2", "Job 2: missing month (T unobserved, ≥ 8 of the other 11 observed)"),
             ("job4_either", "Job 4: estimation now where VCI changed most (top quartile of the 1- or 3-month change)")]
    for key, title in names:
        sub = sc["subsets"].get(key)
        if not sub:
            continue
        L += [f"### {title}", "", f"{sub['n']:,} windows, {sub['n_pos']:,} in drought, "
              f"{sub['sides']['all']['regions']} regions (West {sub['sides']['west']['regions']} / {sub['sides']['west']['n']:,} windows; "
              f"East {sub['sides']['east']['regions']} / {sub['sides']['east']['n']:,}).", ""]
        L += model_table(sub, all_models(sub)) + [""]
        comp = "lr_latest_age" if key == "job2" else "lr_vci_t"
        L += [f"Paired differences against {lab(comp)}:", ""] + paired_table(sub, comp, all_models(sub)) + [""]
        bs = sel.get("job2" if key == "job2" else ("job4" if key.startswith("job4") else "job1"), {}).get("best_simple")
        if bs and bs != comp:
            L += [f"Paired differences against the best simple model ({lab(bs)}):", ""] + \
                 paired_table(sub, bs, all_models(sub)) + [""]
        pf = ["| Model | " + " | ".join(f"Fold {k}" for k in range(1, 5)) + " |", "|---|---|---|---|---|"]
        for m in all_models(sub):
            v = sub["models"][m]["per_fold_auc"]
            pf.append(f"| {lab(m)} | " + " | ".join(f3(x) for x in v) + " |")
        L += ["AUC within each fold (point estimates):", ""] + pf + [""]
    # job 3
    j3 = sc.get("job3_mean", {})
    if j3:
        L += ["### Job 3: forecast T+1, T+2, T+3", ""]
        rows = ["| Lead | Subset | " + " | ".join(lab(m) for m in all_models(sc["subsets"]["job3_lead1_onset"])) + " |",
                "|---|---|" + "---|" * len(all_models(sc["subsets"]["job3_lead1_onset"]))]
        for h in LEADS:
            for kind in ("all", "onset"):
                sub = sc["subsets"][f"job3_lead{h}_{kind}"]
                rows.append(f"| {h} | {'all months' if kind == 'all' else 'onsets'} (n {sub['n']:,}) | " +
                            " | ".join(ci(sub["models"][m]["all"]["auc_within"]) if m in sub["models"] else "—"
                                       for m in all_models(sc["subsets"]["job3_lead1_onset"])) + " |")
        L += ["Within-fold AUC by lead (persistence uses the FLDAS flag at T, i.e. ground data):", ""] + rows + [""]
        if "onset" in j3:
            L += ["Mean onset within-fold AUC over leads 1-3 (the job-3 gate metric):", "",
                  "| Model | All | West | East |", "|---|---|---|---|"]
            for m, v in j3["onset"]["within"].items():
                L.append(f"| {lab(m)} | {ci(v['all'])} | {ci(v['west'])} | {ci(v['east'])} |")
            L += [""]
            bs = sel.get("job3", {}).get("best_simple")
            if bs:
                L += [f"Paired, mean onset metric, against {lab(bs)}:", "", "| Model | Δ all | Δ West | Δ East |", "|---|---|---|---|"]
                for p, v in j3["onset"]["paired"].items():
                    if p.endswith(f" - {bs}"):
                        L.append(f"| {lab(p.split(' - ')[0])} | {ci(v['all'], signed=True)} | {ci(v['west'], signed=True)} | {ci(v['east'], signed=True)} |")
                L += [""]
    j4 = {k: v for k, v in sc["subsets"].items() if k.startswith("job4_") and k != "job4_either"}
    if j4:
        L += ["Job 4 strata (within-fold AUC, all regions):", "", "| Stratum | n | " +
              " | ".join(lab(m) for m in ("lr_vci_t",) + tuple(FAMILIES)) + " |", "|---|---|" + "---|" * (1 + len(FAMILIES))]
        thr = sc.get("job4_thresholds", {})
        for k, v in j4.items():
            L.append(f"| {k.replace('job4_', '')} | {v['n']:,} | " + " | ".join(
                ci(v["models"][m]["all"]["auc_within"]) if m in v["models"] else "—" for m in ("lr_vci_t",) + tuple(FAMILIES)) + " |")
        L += ["", f"Thresholds (75th percentiles over pooled job-1 windows): |ΔVCI| over 1 month {thr.get('d1_q75', 0):.2f}, "
              f"over 3 months {thr.get('d3_q75', 0):.2f}.", ""]
    return L


def sec_checks(d: dict) -> list[str]:
    c = d["checks"]
    L = ["## 5. Checks on the new models (job 1)", ""]
    if not c:
        return L + ["Not run yet.", ""]
    ot = c.get("order_test")
    if ot:
        L += ["**Order test** (months T-11..T-1 shuffled within each window, 5 permutations, final 10-seed models):", "",
              "| Family | Original | Shuffled (mean of 5) | Shuffled − original (permutation 1) | Mean |Δp| |", "|---|---|---|---|---|"]
        for f, v in ot["families"].items():
            L.append(f"| {lab(f)} | {ci(v['original'])} | {sum(x[0] for x in v['shuffled']) / len(v['shuffled']):.3f} | "
                     f"{ci(v['shuffled_minus_original'][0], signed=True)} | {v['mean_abs_prob_change']:.3f} |")
        L += [""]
    pc = c.get("positive_control")
    if pc:
        L += [f"**Positive control** (fold 1; label: {pc['label']}; {pc['n_fit']:,} training and {pc['n_eval']:,} "
              f"validation windows, rate {pc['rate_eval']:.3f}):", "", "| Model | AUC | Δ vs LR on VCI(T) |", "|---|---|---|"]
        for m, v in pc["scores"].items():
            p = pc["paired"].get(f"{m} - lr_vci_t")
            L.append(f"| {lab(m)} | {ci(v)} | {ci(p, signed=True) if p else '—'} |")
        L += [""]
    return L


def sec_whatif(d: dict) -> list[str]:
    w = d["whatif"]
    L = ["## 6. What-ifs (secondary, job 1)", ""]
    if not w:
        return L + ["Not run yet.", ""]
    for key, title in (("ndmi", "NDMI added (SWIR, **not DEWSat's camera**)"), ("vci_cell_vs_pixel", "VCI per cell vs per pixel")):
        s = w.get(key)
        if not s:
            continue
        L += [f"### {title}", "", f"{s['n']:,} windows in {s['regions']} regions (same windows for every row).", ""]
        L += model_table(s, list(s["models"])) + [""]
        L += ["| Comparison | ΔAUC within-fold, all | West | East |", "|---|---|---|---|"]
        for p, v in s["paired"].items():
            a, b = p.split(" - ")
            L.append(f"| {lab(a)} − {lab(b)} | {ci(v['all']['dauc_within'], signed=True)} | "
                     f"{ci(v['west']['dauc_within'], signed=True)} | {ci(v['east']['dauc_within'], signed=True)} |")
        L += [""]
    return L


def sec_weekly(d: dict) -> list[str]:
    w = d["weekly"]
    L = ["## 7. Weekly-input experiment (secondary; amendment 3)", ""]
    if not w:
        return L + ["Not run yet.", ""]
    s = w["scores"]
    L += [f"52 weekly steps ending on the label month's last day, from single clear views (>= 30% of the cell clear); "
          f"{s['n']:,} validation cell-months with at least one clear view in the 52 weeks, "
          f"{s['sides']['all']['regions']} regions. Training windows capped at {w['max_train_windows']:,} per fold.", ""]
    L += model_table(s, list(s["models"])) + [""]
    for p, v in s["paired"].items():
        a, b = p.split(" - ")
        L.append(f"- {lab(a)} − {lab(b)}: {ci(v['all']['dauc_within'], signed=True)} (West "
                 f"{ci(v['west']['dauc_within'], signed=True)}, East {ci(v['east']['dauc_within'], signed=True)}).")
    L += ["", "| Fold | Validation windows | With a clear view | Share of weeks with a new view |", "|---|---|---|---|"]
    for k, v in w["coverage"].items():
        L.append(f"| {k} | {v['validation_windows']:,} | {v['with_a_clear_view']:,} | {v['weeks_with_new_view_share']:.3f} |")
    return L + [""]


def sec_fresh(d: dict) -> list[str]:
    p = d["phase5"]
    L = ["## 8. Fresh test (Phase 5, scored once)", ""]
    if not p:
        return L + ["Not run yet. The fresh-test labels are locked (`data/locked/LOCK.json`).", ""]
    fr = p["fresh"]
    import report_extra as X
    L += X.fresh_verdict_table(d)
    am = p.get("amendment6") or {}
    if am:
        L += ["Rows added after the development results (DESIGN.md amendment 6; not part of the gate): " +
              "; ".join(f"{j}: {', '.join(lab(m) for m in ms)}" for j, ms in am.get("extra", {}).items() if ms) +
              ". The pre-registered rows are each job's best network and best Bi-LSTM variant (section 4).", ""]
    if fr.get("job1"):
        import report_extra as X
        j1 = fr["job1"]

        def auc(m):
            return ((j1["models"].get(m) or {}).get("all") or {}).get("auc_within")

        def diff(a, b):
            return ((j1["paired"].get(f"{a} - {b}") or {}).get("all") or {}).get("dauc_within")
        L += X.two_by_two(auc, diff, "job 1, fresh test, AUC, all regions")
    for key, title in (("job1", "Job 1"), ("job2", "Job 2"), ("job4", "Job 4")):
        s = fr.get(key)
        if not s:
            continue
        L += [f"### {title}", "", f"{s['n']:,} windows, {s['n_pos']:,} in drought, {s['sides']['all']['regions']} regions.", ""]
        L += [r.replace("Within-fold AUC", "AUC") for r in model_table(s, list(s["models"]))] + [""]
        L += ["| Comparison | ΔAUC, all | West | East |", "|---|---|---|---|"]
        for pp, v in s["paired"].items():
            a, b = pp.split(" - ")
            L.append(f"| {lab(a)} − {lab(b)} | {ci(v['all']['dauc_within'], signed=True)} | {ci(v['west']['dauc_within'], signed=True)} | "
                     f"{ci(v['east']['dauc_within'], signed=True)} |")
        L += [""]
    m3 = fr.get("job3_mean_onset")
    if m3:
        L += ["### Job 3 (mean onset AUC over leads 1-3)", "", "| Model | All | West | East |", "|---|---|---|---|"]
        for m, v in m3["models"].items():
            L.append(f"| {lab(m)} | {ci(v['all'])} | {ci(v['west'])} | {ci(v['east'])} |")
        L += ["", "| Comparison | Δ all | Δ West | Δ East |", "|---|---|---|---|"]
        for pp, v in m3["paired"].items():
            a, b = pp.split(" - ")
            L.append(f"| {lab(a)} − {lab(b)} | {ci(v['all'], signed=True)} | {ci(v['west'], signed=True)} | {ci(v['east'], signed=True)} |")
        L += [""]
    L += sec_realistic(p)
    return L


def sec_realistic(p: dict) -> list[str]:
    L = ["### Realistic sampling (fresh test, per-scene values)", ""]
    cov = p.get("coverage")
    if cov:
        L += [f"Share of fresh-test cell-months (2025-01..2026-09, labelled; {cov['total_cell_months']:,}) with a fresh "
              f"clear view: full HLS record (monthly composite) {cov['full_record_composite']:.1%}; " +
              "; ".join(f"one pass every {N} days {cov[f'N{N}']['share_mean']:.1%} (starts: " +
                        ", ".join(f"{x:.1%}" for x in cov[f"N{N}"]["share_by_start"]) + ")" for N in (7, 16, 32)) + ".", ""]
    for job, r in p.get("realistic", {}).items():
        for ver, title in (("composite_trained", "trained on composites"), ("thin_trained", "trained on development regions thinned the same way")):
            v = r.get(ver)
            if not v:
                continue
            L += [f"**{job}, models {title}** (AUC per start; windows pooled over starts below):", "",
                  "| Set | n | " + " | ".join(lab(m) for m in next(iter(v.values()))["models"]) + " |",
                  "|---|---|" + "---|" * len(next(iter(v.values()))["models"])]
            for nm, s in v.items():
                L.append(f"| {nm} | {s['n']:,} | " + " | ".join(ci(s["models"][m]["all"]["auc_within"]) if m in s["models"] else "—"
                                                                 for m in next(iter(v.values()))["models"]) + " |")
            L += [""]
    return L


def sec_data(d: dict) -> list[str]:
    L = ["## 2. Data (Phase 1)", ""]
    plan, pull, fz, ds = d["plan"], d["pull"], d["frozen"] or d["frozen_regions"], d["dataset"]
    if plan:
        L += [f"- Plan: {plan['in_batches']} new regions in {plan['n_batches']} balanced batches of 50 (25 West / 25 East, "
              f"each side 6 fresh-test + 19 development), from {plan['candidate_corners']:,} candidate box positions "
              f"(ESA CCI 2020 vegetated >= 75%, all cell centres in CONUS); {plan['not_in_any_batch']} further planned "
              "regions could not form another balanced batch.", ""]
    if pull:
        c = pull.get("counts", {})
        L += [f"- Pull started {pull['started']}; {c.get('elapsed_h', '?')} h so far; batches: " +
              ", ".join(f"{b}: {v.get('state')}" for b, v in sorted(pull.get("batches", {}).items(), key=lambda kv: int(kv[0]))) + ".", ""]
    if fz:
        L += [f"- Frozen: {json.dumps(fz.get('counts', {}))}.", ""]
    if ds and "dev" in ds:
        L += [f"- Development: {len(ds['dev']['regions'])} regions, {ds['dev']['n_cells']:,} cells "
              f"({ds['dev']['cells_dropped_no_label']} dropped for missing label history); cells with SWIR/NDMI "
              f"{ds['dev']['cells_with_ndmi']:,}, with per-scene values {ds['dev']['cells_with_views']:,}.", ""]
        if "test" in ds:
            L += [f"- Fresh test: {len(ds['test']['regions'])} regions, {ds['test']['n_cells']:,} cells.", ""]
    if d["fldas_check"]:
        fc = d["fldas_check"]
        L += [f"- FLDAS labels reproduce the paper run's for the 102 regions: {fc['values']:,} values, max abs difference "
              f"{fc['max_abs_diff']:.1e}.", ""]
    dc = d["datacount"]
    if dc:
        L += ["- Scenes and views: " + json.dumps(dc.get("summary", {})) + ".", ""]
    return L


def sec_phase0(d: dict) -> list[str]:
    a = d["audit"]
    L = ["## 1. Phase 0: folders, teammates' repositories, audit of the last run", "",
         "- Folders: see `FOLDERS.md` (nothing was changed in them).",
         "- Teammates' repositories (cloned in `external/`): see `DECISIONS.md` #11-#12. No teammate data fits this "
         "setup (none ships data in git; their pipelines use station or NLDAS labels, Earth Engine chips, or FLDAS "
         "climatology as an input).", ""]
    if a:
        L += ["- Audit (details `audit/phase0_audit.md`): the old networks learn (training AUC well above LR) but "
              "overfit the training years; the inner set shared every month with training, so it could not see it. "
              f"Positive control with the old code: LR {a['e']['auc_lr_vci_t']:.3f}, Bi-LSTM "
              f"{a['e']['auc_bilstm_h32_mean_of_seeds']:.4f}. Order test: shuffling the earlier months changed the old "
              f"Bi-LSTM's within-fold AUC by {ci(a['d']['bilstm_h32']['paired_shuffle0_minus_original_within']['diff'], signed=True)}.", ""]
    return L


def write_all() -> None:
    d = load_all()
    gs = git_state()
    head = ["# DEWSat_BiLSTM_v2: report", "",
            f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `code/dewsat2/report.py` at commit `{gs['short']}`"
            f"{' (uncommitted changes)' if gs['dirty'] else ''}. Every number is read from a logged run._", ""]
    import report_extra as X
    body = head + sec_answer(d) + sec_phase0(d) + sec_data(d) + sec_design() + sec_results(d) + X.sec_posthoc(d) + \
        sec_checks(d) + sec_whatif(d) + sec_weekly(d) + X.sec_lcurve(d) + sec_fresh(d) + X.sec_fresh_posthoc(d) + \
        X.sec_sizes(d) + \
        sec_extras(d) + X.sec_fresh_labels(d) + sec_limits(d) + sec_sources()
    write_text(OUT / "report.md", "\n".join(body) + "\n")
    try:
        import report_short
        report_short.write(d)
    except Exception as exc:  # noqa: BLE001
        write_text(OUT / "logs" / "report_short_error.txt", f"{type(exc).__name__}: {exc}\n")


def sec_design() -> list[str]:
    return ["## 3. Design (Phase 2, committed before training)", "",
            "See `code/design/DESIGN.md` and `code/design/gate.json` (commit `f5e92aa` and any dated amendments). "
            "Four outer folds = four region groups (balanced East/West) x the year blocks 2016-17, 2018-19, 2020-21, "
            "2022-23, 12-month buffers (15 for forecasts); VCI tables and scalers per fold from training data only; inner "
            "validation holds out the next region group and one year, with the same buffers; within-fold AUC primary.", ""]


def sec_answer(d: dict) -> list[str]:
    sc = d["scores"]
    L = ["## 0. Answer", ""]
    if not sc:
        return L + ["Development results not in yet.", ""]
    g1 = next((g for g in sc["gate"] if g["job"] == "job1"), None)
    if g1 and g1.get("rows"):
        for r in g1["rows"]:
            if r["kind"] in ("vs_lr_vci_t_bilstm", "gate"):
                L.append(f"- Development, job 1, {lab(r['model'])} − {lab(r['comparator'])}: within-fold ΔAUC "
                         f"{ci(r['dauc_within']['all'], signed=True)} → **{r['verdict']}**.")
    p5 = d["phase5"]
    if p5 and p5["fresh"].get("job1"):
        import report_extra as X
        for r in X.fresh_verdict_rows(d):
            if r["job"] == "job1":
                L.append(f"- **Fresh test (pre-registered), job 1, {r['kind']}:** {lab(r['model'])} − {lab(r['comparator'])} "
                         f"ΔAUC {ci(r['diff']['all'], signed=True)} (West {ci(r['diff']['west'], signed=True)}, East "
                         f"{ci(r['diff']['east'], signed=True)}) → **{r['verdict']}**.")
        s = p5["fresh"]["job1"]
        added = ((d.get("p5set") or {}).get("models", {}).get("job1") or {}).get("extra", [])
        for a in added:
            v = s["paired"].get(f"{a} - lr_vci_t")
            if v:
                L.append(f"- Fresh test, job 1, {lab(a)} − {lab('lr_vci_t')} (added after the development results; not "
                         f"in the gate): ΔAUC {ci(v['all']['dauc_within'], signed=True)} (West "
                         f"{ci(v['west']['dauc_within'], signed=True)}, East {ci(v['east']['dauc_within'], signed=True)}).")
        L += ["", "Development and the fresh test disagree; section 8a shows why (the networks' gap to LR follows the "
              "training-set size) and section 10 lists the limits of a single test period."]
    return L + [""]


def sec_extras(d: dict) -> list[str]:
    p = d["phase6"]
    L = ["## 9. Extras (Phase 6)", ""]
    if not p:
        return L + ["Not run yet.", ""]
    for k, v in p.items():
        if isinstance(v, dict) and "lines" in v:
            L += [f"### {v.get('title', k)}", ""] + v["lines"] + [""]
    return L


def sec_limits(d: dict) -> list[str]:
    lim = read_json(SCORES / "limitations.json", {}) or {}
    L = ["## 10. Problems and limitations", ""]
    for x in lim.get("items", []):
        L.append(f"- {x}")
    return L + [""]


def sec_sources() -> list[str]:
    return ["## 11. Sources of the numbers", "",
            "`scores/scores.json` (development), `scores/checks.json`, `scores/whatif.json`, `scores/weekly.json`, "
            "`scores/posthoc.json` (post-hoc models, development), `scores/lcurve.json` (learning curve), "
            "`scores/phase5.json` and `scores/phase5_settings.json` (fresh test), `scores/fresh_posthoc.json` (post-hoc "
            "slices of the fresh test), `scores/fresh_labels_posthoc.json` (fresh test against other labels), `scores/phase6.json`, "
            "`audit/phase0_results.json`, `data/plan/plan.json`, `data/pull_state.json`, `data/FROZEN.json`, "
            "`cache/dataset.json`, `logs/pull/fldas_check.json`, `scores/data_counts.json`; per-run metadata under "
            "`runs/` (configuration, seed, device, commit, timings).", ""]


def _unused():
    return Path
