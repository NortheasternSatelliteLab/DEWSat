"""Build a paper-results folder from the published score files of the DEWSat v2 study.

Every number is read from results/scores/*.json in ~/Desktop/DEWSat_BiLSTM_github, through the repository's own
export_results.py builders at full precision or straight from the JSON, and rounded once (3 decimals); the count and
weight files of the Bi-LSTM's copies come from its saved runs in runs/phase5. Nothing is typed in.

The paper's model is the plain Bi-LSTM, so the claims, figures 1 and 4 and tables T4 and T5 lead with it, and the
Residual Bi-LSTM and the GRU follow as the pre-registered networks. Figures 2, 3, 5 and 6 are the repository's figures
redrawn without their title, subtitle and source lines; figures 1 and 4 are drawn here in the same style, because the
repository's versions put the Residual Bi-LSTM first and take no order or highlight option.

    python -I -B build_paper_results.py <new output folder>
"""
from __future__ import annotations

import csv
import html
import json
import shutil
import sys
from pathlib import Path

REPO = Path("/Users/eshan/Desktop/DEWSat_BiLSTM_github")
OUT = Path(sys.argv[1])
if OUT.exists():
    raise SystemExit(f"{OUT} already exists; not overwriting it")
sys.path.insert(0, str(REPO / "code" / "dewsat2"))
import export_results as ex  # noqa: E402

# The repository's own table builders, captured at full precision instead of written out.
CAP: dict[str, tuple[list, list]] = {}
ex.r4 = lambda x: None if x is None else float(x)
ex.table = lambda name, what, source, header, rows: CAP.__setitem__(name, (header, rows))

L = ex.load
sc, ph, rl, p5 = L("scores.json"), L("posthoc.json"), L("relu.json"), L("phase5.json")
fp, fl, lc = L("fresh_posthoc.json"), L("fresh_labels_posthoc.json"), L("lcurve.json")
ck, wi, wk = L("checks.json"), L("whatif.json"), L("weekly.json")
dc, p6 = L("data_counts.json"), L("phase6.json")

# SPEI-12 (gridMET 1y), scored later into the v2 working folder (in the GitHub copy since 2026-11-12). When its run used the
# same windows and reproduced every published row exactly, it replaces fresh_labels_posthoc.json here (one more label).
S12_PATH = Path("/Users/eshan/Desktop/DEWSat_BiLSTM_v2/scores/fresh_spei12_posthoc.json")
S12, S12_SAME = None, False
if S12_PATH.exists():
    S12 = json.loads(S12_PATH.read_text())
    S12_SAME = all(c["common_windows"] == c["published_common_windows"] and all(c["published_rows_identical"].values())
                   for c in S12["check_against_fresh_labels_posthoc"].values())
    if S12_SAME:
        fl = S12
ex.fresh_tables(sc, p5)
ex.development_tables(sc, ph, rl)
ex.fresh_diagnostics(fp)
pts = ex.training_size(fp)
ex.learning_curve(lc)
ex.other_labels(fl)
ex.revisit(p5)
ex.relu(rl)
ex.checks(ck, wi, wk, ph, p5)
ex.model_sizes()


def rows(name: str) -> list[dict]:
    header, rs = CAP[name]
    return [dict(zip(header, r)) for r in rs]


def a3(x) -> str:
    return "" if x is None else f"{x:.3f}"


def s3(x) -> str:
    return "" if x is None else f"{'−' if x < 0 else '+'}{abs(x):.3f}"


def ci(v, signed: bool) -> str:
    if not v or v[0] is None:
        return ""
    f = s3 if signed else a3
    return f"{f(v[0])} [{f(v[1])}, {f(v[2])}]"


def tri(r: dict, key: str) -> list:
    return [r[key], r[key + "_lo"], r[key + "_hi"]]


def pct(x) -> str:
    return f"{100 * x:.1f}%"


def est(v):
    """[estimate, lo, hi] from either a list or a {'auc_within': [...]} entry."""
    return v["auc_within"] if isinstance(v, dict) else v


TABLES: list[tuple[str, str, str, list[dict]]] = []   # (file stem, title, source, rows)
R = dc["regions"]
N_TEST = R["test_new_west"] + R["test_new_east"]
N_ORIG = R["dev_existing_west"] + R["dev_existing_east"]
N_DEV = N_ORIG + R["dev_new_west"] + R["dev_new_east"]

# T1. Fresh test, job 1: every model.
e1 = p5["fresh"]["job1"]
sel = sc["selection"]["job1"]
ROLE = {sel["best_network"]: "pre-registered: best network",
        sel["best_bilstm"]: "pre-registered: best Bi-LSTM variant",
        sel["best_simple"]: "pre-registered: best simple model (gate comparator)",
        "lr_vci_t": "pre-registered comparator"}
ADDED = {"bilstm", "bilstm_vci", "cnn_lr", "cnn_bilstm"}          # DESIGN.md amendment 6, after development
t1 = []
for k, v in sorted(e1["models"].items(), key=lambda kv: -kv[1]["all"]["auc_within"][0]):
    if k == "lr_latest_age":           # the same model as LR on VCI(T) when month T is observed
        continue
    pr = e1["paired"].get(f"{k} - lr_vci_t")
    t1.append({"Model": ex.lab(k),
               "Role": ROLE.get(k, "added after development" if k in ADDED else "baseline"),
               "AUC [95% CI]": ci(v["all"]["auc_within"], False),
               "Minus LR on VCI(T) [95% CI]": ci(pr["all"]["dauc_within"], True) if pr else "",
               "Brier score": a3(v["all"]["brier"][0]),
               "ECE": a3(v["all"]["ece"]),
               "Mean predicted probability": a3(v["all"]["mean_pred"])})
TABLES.append(("T1_fresh_test_job1",
               f"Fresh test, job 1 (estimate this month's drought): {N_TEST} regions, {e1['n']:,} windows, "
               f"{e1['n_pos']:,} in drought ({pct(e1['n_pos'] / e1['n'])})",
               "phase5.json → fresh.job1", t1))

# T2. Development, job 1: every model (4 outer folds, within-fold AUC).
t2 = [{"Model": r["model"], "Kind": r["kind"], "Within-fold AUC [95% CI]": ci(tri(r, "auc"), False),
       "West": ci(tri(r, "auc_west"), False), "East": ci(tri(r, "auc_east"), False)}
      for r in rows("development_auc_job1.csv")]
dev_win = sum(dc["windows"][f"job1_fold{i}"]["outer_val"] for i in range(1, 5))
TABLES.append(("T2_development_job1",
               f"Development, job 1: {N_DEV} regions, 4 outer folds (regions and years held out together), "
               f"{dev_win:,} scored windows, {pct(sc['subsets']['job1']['n_pos'] / sc['subsets']['job1']['n'])} in drought",
               "scores.json → subsets.job1.models; posthoc.json; relu.json", t2))

# T3. Pre-registered verdicts, development against the fresh test, all four jobs.
dev = {(r["job"], r["model"], r["comparator"]): r for r in rows("development_gate.csv")}
t3 = []
for r in rows("fresh_test_verdicts.csv"):
    d = dev[(r["job"], r["model"], r["comparator"])]
    t3.append({"Job": r["job_label"], "Pre-registered row": r["row"],
               "Comparison": f"{r['model']} − {r['comparator']}",
               "Development [95% CI]": ci(tri(d, "diff"), True), "Development verdict": d["verdict"],
               "Fresh test [95% CI]": ci(tri(r, "diff"), True),
               "Fresh test, West": ci(tri(r, "diff_west"), True), "Fresh test, East": ci(tri(r, "diff_east"), True),
               "Fresh-test verdict": r["verdict"]})
TABLES.append(("T3_preregistered_verdicts",
               "Pre-registered comparisons, development against the fresh test (AUC differences; passed = interval "
               "above zero overall and neither side's interval entirely below zero)",
               "scores.json → gate; phase5.json → fresh", t3))

# T4. Fresh-test predictions judged by other drought labels (post hoc).
LABN = {"fldas": "FLDAS soil moisture (training label)", "nldas": "NLDAS-2 soil moisture",
        "spei_90d": "SPEI, 90 days", "spei_180d": "SPEI, 180 days", "spei_270d": "SPEI, 270 days",
        "spei_1y": "SPEI, 1 year (SPEI-12)"}
ol = {r["label"]: r for r in rows("other_labels.csv") if r["job"] == "job1"}
t4 = [{"Drought label": LABN[k], "Drought rate": pct(ol[k]["drought_rate"]),
       "Flags agree with FLDAS": pct(ol[k]["agree_with_fldas"]),
       "LR on VCI(T) AUC [95% CI]": ci(tri(ol[k], "lr_auc"), False),
       "Bi-LSTM − LR [95% CI]": ci(tri(ol[k], "bilstm_minus_lr"), True),
       "Residual Bi-LSTM − LR [95% CI]": ci(tri(ol[k], "resbilstm_minus_lr"), True),
       "GRU − LR [95% CI]": ci(tri(ol[k], "gru_minus_lr"), True)} for k in LABN if k in ol]
TABLES.append(("T4_other_labels_job1",
               f"Fresh-test predictions (job 1) judged by other drought labels, scored after the lock: "
               f"{ol['fldas']['common_windows']:,} windows common to all labels",
               "fresh_labels_posthoc.json → jobs.job1.labels"
               + ("; SPEI-12 row: DEWSat_BiLSTM_v2/scores/fresh_spei12_posthoc.json" if "spei_1y" in ol else ""),
               t4))
if S12 is not None and not S12_SAME:      # SPEI-12 on its own windows: a separate table, nothing mixed into T4
    e12 = S12["jobs"]["job1"]
    lr12 = e12["lr"]
    t4b = [{"Drought label": LABN[k], "Drought rate": pct(e12["labels"][k]["drought_rate"]),
            "Flags agree with FLDAS": pct(e12["labels"][k]["agree_with_fldas"]),
            "LR on VCI(T) AUC [95% CI]": ci(e12["labels"][k]["auc"][lr12], False),
            **{f"{n} − LR [95% CI]": ci(e12["labels"][k]["paired"][f"{m} - {lr12}"]["all"], True)
               for m, n in (("bilstm", "Bi-LSTM"), ("resbilstm", "Residual Bi-LSTM"), ("gru", "GRU"))}}
           for k in ("fldas", "spei_1y")]
    TABLES.append(("T4b_spei12_job1", f"Fresh-test predictions (job 1) judged by SPEI-12, scored after the lock: "
                   f"{e12['common_windows']:,} windows with every label", "DEWSat_BiLSTM_v2/scores/"
                   "fresh_spei12_posthoc.json → jobs.job1.labels", t4b))

# T5. Revisit: single clear views every 7, 16 or 32 days (models trained on monthly composites).
cov = p5["coverage"]


def pair1(m):
    return ci(e1["paired"][f"{m} - lr_vci_t"]["all"]["dauc_within"], True)


t5 = [{"One clear view every": "full HLS record (monthly composites)", "Start": "",
       "Cell-months with a fresh view": pct(cov["full_record_composite"]),
       "LR on VCI(T) AUC [95% CI]": ci(e1["models"]["lr_vci_t"]["all"]["auc_within"], False),
       "Bi-LSTM − LR [95% CI]": pair1("bilstm"), "Residual Bi-LSTM − LR [95% CI]": pair1("resbilstm"),
       "GRU − LR [95% CI]": pair1("gru")}]
REV = [r for r in rows("revisit.csv") if r["trained_on"] == "composite"]
for r in REV:
    t5.append({"One clear view every": f"{r['revisit_days']} days", "Start": str(r["start"]),
               "Cell-months with a fresh view": pct(r["share_seen"]),
               "LR on VCI(T) AUC [95% CI]": ci(tri(r, "lr_auc"), False),
               "Bi-LSTM − LR [95% CI]": ci(tri(r, "bilstm_minus_lr"), True),
               "Residual Bi-LSTM − LR [95% CI]": ci(tri(r, "resbilstm_minus_lr"), True),
               "GRU − LR [95% CI]": ci(tri(r, "gru_minus_lr"), True)})
TABLES.append(("T5_revisit_job1",
               "Fresh test, job 1, sampled like a single satellite: one clear view every 7, 16 or 32 days "
               "(three start dates each); models trained on monthly composites",
               "phase5.json → realistic.job1.composite_trained, coverage", t5))

# T6. Model size, job 1 networks.
SIZES = {r["model"]: r for r in rows("model_sizes.csv") if r["job"] == "job1"}
t6 = [{"Model": r["model"], "Setting": r["setting"], "Parameters": f"{r['parameters']:,}",
       "Multiply-adds per prediction": f"{r['multiply_adds']:,}",
       "Weights, float32 (KiB)": f"{r['model_pt_kib']:.1f}"}
      for r in sorted(SIZES.values(), key=lambda r: r["multiply_adds"])]
TABLES.append(("T6_model_size_job1",
               "Job 1 networks as trained for the fresh test: parameters, multiply-adds per prediction (one 12-month "
               "window), weight file size (float32, no compression)",
               "runs/phase5/composite/job1/<model>/seed42/meta.json and model.pt", t6))

# T7. Design checks (ablations).
iv = {r["pair"]: r for r in rows("image_vs_cell_average.csv")}
npair = {(r["job"], r["comparison"]): r for r in rows("fresh_network_pairs.csv")}
ww, rr = rows("whatifs_and_weekly.csv"), rows("relu.csv")
lc_rows, (lch_h, lch_rows) = CAP["learning_curve.csv"][1], CAP["learning_curve_changes.csv"]


def ivr(pair):
    r = iv[pair]
    return [r["development"], r["dev_lo"], r["dev_hi"]], [r["fresh_test"], r["fresh_lo"], r["fresh_hi"]]


def wrow(exp, start):
    return next(tri(r, "diff") for r in ww if r["experiment"] == exp and r["comparison"].startswith(start))


def lchange(axis, net):
    return next([r[2], r[3], r[4]] for r in lch_rows if r[0] == axis and r[1] == net)


yrs = sorted({r[1] for r in lc_rows if r[0] == "years"})
regs = sorted({r[1] for r in lc_rows if r[0] == "regions"})
CHECKS = [
    ("Image tiles (CNN) before the 12-month model", "CNN + Bi-LSTM − VCI-only Bi-LSTM", *ivr("cnn_bilstm - bilstm_vci")),
    ("Image tiles (CNN) for the month-T model", "CNN + LR − LR on VCI(T)", *ivr("cnn_lr - lr_vci_t")),
    ("12 months of history instead of month T only (VCI cell average)", "VCI-only Bi-LSTM − LR on VCI(T)",
     *ivr("bilstm_vci - lr_vci_t")),
    ("All inputs instead of VCI only", "Bi-LSTM − VCI-only Bi-LSTM", None,
     tri(npair[("job1", "Bi-LSTM − VCI-only Bi-LSTM")], "diff")),
    ("Full-input Bi-LSTM against the CNN version", "Bi-LSTM − CNN + Bi-LSTM", None,
     tri(npair[("job1", "Bi-LSTM − CNN + Bi-LSTM")], "diff")),
    ("Add NDMI (needs SWIR, not on DEWSat's camera)", "GRU with NDMI − GRU without",
     wrow("NDMI added (SWIR)", "gru_ndmi_on"), None),
    ("VCI per 0.1° cell instead of per pixel", "GRU, per-cell VCI − per-pixel VCI",
     wrow("VCI per cell against per pixel", "gru_cell"), None),
    ("52 weekly steps instead of 12 monthly", "GRU − LR on latest VCI + age, both weekly",
     wrow("Weekly inputs (52 weekly steps)", "gru_weekly"), None),
    ("ReLU instead of tanh in the Bi-LSTM", "ReLU Bi-LSTM − Bi-LSTM",
     next(tri(r, "diff") for r in rr if r["job"] == "job1" and r["comparison"] == "ReLU Bi-LSTM − Bi-LSTM"), None),
    (f"More training years per fold ({yrs[0]} to {yrs[-1]})", "Change in Bi-LSTM − LR on VCI(T)",
     lchange("years", "Bi-LSTM"), None),
    (f"More regions per fold (1/{round(1 / regs[0])} to all)", "Change in Bi-LSTM − LR on VCI(T)",
     lchange("regions", "Bi-LSTM"), None),
]
t7 = [{"Question": q, "Comparison": c, "Development [95% CI]": ci(d, True) if d else "not run",
       "Fresh test [95% CI]": ci(f, True) if f else "not run"} for q, c, d, f in CHECKS]
TABLES.append(("T7_design_checks_job1", "Design checks, job 1: paired AUC differences",
               "posthoc.json, phase5.json, fresh_posthoc.json, whatif.json, weekly.json, relu.json, lcurve.json", t7))

# T8. Data summary.
spei_dev = p6["spei"]["auc_minus_spei_for_fldas_flag_dev"]
nl = p6["nldas"]["dev"]
cd = dc["clear_dates_per_region_month"]
t8 = [
    {"Quantity": "Development regions", "Value": f"{N_DEV} ({N_ORIG} original + {N_DEV - N_ORIG} new; West "
     f"{R['dev_existing_west'] + R['dev_new_west']}, East {R['dev_existing_east'] + R['dev_new_east']})",
     "Source": "data_counts.json → regions"},
    {"Quantity": "Fresh-test regions (new, never used in development)",
     "Value": f"{N_TEST} (West {R['test_new_west']}, East {R['test_new_east']})", "Source": "data_counts.json → regions"},
    {"Quantity": "0.1° cells: development / fresh test", "Value": f"{dc['cells']['dev']:,} / {dc['cells']['test']:,}",
     "Source": "data_counts.json → cells"},
    {"Quantity": "HLS granules searched", "Value": f"{dc['summary']['granules_searched']:,}",
     "Source": "data_counts.json → summary"},
    {"Quantity": "Clear dates per region-month: median (range)",
     "Value": f"{cd['median']:.0f} ({cd['min']}–{cd['max']})", "Source": "data_counts.json"},
    {"Quantity": "Development windows scored, job 1 (4 outer folds)", "Value": f"{dev_win:,}",
     "Source": "data_counts.json → windows"},
    {"Quantity": "Drought rate, development cell-months (all labelled)", "Value": f"{pct(dc['drought_rate_dev']['all'])} (West "
     f"{pct(dc['drought_rate_dev']['west'])}, East {pct(dc['drought_rate_dev']['east'])})",
     "Source": "data_counts.json → drought_rate_dev"},
    {"Quantity": "Fresh-test windows, job 1", "Value": f"{e1['n']:,}, {e1['n_pos']:,} in drought "
     f"({pct(e1['n_pos'] / e1['n'])})", "Source": "phase5.json → fresh.job1"},
    {"Quantity": "FLDAS drought flag against NLDAS-2, development: agreement / kappa / AUC",
     "Value": f"{pct(nl['flag_agreement'])} / {nl['kappa_drought']:.3f} / {nl['auc_nldas_pct_for_fldas_flag']:.3f}",
     "Source": "phase6.json → nldas.dev"},
    {"Quantity": "AUC of −SPEI for the FLDAS drought flag, development: 90 d / 180 d / 270 d / 1 y",
     "Value": " / ".join(f"{spei_dev[k]:.3f}" for k in ("90d", "180d", "270d", "1y")),
     "Source": "phase6.json → spei.auc_minus_spei_for_fldas_flag_dev"},
]
TABLES.append(("T8_data_summary", "Data summary", "data_counts.json, phase5.json, phase6.json", t8))

# Write the tables: one CSV each (UTF-8 with BOM so Excel shows − and °) and one HTML page for copy and paste.
(OUT / "tables").mkdir(parents=True)
for stem, title, source, rs in TABLES:
    with open(OUT / "tables" / f"{stem}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
        w.writeheader()
        w.writerows(rs)
parts = ["<!doctype html><meta charset='utf-8'><title>DEWSat paper tables</title><style>"
         "body{font:14px/1.4 -apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#111;background:#fff}"
         "table{border-collapse:collapse;margin:6px 0 28px}td,th{border:1px solid #999;padding:3px 7px;"
         "text-align:left;vertical-align:top}th{background:#eee}h2{font-size:15px;margin:22px 0 2px}"
         "p{margin:0;color:#555;font-size:12px}</style>",
         "<p>Select a table, copy it, and paste it into Google Docs. Titles and source lines are labels for you, "
         "not text for the paper.</p>"]
for stem, title, source, rs in TABLES:
    head = "".join(f"<th>{html.escape(k)}</th>" for k in rs[0])
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r.values()) + "</tr>" for r in rs)
    parts.append(f"<h2>{html.escape(stem)}: {html.escape(title)}</h2><p>Source: results/scores/"
                 f"{html.escape(source)}</p><table><tr>{head}</tr>{body}</table>")
(OUT / "tables.html").write_text("\n".join(parts), encoding="utf-8")

# Figures: the repository's six figures without title, subtitle and source (captions are the authors' to write).
FIGDIR = OUT / "figures"
ex.FIG = FIGDIR
ex.heading = lambda fig, title, subtitle, source: None
_frame = ex.frame


def _frame_no_heading(fig, left, right=0.97, bottom_in=0.75, top_in=0.95, **kw):
    return _frame(fig, left + 0.02, right, bottom_in=max(0.5, bottom_in - 0.25), top_in=max(0.2, top_in - 0.7), **kw)


def _save(fig, name, what, source):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=300)
    ex.plt.close(fig)


def fig_gap_paper():
    """Figure 1: the repository's fig_gap (same marks, colours and layout) with the paper's model first: the Bi-LSTM,
    then the two pre-registered networks, the models added after development, and the best simple model."""
    order = ["bilstm", "resbilstm", "gru", "bilstm_vci", "cnn_bilstm", "cnn_lr", "gbt_mono"]
    name = {"bilstm": "Bi-LSTM, the paper's model"}
    note = {"bilstm": "added before fresh-test scoring", "resbilstm": "pre-registered", "gru": "pre-registered",
            "gbt_mono": "best simple model"}
    rows = [(m, ph["jobs"]["job1"]["paired"][f"{m} - lr_vci_t"]["all"] if m in ("bilstm_vci", "cnn_lr", "cnn_bilstm")
             else sc["subsets"]["job1"]["paired"][f"{m} - lr_vci_t"]["all"]["dauc_within"],
             e1["paired"][f"{m} - lr_vci_t"]["all"]["dauc_within"]) for m in order]
    fig, ax = ex.plt.subplots(figsize=(8, 5.2))
    ex.frame(fig, left=0.27)
    for i, (m, dv, fr) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([dv[1], dv[2]], [y - 0.16] * 2, color=ex.GREY, lw=1.6, solid_capstyle="round")
        ax.plot(dv[0], y - 0.16, "o", ms=6.5, mfc="white", mec=ex.GREY, mew=1.6)
        ax.plot([fr[1], fr[2]], [y + 0.16] * 2, color=ex.BLUE, lw=2, solid_capstyle="round")
        ax.plot(fr[0], y + 0.16, "o", ms=7, color=ex.BLUE)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{name.get(m, ex.lab(m))}\n({note.get(m, 'added after development')})"
                        for m, _, _ in reversed(rows)], fontsize=9, color=ex.INK)
    ax.axvline(0, color=ex.ZERO, lw=1.2)
    ax.text(0.002, len(rows) - 0.45, "0 = as good as LR on VCI(T)", fontsize=8.5, color=ex.QUIET)
    ax.set_xlabel("AUC difference against LR on VCI(T), job 1", fontsize=9, color=ex.QUIET)
    ax.plot([], [], "o-", color=ex.BLUE, label="Fresh test, 2025-01 to 2026-09")
    ax.plot([], [], "o-", color=ex.GREY, mfc="white", label="Development, four folds")
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    ex.style(ax)
    ex.save(fig, "fig1_gap_to_lr_dev_vs_fresh", "Gap to LR on VCI(T) per model, development against the fresh test",
            "scores.json, posthoc.json, phase5.json")


def fig_labels_paper():
    """Figure 4: the repository's fig_labels (same marks, colours and layout) with the paper's model highlighted: the
    Bi-LSTM in blue and first in the legend, the two pre-registered networks in greys. Without the repository's
    "Phase 6's choice" line (the study's internal stage name); SPEI-12 as a sixth row when it shares the windows."""
    j = fl["jobs"]["job1"]
    names = [("fldas", "FLDAS soil moisture\n(the training label)"), ("nldas", "NLDAS-2 soil moisture"),
             ("spei_90d", "SPEI, 90 days"), ("spei_180d", "SPEI, 180 days"), ("spei_270d", "SPEI, 270 days")]
    if "spei_1y" in j["labels"]:
        names.append(("spei_1y", "SPEI, 1 year (SPEI-12)"))
    series = (("bilstm", ex.BLUE, 0.18), ("resbilstm", ex.GREY, 0.0), ("gru", ex.LIGHT, -0.18))
    fig, ax = ex.plt.subplots(figsize=(8, 5.6 if "spei_1y" in j["labels"] else 4.8))
    ex.frame(fig, left=0.25)
    for i, (k, _) in enumerate(names):
        y = len(names) - 1 - i
        for m, c, dy in series:
            v = j["labels"][k]["paired"][f"{m} - {j['lr']}"]["all"]
            ax.plot([v[1], v[2]], [y + dy] * 2, color=c, lw=1.8 if m == "bilstm" else 1.4)
            ax.plot(v[0], y + dy, "o", ms=6.5 if m == "bilstm" else 5, color=c)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}\ndrought {j['labels'][k]['drought_rate']:.0%}, "
                        f"agrees {j['labels'][k]['agree_with_fldas']:.0%}" for k, n in reversed(names)],
                       fontsize=8.5, color=ex.INK)
    ax.axvline(0, color=ex.ZERO, lw=1.2)
    for m, c, _ in series:
        ax.plot([], [], "o-", color=c, label=ex.lab(m))
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.set_xlabel("AUC difference against LR on VCI(T), fresh test, job 1", fontsize=9, color=ex.QUIET)
    ex.style(ax)
    ex.save(fig, "fig4_other_labels", "Fresh-test predictions judged by other drought labels",
            "fresh_labels_posthoc.json → jobs.job1.labels")


ex.frame, ex.save = _frame_no_heading, _save
fig_gap_paper()
ex.fig_size(pts)
ex.fig_lcurve(lc)
fig_labels_paper()
ex.fig_revisit(p5)
ex.fig_relu(rl)

# Claims in the manuscript (NortheasternSatelliteLab_DroughtMonitoring.pdf, 05/04/2026) against the results. Each
# finding leads with the plain Bi-LSTM, the paper's model; the Residual Bi-LSTM and the GRU follow as the job's
# pre-registered networks.
gate = {(r["job"], r["row"], r["model"]): r for r in rows("fresh_test_verdicts.csv")}
devg = {(r["job"], r["row"], r["model"]): r for r in rows("development_gate.csv")}
m1 = e1["models"]
eces = [m1[k]["all"]["ece"] for k in m1]
dauc = {r["model"]: r for r in rows("development_auc_job1.csv")}
j3 = p5["fresh"]["job3_mean_onset"]["models"]
fa = {r["model"]: r for r in rows("fresh_test_auc.csv") if r["job"] == "job4"}


def fr1(m: str, comp: str = "lr_vci_t", side: str = "all") -> list:
    """Fresh test, job 1: AUC difference m − comp, [estimate, lo, hi]."""
    return e1["paired"][f"{m} - {comp}"][side]["dauc_within"]


def dev(subset: str, m: str, comp: str = "lr_vci_t") -> list:
    """Development (4 outer folds): within-fold AUC difference m − comp, [estimate, lo, hi]."""
    return sc["subsets"][subset]["paired"][f"{m} - {comp}"]["all"]["dauc_within"]


# Job 1's pre-registered networks: (row against LR on VCI(T), row against boosted trees, role).
PRE1 = {"Residual Bi-LSTM": (gate[("job1", "against LR on VCI(T)", "Residual Bi-LSTM")],
                             gate[("job1", "best Bi-LSTM variant", "Residual Bi-LSTM")], "best Bi-LSTM variant"),
        "GRU": (gate[("job1", "against LR on VCI(T)", "GRU")], gate[("job1", "gate", "GRU")], "best network")}
PRE1_PASSED = all(r["verdict"] == "passed" for lr_row, bt_row, _ in PRE1.values() for r in (lr_row, bt_row))
# The three full-input networks against each other on the fresh test.
NETS3 = {c: tri(npair[("job1", c)], "diff")
         for c in ("Bi-LSTM − Residual Bi-LSTM", "Bi-LSTM − GRU", "Residual Bi-LSTM − GRU")}
NETS3_SPREAD = max(abs(v[0]) for v in NETS3.values())
# Jobs 3 and 4: the plain Bi-LSTM is job 3's pre-registered best Bi-LSTM variant; job 4's are the other two networks.
G3, D3 = gate[("job3", "best Bi-LSTM variant", "Bi-LSTM")], devg[("job3", "best_bilstm", "Bi-LSTM")]
PRE4 = [(n, gate[("job4", r, n)], devg[("job4", k, n)])
        for n, r, k in (("Residual Bi-LSTM", "best Bi-LSTM variant", "best_bilstm"), ("GRU", "gate", "gate"))]
# The scored Bi-LSTM prediction is the mean probability of its saved copies (seeds; DESIGN.md section 5, amendment 6).
COPIES = [p.parent for p in sorted((REPO / "runs" / "phase5" / "composite" / "job1" / "bilstm").glob("seed*/meta.json"))
          if "fresh" in json.loads(p.read_text())["evals"]]
assert COPIES and all(json.loads((p / "meta.json").read_text())["n_params"] == SIZES["Bi-LSTM"]["parameters"]
                      for p in COPIES)
N_COPIES = len(COPIES)
COPIES_KIB = sum((p / "model.pt").stat().st_size for p in COPIES) / 1024
MACS = SIZES["Bi-LSTM"]["multiply_adds"]


def spei12_claim() -> tuple[str, str]:
    """The SPEI-12 claim's finding and status, led by the Bi-LSTM, from the SPEI-12 score when it exists."""
    b90, b180, b270, bnl = (ci(tri(ol[k], "bilstm_minus_lr"), True)
                            for k in ("spei_90d", "spei_180d", "spei_270d", "nldas"))
    base = (f"Against SPEI at 90, 180 and 270 days, Bi-LSTM − LR is {b90}, {b180} and {b270}; against NLDAS-2 soil "
            f"moisture the gain holds: {bnl}.")
    if "spei_1y" in ol:
        r = ol["spei_1y"]
        b, d, g, lra = (tri(r, k) for k in ("bilstm_minus_lr", "resbilstm_minus_lr", "gru_minus_lr", "lr_auc"))
        n, rate, agree = ol["fldas"]["common_windows"], r["drought_rate"], r["agree_with_fldas"]
    elif S12 is not None:
        e = S12["jobs"]["job1"]
        r = e["labels"]["spei_1y"]
        b, d, g = (r["paired"][f"{m} - {e['lr']}"]["all"] for m in ("bilstm", "resbilstm", "gru"))
        lra, n, rate, agree = r["auc"][e["lr"]], e["common_windows"], r["drought_rate"], r["agree_with_fldas"]
    else:
        return (f"The predictions were never scored against SPEI-12. {base} SPEI over 1 year appears only in a "
                f"development label check (AUC {spei_dev['1y']:.3f} for the FLDAS flag).", "Not supported as written")
    e1y = S12["jobs"]["job1"]
    sd = e1y["labels"]["spei_1y"]["paired"][f"bilstm - {e1y['lr']}"]
    ws, es = ci(sd["west"], True), ci(sd["east"], True)
    sides = (f"Within each side the Bi-LSTM's difference is not distinguishable from zero (West {ws}, East {es}), so "
             f"the overall gap comes mostly from ranking West windows against East ones."
             if all(sd[s][1] <= 0 <= sd[s][2] for s in ("west", "east")) else f"West {ws}, East {es}.")
    txt = (f"Scored after the lock against SPEI-12 drought (gridMET 1-year SPEI ≤ −0.84; {pct(rate)} of {n:,} windows "
           f"in drought, flags agree with FLDAS on {pct(agree)}): LR on VCI(T) AUC {ci(lra, False)}; Bi-LSTM − LR "
           f"{ci(b, True)}. {sides} Pre-registered networks: Residual Bi-LSTM − LR {ci(d, True)}, GRU − LR "
           f"{ci(g, True)}. {base} AUC tests ranking, not calibration.")
    if b[1] > 0:
        return txt, "Supported as a ranking benchmark (post hoc); calibration not tested"
    if b[2] < 0:
        return txt, "Not supported: the gain reverses against SPEI-12"
    return txt, "Not supported: no detectable gain against SPEI-12"


def rev_above(n: int, key: str) -> int:
    """Start dates, at one clear view every n days, whose interval for key lies above zero."""
    return sum(1 for r in REV if r["revisit_days"] == n and r[key + "_lo"] > 0)


def rev_all(key: str) -> str:
    """Per revisit: the range over the start dates, and how many of them have the interval above zero."""
    out = []
    for n in (7, 16, 32):
        rs = [r for r in REV if r["revisit_days"] == n]
        out.append(f"{n} days {s3(min(r[key] for r in rs))} to {s3(max(r[key] for r in rs))} "
                   f"({rev_above(n, key)} of {len(rs)})")
    return ", ".join(out)


def rev_status() -> str:
    starts = len({r["start"] for r in REV})
    b = [rev_above(n, "bilstm_minus_lr") for n in (7, 16, 32)]
    st = ("Bi-LSTM: no detectable gain at 7, 16 or 32 days" if not any(b) else
          f"Bi-LSTM: interval above zero for {b[0]}, {b[1]} and {b[2]} of {starts} start dates at 7, 16 and 32 days")
    if all(rev_above(7, k) == starts and rev_above(32, k) == 0 for k in ("resbilstm_minus_lr", "gru_minus_lr")):
        st += "; the pre-registered networks' gain holds at 7 days and fades by 32"
    return st


CLAIMS = [
    ("2.2", "A Bi-LSTM on HLS-derived NDVI/VCI, with FLDAS labels, beats simpler classifiers (logistic regression, "
     "decision trees)",
     f"Fresh test: Bi-LSTM − LR on VCI(T) {ci(fr1('bilstm'), True)} (West {ci(fr1('bilstm', side='west'), True)}, "
     f"East {ci(fr1('bilstm', side='east'), True)}) and − boosted trees {ci(fr1('bilstm', 'gbt_mono'), True)}. For "
     f"job 1 the Bi-LSTM is not a pre-registered row: it was added to the single fresh-test scoring before any "
     f"fresh-test label was read (DESIGN.md amendment 6). The pre-registered rows "
     f"{'all passed' if PRE1_PASSED else 'did not all pass'}: "
     + "; ".join(f"{n} ({role}) − LR {ci(tri(lr_row, 'diff'), True)} and − boosted trees "
                 f"{ci(tri(bt_row, 'diff'), True)}" for n, (lr_row, bt_row, role) in PRE1.items())
     + f". The three full-input networks are within {a3(NETS3_SPREAD)} AUC of each other (Bi-LSTM − Residual Bi-LSTM "
       f"{ci(NETS3['Bi-LSTM − Residual Bi-LSTM'], True)}, Bi-LSTM − GRU {ci(NETS3['Bi-LSTM − GRU'], True)}). "
       f"Development (4 folds): Bi-LSTM − LR {ci(dev('job1', 'bilstm'), True)}, Residual Bi-LSTM − LR "
       f"{ci(tri(devg[('job1', 'vs_lr_vci_t_bilstm', 'Residual Bi-LSTM')], 'diff'), True)}, both worse; every network "
       f"trailed LR. The fresh period was drier: {pct(e1['n_pos'] / e1['n'])} of windows in drought against "
       f"{pct(sc['subsets']['job1']['n_pos'] / sc['subsets']['job1']['n'])} of development windows.",
     "Supported on the fresh test only"),
    ("2.2", "The network predicts a drought-stress probability",
     f"AUC measures ranking only. On the fresh test the probabilities run low: Bi-LSTM mean prediction "
     f"{a3(m1['bilstm']['all']['mean_pred'])} against an observed rate of {a3(e1['n_pos'] / e1['n'])}, ECE "
     f"{a3(m1['bilstm']['all']['ece'])}, Brier {a3(m1['bilstm']['all']['brier'][0])} (the same three for the "
     f"pre-registered networks: "
     + "; ".join(f"{ex.lab(k)} {a3(m1[k]['all']['mean_pred'])}, {a3(m1[k]['all']['ece'])}, "
                 f"{a3(m1[k]['all']['brier'][0])}" for k in ("resbilstm", "gru"))
     + f"). ECE of every model {a3(min(eces))}–{a3(max(eces))}; Brier of LR {a3(m1['lr_vci_t']['all']['brier'][0])}, "
       f"climatology {a3(m1['climatology']['all']['brier'][0])}.",
     "Ranking supported; calibrated probabilities not"),
    ("2.2", "Calibrated against SPEI-12 as an independent benchmark", *spei12_claim()),
    ("2.2", "Bidirectional processing improves on a conventional LSTM",
     f"Development, within-fold AUC: Bi-LSTM {ci(tri(dauc['Bi-LSTM'], 'auc'), False)} against one-direction LSTM "
     f"{ci(tri(dauc['One-direction LSTM'], 'auc'), False)} (not a paired test); Residual Bi-LSTM "
     f"{ci(tri(dauc['Residual Bi-LSTM'], 'auc'), False)}. The one-direction LSTM was not carried to the fresh test.",
     "Not supported in development; not tested on the fresh test"),
    ("2.2", "A CNN spatial step on image tiles before the temporal model",
     f"CNN + Bi-LSTM − VCI-only Bi-LSTM: {ci(ivr('cnn_bilstm - bilstm_vci')[1], True)} on the fresh test, "
     f"{ci(ivr('cnn_bilstm - bilstm_vci')[0], True)} in development. That comparison is against the VCI-only "
     f"Bi-LSTM (both read VCI alone, as image tiles or as cell averages), not the paper's model. The full-input "
     f"Bi-LSTM beats the CNN version by {ci(tri(npair[('job1', 'Bi-LSTM − CNN + Bi-LSTM')], 'diff'), True)} on the "
     f"fresh test, with {MACS:,} multiply-adds per copy against {SIZES['CNN + Bi-LSTM']['multiply_adds']:,}.",
     "Not supported"),
    ("2.2", "Distillation, pruning and INT8 quantization-aware training compress the model from about 1–2 MB to "
     "30–100 KB with under 3% accuracy loss",
     f"One copy of the trained Bi-LSTM has {SIZES['Bi-LSTM']['parameters']:,} parameters and "
     f"{SIZES['Bi-LSTM']['model_pt_kib']:.1f} KiB of float32 weights, already within 30–100 KB with no compression. "
     f"The scored prediction averages {N_COPIES} copies (seeds), so as scored it is about {COPIES_KIB:.0f} KiB of "
     f"weights ({'above' if COPIES_KIB * 1.024 > 100 else 'within'} that range) and {N_COPIES * MACS:,} multiply-adds "
     f"per prediction. Distillation, pruning and INT8 were not tested in this study (the earlier pipeline's distilled "
     f"and INT8 models were trained on synthetic data).",
     "Starting size doesn't match; compression not tested"),
    ("2.2", "Bi-LSTM and CNN inference in under 10 ms per tile on the NPU",
     f"Not measured on any flight processor. One Bi-LSTM prediction is {MACS:,} multiply-adds per copy, "
     f"{N_COPIES * MACS:,} for the {N_COPIES}-copy average that was scored.", "Not tested"),
    ("2.2", "Onboard triage gives up to 90% data reduction", "Cited from reference [18]; not part of this study.",
     "Not tested"),
    ("3.1", "The Bi-LSTM detects flash-drought anomalies earlier by modelling temporal change",
     f"Onset 1–3 months ahead (job 3), fresh test: Bi-LSTM AUC {ci(est(j3['bilstm']['all']), False)}, LR + trend "
     f"features {ci(est(j3['lr_trend']['all']), False)}; Bi-LSTM − LR + trend {ci(tri(G3, 'diff'), True)} "
     f"{G3['verdict']} (for job 3 the Bi-LSTM is the pre-registered best Bi-LSTM variant), but development "
     f"{ci(tri(D3, 'diff'), True)}, {D3['verdict']}; AUC near 0.5 is near chance. Where VCI changed most (job 4): "
     f"Bi-LSTM − LR on VCI(T) {ci(tri(fa['Bi-LSTM'], 'minus_lr'), True)} on the fresh test, "
     f"{ci(dev('job4_either', 'bilstm'), True)} in development. Pre-registered rows against boosted trees, fresh "
     f"test (development): "
     + "; ".join(f"{n} {ci(tri(g, 'diff'), True)}, {g['verdict']} ({ci(tri(d, 'diff'), True)}, {d['verdict']})"
                 for n, g, d in PRE4)
     + ".",
     "Onset: not supported. Rapid change: fresh test only"),
    ("3.2", "A single CubeSat revisits every ≈16–40+ days",
     f"Models trained on monthly composites, then given one clear view every 7, 16 or 32 days; each range spans the "
     f"three start dates, with how many have the interval above zero. Bi-LSTM − LR on VCI(T): "
     f"{rev_all('bilstm_minus_lr')}, against {ci(fr1('bilstm'), True)} with the full record. Pre-registered "
     f"networks: Residual Bi-LSTM {rev_all('resbilstm_minus_lr')}; GRU {rev_all('gru_minus_lr')}.",
     rev_status()),
]
md = ["# Manuscript claims against the v2 results", "",
      "For the authors, not text for the paper. Claims are paraphrased from NortheasternSatelliteLab_DroughtMonitoring"
      ".pdf (05/04/2026); numbers come from the same score files as the tables. AUC differences carry 95% intervals "
      "from 2,000 whole-region resamples.", "",
      "Each finding leads with the plain Bi-LSTM, the paper's model. For job 1 it is not a pre-registered row: it was "
      "added to the single fresh-test scoring before any fresh-test label was read (DESIGN.md amendment 6). The "
      "Residual Bi-LSTM (best Bi-LSTM variant) and the GRU (best network) follow as job 1's pre-registered networks.",
      "", "| Section | Claim | What the study found | Status |", "|---|---|---|---|"]
md += [f"| {s} | {c} | {f} | {st} |" for s, c, f, st in CLAIMS]
(OUT / "claims_vs_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

FILES = [
    ("tables/T1_fresh_test_job1.csv", "Main result: every model on the fresh test, job 1 (AUC, difference from LR on "
     "VCI(T), Brier score, ECE, mean predicted probability)"),
    ("tables/T2_development_job1.csv", "Every model in development, job 1 (4 outer folds)"),
    ("tables/T3_preregistered_verdicts.csv", "The pre-registered comparisons for all four jobs, development against "
     "the fresh test"),
    ("tables/T4_other_labels_job1.csv", "The fresh-test predictions judged by NLDAS-2 and by SPEI at 90, 180 and 270 "
     "days" + (" and 1 year (SPEI-12)" if "spei_1y" in ol else "")),
    *([("tables/T4b_spei12_job1.csv", "The fresh-test predictions judged by SPEI-12, on its own windows")]
      if S12 is not None and not S12_SAME else []),
    ("tables/T5_revisit_job1.csv", "One clear view every 7, 16 or 32 days instead of monthly composites"),
    ("tables/T6_model_size_job1.csv", "Parameters, multiply-adds per prediction, weight size"),
    ("tables/T7_design_checks_job1.csv", "Design checks: image tiles, 12-month history, NDMI, weekly steps, ReLU, "
     "more years against more regions"),
    ("tables/T8_data_summary.csv", "Regions, cells, windows, drought rates, label checks"),
    ("tables.html", "All eight tables on one page: open it in a browser, select a table, copy it, paste it into "
     "Google Docs"),
    ("figures/", "The study's six figures, PNG at 300 dpi (for Google Docs) and PDF, without title text"),
    ("claims_vs_results.md", "Each results claim in the manuscript against what the study found"),
    ("build_paper_results.py", "Rebuilds this folder from the score files"),
]
readme = [
    "# DEWSat paper results", "",
    f"Tables and figures for the paper from the v2 study (`~/Desktop/DEWSat_BiLSTM_v2`; published copy "
    f"`~/Desktop/DEWSat_BiLSTM_github`): {N_DEV} development regions and a fresh test in {N_TEST} new regions, "
    f"scored once. This is not the earlier 102-region run described in this folder's `report.md`.", "",
    f"The paper's model is the plain Bi-LSTM, labelled \"Bi-LSTM\" in the tables and figures. For job 1 it is not a "
    f"pre-registered row: it was added to the single fresh-test scoring before any fresh-test label was read "
    f"(`code/design/DESIGN.md`, amendment 6), so T1 marks it \"added after development\". Job 1's pre-registered "
    f"networks are the Residual Bi-LSTM (best Bi-LSTM variant) and the GRU (best network); "
    f"{'both passed on the fresh test' if PRE1_PASSED else 'T3 has their verdicts'}, and the three are within "
    f"{a3(NETS3_SPREAD)} AUC of each other there. In job 3 the plain Bi-LSTM is the pre-registered best Bi-LSTM "
    f"variant.", "",
    "Every result is read from `results/scores/*.json` and rounded once, to 3 decimals. There is no prose for the "
    "paper here, only table headings and figure labels, so the text and captions can be the authors' own.", "",
    "| File | What it holds |", "|---|---|"]
readme += [f"| `{f}` | {w} |" for f, w in FILES]
readme += [
    "", "## Notes", "",
    "- AUC: how well a model ranks drought windows above non-drought ones; 0.5 is chance. Differences are paired, "
    "with 95% intervals from 2,000 resamples of whole regions. West and East are the two sides of 98° W.",
    "- Pre-registered: fixed in `code/design/DESIGN.md` before any training. Rows marked \"added after development\" "
    "were added after the development results and before the fresh test.",
    "- T1 leaves out \"LR on latest VCI + age\": with month T observed it is the same model as LR on VCI(T).",
    f"- T4 uses the {ol['fldas']['common_windows']:,} windows that have every label, so its FLDAS row can differ "
    "from T1 in the third decimal.",
    "- Figures 2, 3, 5 and 6 are redrawn from the same score files by the repository's own figure code. Figures 1 "
    "and 4 are drawn by `build_paper_results.py` in the same style with the Bi-LSTM first, since the repository's "
    "versions lead with the Residual Bi-LSTM. Figure 5 plots the Residual Bi-LSTM only; T5 has the Bi-LSTM's revisit "
    "numbers.",
]
if S12 is not None:
    readme.append(
        f"- The SPEI-12 results come from `~/Desktop/DEWSat_BiLSTM_v2/scores/fresh_spei12_posthoc.json` (generated "
        f"{S12['generated']}): the saved fresh-test predictions scored after the lock against gridMET's 1-year SPEI, "
        f"with the same models and resamples as the other labels"
        + (", on the same windows; every published row came out identical." if S12_SAME else
           ", on their own windows (table T4b).")
        + " The GitHub copy has it at `results/scores/fresh_spei12_posthoc.json`, with its script at"
          " `code/dewsat2/fresh_spei12_posthoc.py`.")
(OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
shutil.copy2(__file__, OUT / "build_paper_results.py")
print(f"wrote {len(TABLES)} tables, {len(list(FIGDIR.glob('*.png')))} figures and the claims table to {OUT}")
