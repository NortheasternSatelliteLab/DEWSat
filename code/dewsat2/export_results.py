"""Export the study's key results as tables (results/tables/*.csv) and figures (results/figures/*.png and *.pdf), read
from the score files in results/scores/ and the saved fresh-test models in runs/phase5/. Also writes
results/README.md, the index of both.

    python code/dewsat2/export_results.py

Every number is read from a score file; nothing is typed in. Differences are AUC differences with 95% intervals from
2,000 whole-region resamples, as in report.md.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import LABELS, LEADS  # noqa: E402
from report_extra import PH_LABELS, macs_for  # noqa: E402

ROOT = HERE.parents[1]
SC, TAB, FIG = ROOT / "results" / "scores", ROOT / "results" / "tables", ROOT / "results" / "figures"
SIDES = ("all", "west", "east")
BLUE, GREY, LIGHT, INK, QUIET, GRID, ZERO = "#2a78d6", "#8f968f", "#c4c9c5", "#1d2422", "#5f6862", "#e6e8e4", "#8d958f"
JOBS = {"job1": "Job 1, estimate now", "job2": "Job 2, missing month",
        "job3": "Job 3, onset forecast (mean over leads 1-3)", "job4": "Job 4, where VCI changed most"}
INDEX: list[tuple[str, str, str]] = []           # (file, what it shows, where the numbers come from)


def lab(m: str) -> str:
    return LABELS.get(m, PH_LABELS.get(m, m))


def load(name: str) -> dict:
    return json.loads((SC / name).read_text())


def r4(x):
    return None if x is None else round(float(x), 4)


def tri(v) -> list:
    v = list(v or [])
    return [r4(x) for x in (v + [None, None, None])[:3]]


def verdict(a, w, e) -> str:
    """The pre-registered wording (code/design/DESIGN.md section 9)."""
    if not a or a[1] is None:
        return "not scored"
    if a[1] > 0:
        worse = [s for s, v in (("West", w), ("East", e)) if v and v[2] is not None and v[2] < 0]
        return "passed" if not worse else f"did not pass (better overall, worse in the {' and '.join(worse)})"
    if a[2] < 0:
        return "worse"
    return "no detectable difference"


def table(name: str, what: str, source: str, header: list[str], rows: list[list]) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    with open(TAB / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    INDEX.append((f"tables/{name}", what, source))


def side_cols(prefix: str) -> list[str]:
    """Column names for a value on all regions, the West and the East, each with its interval: diff, diff_lo, ..."""
    base = prefix.rstrip("_")
    return [f"{base}{s}{k}" for s in ("", "_west", "_east") for k in ("", "_lo", "_hi")]


def sides(d: dict, key: str | None = None) -> list:
    out = []
    for s in SIDES:
        v = (d or {}).get(s)
        out += tri(v.get(key) if (key and isinstance(v, dict)) else v)
    return out


# ------------------------------------------------------------------------------------------------ tables
def development_tables(sc: dict, ph: dict, rl: dict) -> None:
    rows = []
    for g in sc["gate"]:
        for r in g.get("rows", []):
            rows.append([g["job"], g.get("label"), r["kind"], lab(r["model"]), lab(r["comparator"])]
                        + sides(r["dauc_within"]) + [r["verdict"]])
    table("development_gate.csv", "Development, pre-registered rows for all four jobs: AUC difference, verdict",
          "scores.json → gate", ["job", "job_label", "row", "model", "comparator"] + side_cols("diff_") + ["verdict"],
          rows)
    rows = []
    m = sc["subsets"]["job1"]["models"]
    for k, v in m.items():
        rows.append([lab(k), "network" if k in ("bilstm", "resbilstm", "gru", "lstm", "tcn") else "simple"]
                    + sides(v, "auc_within"))
    for k in ("bilstm_vci", "cnn_lr", "cnn_bilstm"):
        rows.append([lab(k), "network, added after development"] + sides(ph["jobs"]["job1"]["models"][k]))
    rows.append([lab("bilstm_relu"), "network, amendment 7"] + sides(rl["jobs"]["job1"]["models"]["bilstm_relu"]))
    rows.sort(key=lambda r: -(r[2] or 0))
    table("development_auc_job1.csv", "Development, job 1: within-fold AUC of every model",
          "scores.json → subsets.job1.models; posthoc.json; relu.json", ["model", "kind"] + side_cols("auc_"), rows)


def fresh_tables(sc: dict, p5: dict) -> None:
    fr, rows = p5["fresh"], []
    for job in ("job1", "job2", "job3", "job4"):
        sel = (sc.get("selection") or {}).get(job) or {}
        net, bil, simp = sel.get("best_network"), sel.get("best_bilstm"), sel.get("best_simple")
        sub = fr.get("job3_mean_onset" if job == "job3" else job) or {}
        pairs = [("gate", net, simp), ("best Bi-LSTM variant", bil, simp)]
        if job == "job1":
            pairs += [("against LR on VCI(T)", net, "lr_vci_t"), ("against LR on VCI(T)", bil, "lr_vci_t")]
        for kind, a, b in pairs:
            v = (sub.get("paired") or {}).get(f"{a} - {b}")
            if not v:
                continue
            d = {s: (v[s] if job == "job3" else v[s]["dauc_within"]) for s in SIDES}
            rows.append([job, JOBS[job], kind, lab(a), lab(b)] + sides(d) + [verdict(d["all"], d["west"], d["east"])])
    table("fresh_test_verdicts.csv", "Fresh test (scored once), pre-registered rows for all four jobs",
          "phase5.json → fresh; selection in scores.json", ["job", "job_label", "row", "model", "comparator"]
          + side_cols("diff_") + ["verdict"], rows)
    rows = []
    for job, lr in (("job1", "lr_vci_t"), ("job2", "lr_latest_age"), ("job4", "lr_vci_t")):
        e = fr[job]
        for k, v in sorted(e["models"].items(), key=lambda kv: -kv[1]["all"]["auc_within"][0]):
            pr = e["paired"].get(f"{k} - {lr}")
            rows.append([job, lab(k)] + sides(v, "auc_within") + (sides(pr, "dauc_within") if pr else [None] * 9)
                        + [lab(lr), e["n"], e["n_pos"]])
    table("fresh_test_auc.csv", "Fresh test, jobs 1, 2 and 4: AUC of every model and its difference to the job's LR",
          "phase5.json → fresh.<job>", ["job", "model"] + side_cols("auc_") + side_cols("minus_lr_")
          + ["lr_model", "windows", "in_drought"], rows)


def fresh_diagnostics(fp: dict) -> None:
    rows = []
    for s in SIDES:
        e = fp["within_region"][s]
        for k, v in e["models"].items():
            rows.append([s, lab(k), r4(v["auc"])] + tri(v["minus_lr"]) + [e["regions"], e["pairs"]])
    table("fresh_within_region.csv", "Fresh test, job 1, pairs within the same region only (ranking over time)",
          "fresh_posthoc.json → within_region", ["side", "model", "auc", "minus_lr", "minus_lr_lo", "minus_lr_hi",
                                                  "regions", "pairs"], rows)
    rows = []
    for sub_name, sub in fp["network_pairs"].items():
        for k, v in sub["paired"].items():
            a, b = k.split(" - ")
            rows.append([sub_name.replace("fresh_", ""), f"{lab(a)} − {lab(b)}"] + sides(v))
    table("fresh_network_pairs.csv", "Fresh test, networks against each other", "fresh_posthoc.json → network_pairs",
          ["job", "comparison"] + side_cols("diff_"), rows)
    models = list(next(iter(fp["by_year"].values()))["auc"])
    rows = [[k, v["n"], r4(v["drought_rate"])] + [r4(v["auc"][m]) for m in models] for k, v in fp["by_year"].items()]
    table("fresh_by_year.csv", "Fresh test, job 1, by year and side: pooled AUC per model",
          "fresh_posthoc.json → by_year", ["slice", "windows", "drought_rate"] + [f"auc_{m}" for m in models], rows)


def training_size(fp: dict) -> list[dict]:
    df = fp["dev_folds"]
    pts = [{"point": f"Fold {f['fold']}", "held": f["validation_years"], "windows": f["training_windows"],
            "lr": f["auc"]["lr_vci_t"], "bilstm": f["auc"]["bilstm"]} for f in df["folds"]]
    pts.append({"point": "Fresh test", "held": "trained on all 8 years", "windows": df["phase5_training_windows"],
                "lr": df["fresh_auc"]["lr_vci_t"], "bilstm": df["fresh_auc"]["bilstm"]})
    pts.sort(key=lambda p: p["windows"])
    table("training_size.csv", "Bi-LSTM minus LR on VCI(T) against training windows: 4 development folds, fresh test",
          "fresh_posthoc.json → dev_folds", ["point", "held_out_or_note", "training_windows", "lr_auc", "bilstm_auc",
                                             "bilstm_minus_lr"],
          [[p["point"], p["held"], p["windows"], r4(p["lr"]), r4(p["bilstm"]), r4(p["bilstm"] - p["lr"])] for p in pts])
    return pts


def learning_curve(lc: dict) -> None:
    rows, ch = [], []
    for axis, e in lc["axes"].items():
        for r in e["sizes"]:
            for m in lc["networks"]:
                rows.append([axis, r["size"], round(r["mean_training_windows"]), lab(m)]
                            + tri(r["network_minus_lr"][m]["all"]))
        for m, c in e["change_in_difference"].items():
            ch.append([axis, lab(m)] + tri(c["smallest_to_full"]) + tri(c["second_largest_to_full"]))
    table("learning_curve.csv", "Learning curve (development, job 1): network minus LR as training years or regions grow",
          "lcurve.json → axes.<axis>.sizes", ["axis", "size", "mean_training_windows", "network", "minus_lr",
                                              "minus_lr_lo", "minus_lr_hi"], rows)
    table("learning_curve_changes.csv", "Learning curve: change in the gap, smallest to full and second-largest to full",
          "lcurve.json → axes.<axis>.change_in_difference", ["axis", "network", "smallest_to_full", "lo", "hi",
                                                             "second_largest_to_full", "lo", "hi"], ch)


def other_labels(fl: dict) -> None:
    rows = []
    for job, j in fl["jobs"].items():
        for name, e in j["labels"].items():
            row = [job, name, r4(e["drought_rate"]), r4(e["agree_with_fldas"])] + tri(e["auc"].get(j["lr"]))
            for m in ("resbilstm", "gru", "bilstm"):
                row += tri(e["paired"].get(f"{m} - {j['lr']}", {}).get("all"))
            rows.append(row + [j["common_windows"]])
    table("other_labels.csv", "Fresh-test predictions judged by other drought labels (FLDAS, NLDAS-2, SPEI 90/180/270 d)",
          "fresh_labels_posthoc.json → jobs.<job>.labels", ["job", "label", "drought_rate", "agree_with_fldas",
                                                            "lr_auc", "lr_auc_lo", "lr_auc_hi"]
          + [f"{m}_minus_lr{k}" for m in ("resbilstm", "gru", "bilstm") for k in ("", "_lo", "_hi")]
          + ["common_windows"], rows)


def revisit(p5: dict) -> None:
    cov, rows = p5["coverage"], []
    for trained in ("composite_trained", "thin_trained"):
        for n in (7, 16, 32):
            for s in range(3):
                e = p5["realistic"]["job1"][trained][f"thin{n}_s{s}"]
                row = [trained.replace("_trained", ""), n, s + 1, r4(cov[f"N{n}"]["share_by_start"][s])]
                row += tri(e["models"]["lr_vci_t"]["all"]["auc_within"])
                for m in ("resbilstm", "gru", "bilstm"):
                    v = e["paired"].get(f"{m} - lr_vci_t")
                    row += tri(v["all"]["dauc_within"]) if v else [None] * 3
                rows.append(row)
    table("revisit.csv", "Realistic sampling, fresh test, job 1: single clear views every 7, 16 or 32 days",
          "phase5.json → realistic.job1 and coverage", ["trained_on", "revisit_days", "start", "share_seen",
                                                        "lr_auc", "lr_auc_lo", "lr_auc_hi"]
          + [f"{m}_minus_lr{k}" for m in ("resbilstm", "gru", "bilstm") for k in ("", "_lo", "_hi")], rows)


def relu(rl: dict) -> None:
    rows = []
    for job, e in rl["jobs"].items():
        for k, v in e["paired"].items():
            a, b = k.split(" - ")
            rows.append([job, f"{lab(a)} − {lab(b)}"] + sides(v) + [e["verdicts"].get(k)])
    table("relu.csv", "ReLU instead of tanh in the Bi-LSTM (development, jobs 1 and 4)", "relu.json → jobs.<job>.paired",
          ["job", "comparison"] + side_cols("diff_") + ["reading"], rows)
    m = rl["jobs"]["job1"]["models"]
    rows = [[z["fold"], z["cfg"]["hidden"], z["cfg"].get("layers", 1), z["n_params"],
             r4(m["bilstm"]["per_fold_auc"][i]), r4(m["bilstm_relu"]["per_fold_auc"][i])]
            for i, z in enumerate(rl["sizes"])]
    table("relu_per_fold.csv", "ReLU test per development fold: pooled AUC, tanh against ReLU", "relu.json",
          ["fold", "hidden", "layers", "parameters", "bilstm_tanh_auc", "bilstm_relu_auc"], rows)


def checks(ck: dict, wi: dict, wk: dict, ph: dict, p5: dict) -> None:
    rows = []
    for f, e in ck["order_test"]["families"].items():
        rows.append([lab(f)] + tri(e["original"]) + [r4(sum(x[0] for x in e["shuffled"]) / len(e["shuffled"]))]
                    + tri(e["shuffled_minus_original"][0]) + [r4(e["mean_abs_prob_change"])])
    table("order_test.csv", "Order test: earlier months shuffled within each window (development, job 1)",
          "checks.json → order_test", ["network", "auc", "auc_lo", "auc_hi", "auc_shuffled_mean",
                                       "shuffled_minus_original", "lo", "hi", "mean_abs_prob_change"], rows)
    pc = ck["positive_control"]
    rows = []
    for k, v in pc["scores"].items():
        pv = (pc.get("paired") or {}).get(f"{k} - lr_vci_t")
        rows.append([lab(k)] + tri(v) + (tri(pv.get("all") if isinstance(pv, dict) else pv) if pv else [None] * 3))
    table("positive_control.csv", f"Positive control: {pc['label']}, fold {pc['fold']}", "checks.json → positive_control",
          ["model", "auc", "auc_lo", "auc_hi", "minus_lr", "minus_lr_lo", "minus_lr_hi"], rows)
    rows = []
    for key, name in (("ndmi", "NDMI added (SWIR)"), ("vci_cell_vs_pixel", "VCI per cell against per pixel")):
        for k, v in wi[key]["paired"].items():
            a, b = k.split(" - ")
            rows.append([name, f"{lab(a)} − {lab(b)}"] + sides(v, "dauc_within"))
    for k, v in wk["scores"]["paired"].items():
        a, b = k.split(" - ")
        rows.append(["Weekly inputs (52 weekly steps)", f"{lab(a)} − {lab(b)}"] + sides(v, "dauc_within"))
    table("whatifs_and_weekly.csv", "What-ifs (NDMI, VCI per cell or pixel) and weekly inputs (development, job 1)",
          "whatif.json → ndmi, vci_cell_vs_pixel; weekly.json → scores", ["experiment", "comparison"]
          + side_cols("diff_"), rows)
    rows = []
    for key, name in (("cnn_lr - lr_vci_t", "Image minus cell average, month T only"),
                      ("cnn_bilstm - bilstm_vci", "Image minus cell average, 12 months"),
                      ("bilstm_vci - lr_vci_t", "12 months minus month T, cell average"),
                      ("cnn_bilstm - cnn_lr", "12 months minus month T, image")):
        dev = ph["jobs"]["job1"]["paired"].get(key, {}).get("all")
        fr = p5["fresh"]["job1"]["paired"].get(key, {}).get("all", {}).get("dauc_within")
        rows.append([name, key] + tri(dev) + tri(fr))
    table("image_vs_cell_average.csv", "Per-pixel VCI image tiles against cell averages, development and fresh test",
          "posthoc.json → jobs.job1.paired; phase5.json → fresh.job1.paired", ["comparison", "pair", "development",
                                                                             "dev_lo", "dev_hi", "fresh_test",
                                                                             "fresh_lo", "fresh_hi"], rows)


def model_sizes() -> None:
    rows = []
    for meta_p in sorted((ROOT / "runs" / "phase5" / "composite").glob("*/*/seed42/meta.json")):
        m = json.loads(meta_p.read_text())
        if "spec" not in m:
            continue
        cfg, job = m["spec"]["cfg"], m["spec"]["job"]
        heads = len(LEADS) if job == "job3" else 1
        macs = m.get("multiply_adds") or macs_for(cfg, 4 if cfg["family"] == "bilstm_vci" else 9, heads)
        pt = meta_p.parent / "model.pt"
        setting = ", ".join(f"{k} {cfg[k]}" for k in ("hidden", "layers", "width", "depth") if k in cfg)
        rows.append([job, lab(cfg["family"]), setting, m.get("n_params"), macs,
                     round(pt.stat().st_size / 1024, 1) if pt.exists() else None,
                     str(meta_p.parent.parent.relative_to(ROOT))])
    table("model_sizes.csv", "Fresh-test networks: parameters, multiply-adds per prediction, weight file size",
          "runs/phase5/composite/<job>/<model>/seed42/meta.json", ["job", "model", "setting", "parameters",
                                                                   "multiply_adds", "model_pt_kib", "folder"], rows)


# ------------------------------------------------------------------------------------------------ figures
def style(ax, xgrid=True, ygrid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=QUIET, labelsize=9)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def heading(fig, title: str, subtitle: str, source: str):
    """Title (the finding), subtitle (its numbers) and source line, placed in inches so short figures fit too."""
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / h, title, ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.42 / h, subtitle, ha="left", va="top", fontsize=9.5, color=QUIET)
    fig.text(0.01, 0.08 / h, source, ha="left", va="bottom", fontsize=7.5, color=QUIET)


def frame(fig, left: float, right: float = 0.97, bottom_in: float = 0.75, top_in: float = 0.95, **kw):
    h = fig.get_figheight()
    fig.subplots_adjust(left=left, right=right, top=1 - top_in / h, bottom=bottom_in / h, **kw)


def save(fig, name: str, what: str, source: str):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    INDEX.append((f"figures/{name}.png", what, source))


def sg(v: float) -> str:
    return f"{'−' if v < 0 else '+'}{abs(v):.3f}"


def fig_gap(sc: dict, ph: dict, p5: dict):
    order = ["resbilstm", "gru", "bilstm", "bilstm_vci", "cnn_bilstm", "cnn_lr", "gbt_mono"]
    note = {"resbilstm": "pre-registered", "gru": "pre-registered", "gbt_mono": "best simple model"}
    rows = []
    for m in order:
        fr = p5["fresh"]["job1"]["paired"].get(f"{m} - lr_vci_t")
        dv = (ph["jobs"]["job1"]["paired"].get(f"{m} - lr_vci_t", {}).get("all")
              if m in ("bilstm_vci", "cnn_lr", "cnn_bilstm") else
              sc["subsets"]["job1"]["paired"].get(f"{m} - lr_vci_t", {}).get("all", {}).get("dauc_within"))
        if fr and dv:
            rows.append((m, dv, fr["all"]["dauc_within"]))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    frame(fig, left=0.27)
    for i, (m, dv, fr) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([dv[1], dv[2]], [y - 0.16] * 2, color=GREY, lw=1.6, solid_capstyle="round")
        ax.plot(dv[0], y - 0.16, "o", ms=6.5, mfc="white", mec=GREY, mew=1.6)
        ax.plot([fr[1], fr[2]], [y + 0.16] * 2, color=BLUE, lw=2, solid_capstyle="round")
        ax.plot(fr[0], y + 0.16, "o", ms=7, color=BLUE)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{lab(m)}\n({note.get(m, 'added after development')})" for m, _, _ in reversed(rows)],
                       fontsize=9, color=INK)
    ax.axvline(0, color=ZERO, lw=1.2)
    ax.text(0.002, len(rows) - 0.45, "0 = as good as LR on VCI(T)", fontsize=8.5, color=QUIET)
    ax.set_xlabel("AUC difference against LR on VCI(T), job 1", fontsize=9, color=QUIET)
    ax.plot([], [], "o-", color=BLUE, label="Fresh test, 2025-01 to 2026-09")
    ax.plot([], [], "o-", color=GREY, mfc="white", label="Development, four folds")
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    style(ax)
    res = dict((m, (dv, fr)) for m, dv, fr in rows)
    heading(fig, "The networks lost to LR in development and beat it on the fresh test",
            f"Residual Bi-LSTM {sg(res['resbilstm'][0][0])} in development, {sg(res['resbilstm'][1][0])} on the "
            f"fresh test; GRU {sg(res['gru'][0][0])} and {sg(res['gru'][1][0])}",
            "Source: results/scores/scores.json, posthoc.json (development), phase5.json (fresh test). "
            "Lines: 95% intervals, whole-region resamples.")
    save(fig, "fig1_gap_to_lr_dev_vs_fresh", "Gap to LR on VCI(T) per model, development against the fresh test",
         "scores.json, posthoc.json, phase5.json")


def fig_size(pts: list[dict]):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    frame(fig, left=0.1)
    for p in pts:
        fresh = p["point"] == "Fresh test"
        gap = p["bilstm"] - p["lr"]
        ax.plot(p["windows"] / 1e3, gap, "o", ms=8.5, color=BLUE if fresh else GREY)
        above = p["point"] == "Fold 2"                   # Fold 2 sits just left of Fold 4: its label goes above
        ax.annotate(f"{p['point']} {sg(gap)}", (p["windows"] / 1e3, gap),
                    xytext=(-10 if fresh else 9, 9 if above else -3), textcoords="offset points",
                    ha="right" if fresh else "left", fontsize=9, color=INK if fresh else QUIET,
                    fontweight="bold" if fresh else "normal")
    ax.axhline(0, color=ZERO, lw=1.2)
    ax.text(1180, 0.004, "0 = as good as LR on VCI(T)", ha="right", fontsize=8.5, color=QUIET)
    ax.set_xlim(250, 1200)
    ax.set_xlabel("Training windows (thousands)", fontsize=9, color=QUIET)
    ax.set_ylabel("Bi-LSTM AUC minus LR", fontsize=9, color=QUIET)
    style(ax, xgrid=False, ygrid=True)
    first, last = pts[0], pts[-1]
    heading(fig, "The Bi-LSTM's gap to LR closes as training data grow",
            f"{sg(first['bilstm'] - first['lr'])} at {first['windows'] / 1e3:.0f}K training windows, "
            f"{sg(last['bilstm'] - last['lr'])} at {last['windows'] / 1e6:.2f}M. Grey: development folds (4-5 "
            f"training years); blue: fresh test (all 8).",
            "Source: results/scores/fresh_posthoc.json → dev_folds. Five points; training size moves with each "
            "fold's years and regions.")
    save(fig, "fig2_training_size", "Bi-LSTM minus LR against training windows", "fresh_posthoc.json → dev_folds")


def fig_lcurve(lc: dict):
    nets = lc["networks"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
    frame(fig, left=0.08, right=0.84, bottom_in=0.95, top_in=1.15, wspace=0.12)
    tone = {"bilstm": (BLUE, 2.0, 1.0), "cnn_bilstm": (GREY, 1.5, 1.0), "bilstm_vci": (LIGHT, 1.5, 1.0)}
    for ax, axis, name in ((axes[0], "years", "More years (folds 1 and 4)"),
                           (axes[1], "regions", "More regions (all four folds)")):
        sizes = lc["axes"][axis]["sizes"]
        xs = range(len(sizes))
        for m in ("cnn_bilstm", "bilstm_vci", "bilstm"):
            if m not in nets:
                continue
            c, lw, _ = tone[m]
            v = [s["network_minus_lr"][m]["all"] for s in sizes]
            if m == "bilstm":
                ax.errorbar(xs, [a[0] for a in v], yerr=[[a[0] - a[1] for a in v], [a[2] - a[0] for a in v]],
                            color=c, lw=lw, marker="o", ms=6, capsize=0, elinewidth=1.4)
            else:
                ax.plot(xs, [a[0] for a in v], color=c, lw=lw, marker="o", ms=4.5)
            if axis == "regions":
                ax.annotate(lab(m), (len(sizes) - 1, v[-1][0]), xytext=(8, {"bilstm": 0, "cnn_bilstm": 6,
                                                                            "bilstm_vci": -6}[m]),
                            textcoords="offset points", fontsize=8.5, color=INK if m == "bilstm" else QUIET,
                            va="center", fontweight="bold" if m == "bilstm" else "normal")
        ax.axhline(0, color=ZERO, lw=1.2)
        if axis == "years":
            ax.text(0, -0.0025, "0 = as good as LR on VCI(T)", fontsize=8.5, color=QUIET, va="top")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([(f"{s['size']} yr" if axis == "years" else
                             ("all" if s["size"] == 1 else f"1/{round(1 / s['size'])}"))
                            + f"\n{s['mean_training_windows'] / 1e3:.0f}K" for s in sizes], fontsize=8.5)
        ax.set_title(name, fontsize=10, color=INK, loc="left")
        style(ax, xgrid=False, ygrid=True)
    axes[0].set_ylabel("Network AUC minus LR", fontsize=9, color=QUIET)
    yb = [s["network_minus_lr"]["bilstm"]["all"][0] for s in lc["axes"]["years"]["sizes"]]
    rb = [s["network_minus_lr"]["bilstm"]["all"][0] for s in lc["axes"]["regions"]["sizes"]]
    heading(fig, "More years narrowed every network's gap to LR; more regions did not",
            f"Bi-LSTM (blue, 95% intervals) {sg(yb[0])} to {sg(yb[-1])} from 2 to 5 years; {sg(rb[0])} to "
            f"{sg(rb[-1])} from 1/8 to all regions",
            "Source: results/scores/lcurve.json → axes (development, job 1, seeds 42-44). Under each size: mean "
            "training windows.")
    save(fig, "fig3_learning_curve", "Learning curve: network minus LR as training years or regions grow",
         "lcurve.json → axes")


def fig_labels(fl: dict):
    j = fl["jobs"]["job1"]
    names = [("fldas", "FLDAS soil moisture\n(the training label)"), ("nldas", "NLDAS-2 soil moisture"),
             ("spei_90d", "SPEI, 90 days"), ("spei_180d", "SPEI, 180 days\n(Phase 6's choice)"),
             ("spei_270d", "SPEI, 270 days")]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    frame(fig, left=0.25)
    for i, (k, name) in enumerate(names):
        y = len(names) - 1 - i
        for dy, m, c in ((0.18, "resbilstm", BLUE), (0.0, "gru", GREY), (-0.18, "bilstm", LIGHT)):
            v = j["labels"][k]["paired"][f"{m} - {j['lr']}"]["all"]
            ax.plot([v[1], v[2]], [y + dy] * 2, color=c, lw=1.8 if m == "resbilstm" else 1.4)
            ax.plot(v[0], y + dy, "o", ms=6.5 if m == "resbilstm" else 5, color=c)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}\ndrought {j['labels'][k]['drought_rate']:.0%}, agrees {j['labels'][k]['agree_with_fldas']:.0%}"
                        for k, n in reversed(names)], fontsize=8.5, color=INK)
    ax.axvline(0, color=ZERO, lw=1.2)
    for m, c in (("Residual Bi-LSTM", BLUE), ("GRU", GREY), ("Bi-LSTM", LIGHT)):
        ax.plot([], [], "o-", color=c, label=m)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.set_xlabel("AUC difference against LR on VCI(T), fresh test, job 1", fontsize=9, color=QUIET)
    style(ax)
    nl = j["labels"]["nldas"]["paired"][f"resbilstm - {j['lr']}"]["all"][0]
    sp = j["labels"]["spei_180d"]["paired"][f"resbilstm - {j['lr']}"]["all"][0]
    heading(fig, "The gain holds for soil-moisture drought and reverses for climatic drought",
            f"Residual Bi-LSTM minus LR: {sg(nl)} judged by NLDAS-2, {sg(sp)} judged by SPEI over 180 days",
            f"Source: results/scores/fresh_labels_posthoc.json → jobs.job1.labels; the {j['common_windows']:,} windows "
            f"where every label exists.")
    save(fig, "fig4_other_labels", "Fresh-test predictions judged by five drought labels",
         "fresh_labels_posthoc.json → jobs.job1.labels")


def fig_revisit(p5: dict):
    cov, comp = p5["coverage"], p5["realistic"]["job1"]["composite_trained"]
    full = p5["fresh"]["job1"]["paired"]["resbilstm - lr_vci_t"]["all"]["dauc_within"]
    cats = [("Full record", cov["full_record_composite"], [full])]
    for n in (7, 16, 32):
        cats.append((f"{n} days", cov[f"N{n}"]["share_mean"],
                     [comp[f"thin{n}_s{s}"]["paired"]["resbilstm - lr_vci_t"]["all"]["dauc_within"] for s in range(3)]))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    frame(fig, left=0.1, bottom_in=0.95)
    means = []
    for i, (name, share, vals) in enumerate(cats):
        offs = [0.0] if len(vals) == 1 else [-0.12, 0.0, 0.12]
        for o, v in zip(offs, vals):
            ax.plot([i + o] * 2, [v[1], v[2]], color=BLUE, lw=1.6)
            ax.plot(i + o, v[0], "o", ms=6.5, color=BLUE)
        mean = sum(v[0] for v in vals) / len(vals)
        means.append(mean)
        ax.annotate(sg(mean), (i + (0.2 if len(vals) > 1 else 0.08), mean), fontsize=9, color=INK, va="center",
                    fontweight="bold")
    ax.axhline(0, color=ZERO, lw=1.2)
    ax.text(-0.3, -0.008, "0 = as good as LR on VCI(T)", fontsize=8.5, color=QUIET)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{n}\n{s:.0%} seen" for n, s, _ in cats], fontsize=9)
    ax.set_xlim(-0.4, len(cats) - 0.4)
    ax.set_ylabel("Residual Bi-LSTM AUC minus LR", fontsize=9, color=QUIET)
    style(ax, xgrid=False, ygrid=True)
    heading(fig, "The network's lead over LR fades as clear views get rarer",
            f"{sg(means[0])} with the full record, {sg(means[1])} at 7 days, {sg(means[3])} at 32 days (mean of 3 "
            f"start dates)",
            "Source: results/scores/phase5.json → realistic.job1, coverage. One point per start date; seen = "
            "cell-months with a clear view.")
    save(fig, "fig5_revisit", "Network minus LR with single clear views every 7, 16 or 32 days",
         "phase5.json → realistic.job1")


def fig_relu(rl: dict):
    e = rl["jobs"]["job1"]
    rows = [("ReLU minus tanh Bi-LSTM", e["paired"]["bilstm_relu - bilstm"]["all"]),
            ("ReLU Bi-LSTM minus LR", e["paired"]["bilstm_relu - lr_vci_t"]["all"]),
            ("tanh Bi-LSTM minus LR", e["paired"]["bilstm - lr_vci_t"]["all"])]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    frame(fig, left=0.27, right=0.93)
    for i, (name, v) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([v[1], v[2]], [y, y], color=BLUE, lw=2)
        ax.plot(v[0], y, "o", ms=7, color=BLUE)
        ax.annotate(sg(v[0]), (v[2], y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                    color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in reversed(rows)], fontsize=9, color=INK)
    ax.axvline(0, color=ZERO, lw=1.2)
    lo, hi = min(v[1] for _, v in rows), max(v[2] for _, v in rows)
    ax.set_xlim(lo - 0.004, hi + 0.012)                  # room for the value printed right of each interval
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlabel("Within-fold AUC difference, development, job 1", fontsize=9, color=QUIET)
    style(ax)
    heading(fig, "ReLU in place of tanh changed nothing detectable",
            f"ReLU minus tanh {sg(rows[0][1][0])} [{sg(rows[0][1][1])}, {sg(rows[0][1][2])}]: "
            f"{e['verdicts']['bilstm_relu - bilstm']} (reading fixed in advance)",
            "Source: results/scores/relu.json → jobs.job1.paired. Same settings, seeds, starting weights and data "
            "order as the original Bi-LSTM runs.")
    save(fig, "fig6_relu", "ReLU against tanh and LR, development", "relu.json → jobs.job1.paired")


def write_index():
    L = ["# Results", "",
         "Tables and figures of the DEWSat Bi-LSTM study, generated from the score files in `scores/` by "
         "`code/dewsat2/export_results.py`. AUC differences carry 95% intervals from 2,000 whole-region resamples "
         "(columns `_lo`, `_hi`; `west_`, `east_` are the two sides of 98 degrees W). The full write-up is "
         "`../FINDINGS.md`; every table of the run is in `../report.md`.", "",
         "| File | What it shows | Source in `scores/` |", "|---|---|---|"]
    L += [f"| [`{f}`]({f}) | {w} | `{s}` |" for f, w, s in INDEX]
    L += ["", "The score files themselves (`scores/*.json`) hold every number of the run, including those not "
          "exported here."]
    (ROOT / "results" / "README.md").write_text("\n".join(L) + "\n")


def main() -> int:
    sc, ph, rl, p5 = load("scores.json"), load("posthoc.json"), load("relu.json"), load("phase5.json")
    fp, fl, lc = load("fresh_posthoc.json"), load("fresh_labels_posthoc.json"), load("lcurve.json")
    ck, wi, wk = load("checks.json"), load("whatif.json"), load("weekly.json")
    fresh_tables(sc, p5)
    development_tables(sc, ph, rl)
    fresh_diagnostics(fp)
    pts = training_size(fp)
    learning_curve(lc)
    other_labels(fl)
    revisit(p5)
    relu(rl)
    checks(ck, wi, wk, ph, p5)
    model_sizes()
    fig_gap(sc, ph, p5)
    fig_size(pts)
    fig_lcurve(lc)
    fig_labels(fl)
    fig_revisit(p5)
    fig_relu(rl)
    write_index()
    print(f"wrote {sum(1 for f, _, _ in INDEX if f.startswith('tables'))} tables and "
          f"{sum(1 for f, _, _ in INDEX if f.startswith('figures'))} figures to {ROOT / 'results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
