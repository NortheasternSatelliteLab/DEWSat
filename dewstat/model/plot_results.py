"""Charts for a training run. train.py calls this automatically at the end.

Run by hand:
    .venv-train/bin/python plot_results.py runs/h32                      # charts for one run -> runs/h32/plots/
    .venv-train/bin/python plot_results.py --compare --runs-dir runs_pc_satellite_anomaly  # every run side by side

Charts (in runs/<name>/plots/):
    1_training_curves.png    loss and val R2 per epoch: is it learning? when did it stop?
    2_scores.png             RMSE, R2, R2 change: Bi-LSTM vs persistence vs ridge
    3_predicted_vs_actual.png  every test cell-month; dots on the diagonal = perfect
    4_timeseries.png         soil moisture over the test months, one panel per held-out region
    5_error_by_region.png    test error in each held-out region, per model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

RUNS = Path(__file__).resolve().parent / "runs"

# Colors follow the model, the same in every chart.
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
# "seasonal" is the satellite-only runs' yardstick; it takes persistence's slot (never shown together).
COLOR = {"bilstm": "#2a78d6", "persistence": "#eb6834", "seasonal": "#eb6834", "normal": "#eb6834",
         "ridge": "#1baf7a", "actual": INK}
LABEL = {"bilstm": "Bi-LSTM", "persistence": "Persistence", "seasonal": "Seasonal avg", "normal": "Normal map",
         "ridge": "Ridge (linear)", "actual": "Actual"}
TRAIN_C, VAL_C = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "text.color": INK,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 1.6,
    "font.size": 10,
})


def baseline(scores: dict) -> str:
    """Each run's own yardstick: normal map (anomaly runs), seasonal (satellite), persistence (full)."""
    return next((k for k in ("normal", "seasonal") if k in scores), "persistence")


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_curves(run: Path, out: Path) -> None:
    logs = sorted((run / "logs").glob("seed*/metrics.csv"))
    if not logs:
        return
    cfg = run / "config.json"
    inputs = json.loads(cfg.read_text()).get("inputs", "full") if cfg.exists() else "full"
    sat = inputs != "full"
    what = {"satellite": "soil moisture", "satellite_anomaly": "anomaly", "satellite_swir": "anomaly"}.get(
        inputs, "monthly change")
    fig, (ax_loss, ax_r2) = plt.subplots(1, 2, figsize=(11, 4))
    for k, path in enumerate(logs):
        m = pd.read_csv(path).groupby("epoch").mean(numeric_only=True)
        first = k == 0
        ax_loss.plot(m.index, m["train_loss"], color=TRAIN_C, alpha=0.9, label="Train" if first else None)
        ax_loss.plot(m.index, m["val_loss"], color=VAL_C, alpha=0.9, label="Validation regions" if first else None)
        best = m["val_loss"].idxmin()
        ax_loss.plot(best, m.loc[best, "val_loss"], "o", ms=7, color=VAL_C, mec=SURFACE, mew=2)
        if "val_r2" in m:
            ax_r2.plot(m.index, m["val_r2"], color=VAL_C, alpha=0.9)
            ax_r2.plot(best, m.loc[best, "val_r2"], "o", ms=7, color=VAL_C, mec=SURFACE, mew=2)
    ax_loss.set(title="Loss per epoch (dot = best epoch, kept)", xlabel="Epoch", ylabel=f"MSE (scaled {what})")
    ax_loss.legend()
    ax_r2.axhline(0, color=MUTED, lw=1, ls="--")
    ax_r2.text(ax_r2.get_xlim()[1], 0, "average-guess level " if sat else "persistence level ",
               ha="right", va="bottom", color=MUTED, fontsize=9)
    ax_r2.set(title=f"Validation R² of the {what}", xlabel="Epoch", ylabel="R² (higher is better)")
    for ax in (ax_loss, ax_r2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if len(logs) > 1:
        fig.suptitle(f"{len(logs)} seeds, one line each", x=0.01, ha="left", color=INK_2, fontsize=9)
    _save(fig, out / "1_training_curves.png")


def scores_chart(scores: dict, out: Path) -> None:
    models = [baseline(scores), "ridge", "bilstm"]
    metrics = [("rmse", "RMSE (m³/m³), lower is better", "{:.4f}"),
               ("r2", "R² of level, higher is better", "{:.3f}"),
               ("r2_change", "R² of monthly change, higher is better", "{:.3f}")]
    metrics = [mt for mt in metrics if mt[0] in scores["bilstm"]]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    for ax, (key, title, fmt) in zip(axes, metrics):
        vals = [scores[m][key] for m in models]
        bars = ax.bar([LABEL[m] for m in models], vals, color=[COLOR[m] for m in models],
                      width=0.6, edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -12),
                        textcoords="offset points", ha="center", color=INK, fontsize=9)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
        lo, hi = min(0, *vals), max(0, *vals)
        ax.set_ylim(lo - 0.12 * (hi - lo or 1) * (lo < 0), hi + 0.15 * (hi - lo or 1))
    fig.suptitle("Test scores: held-out regions, 2024 onward", x=0.01, ha="left", fontweight="bold", fontsize=12)
    _save(fig, out / "2_scores.png")


def predicted_vs_actual(pred: pd.DataFrame, scores: dict, out: Path) -> None:
    base = baseline(scores)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    lo, hi = pred[["actual", "bilstm", base]].min().min(), pred[["actual", "bilstm", base]].max().max()
    pad = 0.02 * (hi - lo)
    for ax, model in zip(axes, [base, "bilstm"]):
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1, ls="--")
        ax.scatter(pred["actual"], pred[model], s=9, color=COLOR[model], alpha=0.35, linewidths=0)
        s = scores[model]
        ax.set_title(f"{LABEL[model]}   R² {s['r2']:.3f}   RMSE {s['rmse']:.4f}", fontsize=11)
        ax.set_xlabel("Actual soil moisture 10–40 cm (m³/m³)")
        ax.set_aspect("equal")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel("Predicted (m³/m³)")
    axes[1].text(hi, hi, "perfect ", ha="right", va="bottom", color=MUTED, fontsize=9, rotation=45,
                 rotation_mode="anchor")
    _save(fig, out / "3_predicted_vs_actual.png")


def timeseries(pred: pd.DataFrame, base: str, out: Path) -> None:
    """One panel per test region: the region's average soil moisture each month."""
    pred = pred.assign(date=pd.to_datetime(pred["month"]))
    regions = sorted(pred["region"].unique())
    fig, axes = plt.subplots(len(regions), 1, figsize=(11, 2.6 * len(regions) + 0.8), sharex=True, squeeze=False)
    for ax, name in zip(axes[:, 0], regions):
        df = pred[pred["region"] == name].groupby("date")[["actual", base, "ridge", "bilstm"]].mean()
        ax.plot(df.index, df["actual"], color=COLOR["actual"], lw=2, label=LABEL["actual"])
        ax.plot(df.index, df[base], color=COLOR[base], ls="--", label=LABEL[base])
        ax.plot(df.index, df["ridge"], color=COLOR["ridge"], ls=":", lw=2, label=LABEL["ridge"])
        ax.plot(df.index, df["bilstm"], color=COLOR["bilstm"], marker="o", ms=4, label=LABEL["bilstm"])
        ax.set_title(f"{name} (region average)", fontsize=11)
        ax.set_ylabel("m³/m³")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper right")
    fig.suptitle("Soil moisture 10–40 cm, held-out regions, test months", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    _save(fig, out / "4_timeseries.png")


def error_by_region(pred: pd.DataFrame, base: str, out: Path) -> None:
    """Test RMSE per held-out region for each model: does it work everywhere, or only in some places?"""
    models = [base, "ridge", "bilstm"]
    err = {m: pred.assign(e=(pred[m] - pred["actual"]) ** 2).groupby("region")["e"].mean() ** 0.5 for m in models}
    regions = list(err["bilstm"].sort_values().index)
    x = np.arange(len(regions))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(regions) + 2), 4.2))
    for k, m in enumerate(models):
        vals = [err[m][r] for r in regions]
        bars = ax.bar(x + (k - 1) * width, vals, width, color=COLOR[m], label=LABEL[m], edgecolor=SURFACE, linewidth=2)
        if m == "bilstm":
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                            textcoords="offset points", ha="center", color=INK, fontsize=8)
    ax.set_xticks(x, regions)
    ax.set_ylabel("Test RMSE (m³/m³), lower is better")
    ax.set_title("Error in each held-out region")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper left")
    ax.set_ylim(0, max(max(e) for e in err.values()) * 1.2)
    _save(fig, out / "5_error_by_region.png")


def make_plots(run: Path) -> Path:
    run = Path(run)
    out = run / "plots"
    out.mkdir(exist_ok=True)
    training_curves(run, out)
    scores_path, pred_path = run / "scores.json", run / "predictions_test.csv"
    if scores_path.exists() and pred_path.exists():
        scores = json.loads(scores_path.read_text())
        pred = pd.read_csv(pred_path)
        scores_chart(scores, out)
        predicted_vs_actual(pred, scores, out)
        timeseries(pred, baseline(scores), out)
        error_by_region(pred, baseline(scores), out)
    return out


def compare_runs(runs_dir: Path = RUNS) -> Path | None:
    """All finished runs side by side, with the baseline and ridge as reference lines."""
    found = [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*/scores.json"))]
    for s in found:  # best val_loss per seed, averaged: this (not test) is how to pick a winner
        best = [pd.read_csv(f)["val_loss"].min() for f in (runs_dir / s["name"] / "logs").glob("seed*/metrics.csv")]
        s["val_loss"] = float(np.mean(best)) if best else np.inf
    if not found:
        print(f"no finished runs in {runs_dir}")
        return None
    found.sort(key=lambda s: s["val_loss"])  # fair pick on top
    names = [run_label(runs_dir / s["name"], s) for s in found]
    base = baseline(found[0])
    second = (("r2_change", "R² of monthly change (higher is better)") if "r2_change" in found[0]["bilstm"]
              else ("r2", "R² of soil moisture (higher is better)"))
    y = np.arange(len(found))[::-1]  # first run at the top
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.75 * len(found) + 2.4), sharey=True)
    for ax, key, title in ((axes[0], "rmse", "Error, RMSE in m³/m³ (lower is better)"), (axes[1], *second)):
        fmt = "{:.4f}" if key == "rmse" else "{:+.2f}"
        vals = [s["bilstm"][key] for s in found]
        bars = ax.barh(y, vals, height=0.6, color=COLOR["bilstm"], edgecolor=SURFACE, linewidth=2)
        bars[0].set_hatch("//")
        span = max(abs(v) for v in vals + [s[base][key] for s in found] + [s["ridge"][key] for s in found])
        for yy, v in zip(y, vals):  # Bi-LSTM numbers in one column past the right edge, clear of every mark
            ax.annotate(fmt.format(v), (1.02, yy), xycoords=("axes fraction", "data"), ha="left", va="center",
                        color=INK, fontsize=10, fontweight="bold")
        # The yardsticks differ a little per run (same samples, same inputs), so mark them on each row.
        ax.scatter([s[base][key] for s in found], y, marker="|", s=420, linewidths=2.5, color=COLOR[base],
                   label=LABEL[base], zorder=3)
        ax.scatter([s["ridge"][key] for s in found], y, marker="D", s=40, color=COLOR["ridge"],
                   edgecolors=SURFACE, linewidths=1.5, label=LABEL["ridge"], zorder=3)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
        lo = min(0, *vals, *(s["ridge"][key] for s in found), *(s[base][key] for s in found))
        hi = max(0, *vals, *(s[base][key] for s in found), *(s["ridge"][key] for s in found))
        ax.set_xlim(lo - 0.08 * span * (lo < 0), hi + 0.08 * span)
    axes[0].set_yticks(y, names)
    axes[0].tick_params(axis="y", length=0, labelsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(color=COLOR["bilstm"]))
    labels.append("Bi-LSTM")
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0))
    where = "new regions, 2024 onward"
    fig.suptitle(f"Test scores on {where}  ·  top row (striped) = best on validation, the fair pick",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = runs_dir / "compare_runs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_label(run: Path, s: dict) -> str:
    """'12-month window, size 16' etc. from the run's config, plus its validation loss."""
    cfg_path = run / "config.json"
    if not cfg_path.exists():
        return f"{s['name']}\nval loss {s['val_loss']:.3f}"
    cfg = json.loads(cfg_path.read_text())
    text = f"{cfg['window']}-month window, size {cfg['hidden']}"
    if cfg.get("no_location"):
        text += ", no location"
    return f"{text}\nval loss {s['val_loss']:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", type=Path, help="a run folder, e.g. runs/h32")
    ap.add_argument("--compare", action="store_true", help="chart every run in runs/ side by side")
    ap.add_argument("--runs-dir", type=Path, default=RUNS, help="folder of runs for --compare")
    args = ap.parse_args()
    if args.compare:
        print(f"wrote {compare_runs(args.runs_dir)}")
    if args.run:
        print(f"wrote {make_plots(args.run)}")
    if not (args.run or args.compare):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
