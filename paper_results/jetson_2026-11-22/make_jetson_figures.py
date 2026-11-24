#!/usr/bin/env python3
"""Figures for the paper from the Jetson tables (run make_jetson_tables.py first).

  figures/jetson_heat_power_soaks.png/.pdf   hottest on-chip sensor and module power over time in the runbook soaks
                                             (F10 15W continuous, F11 15W duty cycle, F12 MAXN_SUPER pinned worst case)
  figures/jetson_power_by_mode.png/.pdf      idle vs running the paper model, module power, every power mode
  figures/<figure name>_data.csv             the exact values each figure plots (one CSV per figure, UTF-8 with BOM
                                             like the tables), to re-make the chart in Excel. Sources: the soak figure
                                             reads J8 (tests F10-F12), the mode figure reads J0.

Same look as the paper's other figures (DEWSat_BiLSTM_github/code/dewsat2/export_results.py: colours, spines, 300 dpi,
no title text). Categorical colours are the first three slots of the dataviz reference palette, the subset documented
as passing every pair in both modes (slot 1 is the paper's own blue). Needs matplotlib: run with v2's venv:
    ~/Desktop/DEWSat_BiLSTM_v2/.venv/bin/python make_jetson_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
INK, QUIET, GRID, ZERO, LIGHT = "#1d2422", "#5f6862", "#e6e8e4", "#8d958f", "#c4c9c5"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # blue, orange, aqua (fixed order)
TRIP = 70.0


def rows(name):
    with (HERE / name).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=QUIET, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  wrote figures/{name}.png and .pdf")


def write_data(name, header, body):
    """The plotted values as copied from the source table (no re-rounding), for re-making the chart in Excel."""
    FIG.mkdir(exist_ok=True)
    with (FIG / f"{name}_data.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)
    print(f"  wrote figures/{name}_data.csv ({len(body)} rows)")


def fig_soaks():
    ts = rows("J8_v2_soak_timeseries.csv")
    want = [("F10 flight15w", "F10: 15W, dynamic clocks, continuous, 120 min"),
            ("F11 duty15w", "F11: 15W, dynamic clocks, 60 s on / 240 s off"),
            ("F12 worstcase", "F12: MAXN_SUPER, pinned clocks, batch 4096")]
    have = [(k, lab) for k, lab in want if any(r["Test"] == k for r in ts)]
    if not have:
        print("  soak figure: no F10-F12 data yet; skipped")
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.35, 1]})
    fig.subplots_adjust(left=0.09, right=0.97, top=0.98, bottom=0.09, hspace=0.10)
    tmax = 0
    head, cols = [], []
    for i, (k, lab) in enumerate(have):
        rs = [r for r in ts if r["Test"] == k]
        short, duty = k.split()[0], any(r["Phase"] == "off" for r in rs)
        head += ([f"{short} minutes since the start ({lab.split(': ', 1)[1]})"]
                 + ([f"{short} phase (on = running the model, off = idle)"] if duty else [])
                 + [f"{short} hottest on-chip sensor (°C)", f"{short} module input power (W)"])
        cols.append([[r["Minutes"]] + ([r["Phase"]] if duty else []) + [r["Hottest sensor (C)"], r["Module power (W)"]]
                     for r in rs])
        t = [float(r["Minutes"]) for r in rs]
        hot = [float(r["Hottest sensor (C)"]) for r in rs]
        pw = [float(r["Module power (W)"]) for r in rs]
        tmax = max(tmax, t[-1])
        a1.plot(t, hot, color=SERIES[i], linewidth=1.6, label=lab, solid_capstyle="round")
        a2.plot(t, pw, color=SERIES[i], linewidth=1.1 if "duty" in k else 1.6, label=lab, solid_capstyle="round")
        j = max(range(len(hot)), key=lambda q: hot[q])
        if "worstcase" in k:      # flat near the top: label under the line, at its end
            a1.text(t[-1], hot[-1] - 2.2, f"{hot[j]:.1f} °C max", fontsize=8.5, color=INK, ha="right", va="top")
        elif "duty" in k:         # spiky: label above the hottest spike
            a1.text(t[j], hot[j] + 1.2, f"{hot[j]:.1f} °C max", fontsize=8.5, color=INK, ha="center", va="bottom")
        else:                     # long and flat: label above its end
            a1.text(t[-1], hot[-1] + 1.4, f"{hot[j]:.1f} °C max", fontsize=8.5, color=INK, ha="right", va="bottom")
        on = [w for w, r in zip(pw, rs) if r["Phase"] != "off"]
        off = [w for w, r in zip(pw, rs) if r["Phase"] == "off"]
        if "duty" in k and off:
            a2.text(t[-1] + 1.5, sum(off) / len(off), f"on {sum(on) / len(on):.1f} W, off {sum(off) / len(off):.1f} W",
                    fontsize=8.5, color=INK, ha="left", va="center")
        elif "worstcase" in k:
            a2.text(t[-1] + 1.5, sum(on) / len(on), f"{sum(on) / len(on):.1f} W", fontsize=8.5, color=INK,
                    ha="left", va="center")
        else:
            a2.text(t[-1], sum(on) / len(on) + 1.0, f"{sum(on) / len(on):.1f} W", fontsize=8.5, color=INK,
                    ha="right", va="bottom")
    a1.axhline(TRIP, color=ZERO, linewidth=1.0, linestyle=(0, (4, 3)))
    a1.text(max(tmax, 60), TRIP + 0.8, "70 °C passive (throttling) trip, the paper's limit", fontsize=8.5,
            color=QUIET, ha="right", va="bottom")
    a1.set_ylim(25, 75)
    a1.set_ylabel("Hottest on-chip sensor (°C)", fontsize=9, color=QUIET)
    a2.set_ylim(0, 20)
    a2.set_ylabel("Module input power (W)", fontsize=9, color=QUIET)
    a2.set_xlabel("Minutes since the test started", fontsize=9, color=QUIET)
    a2.set_xlim(0, max(tmax, 60) * 1.01)
    for a in (a1, a2):
        style(a)
    a1.legend(loc="lower right", fontsize=8.5, frameon=False, labelcolor=INK, handlelength=1.6)
    save(fig, "jetson_heat_power_soaks")
    # one block of columns per test, side by side (the tests have different lengths: blank cells pad the shorter ones)
    body = []
    for j in range(max(len(c) for c in cols)):
        row = []
        for c in cols:
            row += c[j] if j < len(c) else [""] * len(c[0])
        body.append(row)
    write_data("jetson_heat_power_soaks", head, body)


def fig_modes():
    j0 = rows("J0_v2_power_budget_by_mode.csv")
    if not j0:
        return
    order = {"7W": 0, "15W": 1, "25W": 2, "MAXN_SUPER": 3}
    j0.sort(key=lambda r: (order.get(r["Power mode"], 9), r["Clocks"]))
    labels = [f"{r['Power mode']}, {'pinned' if 'pinned' in r['Clocks'] else 'dynamic'} clocks" for r in j0]
    idle = [float(r["Idle for the budget (W)"]) for r in j0]
    rested = [r["Which idle reading that is"].startswith("at rest") for r in j0]
    run = [float(r["Paper model, module power during tiles (W)"] or "nan") for r in j0]
    ms = [r["Paper model, one tile (100 windows) p50 (ms)"] for r in j0]
    h = 0.42 * len(j0) + 1.55
    fig, ax = plt.subplots(figsize=(7.0, h))
    fig.subplots_adjust(left=0.32, right=0.97, top=1 - 0.5 / h, bottom=0.85 / h)
    y = list(range(len(j0)))[::-1]
    for yy, a, b in zip(y, idle, run):
        ax.plot([a, b], [yy, yy], color=LIGHT, linewidth=2.0, solid_capstyle="round", zorder=1)
    ax.scatter(idle, y, s=46, color=ZERO, zorder=2, label="Idle", edgecolors="white", linewidths=1.5)
    ax.scatter(run, y, s=46, color=SERIES[0], zorder=3, label="Running the paper model on 100-window tiles",
               edgecolors="white", linewidths=1.5)
    for yy, b, m in zip(y, run, ms):
        ax.text(b + 0.35, yy, f"{b:.1f} W  ({m} ms per tile)", fontsize=8.5, color=INK, va="center")
    for yy, a, ok in zip(y, idle, rested):
        ax.text(a - 0.35, yy, f"{a:.1f} W" + ("" if ok else "*"), fontsize=8.5, color=QUIET, va="center", ha="right")
    style(ax)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlim(0, max(r for r in run if r == r) * 1.85)
    ax.set_xlabel("Module input power (W)", fontsize=9, color=QUIET)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_ylim(-0.6, len(j0) - 0.4)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, fontsize=8.5, frameon=False, labelcolor=INK,
              handletextpad=0.3, columnspacing=1.6)
    if not all(rested):
        fig.text(0.32, 0.12 / h, "* idle read at the start of the benchmark, not after 4-10 min with no load",
                 fontsize=8, color=QUIET, va="bottom")
    save(fig, "jetson_power_by_mode")
    write_data("jetson_power_by_mode",
               ["Power mode and clocks (figure order, top to bottom)", "Idle module power (W)",
                "Which idle reading that is (marked * in the figure unless at rest)",
                "Module power running the paper model on 100-window tiles (W)", "Time per tile, p50 (ms)"],
               [[lab, r["Idle for the budget (W)"], r["Which idle reading that is"],
                 r["Paper model, module power during tiles (W)"], r["Paper model, one tile (100 windows) p50 (ms)"]]
                for lab, r in zip(labels, j0)])


if __name__ == "__main__":
    fig_soaks()
    fig_modes()
