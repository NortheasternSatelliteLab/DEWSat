"""Results table for the sweeps in run_sweeps.sh: V1/V2/V3 on 102 regions, HLS 2020+ only (pc) and HLS 2016+ (full).

Every version's setting is picked by validation loss (mean over seeds), never by test score. Test = held-out regions,
2024+. Persistence (last month's FLDAS value, needs ground data the satellite lacks) is computed per region from fldas.csv.

Run from the model folder after run_sweeps.sh:   .venv-train/bin/python make_results.py   (missing sweeps are skipped)
Writes results/: results.md (tables + findings), per_region.csv, swir_by_region.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dew_stat_data import REGIONS_DIR, TARGET

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
# label -> runs folder. "pc": HLS 2020+ only (Planetary Computer; the chosen setup); "full": + NASA 2016-2019.
RUNS = {
    "V1 102 pc": "runs_pc_satellite",
    "V2 102 pc": "runs_pc_satellite_anomaly",
    "V3 102 pc": "runs_pc_satellite_swir",
    "V1 102 full": "runs_full_satellite",
    "V2 102 full": "runs_full_satellite_anomaly",
    "V3 102 full": "runs_full_satellite_swir",
}
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated default palette)
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def fair_pick(runs_dir: Path) -> tuple[str, float] | None:
    """Run with the lowest mean validation loss over its seeds."""
    best = None
    for run in sorted(p for p in runs_dir.iterdir() if (p / "scores.json").exists()):
        vals = [pd.read_csv(f)["val_loss"].min() for f in (run / "logs").glob("seed*/metrics.csv")]
        if vals and (best is None or np.mean(vals) < best[1]):
            best = (run.name, float(np.mean(vals)))
    return best


def persistence_lookup(regions: list[str]) -> dict[tuple, float]:
    """(region, lon, lat, month) -> last month's soil moisture, from each region's fldas.csv."""
    out = {}
    for name in regions:
        fl = pd.read_csv(REGIONS_DIR / name / "fldas.csv")
        lon = "cell_longitude" if "cell_longitude" in fl else "longitude"
        lat = "cell_latitude" if "cell_latitude" in fl else "latitude"
        fl = fl.assign(lon=fl[lon].round(2), lat=fl[lat].round(2), month=fl["time"].str[:7])
        fl[TARGET] = fl[TARGET].where(fl[TARGET] > -9000)
        for (x, y), g in fl.groupby(["lon", "lat"]):
            s = g.set_index("month")[TARGET].sort_index()
            months = list(s.index)
            for prev, cur in zip(months[:-1], months[1:]):
                out[(name, x, y, cur)] = s[prev]
    return out


def rmse(a, b) -> float:
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def r2(pred, y) -> float:
    pred, y = np.asarray(pred), np.asarray(y)
    return float(1 - np.nansum((y - pred) ** 2) / np.nansum((y - np.nanmean(y)) ** 2))


def binom_two_sided(k: int, n: int) -> float:
    """P-value of k wins out of n if wins and losses were a coin flip."""
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rows, per_region = [], []
    for label, folder in RUNS.items():
        d = HERE / folder
        pick = fair_pick(d) if d.exists() else None
        if pick is None:
            print(f"{label:<14} not run yet ({folder})")
            continue
        name, val = pick
        sc = json.loads((d / name / "scores.json").read_text())
        pred = pd.read_csv(d / name / "predictions_test.csv")
        prev = persistence_lookup(sorted(pred["region"].unique()))
        pred["persistence"] = [prev.get((r, round(x, 2), round(y, 2), m), np.nan) for r, x, y, m in
                               zip(pred["region"], pred["cell_longitude"], pred["cell_latitude"], pred["month"])]
        if "refs" in sc and abs(rmse(pred["persistence"], pred["actual"]) - sc["refs"]["persistence"]["rmse"]) > 1e-3:
            print(f"  warning {label}: persistence from fldas.csv differs from train.py's")
        normal = pred["normal"] if "normal" in pred else pd.Series(np.nan, index=pred.index)
        pers = rmse(pred["persistence"], pred["actual"])
        rows.append({
            "label": label, "run": f"{folder}/{name}", "val_loss": val, "regions": pred["region"].nunique(),
            "test_rmse": rmse(pred["bilstm"], pred["actual"]), "test_r2": r2(pred["bilstm"], pred["actual"]),
            "x_persistence": (rmse(pred["bilstm"], pred["actual"]) / pers) ** 2,  # MSE ratio, as the team report
            "normal_r2": r2(normal, pred["actual"]) if normal.notna().any() else np.nan,
            "persistence_rmse": pers, "persistence_r2": r2(pred["persistence"], pred["actual"]),
            "train_regions_2024_r2": sc["same_place"]["bilstm"]["r2"],
        })
        for reg, g in pred.groupby("region"):
            per_region.append({"label": label, "region": reg, "rmse": rmse(g["bilstm"], g["actual"]),
                               "r2": r2(g["bilstm"], g["actual"]),
                               "persistence_rmse": rmse(g["persistence"], g["actual"])})
    if not rows:
        print("no runs found")
        return 1
    res = pd.DataFrame(rows)
    pr = pd.DataFrame(per_region)
    pr["x_persistence"] = (pr["rmse"] / pr["persistence_rmse"]) ** 2  # MSE ratio, as the team report
    pr.round(5).to_csv(OUT / "per_region.csv", index=False)
    data = {"102 pc": "102 regions, HLS 2020+", "102 full": "102 regions, HLS 2016+"}

    # The SWIR question: V3 vs V2 region by region, same data and test regions. And the effect of 2016-2019.
    findings, by_region = [], []
    for tag in ("102 pc", "102 full"):
        a, b = pr[pr["label"] == f"V2 {tag}"], pr[pr["label"] == f"V3 {tag}"]
        if a.empty or b.empty:
            continue
        m = a.merge(b, on="region", suffixes=("_v2", "_v3"))
        wins = int((m["rmse_v3"] < m["rmse_v2"]).sum())
        findings.append(f"- **SWIR ({data[tag]}):** V3 beats V2 in {wins} of {len(m)} test regions "
                        f"(chance of a split this lopsided if SWIR did nothing: p = {binom_two_sided(wins, len(m)):.2f}); "
                        f"median RMSE change {np.median(m['rmse_v3'] - m['rmse_v2']):+.4f} m³/m³.")
        if tag == "102 pc":
            by_region = ["", "## V2 vs V3 by test region (102 regions, HLS 2020+)", "",
                         "| Region | V2 RMSE | V3 RMSE | Persistence RMSE | V2 ÷ persistence | V3 ÷ persistence |",
                         "|---|---|---|---|---|---|"]
            for _, r in m.sort_values("x_persistence_v2").iterrows():
                by_region.append(f"| {r['region']} | {r['rmse_v2']:.4f} | {r['rmse_v3']:.4f} | "
                                 f"{r['persistence_rmse_v2']:.4f} | {r['x_persistence_v2']:.2f}× | "
                                 f"{r['x_persistence_v3']:.2f}× |")
            by_region += ["", "![V2 vs V3 by region](swir_by_region.png)"]
            swir_chart(m, OUT / "swir_by_region.png")
    for v in ("V2", "V3"):
        a, b = res[res["label"] == f"{v} 102 pc"], res[res["label"] == f"{v} 102 full"]
        if len(a) and len(b):
            findings.append(f"- **Effect of adding 2016-2019 ({v}, same 102 regions and test set):** test R² "
                            f"{a['test_r2'].iloc[0]:.2f} → {b['test_r2'].iloc[0]:.2f}, RMSE "
                            f"{a['test_rmse'].iloc[0]:.4f} → {b['test_rmse'].iloc[0]:.4f}.")

    md = ["# DewStat results: 102 regions", "",
          "Test = held-out regions, 2024+. Settings picked by validation loss only. RMSE in m³/m³ (lower is better); "
          "R² = share of the soil-moisture variation explained (1 = perfect). ÷ persistence = model MSE (squared error) divided by "
          "the MSE of 'same as last month', as in the team report (needs ground data the satellite lacks; below 1 would beat it).",
          "", "| Version | Data | Picked run | Val loss | Test regions | Test RMSE | Test R² | ÷ persistence "
          "| Normal map R² | Persistence R² | Train regions 2024+ R² |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        v, tag = r["label"].split(" ", 1)
        nr = "n/a" if np.isnan(r["normal_r2"]) else f"{r['normal_r2']:.2f}"
        md.append(f"| {v} | {data[tag]} | `{r['run'].split('/')[-1]}` | {r['val_loss']:.3f} | {r['regions']} | "
                  f"{r['test_rmse']:.4f} | {r['test_r2']:.2f} | {r['x_persistence']:.2f}× | {nr} | "
                  f"{r['persistence_r2']:.2f} | {r['train_regions_2024_r2']:.2f} |")
    md += ["", "Validation loss compares settings within one row's sweep only (each data setup has its own "
               "val regions and target scale)."]
    if findings:
        md += ["", "## Findings (computed from the runs)", "", *findings]
    md += by_region
    (OUT / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


def swir_chart(m: pd.DataFrame, path: Path) -> None:
    """Per test region: V2 and V3 squared error divided by persistence's (1 = as good as persistence)."""
    m = m.sort_values("x_persistence_v2")
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(m) + 1.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvline(1.0, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(1.0, len(m) - 0.3, " persistence", color=MUTED, fontsize=8, va="bottom")
    for a, b, yy in zip(m["x_persistence_v2"], m["x_persistence_v3"], y):
        ax.plot([a, b], [yy, yy], color=GRID, lw=2, zorder=2)
    ax.scatter(m["x_persistence_v2"], y, s=42, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="V2 (NDVI + VCI)")
    ax.scatter(m["x_persistence_v3"], y, s=42, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="V3 (NDVI + NDMI/NDWI)")
    ax.set_yticks(y, [r.replace("_", " ") for r in m["region"]], fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("model MSE ÷ persistence MSE (log scale; lower is better)", fontsize=9, color=MUTED)
    ax.set_title("SWIR check on held-out regions, 2024+", fontsize=11, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
