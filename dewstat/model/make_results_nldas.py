"""Weekly vs monthly results with NLDAS-2 as the answer key (runs from run_nldas.sh), next to the FLDAS monthly model.

Per step, the setting is picked by validation loss (fair_pick), never by test. Test = 22 test regions, 2024-2025.
Persistence (last step's NLDAS value) is rebuilt from nldas_daily.npz to count per-region wins.

Run from the model folder:   .venv-train/bin/python make_results_nldas.py
Writes report/weekly/weekly.md and report/weekly/3_weekly_vs_monthly.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dew_stat_nldas import step_start  # noqa: E402
from compare_setups import fair_pick  # noqa: E402

HERE = Path(__file__).resolve().parent
PRACTICE = HERE.parent / "data-pipeline" / "practice"
OUT = HERE / "report" / "weekly"
FLDAS_RUN = HERE / "models" / "v2_satellite_anomaly"  # the chosen V2 (FLDAS, monthly)
INK, MUTED, GRAY, BLUE, ORANGE = "#1f2328", "#59636e", "#8c959f", "#2a6fdb", "#d9822b"


def val_loss(run: Path) -> float:
    return float(np.mean([pd.read_csv(f)["val_loss"].min() for f in (run / "logs").glob("seed*/metrics.csv")]))


def persistence_lookup(step: str) -> pd.Series:
    """NLDAS value of the previous step, keyed by (lon, lat, step start)."""
    z = np.load(PRACTICE / "nldas_daily.npz")
    days = pd.DatetimeIndex(pd.to_datetime(z["days"]))
    cells = pd.read_csv(PRACTICE / "nldas_cells.csv")
    s = pd.DataFrame(z["sm"], index=days).groupby(step_start(days, step)).mean().shift(1)
    s.columns = pd.MultiIndex.from_arrays([cells["lon"].round(2), cells["lat"].round(2)])
    s.index = [d.strftime("%Y-%m-%d") for d in s.index]
    out = s.stack([0, 1], future_stack=True)
    out.index = out.index.set_names(["when", "lon", "lat"])
    return out


def step_result(step: str) -> dict:
    runs = HERE / f"runs_nldas_{step}"
    name = fair_pick(runs)
    sc = json.loads((runs / name / "scores.json").read_text())
    p = pd.read_csv(runs / name / "predictions_test.csv")
    prev = persistence_lookup(step)
    key = pd.MultiIndex.from_arrays([p["month"], p["cell_longitude"].round(2), p["cell_latitude"].round(2)])
    p["persistence"] = prev.reindex(key).to_numpy()
    p = p.dropna(subset=["persistence"])
    anom_a, anom_p = p["actual"] - p["normal"], p["bilstm"] - p["normal"]
    reg = p.groupby("region").apply(lambda g: pd.Series({
        "model": np.mean((g["bilstm"] - g["actual"]) ** 2),
        "persist": np.mean((g["persistence"] - g["actual"]) ** 2),
        "normal": np.mean((g["normal"] - g["actual"]) ** 2)}), include_groups=False)
    settings = [{"run": r.name, "val_loss": val_loss(r), **json.loads((r / "scores.json").read_text())["bilstm"]}
                for r in sorted(runs.iterdir()) if (r / "scores.json").exists()]
    return {
        "name": name, "n": len(p), "model": sc["bilstm"], "normal": sc["normal"], "persist": sc["refs"]["persistence"],
        "anom_corr": float(np.corrcoef(anom_a, anom_p)[0, 1]),
        "beats_persist": int((reg["model"] < reg["persist"]).sum()),
        "beats_normal": int((reg["model"] < reg["normal"]).sum()), "regions": len(reg),
        "ratio_persist": float(np.mean((p["bilstm"] - p["actual"]) ** 2) / np.mean((p["persistence"] - p["actual"]) ** 2)),
        "settings": settings,
    }


def main() -> int:
    OUT.mkdir(exist_ok=True)
    res = {"month": step_result("month"), "week": step_result("week")}
    fl = json.loads((FLDAS_RUN / "scores.json").read_text())
    rows = [("FLDAS, monthly (current model)", fl["bilstm"], fl["normal"], fl["refs"]["persistence"], None),
            ("NLDAS, monthly", res["month"]["model"], res["month"]["normal"], res["month"]["persist"], res["month"]),
            ("NLDAS, weekly", res["week"]["model"], res["week"]["normal"], res["week"]["persist"], res["week"])]

    md = ["# Weekly vs monthly (NLDAS-2 answer key)", "",
          "Same V2 model and inputs (NDVI, NDVI anomaly, VCI, has_photo, season, location), same 102 regions and "
          "splits. Each step's setting picked by validation loss. Test = 22 test regions; FLDAS 2024-01..2026-08, "
          "NLDAS 2024-2025. Persistence needs ground data the satellite does not have.", "",
          "| Answer key, step | Picked setting | Model R² | Normal map R² | Persistence R² | Model RMSE | "
          "Persistence RMSE | Model ÷ persistence (MSE) | Anomaly correlation | Beats persistence | Beats normal map |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, m, n, p, r in rows:
        extra = (f"{r['ratio_persist']:.2f}× | {r['anom_corr']:.2f} | {r['beats_persist']}/{r['regions']} | "
                 f"{r['beats_normal']}/{r['regions']}") if r else "0.95× | 0.30 | 11/22 | n/a"
        md.append(f"| {label} | `{r['name'] if r else FLDAS_RUN.name}` | **{m['r2']:.3f}** | {n['r2']:.3f} | "
                  f"{p['r2']:.3f} | {m['rmse']:.4f} | {p['rmse']:.4f} | {extra} |")
    md += ["", "## Every setting (picked = lowest validation loss; test shown for reference only)", "",
           "| Step | Setting | Val loss | Test RMSE | Test R² |", "|---|---|---|---|---|"]
    for step, r in res.items():
        for s in sorted(r["settings"], key=lambda s: s["val_loss"]):
            pick = " (picked)" if s["run"] == r["name"] else ""
            md.append(f"| {step} | `{s['run']}`{pick} | {s['val_loss']:.4f} | {s['rmse']:.4f} | {s['r2']:.3f} |")
    (OUT / "weekly.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(rows))
    series = (("Normal map alone", 1, GRAY), ("Model (V2)", 0, BLUE), ("Persistence (needs ground data)", 2, ORANGE))
    for i, (lab, k, col) in enumerate(series):
        vals = [row[1 + k]["r2"] for row in rows]
        bars = ax.bar(x + (i - 1) * 0.27, vals, 0.25, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [r[0].replace(", ", "\n") for r in rows])
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Test R² (22 test regions)", color=MUTED)
    ax.set_title("Test R² by answer key and time step", loc="left", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#eaeef2")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "3_weekly_vs_monthly.png", dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
