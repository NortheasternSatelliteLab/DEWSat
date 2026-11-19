"""With vs without the normal map, for each answer key (runs from run_sweeps.sh / run_nldas.sh and run_nomap.sh).

"Without the map" = inputs "satellite" (as V1): the model predicts the soil-moisture level itself from the same inputs
(NDVI, NDVI anomaly, VCI, has_photo, season, location). "With the map" = V2: it predicts wetter/drier than the cell's
normal, and the normal map is added back. HLS 2020+ data, 22 test regions. Each run picked by validation loss.

Anomaly skill = correlation between (prediction - normal) and (actual - normal), on the cell-steps both runs scored.

Run from the model folder:   .venv-train/bin/python make_results_nomap.py
Writes report/nomap/results.md and report/nomap/1_with_without_map.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from compare_setups import fair_pick  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "report" / "nomap"
KEYS = {  # label: (with map, without map)
    "FLDAS, monthly": ("runs_pc_satellite_anomaly", "runs_pc_satellite"),
    "NLDAS, monthly": ("runs_nldas_month", "runs_nldas_month_nomap"),
    "NLDAS, weekly": ("runs_nldas_week", "runs_nldas_week_nomap"),
}
INK, MUTED, GRAY, BLUE, TEAL, ORANGE = "#1f2328", "#59636e", "#8c959f", "#2a6fdb", "#1a9e8f", "#d9822b"
KEY_COLS = ["region", "cell_longitude", "cell_latitude", "month"]


def val_loss(run: Path) -> float:
    return float(np.mean([pd.read_csv(f)["val_loss"].min() for f in (run / "logs").glob("seed*/metrics.csv")]))


def compare(with_dir: str, without_dir: str) -> dict:
    wd, nd = HERE / with_dir, HERE / without_dir
    wn, nn = fair_pick(wd), fair_pick(nd)
    ws, ns = (json.loads((d / n / "scores.json").read_text()) for d, n in ((wd, wn), (nd, nn)))
    pw = pd.read_csv(wd / wn / "predictions_test.csv")
    pn = pd.read_csv(nd / nn / "predictions_test.csv")
    j = pw[KEY_COLS + ["actual", "normal", "bilstm"]].merge(
        pn[KEY_COLS + ["bilstm"]].rename(columns={"bilstm": "nomap"}), on=KEY_COLS)
    a = j["actual"] - j["normal"]
    reg = j.groupby("region").apply(lambda g: pd.Series({
        "with": np.mean((g["bilstm"] - g["actual"]) ** 2), "without": np.mean((g["nomap"] - g["actual"]) ** 2)}),
        include_groups=False)
    persist = ws["refs"]["persistence"]
    return {
        "with_run": wn, "without_run": nn, "with": ws["bilstm"], "without": ns["bilstm"], "normal": ws["normal"],
        "persist": persist, "n": len(j),
        "anom_with": float(np.corrcoef(a, j["bilstm"] - j["normal"])[0, 1]),
        "anom_without": float(np.corrcoef(a, j["nomap"] - j["normal"])[0, 1]),
        "without_wins": int((reg["without"] < reg["with"]).sum()), "regions": len(reg),
        "settings": [{"run": r.name, "val_loss": val_loss(r), **json.loads((r / "scores.json").read_text())["bilstm"]}
                     for r in sorted(nd.iterdir()) if (r / "scores.json").exists()],
    }


def main() -> int:
    OUT.mkdir(exist_ok=True)
    res = {k: compare(*v) for k, v in KEYS.items()}
    md = ["# With vs without the normal map", "",
          "Without the map, the model predicts soil moisture itself from the same satellite inputs (it still gets "
          "season and location, so it can learn the usual pattern on its own). With the map (V2), it predicts wetter "
          "or drier than the cell's normal, and the map is added back. HLS 2020+ data, 22 test regions; FLDAS "
          "2024-01..2026-08, NLDAS 2024-2025. Each run picked by validation loss, never by test.", "",
          "| Answer key, step | With map R² | **Without map R²** | Normal map alone R² | Persistence R² | "
          "With map RMSE | Without map RMSE | Without ÷ persistence (MSE) | Anomaly correlation with / without | "
          "Without map better in |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        ratio = (r["without"]["rmse"] / r["persist"]["rmse"]) ** 2
        md.append(f"| {k} | {r['with']['r2']:.3f} | **{r['without']['r2']:.3f}** | {r['normal']['r2']:.3f} | "
                  f"{r['persist']['r2']:.3f} | {r['with']['rmse']:.4f} | {r['without']['rmse']:.4f} | {ratio:.2f}× | "
                  f"{r['anom_with']:.2f} / {r['anom_without']:.2f} | {r['without_wins']}/{r['regions']} regions |")
    md += ["", "Picked runs: " + "; ".join(f"{k}: `{r['with_run']}` (with), `{r['without_run']}` (without)"
                                            for k, r in res.items()), "",
           "## Every setting without the map (picked = lowest validation loss; test for reference only)", "",
           "| Answer key | Setting | Val loss | Test RMSE | Test R² |", "|---|---|---|---|---|"]
    for k, r in res.items():
        for s in sorted(r["settings"], key=lambda s: s["val_loss"]):
            pick = " (picked)" if s["run"] == r["without_run"] else ""
            md.append(f"| {k} | `{s['run']}`{pick} | {s['val_loss']:.4f} | {s['rmse']:.4f} | {s['r2']:.3f} |")
    (OUT / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(res))
    series = (("Normal map alone", "normal", GRAY), ("Model with map (V2)", "with", BLUE),
              ("Model without map", "without", TEAL), ("Persistence (needs ground data)", "persist", ORANGE))
    for i, (lab, key, col) in enumerate(series):
        vals = [r[key]["r2"] for r in res.values()]
        bars = ax.bar(x + (i - 1.5) * 0.2, vals, 0.18, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.012, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [k.replace(", ", "\n") for k in res])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test R² (22 test regions)", color=MUTED)
    ax.set_title("Test R² with and without the normal map", loc="left", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#eaeef2")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "1_with_without_map.png", dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
