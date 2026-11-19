"""Validation on all years (as first run) vs validation in 2023 only (run_valgap.sh), same everything else.

With val_years="all", the val regions' 2020-2022 (training years) and 2024+ (test years) weeks also chose the best
epoch and the setting. With "gap", validation is the val regions in 2023 only: no year shared with train or test.

Run from the model folder:   .venv-train/bin/python make_results_valgap.py
Writes report/validation_years/results.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import make_results
import make_results_nldas
from compare_setups import fair_pick
from make_results_nomap import val_loss

HERE = Path(__file__).resolve().parent
OUT = HERE / "report" / "validation_years"
PAIRS = {  # label: (validation all years, validation 2023 only)
    "FLDAS monthly, with map (V2)": ("runs_pc_satellite_anomaly", "runs_valgap_fldas"),
    "FLDAS monthly, no map": ("runs_pc_satellite", "runs_valgap_fldas_nomap"),
    "NLDAS weekly, with map (final)": ("runs_nldas_week", "runs_valgap_week"),
    "NLDAS weekly, no map": ("runs_nldas_week_nomap", "runs_valgap_week_nomap"),
}


_weekly_prev: pd.Series | None = None


def with_persistence(p: pd.DataFrame, weekly: bool) -> np.ndarray:
    """Last step's answer-key value for every prediction row (persistence)."""
    global _weekly_prev
    if weekly:
        if _weekly_prev is None:
            _weekly_prev = make_results_nldas.persistence_lookup("week")
        key = pd.MultiIndex.from_arrays([p["month"], p["cell_longitude"].round(2), p["cell_latitude"].round(2)])
        return _weekly_prev.reindex(key).to_numpy()
    prev = make_results.persistence_lookup(sorted(p["region"].unique()))
    return np.array([prev.get((r, round(x, 2), round(y, 2), m), np.nan) for r, x, y, m in
                     zip(p["region"], p["cell_longitude"], p["cell_latitude"], p["month"])])


def summary(runs: Path) -> dict | None:
    if not runs.exists() or not any((r / "scores.json").exists() for r in runs.iterdir()):
        return None
    name = fair_pick(runs)
    s = json.loads((runs / name / "scores.json").read_text())
    p = pd.read_csv(runs / name / "predictions_test.csv")
    mse = p.groupby("region").apply(lambda g: pd.Series({
        "model": np.mean((g["bilstm"] - g["actual"]) ** 2)}), include_groups=False)["model"]
    done = sorted(r.name for r in runs.iterdir() if (r / "scores.json").exists())
    return {"name": name, "r2": s["bilstm"]["r2"], "rmse": s["bilstm"]["rmse"], "persist": s["refs"]["persistence"],
            "normal": s.get("normal", s["refs"].get("normal")), "mse": mse, "done": done,
            "settings": {r: (val_loss(runs / r), json.loads((runs / r / "scores.json").read_text())["bilstm"]["r2"])
                         for r in done}}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    md = ["# Validation years: all years vs 2023 only", "",
          "First runs: validation = the 13 val regions in **every** year, 2020-2025, so it shared years with training "
          "(2020-2022) and with test (2024-2025), though never places. Re-run: validation = the val regions in **2023 "
          "only**, a year used by neither training nor test. Everything else is the same. 22 test regions, 2024+.", "",
          "| Model | Validation | Picked setting | Test R² | Test RMSE | Normal map R² | Persistence R² | "
          "Beats persistence in |", "|---|---|---|---|---|---|---|---|"]
    detail = []
    for label, dirs in PAIRS.items():
        for tag, d in zip(("all years", "2023 only"), dirs):
            r = summary(HERE / d)
            if r is None:
                md.append(f"| {label} | {tag} | (not run yet) | | | | | |")
                continue
            p = pd.read_csv(HERE / d / r["name"] / "predictions_test.csv")
            p["persistence"] = with_persistence(p, weekly="week" in d)
            g = p.dropna(subset=["persistence"]).groupby("region").apply(
                lambda x: np.mean((x["bilstm"] - x["actual"]) ** 2) < np.mean((x["persistence"] - x["actual"]) ** 2),
                include_groups=False)
            wins = f"{int(g.sum())}/{len(g)}"
            normal = f"{r['normal']['r2']:.3f}" if r["normal"] else "n/a"
            note = "" if len(r["done"]) == 5 else f" ({len(r['done'])}/5 settings done)"
            md.append(f"| {label} | {tag} | `{r['name']}`{note} | **{r['r2']:.3f}** | {r['rmse']:.4f} | {normal} | "
                      f"{r['persist']['r2']:.3f} | {wins} |")
            detail += [f"| {label} | {tag} | `{s}`{' (picked)' if s == r['name'] else ''} | {v:.4f} | {t:.3f} |"
                       for s, (v, t) in sorted(r["settings"].items(), key=lambda kv: kv[1][0])]
    md += ["", "## Every setting (picked = lowest validation loss; test R² for reference only)", "",
           "| Model | Validation | Setting | Val loss | Test R² |", "|---|---|---|---|---|", *detail]
    (OUT / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
