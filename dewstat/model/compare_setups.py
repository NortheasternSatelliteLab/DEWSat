"""Pick the HLS data setup without looking at test: compare setups on the validation regions, 2020-2023.

Setups: HLS 2020+ only (pc) and HLS 2016+ (full).
Validation loss cannot be compared across setups (different years in val, different target scale), so each setup's
picked run (lowest validation loss within its own sweep) is scored here on the same cell-months: validation regions,
2020-01 .. 2023-12 (the years every setup has), RMSE in m³/m³. Test is printed for reference only.

Run from the model folder after run_sweeps.sh (needs the pc and full sweeps):
    .venv-train/bin/python compare_setups.py
Writes results/setups.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dew_stat_data import DewStatData
from dew_stat_model_fast import BiLSTM

HERE = Path(__file__).resolve().parent
SETUPS = {"HLS 2020+ only": "pc", "HLS 2016+": "full"}
VERSIONS = {"V2": "satellite_anomaly", "V3": "satellite_swir"}


def fair_pick(runs_dir: Path) -> str:
    """The run with the lowest validation loss (averaged over seeds), chosen without looking at test."""
    best, best_name = np.inf, None
    for run in sorted(p for p in runs_dir.iterdir() if (p / "scores.json").exists()):
        vals = [pd.read_csv(f)["val_loss"].min() for f in (run / "logs").glob("seed*/metrics.csv")]
        if vals and np.mean(vals) < best:
            best, best_name = float(np.mean(vals)), run.name
    return best_name


def val_predictions(run: Path, inputs: str) -> pd.DataFrame:
    """Seed-averaged predictions on the validation split, in m³/m³, keyed by region/cell/month."""
    cfg = json.loads((run / "config.json").read_text())
    dm = DewStatData(window=cfg["window"], inputs=inputs, location=not cfg.get("no_location"),
                     regions_csv=[Path(p) for p in cfg["regions_csv"]], hls=cfg.get("hls", "pc"),
                     sat_baseline_start=cfg.get("sat_baseline_start"))
    dm.setup()
    m = dm.split["val"]
    X = torch.from_numpy(np.ascontiguousarray(dm.X[m]))
    seeds = []
    for ck in sorted((run / "checkpoints").glob("best_seed*.ckpt")):
        net = BiLSTM.load_from_checkpoint(ck, map_location="cpu").eval()
        with torch.no_grad():
            seeds.append(dm.to_soil_moisture(net(X).numpy(), "val"))
    idx = np.where(m)[0]
    return pd.DataFrame({
        "region": [dm.regions[dm.region[j]].name for j in idx], "cell": dm.cell[idx], "month": dm.when[idx],
        "actual": dm.y[idx], "pred": np.mean(seeds, axis=0),
    })


def main() -> int:
    rows = []
    for v, inputs in VERSIONS.items():
        preds = {}
        for label, tag in SETUPS.items():
            d = HERE / f"runs_{tag}_{inputs}"
            name = fair_pick(d)
            p = val_predictions(d / name, inputs)
            preds[label] = (name, p[(p["month"] >= "2020-01") & (p["month"] <= "2023-12")].set_index(
                ["region", "cell", "month"]))
            print(v, label, name, flush=True)
        common = sorted(set.intersection(*(set(p.index) for _, p in preds.values())))
        for label, (name, p) in preds.items():
            q = p.loc[common]
            test = json.loads((HERE / f"runs_{SETUPS[label]}_{inputs}" / name / "scores.json").read_text())["bilstm"]
            rows.append({"version": v, "setup": label, "run": name, "n": len(q),
                         "val_rmse": float(np.sqrt(np.mean((q["pred"] - q["actual"]) ** 2))),
                         "test_rmse": test["rmse"], "test_r2": test["r2"]})
    res = pd.DataFrame(rows)
    md = ["# Which HLS data setup? (picked on validation, not test)", "",
          "Each setup's run = lowest validation loss in its own sweep. Compared on the same cell-months: the 13 "
          "validation regions, 2020-01 to 2023-12. Lower validation RMSE wins. Test (22 regions, 2024+) is shown for "
          "reference only and was not used to choose.", "",
          "| Version | Setup | Picked run | Val cell-months | Val RMSE 2020-23 | Test RMSE (ref.) | Test R² (ref.) |",
          "|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        best = r["val_rmse"] == res[res["version"] == r["version"]]["val_rmse"].min()
        md.append(f"| {r['version']} | {r['setup']} | `{r['run']}` | {r['n']} | "
                  f"{'**' if best else ''}{r['val_rmse']:.4f}{'**' if best else ''} | {r['test_rmse']:.4f} | "
                  f"{r['test_r2']:.2f} |")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "setups.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
