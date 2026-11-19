"""Train the Bi-LSTM on soil moisture (10-40 cm) across many regions and score it on held-out regions.

Run from the model folder (after the data pull has finished at least some regions); on Windows use
.venv-train\\Scripts\\python instead of .venv-train/bin/python:
    .venv-train/bin/python train.py --inputs satellite_anomaly --hidden 16 --window 12 --seeds 3   # the chosen V2
    .venv-train/bin/python train.py --inputs satellite_swir --hidden 16 --window 12 --seeds 3      # V3 (SWIR)
    bash run_sweeps.sh                                                                           # all settings

Scores are reported on two test sets (see dew_stat_data.py):
    test        test regions, 2024+   new place AND new time: the honest number
    test_same   train regions, 2024+  same place, new time

Each run writes to runs/<name>/:
    checkpoints/best_seed*.ckpt   weights from the epoch with the lowest val_loss
    logs/seed*/metrics.csv        train/val loss and val R2 per epoch
    predictions_test.csv          actual vs predicted, every test cell-month (with region)
    summary.txt / scores.json     final RMSE / R2 vs the baselines
    plots/*.png                   charts (see plot_results.py)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from dew_stat_data import DewStatData
from dew_stat_nldas import DewStatNLDAS
from dew_stat_model_fast import BiLSTM
from plot_results import make_plots

warnings.filterwarnings("ignore", ".*does not have many workers.*")

RIDGE_ALPHAS = (1, 10, 100, 1000, 10000)
RIDGE_MONTHS = 3  # ridge sees the last 3 months of features, laid out in one row


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def r2(pred, y) -> float:
    """R^2 of the soil-moisture LEVEL: 1 = perfect, 0 = no better than the test-period average."""
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def r2_change(pred, y, prev) -> float:
    """R^2 of the month-to-month CHANGE. Persistence scores ~0 here, so this is the honest one."""
    d = y - prev
    return float(1 - np.sum((y - pred) ** 2) / np.sum((d - d.mean()) ** 2))


def fit_ridge(dm: DewStatData) -> dict[str, np.ndarray]:
    """Linear baseline on the same samples: weighted sum of the last 3 months of features.

    Alpha (how hard the weights are pulled toward zero) is picked on the val regions.
    Returns scaled-change predictions for every split.
    """
    k = min(RIDGE_MONTHS, dm.window)
    flat = dm.X[:, -k:].reshape(len(dm.X), -1).astype(np.float64)
    Xb = np.hstack([flat, np.ones((len(flat), 1))])
    tr, va = dm.split["train"], dm.split["val"]
    XtX, Xty = Xb[tr].T @ Xb[tr], Xb[tr].T @ dm.target[tr, 0]
    best, best_w = np.inf, None
    for alpha in RIDGE_ALPHAS:
        reg = alpha * np.eye(Xb.shape[1])
        reg[-1, -1] = 0  # do not shrink the intercept
        w = np.linalg.solve(XtX + reg, Xty)
        val = float(np.mean((Xb[va] @ w - dm.target[va, 0]) ** 2))
        if val < best:
            best, best_w = val, w
    return {s: Xb[m] @ best_w for s, m in dm.split.items()}


def train_one(dm: DewStatData, args, seed: int, run_dir: Path) -> tuple[dict[str, np.ndarray], str, float]:
    L.seed_everything(seed, workers=True)
    model = BiLSTM(
        input_size=dm.input_size,
        hidden_size=args.hidden,
        lr=args.lr,
        num_layers=args.layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
    )
    ckpt = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", filename=f"best_seed{seed}", monitor="val_loss", mode="min"
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",  # uses the RTX 4050 when CUDA is available
        callbacks=[EarlyStopping("val_loss", patience=args.patience), ckpt, LearningRateMonitor()],
        logger=CSVLogger(run_dir, name="logs", version=f"seed{seed}"),
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=not args.quiet,
        enable_model_summary=not args.quiet,
    )
    trainer.fit(model, datamodule=dm)
    best = BiLSTM.load_from_checkpoint(ckpt.best_model_path)
    trainer.test(best, datamodule=dm, verbose=False)
    preds = {
        s: torch.cat(trainer.predict(best, dataloaders=dm.loader(s))).float().cpu().numpy()[:, 0]
        for s in ("test", "test_same")
        if len(dm.sets[s])
    }
    return preds, ckpt.best_model_path, float(ckpt.best_model_score)


def seasonal(dm: DewStatData, split: str) -> np.ndarray:
    """Baseline that only knows the date: train regions' average soil moisture for that calendar month."""
    cal = np.array([w[5:7] for w in dm.when])
    tr = dm.split["train"]
    avg = {c: float(dm.y[tr & (cal == c)].mean()) for c in np.unique(cal[tr])}
    fallback = float(dm.y[tr].mean())
    return np.array([avg.get(c, fallback) for c in cal[dm.split[split]]])


def score_block(dm: DewStatData, split: str, preds: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    m = dm.split[split]
    y, prev = dm.y[m], dm.prev[m]
    out = {}
    for name, p in preds.items():
        out[name] = {"rmse": rmse(p, y), "r2": r2(p, y)}
        if dm.inputs == "full":  # the satellite model never sees last month, so "change" is not its job
            out[name]["r2_change"] = r2_change(p, y, prev)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=time.strftime("%Y%m%d-%H%M%S"), help="run folder under runs/")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--window", type=int, default=6, help="months of history per sample")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15, help="stop after this many epochs without val improvement")
    ap.add_argument("--seeds", type=int, default=1, help="train N copies and average them (more stable)")
    ap.add_argument("--quiet", action="store_true", help="no progress bars (for overnight logs)")
    ap.add_argument("--regions-csv", type=Path, nargs="+", default=None,
                    help="region lists to use (default: practice/regions.csv and practice/regions_new.csv)")
    ap.add_argument("--regions-dir", type=Path, default=None, help="override practice/regions/")
    ap.add_argument("--inputs", choices=("full", "satellite", "satellite_anomaly", "satellite_swir"), default="full",
                    help="full = FLDAS + satellite; satellite = only what a satellite sees (FLDAS is just the label); "
                         "satellite_anomaly = same inputs, label = wetter/drier than this cell's normal; "
                         "satellite_swir = like satellite_anomaly with VCI swapped for NDMI/NDWI (SWIR bands)")
    ap.add_argument("--no-location", action="store_true", help="satellite inputs without latitude/longitude")
    ap.add_argument("--runs-dir", default="runs", help="folder (next to this file) to put the run in")
    ap.add_argument("--hls", choices=("pc", "full"), default="pc",
                    help="pc = Planetary Computer HLS tables (effectively 2020+); "
                         "full = joined with NASA's 2016-2019 (hls_full.csv / vci_full.csv, merge_hls_years.py)")
    ap.add_argument("--sat-baseline-start", default=None,
                    help="with --hls full: index normals and VCI range from this month on (e.g. 2020-01); needs "
                         "vci_full_b<YYYY>.csv (merge_hls_years.py --vci-baseline)")
    ap.add_argument("--answer", choices=("fldas", "nldas_month", "nldas_week"), default="fldas",
                    help="answer key: FLDAS monthly (default), or NLDAS-2 monthly / weekly (dew_stat_nldas.py; "
                         "satellite_anomaly or satellite inputs, --window counts steps)")
    ap.add_argument("--val-years", choices=("all", "gap"), default="all",
                    help="all = val regions, every year (overlaps train and test years); gap = val regions, 2023 only "
                         "(no year shared with training or test)")
    args = ap.parse_args()
    if args.answer != "fldas" and args.inputs not in ("satellite_anomaly", "satellite"):
        raise SystemExit("--answer nldas_* only supports --inputs satellite_anomaly or satellite")
    if args.sat_baseline_start and args.hls != "full":
        raise SystemExit("--sat-baseline-start only applies with --hls full")

    torch.set_float32_matmul_precision("high")
    kw = {k: v for k, v in (("regions_csv", args.regions_csv), ("regions_dir", args.regions_dir)) if v}
    if args.answer == "fldas":
        dm = DewStatData(window=args.window, batch_size=args.batch, inputs=args.inputs,
                         location=not args.no_location, hls=args.hls, sat_baseline_start=args.sat_baseline_start,
                         val_years=args.val_years, **kw)
    else:
        dm = DewStatNLDAS(step=args.answer.split("_")[1], window=args.window, batch_size=args.batch,
                          location=not args.no_location, inputs=args.inputs, val_years=args.val_years, **kw)
    dm.setup()
    n = {k: len(v) for k, v in dm.sets.items()}
    print(f"regions loaded: {len(dm.regions)}" + (f" (not pulled yet: {', '.join(dm.missing)})" if dm.missing else ""))
    for s in ("train", "val", "test", "test_same"):
        print(f"  {s:<10} {n[s]:>7} samples  {', '.join(dm.region_names(s)) or '-'}")
    if not n["val"] or not n["test"]:
        print("  (need at least one finished val region and one test region for a fair score)")
    print(f"inputs: {args.inputs} ({', '.join(dm.features)})")
    print(f"x shape ({args.window}, {dm.input_size})", flush=True)
    if not n["val"]:
        raise SystemExit("no val regions finished yet; nothing to pick the best epoch with")

    run_dir = Path(__file__).resolve().parent / args.runs_dir / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Everything needed to reuse the model later on new data.
    (run_dir / "config.json").write_text(json.dumps({
        **{k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v
           for k, v in vars(args).items()},
        "regions": {s: dm.region_names(s) for s in dm.split},
        "features": dm.features,
        "feature_mean": dm.feat_mean.tolist(),
        "feature_std": dm.feat_std.tolist(),
        "target": {"full": "change from last month", "satellite": "level"}.get(
            args.inputs, "level minus the cell's normal for the month"),
        "target_center": dm.t_center,
        "target_scale": dm.t_scale,
        "d_scale": dm.d_scale,
        "train_end": dm.train_end,
        "val_end": dm.val_end,
    }, indent=2))

    ridge = fit_ridge(dm)
    all_preds, lines = [], []
    for seed in range(args.seeds):
        preds, path, val_loss = train_one(dm, args, seed, run_dir)
        all_preds.append(preds)
        lines.append(f"seed {seed}: best val_loss {val_loss:.4f} -> {path}")
        print(lines[-1], flush=True)

    full = args.inputs == "full"
    # The yardstick: persistence needs last month's soil moisture, which a satellite does not have,
    # so the satellite model is measured against a guess that only knows the date. The anomaly
    # model is measured against its own starting point: the normal map alone (anomaly = 0).
    base_key = {"full": "persistence", "satellite": "seasonal"}.get(args.inputs, "normal")
    labels = {
        "persistence": "persistence (same as last month)",
        "seasonal": "seasonal average (date only)",
        "normal": "normal map alone (no model)",
        "ridge": "ridge (linear, same data)",
        "bilstm": f"Bi-LSTM ({args.seeds} seed avg)",
    }
    scores: dict = {"name": args.name, "inputs": args.inputs}
    level = {}
    for split, title in (("test", "NEW regions, 2024+ (the honest score)"), ("test_same", "TRAIN regions, 2024+")):
        if not n[split]:
            continue
        m = dm.split[split]
        base_pred = {"persistence": dm.prev[m], "seasonal": seasonal(dm, split), "normal": dm.normal[m]}
        level[split] = {
            base_key: base_pred[base_key],
            "ridge": dm.to_soil_moisture(ridge[split], split),
            "bilstm": dm.to_soil_moisture(np.mean([p[split] for p in all_preds], axis=0), split),
        }
        block = score_block(dm, split, level[split])
        # Every yardstick, for reference (the report can compare against any of them).
        refs = score_block(dm, split, {k: v for k, v in base_pred.items() if k != base_key})
        base = block[base_key]["rmse"]
        vs = {"persistence": "vs persist", "seasonal": "vs season", "normal": "vs normal"}[base_key]
        lines += [
            "",
            f"{title}: {', '.join(dm.region_names(split))} ({m.sum()} cell-months)",
            f"  {'model':<36}{'RMSE':>9}{'R2':>8}" + (f"{'R2 change':>11}" if full else "") + f"{vs:>12}",
        ]
        for key, s in block.items():
            lines.append(f"  {labels[key]:<36}{s['rmse']:>9.5f}{s['r2']:>8.3f}"
                         + (f"{s['r2_change']:>11.3f}" if full else "")
                         + f"{(1 - s['rmse'] / base) * 100:>+11.1f}%")
        if not full:  # for reference: other yardsticks (persistence needs last month's ground data)
            for key, p in refs.items():
                tag = "(ref) persistence, needs ground data" if key == "persistence" else f"(ref) {labels[key]}"
                lines.append(f"  {tag:<36}{p['rmse']:>9.5f}{p['r2']:>8.3f}")
        if split == "test":
            scores.update(block)
            scores["refs"] = refs
        else:
            scores["same_place"] = block
            scores["same_place_refs"] = refs
    lines += ["", "R2        = how much of the soil-moisture level is explained (1 = perfect)"]
    if full:
        lines.append("R2 change = how much of the month-to-month movement is explained (0 = no better than persistence)")
    summary = "\n".join(lines)
    print("\n" + summary)
    (run_dir / "summary.txt").write_text(summary + "\n")
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=2))

    if "test" in level:
        m = np.where(dm.split["test"])[0]
        with (run_dir / "predictions_test.csv").open("w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            models = (base_key, "ridge", "bilstm")
            wr.writerow(["region", "cell_longitude", "cell_latitude", "month", "actual", *models])
            for i, j in enumerate(m):
                g = dm.regions[dm.region[j]]
                lon, lat = g.cells[dm.cell[j]]
                wr.writerow([g.name, lon, lat, dm.when[j], round(dm.y[j], 5),
                             *(round(level["test"][k][i], 5) for k in models)])
        make_plots(run_dir)
        print(f"wrote {run_dir} (charts in {run_dir / 'plots'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
