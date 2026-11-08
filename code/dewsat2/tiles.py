"""Per-cell VCI tiles for the CNN models (DESIGN.md amendment 4).

For every region, VCI table and month, the per-pixel VCI and usable rule behind VCI_mean (features.py) are averaged
onto a TILE_H x TILE_W grid of near-equal pixel bins per cell (rows: 92 pixels -> 23 bins of 4; columns: 61-84
pixels -> 21 bins of 3-4), stored as uint8 0-100 with EMPTY (255) for a bin without usable pixels. A cell-month that
is unobserved under that table (usable pixels < 5% of the cell, the features' rule) gets an all-EMPTY tile.

Outputs, in the dataset's cell order (cache/dataset.json, cache/dev.npz, cache/test.npz):
    cache/tiles/dev_<ref>.npy   (n_dev_cells, DEV_END + 1, TILE_H, TILE_W) uint8 for ref in REFS (f1-f4, all)
    cache/tiles/test_all.npy    (n_test_cells, N_MONTHS, TILE_H, TILE_W) uint8 (table "all", as the features)
    cache/tiles/tiles.json      shapes, the checks below, timings
Checks, for every table and month: each cell's mean VCI recomputed from the same pixels equals the features'
VCI_mean, and the observed flags agree; the assembled cell ids equal the dataset's. No labels are read.

    .venv/bin/python -u code/dewsat2/tiles.py --workers 6            # all regions, then assemble
    .venv/bin/python -u code/dewsat2/tiles.py --names a,b --workers 2  # test: these regions only, no assembly
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CACHE, DEV_END, MIN_USABLE_FRAC, MONTHS, N_MONTHS, NDVI_SCALE, NODATA, REFS, read_json, write_json  # noqa: E402

TILE_H, TILE_W = 23, 21
EMPTY = 255
TILES = CACHE / "tiles"
MEAN_TOL = 1e-3          # VCI units; float32 pixels summed in float64 on both sides


def say(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] tiles: {msg}", flush=True)


def _edges(n: int, k: int) -> np.ndarray:
    e = np.rint(np.linspace(0, n, k + 1)).astype(np.int64)
    if (np.diff(e) <= 0).any():
        raise ValueError(f"{n} pixels cannot fill {k} bins")
    return e


def month_tiles(vci: np.ndarray, usable: np.ndarray, shape: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One month of one region: (n_cells, TILE_H, TILE_W) uint8 tiles, each cell's mean VCI over its usable pixels
    (NaN when unobserved) and the observed flag (usable >= 5% of the cell)."""
    from features import _cells
    rows, cols, py, px = shape
    n = rows * cols
    u = _cells(usable, *shape).reshape(n, py, px)
    v = np.where(u, _cells(vci, *shape).reshape(n, py, px), 0.0).astype(np.float64)
    ey, ex = _edges(py, TILE_H), _edges(px, TILE_W)
    s = np.add.reduceat(np.add.reduceat(v, ey[:-1], axis=1), ex[:-1], axis=2)
    c = np.add.reduceat(np.add.reduceat(u.astype(np.float64), ey[:-1], axis=1), ex[:-1], axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(c > 0, np.rint(s / np.maximum(c, 1.0)), EMPTY).astype(np.uint8)
        cnt = c.sum(axis=(1, 2))
        obs = cnt / (py * px) >= MIN_USABLE_FRAC
        mean = np.where(obs, s.sum(axis=(1, 2)) / np.maximum(cnt, 1.0), np.nan)
    t[~obs] = EMPTY
    return t, mean, obs


def load_ndvi(g: dict, grid: dict) -> np.ndarray:
    """NDVI (months, H, W) float32 with NaN: features.load_composites without the NDMI pass."""
    from regions import composite_dir
    d = composite_dir(g)
    ndvi = np.empty((N_MONTHS, grid["height"], grid["width"]), np.float32)
    for k, m in enumerate(MONTHS):
        with np.load(d / f"{m}.npz") as z:
            raw = z["ndvi"]
            src = json.loads(str(z["meta"]))["source"]
        if src not in ("planetary_computer_cog", "nasa_lpdaac_cog"):
            raise RuntimeError(f"{g['name']} {m}: composite from {src!r}")
        ndvi[k] = np.where(raw == NODATA, np.nan, raw / NDVI_SCALE).astype(np.float32)
    return ndvi


def process_region(args) -> dict:
    """Tiles for one region (cached as cache/tiles/regions/<name>.npz) with the checks against its features."""
    g, role = args
    from features import _ci, _loyo_tables, reference_mask
    from regions import composite_dir
    name = g["name"]
    out = TILES / "regions" / f"{name}.npz"
    if out.exists():
        return {"name": name, "cached": True}
    t0 = time.time()
    grid = json.loads((composite_dir(g) / "grid.json").read_text())
    shape = (grid["rows"], grid["cols"], grid["px_y"], grid["px_x"])
    n = grid["rows"] * grid["cols"]
    refs = list(REFS) if role == "dev" else ["all"]
    n_m = DEV_END + 1 if role == "dev" else N_MONTHS
    ndvi = load_ndvi(g, grid)
    tiles = np.full((len(refs), n, n_m, TILE_H, TILE_W), EMPTY, np.uint8)
    worst, obs_mismatch, compared = 0.0, 0, 0
    with np.load(CACHE / "regions" / f"{name}.npz") as fz, warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ids = fz["ids"]
        if len(ids) != n:
            raise RuntimeError(f"{name}: {len(ids)} cells in the features, {n} in the grid")
        for i, ref in enumerate(refs):
            r = list(REFS).index(ref)
            f_mean = fz["feats"][r, :, :, 2]
            f_obs = fz["obs"][r]
            for k, lo, hi, n_k in _loyo_tables(ndvi, reference_mask(ref)):
                if k >= n_m:
                    continue
                v = ndvi[k]
                vci, ok, _ = _ci(v, lo, hi, n_k)
                t, mean, obs = month_tiles(vci, np.isfinite(v) & ok, shape)
                tiles[i, :, k] = t
                obs_mismatch += int((obs != f_obs[:, k]).sum())
                both = obs & f_obs[:, k]
                if both.any():
                    worst = max(worst, float(np.max(np.abs(mean[both] - f_mean[both, k]))))
                    compared += int(both.sum())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, tiles=tiles, ids=ids, refs=np.array(refs))
    os.replace(tmp, out)
    return {"name": name, "cached": False, "role": role, "cells": n, "seconds": round(time.time() - t0, 1),
            "worst_vci_mean_diff": worst, "obs_mismatch": obs_mismatch, "cell_months_compared": compared,
            "px": [grid["px_y"], grid["px_x"]]}


def assemble(dataset_meta: dict) -> dict:
    """Stack the regions' tiles in the dataset's cell order (development cells after the label-history filter)."""
    out = {}
    for part in ("dev", "test"):
        if part not in dataset_meta:
            continue
        names = dataset_meta[part]["regions"]
        n_cells = dataset_meta[part]["n_cells"]
        refs = list(REFS) if part == "dev" else ["all"]
        n_m = DEV_END + 1 if part == "dev" else N_MONTHS
        with np.load(CACHE / f"{part}.npz") as z:
            ids_ds = z["cell_id"]
        tmp = {ref: TILES / f"{part}_{ref}.tmp.npy" for ref in refs}
        mm = {ref: np.lib.format.open_memmap(tmp[ref], mode="w+", dtype=np.uint8,
                                             shape=(n_cells, n_m, TILE_H, TILE_W)) for ref in refs}
        pos, ids_all = 0, []
        for name in names:
            with np.load(TILES / "regions" / f"{name}.npz") as z, np.load(CACHE / "regions" / f"{name}.npz") as fz:
                keep = fz["keep"] if part == "dev" else np.ones(len(fz["ids"]), bool)
                t = z["tiles"]
                if list(z["refs"]) != refs or t.shape[2] != n_m:
                    raise RuntimeError(f"{name}: tiles built for {list(z['refs'])} x {t.shape[2]} months")
                ids = z["ids"][keep]
                for i, ref in enumerate(refs):
                    mm[ref][pos:pos + len(ids)] = t[i][keep]
                ids_all.append(ids)
                pos += len(ids)
        ids_all = np.concatenate(ids_all)
        if pos != n_cells or not np.array_equal(ids_all, ids_ds):
            raise RuntimeError(f"{part}: assembled cell ids do not match the dataset ({pos} vs {n_cells} cells)")
        for ref in refs:
            mm[ref].flush()
            del mm[ref]
            os.replace(tmp[ref], TILES / f"{part}_{ref}.npy")
        out[part] = {"cells": int(n_cells), "months": int(n_m), "refs": refs, "regions": len(names),
                     "files": [f"cache/tiles/{part}_{ref}.npy" for ref in refs]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--names", default=None, help="comma-separated region names (test: no assembly)")
    a = ap.parse_args()
    from regions import selected_regions
    t0 = time.time()
    meta = read_json(CACHE / "dataset.json")
    role = {nm: part for part in ("dev", "test") if part in meta for nm in meta[part]["regions"]}
    regs = [g for g in selected_regions(frozen=True) if g["name"] in role]
    if len(regs) != len(role):
        raise RuntimeError(f"{len(role)} regions in the dataset, {len(regs)} found")
    if a.names:
        want = set(a.names.split(","))
        regs = [g for g in regs if g["name"] in want]
    say(f"{len(regs)} regions, {a.workers} workers")
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(process_region, [(g, role[g["name"]]) for g in regs]), 1):
            res.append(r)
            if not r.get("cached") and (i % 25 == 0 or i == len(regs) or a.names):
                say(f"{i}/{len(regs)} {r['name']} {r['seconds']}s, worst VCI_mean diff {r['worst_vci_mean_diff']:.2e}, "
                    f"observed mismatches {r['obs_mismatch']}")
    built = [r for r in res if not r.get("cached")]
    worst = max([r["worst_vci_mean_diff"] for r in built], default=0.0)
    mism = sum(r["obs_mismatch"] for r in built)
    say(f"built {len(built)} (cached {len(res) - len(built)}); worst VCI_mean diff {worst:.2e}; observed mismatches {mism}")
    if worst > MEAN_TOL or mism:
        say("CHECK FAILED: tiles do not match the features")
        return 1
    if a.names:
        return 0
    info = assemble(meta)
    commit = subprocess.run(["git", "-C", str(HERE.parent), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    write_json(TILES / "tiles.json", {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "code_commit": commit,
                                      "tile": [TILE_H, TILE_W], "empty": EMPTY, "parts": info,
                                      "check": {"worst_vci_mean_diff_this_build": worst, "tolerance": MEAN_TOL,
                                                "observed_mismatches_this_build": mism,
                                                "regions_built": len(built), "regions_cached": len(res) - len(built)},
                                      "minutes": round((time.time() - t0) / 60, 1)})
    say(f"assembled: {json.dumps(info)}; {round((time.time() - t0) / 60, 1)} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
