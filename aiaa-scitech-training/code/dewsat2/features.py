"""Per-region features for every VCI table, labels, NDMI features, and per-view tables (DESIGN.md section 1).

Pixel VCI rules are the paper's (paper_hls/build_dataset.py) as implemented in the capability run's features.py
(commit e13e7a4, which reproduced the paper's tables on all 291,691 rows), with the reference months changed to
development months only (DESIGN.md 1). Per VCI table r in REFS (f1..f4, all):
  pixel: NDVI_mean, NDVI_std, VCI_mean, VCI_p10, valid_frac over usable pixels (finite NDVI and a valid reference);
         observed = usable >= 5% of the cell.
  cell:  VCIc = VCI of the cell's mean NDVI (all clear pixels) against a per-cell table (same reference months,
         leave-one-year-out, >= 3 years, range >= 0.05), VCIc_p10 = the cell's NDVI 10th percentile against the same
         table; observed_c = clear pixels >= 5% of the cell. NDVIc_mean/std/p10 and clear_frac do not depend on r.
  ndmi:  NDMI_mean over usable pixels with a valid NDMI; NDMI_CI_mean = per-pixel NDMI condition index, built like
         VCI (per-pixel table from the same reference months, leave-one-year-out).
Labels for development regions only (fresh-test labels stay locked).
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np

from common import (CACHE, DEV_END, DROUGHT_QUANTILE, FLDAS_DEV, LABEL_MIN_REF, LABEL_REF, MIN_USABLE_FRAC, MONTHS,
                    N_MONTHS, NDVI_SCALE, NODATA, REF_MIN_RANGE, REF_MIN_YEARS, REFS, BLOCK_IDX, BUFFER_NOWCAST,
                    save_npz)
from regions import cell_centers, composite_dir, scene_dir

CALS = np.array([int(m[5:]) for m in MONTHS])
PIX_F = ["NDVI_mean", "NDVI_std", "VCI_mean", "VCI_p10", "valid_frac"]
CELL_F = ["VCIc", "VCIc_p10"]
CELL_FIXED = ["NDVIc_mean", "NDVIc_std", "NDVIc_p10", "clear_frac"]
NDMI_F = ["NDMI_mean", "NDMI_CI_mean"]
VIEW_FIELDS = ("covered_frac", "clear_frac", "n_clear", "ndvi_mean", "ndvi_std", "ndvi_p10", "ndmi_mean", "n_ndmi")


def reference_mask(ref: str) -> np.ndarray:
    """Months whose NDVI may enter table `ref`: development months outside the fold's block +/- 12 (all: every
    development month)."""
    idx = np.arange(N_MONTHS)
    dev = idx <= DEV_END
    if ref == "all":
        return dev
    k = int(ref[1:]) - 1
    bs, be = BLOCK_IDX[k]
    return dev & ((idx < bs - BUFFER_NOWCAST) | (idx > be + BUFFER_NOWCAST))


def _cells(a, rows, cols, px_y, px_x):
    return a.reshape(rows, px_y, cols, px_x).transpose(0, 2, 1, 3).reshape(rows * cols, px_y * px_x)


def _p10(vals, cnt):
    """10th percentile (numpy linear) along axis 1 of NaN-padded rows with cnt finite values each (the paper's)."""
    s = np.sort(vals, axis=1)
    cnt = cnt.astype(np.int64)
    pos = 0.1 * (np.maximum(cnt, 1) - 1)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, np.maximum(cnt, 1) - 1)
    t = pos - i0
    a = np.take_along_axis(s, i0[:, None], 1)[:, 0]
    b = np.take_along_axis(s, i1[:, None], 1)[:, 0]
    return np.where(t < 0.5, a + t * (b - a), b - (1 - t) * (b - a))


def _aggregate(v, vci, usable, shape):
    """Cell means/std/p10 over usable pixels, float64 (the paper's definitions)."""
    cu = _cells(usable, *shape)
    cnt = cu.sum(axis=1)
    frac = cu.mean(axis=1)
    out = np.full((cu.shape[0], 5), np.nan, np.float32)
    ok = frac >= MIN_USABLE_FRAC
    if not ok.any():
        return out, ok
    cu = cu[ok]
    c = cnt[ok].astype(np.float64)
    nv = np.where(cu, _cells(v, *shape)[ok].astype(np.float64), np.nan)
    vv = np.where(cu, _cells(vci, *shape)[ok].astype(np.float64), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mn = np.nansum(nv, axis=1) / c
        sd = np.sqrt(np.nansum((nv - mn[:, None]) ** 2, axis=1) / c)
        mv = np.nansum(vv, axis=1) / c
    p10 = _p10(vv, c)
    out[ok] = np.stack([mn, sd, mv, p10, frac[ok]], axis=1).astype(np.float32)
    return out, ok


def _loyo_tables(stack_all: np.ndarray, ref_mask: np.ndarray):
    """Generator over months k: (lo, hi, n) per element for month k, leave-one-year-out when k is a reference month.

    stack_all: (months, ...) float array with NaN for missing. Done per calendar month.
    """
    for cal in range(1, 13):
        sel = np.where(CALS == cal)[0]
        rsel = sel[ref_mask[sel]]
        stack = stack_all[rsel]
        n_all = np.isfinite(stack).sum(axis=0)
        if len(rsel) == 0:
            nanv = np.full(stack_all.shape[1:], np.nan, np.float32)
            for k in sel:
                yield k, nanv, nanv, np.zeros(stack_all.shape[1:], np.int64)
            continue
        srt = np.sort(stack, axis=0)
        lo_full = srt[0]
        hi_full = np.take_along_axis(srt, np.clip(n_all - 1, 0, None)[None], 0)[0]
        if len(rsel) >= 2:
            lo_2 = srt[1]
            hi_2 = np.take_along_axis(srt, np.clip(n_all - 2, 0, None)[None], 0)[0]
        else:
            lo_2 = hi_2 = np.full_like(lo_full, np.nan)
        for k in sel:
            v = stack_all[k]
            if ref_mask[k]:
                fin = np.isfinite(v)
                n_k = n_all - fin
                lo = np.where(fin & (lo_full == v), lo_2, lo_full)
                hi = np.where(fin & (hi_full == v), hi_2, hi_full)
            else:
                n_k, lo, hi = n_all, lo_full, hi_full
            yield k, lo, hi, n_k


def _ci(v, lo, hi, n_k):
    ok = (n_k >= REF_MIN_YEARS) & ((hi - lo) >= REF_MIN_RANGE)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = 100.0 * (v - lo) / (hi - lo)
    return np.clip(raw, 0.0, 100.0), ok, raw


def labels_for(name: str, n_cells: int, centers) -> tuple[np.ndarray, np.ndarray]:
    """Drought flag (cells, months) from data/fldas/dev (paper rule), and the keep mask (20 reference years)."""
    z = np.load(FLDAS_DEV / f"{name}.npz")
    months = list(z["months"])
    vals = z["values"]                                 # (cells, 309)
    lon, lat = z["cell_lon"], z["cell_lat"]
    order = {(round(float(a), 2), round(float(b), 2)): i for i, (a, b) in enumerate(zip(lon, lat))}
    idx = [order[c] for c in centers]
    vals = vals[idx]
    yrs = np.array([int(m[:4]) for m in months])
    cal = np.array([int(m[5:]) for m in months])
    drought = np.full((n_cells, N_MONTHS), -1, np.int8)
    keep = np.zeros(n_cells, bool)
    pos = {m: j for j, m in enumerate(months)}
    target_j = np.array([pos[m] for m in MONTHS])
    for i in range(n_cells):
        q20 = np.full(13, np.nan)
        okc = True
        for c in range(1, 13):
            ref = vals[i, (cal == c) & (yrs >= int(LABEL_REF[0])) & (yrs <= int(LABEL_REF[1]))]
            ref = ref[np.isfinite(ref)]
            if len(ref) < LABEL_MIN_REF:
                okc = False
                break
            q20[c] = float(np.quantile(np.sort(ref), DROUGHT_QUANTILE))
        if not okc:
            continue
        keep[i] = True
        v = vals[i, target_j]
        fin = np.isfinite(v)
        drought[i, fin] = (v[fin] < q20[CALS[fin]]).astype(np.int8)
    return drought, keep


def load_composites(g: dict, grid: dict):
    """NDVI (months, H, W) float32 with NaN, and NDMI (or None), clear-date counts, per-month metadata."""
    d = composite_dir(g)
    H, W = grid["height"], grid["width"]
    ndvi = np.empty((N_MONTHS, H, W), np.float32)
    meta = []
    for k, m in enumerate(MONTHS):
        with np.load(d / f"{m}.npz") as z:
            raw = z["ndvi"]
            mt = json.loads(str(z["meta"]))
        if mt["source"] not in ("planetary_computer_cog", "nasa_lpdaac_cog"):
            raise RuntimeError(f"{g['name']} {m}: composite from {mt['source']!r}")
        ndvi[k] = np.where(raw == NODATA, np.nan, raw / NDVI_SCALE).astype(np.float32)
        meta.append({"dates": mt.get("dates", []), "scenes_found": mt.get("scenes_found"), "source": mt["source"]})
    nd = None
    sd = scene_dir(g)
    if sd is not None and all((sd / f"{m}.npz").exists() for m in MONTHS):
        nd = np.empty((N_MONTHS, H, W), np.float32)
        for k, m in enumerate(MONTHS):
            with np.load(sd / f"{m}.npz") as z:
                if "ndmi" not in z.files:
                    nd = None
                    break
                raw = z["ndmi"]
            nd[k] = np.where(raw == NODATA, np.nan, raw / NDVI_SCALE).astype(np.float32)
    return ndvi, nd, meta


def view_table(g: dict, grid: dict) -> dict | None:
    """Per-view, per-cell values over all months (None when the region has no per-scene record)."""
    sd = scene_dir(g)
    if sd is None or not all((sd / f"{m}.npz").exists() for m in MONTHS):
        return None
    parts = {f: [] for f in VIEW_FIELDS}
    date, sensor, tile = [], [], []
    for m in MONTHS:
        with np.load(sd / f"{m}.npz") as z:
            if "view_date" not in z.files:
                return None
            date.append(z["view_date"])
            sensor.append(z["view_sensor"])
            tile.append(z["view_tile"])
            for f in VIEW_FIELDS:
                parts[f].append(z[f"view_{f}"])
    out = {"date": np.concatenate(date), "sensor": np.concatenate(sensor), "tile": np.concatenate(tile)}
    for f in VIEW_FIELDS:
        out[f] = np.concatenate(parts[f]).astype(np.int32 if f.startswith("n_") else np.float32)
    return out


def process_region(g: dict, with_labels: bool) -> dict:
    """Every table's features for one region, cached as one npz."""
    path = CACHE / "regions" / f"{g['name']}.npz"
    if path.exists():
        return {"name": g["name"], "cached": True}
    t0 = time.time()
    grid = json.loads((composite_dir(g) / "grid.json").read_text())
    rows, cols, px_y, px_x = grid["rows"], grid["cols"], grid["px_y"], grid["px_x"]
    shape = (rows, cols, px_y, px_x)
    n_cells = rows * cols
    centers = cell_centers(grid)
    ids = np.array([f"{g['name']}:{lo:.2f}:{la:.2f}" for lo, la in centers])
    ndvi, ndmi, meta = load_composites(g, grid)
    # reference-independent cell values over all clear pixels of the composite
    fixed = np.full((n_cells, N_MONTHS, len(CELL_FIXED)), np.nan, np.float32)
    cell_ndvi = np.full((N_MONTHS, n_cells), np.nan, np.float64)
    cell_p10 = np.full((N_MONTHS, n_cells), np.nan, np.float64)
    obs_c = np.zeros((n_cells, N_MONTHS), bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(N_MONTHS):
            cv = _cells(ndvi[k], *shape).astype(np.float64)
            fin = np.isfinite(cv)
            cnt = fin.sum(axis=1)
            frac = fin.mean(axis=1)
            ok = frac >= MIN_USABLE_FRAC
            mn = np.where(cnt > 0, np.nansum(cv, axis=1) / np.maximum(cnt, 1), np.nan)
            sd = np.sqrt(np.where(cnt > 0, np.nansum((cv - mn[:, None]) ** 2, axis=1) / np.maximum(cnt, 1), np.nan))
            p10 = np.where(cnt > 0, _p10(cv, cnt), np.nan)
            fixed[:, k] = np.stack([mn, sd, p10, frac], axis=1)
            fixed[~ok, k, :3] = np.nan
            obs_c[:, k] = ok
            cell_ndvi[k] = np.where(ok, mn, np.nan)
            cell_p10[k] = np.where(ok, p10, np.nan)
    R = len(REFS)
    feats = np.full((R, n_cells, N_MONTHS, len(PIX_F)), np.nan, np.float32)
    obs = np.zeros((R, n_cells, N_MONTHS), bool)
    cellf = np.full((R, n_cells, N_MONTHS, len(CELL_F)), np.nan, np.float32)
    nd_f = None if ndmi is None else np.full((R, n_cells, N_MONTHS, len(NDMI_F)), np.nan, np.float32)
    clip = np.zeros((R, N_MONTHS, 3), np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for r, ref in enumerate(REFS):
            rm = reference_mask(ref)
            usable_by_month = {}
            for k, lo, hi, n_k in _loyo_tables(ndvi, rm):
                v = ndvi[k]
                vci, ok, raw = _ci(v, lo, hi, n_k)
                usable = np.isfinite(v) & ok
                usable_by_month[k] = usable
                clip[r, k] = (int(usable.sum()), int((usable & (raw < -1e-3)).sum()),
                              int((usable & (raw > 100 + 1e-3)).sum()))
                feats[r, :, k], obs[r, :, k] = _aggregate(v, vci, usable, shape)
            # per-cell table on the cell-mean NDVI (and the cell's NDVI 10th percentile against it)
            for k, lo, hi, n_k in _loyo_tables(cell_ndvi, rm):
                vc, okc, _ = _ci(cell_ndvi[k], lo, hi, n_k)
                good = okc & np.isfinite(cell_ndvi[k])
                cellf[r, :, k, 0] = np.where(good, vc, np.nan)
                vp, _, _ = _ci(cell_p10[k], lo, hi, n_k)
                cellf[r, :, k, 1] = np.where(good & np.isfinite(cell_p10[k]), vp, np.nan)
            if nd_f is not None:
                for k, lo, hi, n_k in _loyo_tables(ndmi, rm):
                    v = ndmi[k]
                    ci, okn, _ = _ci(v, lo, hi, n_k)
                    use = usable_by_month[k] & np.isfinite(v)
                    cu = _cells(use, *shape)
                    nv = np.where(cu, _cells(v, *shape), np.nan)
                    cv = np.where(cu & _cells(okn, *shape), _cells(ci, *shape), np.nan)
                    o = obs[r, :, k]
                    nd_f[r, :, k, 0] = np.where(o & (cu.sum(axis=1) > 0), np.nanmean(nv, axis=1), np.nan)
                    nd_f[r, :, k, 1] = np.where(o & (np.isfinite(cv).sum(axis=1) > 0), np.nanmean(cv, axis=1), np.nan)
    keep = np.ones(n_cells, bool)
    drought = np.full((n_cells, N_MONTHS), -1, np.int8)
    if with_labels:
        drought, keep = labels_for(g["name"], n_cells, centers)
    vt = view_table(g, grid)
    arrays = {"ids": ids, "keep": keep, "drought": drought, "feats": feats, "obs": obs, "cellf": cellf,
              "fixed": fixed, "obs_c": obs_c, "clip": clip,
              "n_dates": np.array([len({x for x in mt["dates"] if x[:7] == m}) for mt, m in zip(meta, MONTHS)], np.int16)}
    if nd_f is not None:
        arrays["ndmi"] = nd_f
    if vt is not None:
        for kk, vv in vt.items():
            arrays[f"view_{kk}"] = vv
    save_npz(path, **arrays)
    return {"name": g["name"], "cached": False, "seconds": round(time.time() - t0, 1), "cells": n_cells,
            "has_ndmi": nd_f is not None, "has_views": vt is not None}
