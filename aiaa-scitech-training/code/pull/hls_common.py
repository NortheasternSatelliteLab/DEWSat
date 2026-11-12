"""Clear-pixel rule, per-pixel monthly composites (NDVI as before, plus NDMI), and per-view, per-cell values.

The cloud rule and the NDVI composite are the paper run's (code/paper_hls/hls_common.py, commit f17e700, after
Austin Schlather's build_table.py): Fmask bits 1-5 (cloud, adjacent cloud/shadow, shadow, snow/ice, water) and
high aerosol are rejected, Red and NIR must be positive, a scene only counts for a cell when at least 30% of the
cell is clear, and the monthly composite is the per-pixel median over the month's distinct acquisition dates
(overlapping tiles of one date averaged first). `clear_ndvi` and `MonthComposite` are unchanged.

New here:
- NDMI = (NIR - SWIR1) / (NIR + SWIR1) on the same clear pixels (SWIR1 also positive), composited the same way.
- Per-view, per-cell values. A view is one (sensor, MGRS tile, date): granules that share all three (Landsat 8/9
  underfly days, adjacent Landsat rows of one overpass) are merged pixel by pixel (mean where both are clear), so a
  view is counted once. Per cell: covered fraction (scene has data), clear fraction (clear pixels / cell pixels,
  no 30% rule applied), and over clear pixels NDVI mean, spread (population SD), 10th percentile (linear) and NDMI
  mean.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np

FMASK_REJECT = 0b0011_1110
FMASK_AEROSOL_HIGH = 0b1100_0000
FMASK_FILL = 255
MIN_CLEAR = 0.3
NDVI_SCALE = 10_000
NODATA = -32768


def clear_mask(red, nir, fmask, valid):
    """The paper's per-pixel clear rule (no 30% cell rule)."""
    red = red.astype(np.int32)
    nir = nir.astype(np.int32)
    fm = fmask.astype(np.int32)
    return (valid & (fm != FMASK_FILL) & ((fm & FMASK_REJECT) == 0)
            & ((fm & FMASK_AEROSOL_HIGH) != FMASK_AEROSOL_HIGH) & (red > 0) & (nir > 0))


def clear_ndvi(red, nir, fmask, valid, px_x: int, px_y: int, rows: int, cols: int,
               min_clear: float = MIN_CLEAR):
    """NDVI where a pixel is clear and its cell is >= 30% clear (NaN elsewhere), and the per-cell clear fraction.

    Unchanged from the paper run.
    """
    red = red.astype(np.int32)
    nir = nir.astype(np.int32)
    fm = fmask.astype(np.int32)
    clear = (valid & (fm != FMASK_FILL) & ((fm & FMASK_REJECT) == 0)
             & ((fm & FMASK_AEROSOL_HIGH) != FMASK_AEROSOL_HIGH) & (red > 0) & (nir > 0))
    fraction = clear.reshape(rows, px_y, cols, px_x).mean(axis=(1, 3))
    keep_cell = fraction >= min_clear
    keep = np.repeat(np.repeat(keep_cell, px_y, axis=0), px_x, axis=1) & clear
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(keep, (nir - red) / (nir + red), np.nan).astype(np.float32)
    return ndvi, fraction


def ndmi_of(nir, swir, ok):
    nir = nir.astype(np.int32)
    swir = swir.astype(np.int32)
    good = ok & (swir > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(good, (nir - swir) / (nir + swir), np.nan).astype(np.float32)


class MonthComposite:
    """Collects clear values per acquisition date, then takes the per-pixel median over dates (unchanged)."""

    def __init__(self, height: int, width: int):
        self.shape = (height, width)
        self.by_date: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.scenes_used: list[str] = []

    def add(self, date: str, scene_id: str, ndvi: np.ndarray, row0: int, col0: int) -> bool:
        ok = np.isfinite(ndvi)
        if not ok.any():
            return False
        total, count = self.by_date.get(date) or (np.zeros(self.shape, np.float32), np.zeros(self.shape, np.uint8))
        h, w = ndvi.shape
        view_t, view_c = total[row0:row0 + h, col0:col0 + w], count[row0:row0 + h, col0:col0 + w]
        view_t += np.where(ok, ndvi, 0.0)
        view_c += ok.astype(np.uint8)
        self.by_date[date] = (total, count)
        self.scenes_used.append(scene_id)
        return True

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        """(value x 10000 as int16 with NODATA, number of dates per pixel as uint8)."""
        if not self.by_date:
            return np.full(self.shape, NODATA, np.int16), np.zeros(self.shape, np.uint8)
        layers = []
        for total, count in self.by_date.values():
            with np.errstate(invalid="ignore", divide="ignore"):
                layers.append(np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32))
        stack = np.stack(layers)
        n = np.isfinite(stack).sum(axis=0).astype(np.uint8)
        with np.errstate(all="ignore"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(stack, axis=0)
        val = np.where(n > 0, np.clip(np.round(med * NDVI_SCALE), -NDVI_SCALE, NDVI_SCALE), NODATA)
        return val.astype(np.int16), n


class ViewAccumulator:
    """Pixel-wise merge of the granules of one (sensor, tile, date) view."""

    def __init__(self, shape):
        self.ndvi_sum = np.zeros(shape, np.float32)
        self.ndvi_cnt = np.zeros(shape, np.uint8)
        self.ndmi_sum = np.zeros(shape, np.float32)
        self.ndmi_cnt = np.zeros(shape, np.uint8)
        self.covered = np.zeros(shape, bool)
        self.granules: list[str] = []
        self.clouds: list = []

    def add(self, granule: str, cloud, covered, ndvi=None, ndmi=None):
        self.granules.append(granule)
        self.clouds.append(cloud)
        self.covered |= covered
        if ndvi is not None:
            ok = np.isfinite(ndvi)
            self.ndvi_sum += np.where(ok, ndvi, 0.0)
            self.ndvi_cnt += ok.astype(np.uint8)
        if ndmi is not None:
            ok = np.isfinite(ndmi)
            self.ndmi_sum += np.where(ok, ndmi, 0.0)
            self.ndmi_cnt += ok.astype(np.uint8)


VIEW_FIELDS = ("covered_frac", "clear_frac", "n_clear", "ndvi_mean", "ndvi_std", "ndvi_p10", "ndmi_mean", "n_ndmi")


def view_cell_stats(acc: ViewAccumulator, rows: int, cols: int, px_y: int, px_x: int) -> dict[str, np.ndarray]:
    """Per-cell values of one merged view (NaN statistics where a cell has no clear pixel)."""
    def cells(a):
        return a.reshape(rows, px_y, cols, px_x).transpose(0, 2, 1, 3).reshape(rows * cols, px_y * px_x)

    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(acc.ndvi_cnt > 0, acc.ndvi_sum / np.maximum(acc.ndvi_cnt, 1), np.nan)
        ndmi = np.where(acc.ndmi_cnt > 0, acc.ndmi_sum / np.maximum(acc.ndmi_cnt, 1), np.nan)
    cv, cn, cm = cells(ndvi), cells(ndmi), cells(acc.covered)
    n_px = cv.shape[1]
    n_clear = np.isfinite(cv).sum(axis=1)
    n_ndmi = np.isfinite(cn).sum(axis=1)
    out = {"covered_frac": (cm.sum(axis=1) / n_px).astype(np.float32),
           "clear_frac": (n_clear / n_px).astype(np.float32),
           "n_clear": n_clear.astype(np.int32), "n_ndmi": n_ndmi.astype(np.int32)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["ndvi_mean"] = np.nanmean(cv, axis=1).astype(np.float32)
        out["ndvi_std"] = np.nanstd(cv, axis=1).astype(np.float32)
        out["ndvi_p10"] = np.nanpercentile(cv, 10, axis=1).astype(np.float32)
        out["ndmi_mean"] = np.nanmean(cn, axis=1).astype(np.float32)
    return out


def save_month(path: Path, arrays: dict, meta: dict) -> None:
    """Atomic write, so a stopped pull never leaves a half-written month that looks finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def load_month(path: Path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files if k != "meta"}, json.loads(str(z["meta"]))
