"""Per-scene clear-pixel NDVI and per-pixel monthly composites on a region's fixed pixel grid.

The cloud rule is Austin Schlather's (data-pipeline/practice/build_table.py, dewstat-northeastern@7adb8d6):
Fmask bits 1-5 (cloud, adjacent to cloud/shadow, shadow, snow/ice, water) and high aerosol are rejected,
Red and NIR must be positive, and a pass only counts for a cell when at least 30% of that cell is clear.
His tables kept per-cell means per pass; here every clear pixel is kept so pixel statistics can be formed.

Monthly composite: per pixel, the median NDVI over the distinct acquisition dates of the month. Overlapping
tiles of the same date are averaged first, so one acquisition is never counted twice.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

FMASK_REJECT = 0b0011_1110
FMASK_AEROSOL_HIGH = 0b1100_0000
FMASK_FILL = 255
MIN_CLEAR = 0.3
NDVI_SCALE = 10_000
NODATA = -32768


def clear_ndvi(red, nir, fmask, valid, px_x: int, px_y: int, rows: int, cols: int,
               min_clear: float = MIN_CLEAR):
    """NDVI where a pixel is clear (NaN elsewhere) and the per-cell clear fraction.

    red, nir: reflectance x 10000 (any int dtype); fmask: HLS v2 Fmask; valid: True where the scene has data.
    The arrays cover `rows` x `cols` whole cells of px_y x px_x pixels each.
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


class MonthComposite:
    """Collects clear NDVI per acquisition date, then takes the per-pixel median over dates."""

    def __init__(self, height: int, width: int):
        self.shape = (height, width)
        self.by_date: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.scenes_used: list[str] = []

    def add(self, date: str, scene_id: str, ndvi: np.ndarray, row0: int, col0: int) -> bool:
        """Place one scene's NDVI (NaN = not clear) at pixel offset (row0, col0). False if nothing clear."""
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
        """(NDVI x 10000 as int16 with NODATA, number of dates per pixel as uint8)."""
        if not self.by_date:
            return np.full(self.shape, NODATA, np.int16), np.zeros(self.shape, np.uint8)
        layers = []
        for total, count in self.by_date.values():
            with np.errstate(invalid="ignore", divide="ignore"):
                layers.append(np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32))
        stack = np.stack(layers)
        n = np.isfinite(stack).sum(axis=0).astype(np.uint8)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(stack, axis=0)
        ndvi = np.where(n > 0, np.clip(np.round(med * NDVI_SCALE), -NDVI_SCALE, NDVI_SCALE), NODATA)
        return ndvi.astype(np.int16), n


def save_month(path: Path, ndvi: np.ndarray, nobs: np.ndarray, meta: dict) -> None:
    """Atomic write, so a stopped pull never leaves a half-written month that looks finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, ndvi=ndvi, nobs=nobs, meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def load_month(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path) as z:
        return z["ndvi"], z["nobs"], json.loads(str(z["meta"]))
