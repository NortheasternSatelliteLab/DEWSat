"""Read one HLS scene's Fmask, Red, NIR and SWIR1 onto a region's fixed pixel grid, identically for both archives.

`warp_band` is the paper run's (code/paper_hls/cog_read.py, commit f17e700; overview-window read adapted from Austin
Schlather's nasa_backfill.py): each band from the COG's 4x overview (~120 m, nearest neighbour, so Fmask bits stay
exact) over the region box, warped with nearest neighbour onto the region's EPSG:4326 grid. The same code reads
NASA LP DAAC (2016-2019) and Planetary Computer (2020+).

Changes: SWIR1 (L30 B06, S30 B11) is read as well, and a scene whose Fmask has no possibly-clear pixel over the box
still returns its Fmask, so the per-view record can say the scene covered the cell but was cloudy.
"""

from __future__ import annotations

import numpy as np

from grid import PixelGrid
from hls_common import FMASK_FILL, FMASK_REJECT

REFL_NODATA = -9999
OVERVIEW_120M = 1   # HLS COG overviews are [2, 4]; level 1 is the 4x (120 m) one


def warp_band(url: str, grid: PixelGrid, nodata: int, dtype) -> np.ndarray | None:
    """One band over the region box on the region grid; None if the file does not reach the box. (Unchanged.)"""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import Window, from_bounds as window_from_bounds

    dst_transform = from_bounds(grid.west, grid.south, grid.east, grid.north, grid.width, grid.height)
    with rasterio.open(url, overview_level=OVERVIEW_120M) as ds:
        bounds = transform_bounds("EPSG:4326", ds.crs, grid.west, grid.south, grid.east, grid.north, densify_pts=21)
        win = window_from_bounds(*bounds, transform=ds.transform).round_offsets().round_lengths()
        full = Window(0, 0, ds.width, ds.height)
        if not rasterio.windows.intersect(win, full):
            return None
        win = Window(win.col_off - 1, win.row_off - 1, win.width + 2, win.height + 2).intersection(full)
        src = ds.read(1, window=win)
        out = np.full((grid.height, grid.width), nodata, dtype=dtype)
        reproject(src, out, src_transform=ds.window_transform(win), src_crs=ds.crs, src_nodata=nodata,
                  dst_transform=dst_transform, dst_crs="EPSG:4326", dst_nodata=nodata, resampling=Resampling.nearest)
        return out


def read_scene(grid: PixelGrid, fmask_url: str, red_url: str, nir_url: str, swir_url: str):
    """None if the scene does not reach the box; else a dict with fmask and, when any pixel could be clear,
    red, nir, swir1 and valid (all on the region grid)."""
    fmask = warp_band(fmask_url, grid, FMASK_FILL, np.uint8)
    if fmask is None:
        return None
    fm = fmask.astype(np.int32)
    out = {"fmask": fmask, "covered": fmask != FMASK_FILL}
    if not ((fm != FMASK_FILL) & ((fm & FMASK_REJECT) == 0)).any():   # all cloud or fill: skip the reflectance
        return out
    red = warp_band(red_url, grid, REFL_NODATA, np.int16)
    nir = warp_band(nir_url, grid, REFL_NODATA, np.int16)
    swir = warp_band(swir_url, grid, REFL_NODATA, np.int16)
    out.update(red=red, nir=nir, swir=swir,
               valid=(fmask != FMASK_FILL) & (red != REFL_NODATA) & (nir != REFL_NODATA),
               swir_valid=swir != REFL_NODATA)
    return out
