"""Read one HLS scene's Red, NIR and Fmask onto a region's fixed pixel grid, identically for both archives.

Each band is read from the Cloud Optimized GeoTIFF's 4x overview (~120 m; HLS overviews are nearest-neighbour, so
Fmask bits stay exact) over the region box, then warped with nearest neighbour onto the region's EPSG:4326 grid.
The same code reads NASA LP DAAC (2016-2019) and Planetary Computer (2020+) files; on the same granule the two give
pixel-identical arrays (logs/probe/nasa_vs_pc_same_scene.log), so the per-pixel VCI reference does not mix two
different samplings. Planetary Computer's own crop endpoint, which Austin Schlather's build_table.py used, samples
differently (logs/probe/seam_check_iowa_2021-07.log) and is not used here.

The overview-window read is adapted from Austin Schlather's data-pipeline/practice/nasa_backfill.py, read_scene()
(dewstat-northeastern@7adb8d6, lines 141-182): the region box transformed into the file's CRS, a window from those
bounds with rounded offsets and lengths, intersected with the file, and the reflectance bands skipped when Fmask shows
no possibly-clear pixel. The warp onto a fixed EPSG:4326 grid is new.
"""

from __future__ import annotations

import numpy as np

from grid import PixelGrid
from hls_common import FMASK_FILL, FMASK_REJECT

REFL_NODATA = -9999
OVERVIEW_120M = 1  # HLS COG overviews are [2, 4]; level 1 is the 4x (120 m) one


def warp_band(url: str, grid: PixelGrid, nodata: int, dtype) -> np.ndarray | None:
    """One band over the region box on the region grid; None if the file does not reach the box."""
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
        # one pixel of margin so nearest-neighbour warping has a source pixel at the box edge
        win = Window(win.col_off - 1, win.row_off - 1, win.width + 2, win.height + 2).intersection(full)
        src = ds.read(1, window=win)
        out = np.full((grid.height, grid.width), nodata, dtype=dtype)
        reproject(src, out, src_transform=ds.window_transform(win), src_crs=ds.crs, src_nodata=nodata,
                  dst_transform=dst_transform, dst_crs="EPSG:4326", dst_nodata=nodata, resampling=Resampling.nearest)
        return out


def read_scene(grid: PixelGrid, fmask_url: str, red_url: str, nir_url: str):
    """(red, nir, fmask, valid) on the region grid, or None when the scene has no possibly-clear pixel there."""
    fmask = warp_band(fmask_url, grid, FMASK_FILL, np.uint8)
    if fmask is None:
        return None
    fm = fmask.astype(np.int32)
    if not ((fm != FMASK_FILL) & ((fm & FMASK_REJECT) == 0)).any():  # all cloud or fill: skip the reflectance
        return None
    red = warp_band(red_url, grid, REFL_NODATA, np.int16)
    nir = warp_band(nir_url, grid, REFL_NODATA, np.int16)
    valid = (fmask != FMASK_FILL) & (red != REFL_NODATA) & (nir != REFL_NODATA)
    return red, nir, fmask, valid
