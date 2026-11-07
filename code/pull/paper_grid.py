"""Regions, the FLDAS 0.1-degree cell grid, and a fixed pixel grid inside it.

Adapted from Austin Schlather's data-pipeline/practice/build_table.py (dewstat-northeastern@7adb8d6):
the box snapping, the cell numbering (row-major from the north-west corner) and the ~120 m pixel size are
his. What is new is that every region gets ONE fixed pixel grid, a whole number of pixels per cell, so the
pixels of every scene and every month line up exactly. That is what pixel-level statistics (spatial standard
deviation, the 10th percentile of VCI, the clear fraction) and a per-pixel VCI reference need.

The East/West split at 98 W (WEST_OF_LON) follows Ryan's (RyanBoomer30/Atlas, branch bilstm-split-v2@227042b,
experiments/bilstm/RESULTS_vci.md and shared_split_us.py WEST/EAST), which assigns whole states by the mean longitude
of their kept cells. Here a region is West when its box centroid lies west of 98 W.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGION_FILES = (HERE / "regions.csv", HERE / "regions_new.csv")
CELL = 0.1  # FLDAS grid spacing in degrees
PIXEL_M = 120.0  # Austin's pixel size: 120 m matched 30 m on two regions (his README)
WEST_OF_LON = -98.0  # East/West split used in the reports


@dataclass(frozen=True)
class Region:
    name: str
    west: float
    south: float
    east: float
    north: float
    split: str

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.west + self.east) / 2, (self.south + self.north) / 2

    @property
    def side(self) -> str:
        return "west" if self.centroid[0] < WEST_OF_LON else "east"


def load_regions(files=REGION_FILES) -> list[Region]:
    out = []
    for path in files:
        with Path(path).open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out.append(Region(r["name"], float(r["west"]), float(r["south"]), float(r["east"]),
                                  float(r["north"]), r["split"]))
    names = [r.name for r in out]
    if len(names) != len(set(names)):
        raise ValueError("duplicate region names across the region files")
    return out


@dataclass(frozen=True)
class PixelGrid:
    """Whole FLDAS cells of one region, each split into px_x by px_y pixels."""

    west: float
    south: float
    east: float
    north: float
    px_x: int
    px_y: int

    @property
    def cols(self) -> int:
        return round((self.east - self.west) / CELL)

    @property
    def rows(self) -> int:
        return round((self.north - self.south) / CELL)

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    @property
    def width(self) -> int:
        return self.cols * self.px_x

    @property
    def height(self) -> int:
        return self.rows * self.px_y

    def cell_center(self, index: int) -> tuple[float, float]:
        row, col = divmod(index, self.cols)
        return round(self.west + (col + 0.5) * CELL, 2), round(self.north - (row + 0.5) * CELL, 2)

    def cell_of_pixel(self):
        """(height, width) array: the cell index each pixel belongs to."""
        import numpy as np

        rows = np.arange(self.height) // self.px_y
        cols = np.arange(self.width) // self.px_x
        return rows[:, None] * self.cols + cols[None, :]

    def sub(self, west: float, south: float, east: float, north: float):
        """The whole cells a footprint touches, as (grid, row offset, col offset) in cells; None if outside."""
        w, e = max(self.west, west), min(self.east, east)
        s, n = max(self.south, south), min(self.north, north)
        if w >= e or s >= n:
            return None
        w = max(self.west, math.floor(w / CELL + 1e-9) * CELL)
        s = max(self.south, math.floor(s / CELL + 1e-9) * CELL)
        e = min(self.east, math.ceil(e / CELL - 1e-9) * CELL)
        n = min(self.north, math.ceil(n / CELL - 1e-9) * CELL)
        sub = PixelGrid(round(w, 6), round(s, 6), round(e, 6), round(n, 6), self.px_x, self.px_y)
        return sub, round((self.north - sub.north) / CELL), round((sub.west - self.west) / CELL)

    def as_dict(self) -> dict:
        return {"west": self.west, "south": self.south, "east": self.east, "north": self.north,
                "px_x": self.px_x, "px_y": self.px_y, "cols": self.cols, "rows": self.rows,
                "width": self.width, "height": self.height, "cell_deg": CELL, "pixel_m_nominal": PIXEL_M,
                "crs": "EPSG:4326", "cell_order": "row-major from the north-west corner"}


def region_grid(region: Region, pixel_m: float = PIXEL_M) -> PixelGrid:
    """Snap the box outward to whole cells; one pixel count per cell for the whole region."""
    w = round(math.floor(region.west / CELL + 1e-9) * CELL, 6)
    s = round(math.floor(region.south / CELL + 1e-9) * CELL, 6)
    e = round(math.ceil(region.east / CELL - 1e-9) * CELL, 6)
    n = round(math.ceil(region.north / CELL - 1e-9) * CELL, 6)
    mid_lat = math.radians((s + n) / 2)
    px_x = max(1, round(CELL * 111_320 * math.cos(mid_lat) / pixel_m))
    px_y = max(1, round(CELL * 110_574 / pixel_m))
    return PixelGrid(w, s, e, n, px_x, px_y)


def month_list(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= end:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out



def earthdata_token() -> str:
    """An Earthdata bearer token, never printed or written anywhere.

    EARTHDATA_TOKEN from the environment if set (the orchestrator passes it to its subprocesses that way);
    otherwise earthaccess logs in with the user's ~/.netrc and returns that account's token.
    """
    import os

    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if token:
        return token
    import earthaccess

    auth = earthaccess.login(strategy="netrc")
    if not auth or not getattr(auth, "authenticated", False) or not getattr(auth, "token", None):
        raise SystemExit("Earthdata login with ~/.netrc failed")
    return auth.token["access_token"]
