"""Regions, the FLDAS 0.1-degree cell grid, a fixed pixel grid per region, and the Earthdata login.

Copied from the paper run's code/paper_hls/grid.py (branch paper-hls-bilstm, commit f17e700; credit there to Austin
Schlather for the box snapping, cell numbering and ~120 m pixel size, and to Ryan for the 98 W split). Unchanged
except `load_regions`, which also reads the planned new regions (data/plan/regions_plan.csv) and returns each
region's role and pull batch.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

V2 = Path.home() / "Desktop" / "DEWSat_BiLSTM_v2"
EXISTING_FILES = (V2 / "inputs" / "paper_code" / "paper_hls" / "regions.csv",
                  V2 / "inputs" / "paper_code" / "paper_hls" / "regions_new.csv")
PLAN_FILE = V2 / "data" / "plan" / "regions_plan.csv"
CELL = 0.1
PIXEL_M = 120.0
WEST_OF_LON = -98.0


@dataclass(frozen=True)
class Region:
    name: str
    west: float
    south: float
    east: float
    north: float
    split: str          # paper split for the existing 102; "dev" or "test" for new regions
    origin: str = "existing"   # "existing" (the paper's 102) or "new"
    batch: int = 0

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.west + self.east) / 2, (self.south + self.north) / 2

    @property
    def side(self) -> str:
        return "west" if self.centroid[0] < WEST_OF_LON else "east"

    @property
    def role(self) -> str:
        """'dev' for every existing region and new development region, 'test' for new fresh-test regions."""
        return "test" if (self.origin == "new" and self.split == "test") else "dev"


def load_regions(include_plan: bool = True, include_unbatched: bool = False) -> list[Region]:
    out = []
    for path in EXISTING_FILES:
        with Path(path).open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out.append(Region(r["name"], float(r["west"]), float(r["south"]), float(r["east"]),
                                  float(r["north"]), r["split"], "existing", 0))
    if include_plan and PLAN_FILE.exists():
        with PLAN_FILE.open() as fh:
            for r in csv.DictReader(fh):
                b = int(r["batch"])
                if b == 0 and not include_unbatched:
                    continue
                out.append(Region(r["name"], float(r["west"]), float(r["south"]), float(r["east"]),
                                  float(r["north"]), r["role"], "new", b))
    names = [r.name for r in out]
    if len(names) != len(set(names)):
        raise ValueError("duplicate region names")
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
    """An Earthdata bearer token, never printed or written anywhere (the paper run's function, unchanged).

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
