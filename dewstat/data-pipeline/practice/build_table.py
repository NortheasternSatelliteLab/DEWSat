"""Practice HLS + FLDAS training table (NOT the team's official pipeline).

Builds one row per FLDAS 0.1-degree grid cell per month:

    cell_longitude, cell_latitude, month, <HLS columns>, <FLDAS columns>

How it works:
  1. Snap the box to the FLDAS 0.1-degree grid so every cell is whole.
  2. Search HLS Sentinel-2 (S30) and Landsat (L30) scenes month by month with Atlas.
  3. For each scene, ask Microsoft Planetary Computer to crop Red, NIR and Fmask
     to the box (in lon/lat, ~30 m pixels) and send it back as a numpy array.
  4. Throw away cloudy / shadowed / snowy / water / bad pixels using Fmask.
  5. Per cell and per satellite pass: average NDVI over the clear pixels.
  6. Per cell and per month: median of those pass averages.
  7. Optionally join a CSV written by `atlas data fldas landsurface`.

Run from this folder:
    uv run --project ../Atlas python build_table.py
    uv run --project ../Atlas python build_table.py --fldas ../Atlas/.atlas/data/fldas/fldas.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import numpy as np

from atlas.data.base import BBox, PullRequest, Scene
from atlas.data.hls_landsat import HLSLandsatClient
from atlas.data.hls_sentinel import HLSSentinelClient

CROP_URL = "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox/{w},{s},{e},{n}/{width}x{height}.npy"
CELL = 0.1  # FLDAS grid spacing in degrees
PIXEL_M = 30.0  # HLS pixel size in meters

# Red / NIR band names differ between the two HLS products.
SOURCES = {
    "S30": (HLSSentinelClient, "B04", "B8A"),
    "L30": (HLSLandsatClient, "B04", "B05"),
}
# Shortwave infrared (--swir): SWIR1 ~1.6 um and SWIR2 ~2.2 um. Water in leaves and soil absorbs here,
# so NIR-vs-SWIR indices track moisture more directly than NDVI (which tracks greenness).
SWIR_BANDS = {"S30": ("B11", "B12"), "L30": ("B06", "B07")}

# Fmask bits (HLS v2): 1 cloud, 2 near cloud/shadow, 3 cloud shadow, 4 snow/ice, 5 water.
FMASK_REJECT = 0b0011_1110
FMASK_AEROSOL_HIGH = 0b1100_0000


@dataclass(frozen=True)
class Grid:
    west: float
    south: float
    east: float
    north: float

    @property
    def cols(self) -> int:
        return round((self.east - self.west) / CELL)

    @property
    def rows(self) -> int:
        return round((self.north - self.south) / CELL)

    def cell_center(self, index: int) -> tuple[float, float]:
        row, col = divmod(index, self.cols)
        lon = self.west + (col + 0.5) * CELL
        lat = self.north - (row + 0.5) * CELL
        return round(lon, 2), round(lat, 2)

    def pixel_size(self, pixel_m: float = PIXEL_M) -> tuple[int, int]:
        """Output width/height so one pixel is about `pixel_m` meters on the ground."""
        mid_lat = math.radians((self.south + self.north) / 2)
        width = (self.east - self.west) * 111_320 * math.cos(mid_lat) / pixel_m
        height = (self.north - self.south) * 110_574 / pixel_m
        return max(1, round(width)), max(1, round(height))


def sub_grid(grid: Grid, scene_bbox: BBox) -> tuple[Grid, int, int] | None:
    """The whole cells of `grid` that a scene's footprint touches, plus their row/col offset.

    Asking for only this part of the box (instead of the whole box) avoids downloading
    empty "no data" pixels where a tile does not reach.
    """
    w = max(grid.west, scene_bbox.west)
    e = min(grid.east, scene_bbox.east)
    s = max(grid.south, scene_bbox.south)
    n = min(grid.north, scene_bbox.north)
    if w >= e or s >= n:
        return None
    sub = snap([w, s, e, n])
    sub = Grid(
        west=max(sub.west, grid.west),
        south=max(sub.south, grid.south),
        east=min(sub.east, grid.east),
        north=min(sub.north, grid.north),
    )
    row0 = round((grid.north - sub.north) / CELL)
    col0 = round((sub.west - grid.west) / CELL)
    return sub, row0, col0


def snap(bbox: list[float]) -> Grid:
    """Grow the box outward to whole FLDAS cells (edges on multiples of 0.1 degree)."""
    w, s, e, n = bbox
    return Grid(
        west=math.floor(w / CELL + 1e-9) * CELL,
        south=math.floor(s / CELL + 1e-9) * CELL,
        east=math.ceil(e / CELL - 1e-9) * CELL,
        north=math.ceil(n / CELL - 1e-9) * CELL,
    )


def cell_index_map(grid: Grid, width: int, height: int) -> np.ndarray:
    """For every output pixel, which grid cell (0..cells-1) its center falls in."""
    col = np.minimum(((np.arange(width) + 0.5) / width * grid.cols).astype(int), grid.cols - 1)
    row = np.minimum(((np.arange(height) + 0.5) / height * grid.rows).astype(int), grid.rows - 1)
    return row[:, None] * grid.cols + col[None, :]


def months(start: str, end: str) -> list[tuple[date, date]]:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        nxt = date(y + (m == 12), m % 12 + 1, 1)
        out.append((date(y, m, 1), date.fromordinal(nxt.toordinal() - 1)))
        y, m = nxt.year, nxt.month
    return out


SEARCH_WAITS = (5, 15, 30, 60, 120, 300)  # seconds between retries of one search request


async def search_month(client, req: PullRequest, label: str):
    """One STAC search, retried: Planetary Computer answers 403/429/503 when it is busy."""
    for attempt, wait in enumerate((*SEARCH_WAITS, None)):
        try:
            return await client.search(req)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if wait is None or (status is not None and status not in (403, 429, 500, 502, 503, 504)):
                raise
            print(f"  {label}: {status or type(exc).__name__}, retry {attempt + 1} in {wait}s", flush=True)
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


async def search_scenes(
    grid: Grid, start: str, end: str, max_cloud: float, cache_path: Path | None = None
) -> list[tuple[str, Scene]]:
    """All HLS scenes over the grid, one search per sensor per month.

    Each finished month is saved to cache_path, so a rerun only searches months it has not done yet.
    """
    bbox = BBox(west=grid.west, south=grid.south, east=grid.east, north=grid.north)
    done: dict[tuple[str, str], list[Scene]] = {}
    if cache_path and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            done[(entry["code"], entry["month"])] = [Scene.model_validate(s) for s in entry["scenes"]]
    found: list[tuple[str, Scene]] = []
    for code, (client_cls, _, _) in SOURCES.items():
        async with client_cls() as client:
            for first, last in months(start, end):
                key = (code, f"{first:%Y-%m}")
                if key not in done:
                    req = PullRequest(
                        start_date=first, end_date=last, bbox=bbox, max_cloud_cover=max_cloud, limit=500
                    )
                    result = await search_month(client, req, f"{code} {key[1]}")
                    done[key] = result.scenes
                    print(f"  {code} {key[1]}: {len(result.scenes)} scenes", flush=True)
                    if cache_path:
                        with cache_path.open("a", encoding="utf-8") as fh:
                            scenes = [s.model_dump(mode="json") for s in result.scenes]
                            fh.write(json.dumps({"code": code, "month": key[1], "scenes": scenes}) + "\n")
                    await asyncio.sleep(0.5)  # be gentle; three regions may be searching at once
                found.extend((code, scene) for scene in done[key])
    return found


def summarize(stack: np.ndarray, cells: np.ndarray, n_cells: int, swir: bool = False) -> list[dict[str, float]]:
    """Per-cell clear-pixel counts and NDVI / Red / NIR means for one scene (+ NDMI / NDWI with swir).

    Band order in `stack` follows fetch_crop: red, nir, [swir1, swir2,] Fmask, then the valid-data mask.
    """
    red, nir = stack[0].astype(np.float64), stack[1].astype(np.float64)
    s1, s2 = (stack[2].astype(np.float64), stack[3].astype(np.float64)) if swir else (None, None)
    fm = stack[4 if swir else 2].astype(np.int32)
    valid = stack[5 if swir else 3]
    clear = (
        (valid > 0)
        & (fm != 255)
        & ((fm & FMASK_REJECT) == 0)
        & ((fm & FMASK_AEROSOL_HIGH) != FMASK_AEROSOL_HIGH)
        & (red > 0)
        & (nir > 0)
    )
    if swir:
        clear &= (s1 > 0) & (s2 > 0)
    red, nir = red * 1e-4, nir * 1e-4  # stored as reflectance x 10000
    ndvi = np.where(clear, (nir - red) / np.maximum(nir + red, 1e-6), 0.0)
    layers = [("ndvi", ndvi), ("red", red), ("nir", nir)]
    if swir:
        s1, s2 = s1 * 1e-4, s2 * 1e-4
        layers += [
            ("ndmi", np.where(clear, (nir - s1) / np.maximum(nir + s1, 1e-6), 0.0)),  # NIR vs SWIR1 (1.6 um)
            ("ndwi", np.where(clear, (nir - s2) / np.maximum(nir + s2, 1e-6), 0.0)),  # NIR vs SWIR2 (2.2 um)
            ("swir1", s1),
            ("swir2", s2),
        ]

    flat = cells.ravel()
    total = np.bincount(flat, minlength=n_cells)
    covered = np.bincount(flat, weights=(valid > 0).ravel(), minlength=n_cells)
    n_clear = np.bincount(flat, weights=clear.ravel(), minlength=n_cells)
    sums = {
        name: np.bincount(flat, weights=np.where(clear, arr, 0.0).ravel(), minlength=n_cells)
        for name, arr in layers
    }
    out = []
    for i in range(n_cells):
        k = n_clear[i]
        rec = {"cell": i, "pixels": int(total[i]), "covered": int(covered[i]), "clear": int(k)}
        rec.update({name: float(sums[name][i] / k) if k else math.nan for name, _ in layers})
        out.append(rec)
    return out


async def fetch_crop(
    http: httpx.AsyncClient, grid: Grid, size: tuple[int, int], code: str, scene: Scene, swir: bool = False
) -> np.ndarray:
    _, red_band, nir_band = SOURCES[code]
    url = CROP_URL.format(
        w=grid.west, s=grid.south, e=grid.east, n=grid.north, width=size[0], height=size[1]
    )
    extra = [("assets", b) for b in SWIR_BANDS[code]] if swir else []
    params = [
        ("collection", scene.collection),
        ("item", scene.id),
        ("assets", red_band),
        ("assets", nir_band),
        *extra,
        ("assets", "Fmask"),
        ("dst_crs", "epsg:4326"),
        ("resampling", "nearest"),
    ]
    for attempt in range(5):
        try:
            resp = await http.get(url, params=params)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
            resp.raise_for_status()
            return np.load(io.BytesIO(resp.content))
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            # One line per retry so a pull's error rate can be counted from its log ("retry crop").
            status = getattr(getattr(exc, "response", None), "status_code", None)
            print(f"  retry crop {status or type(exc).__name__} {scene.id}", file=sys.stderr, flush=True)
            if attempt == 4:
                raise
            await asyncio.sleep(2**attempt)
    raise AssertionError("unreachable")


async def collect_observations(
    grid: Grid, scenes: list[tuple[str, Scene]], cache_path: Path, workers: int, pixel_m: float, swir: bool = False
) -> list[dict]:
    """One record per (scene, cell). Cached so a rerun skips finished scenes."""
    cache: dict[str, list[dict]] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            cache[entry["scene"]] = entry["cells"]

    todo = [(c, s) for c, s in scenes if s.id not in cache]
    print(
        f"Fetching {len(todo)} scene crops ({len(scenes) - len(todo)} cached), "
        f"~{pixel_m:.0f} m pixels, only where each tile overlaps the box"
    )

    sem = asyncio.Semaphore(workers)
    done = 0
    failed = 0
    async with httpx.AsyncClient(timeout=180) as http:

        async def one(code: str, scene: Scene) -> None:
            nonlocal done, failed
            part = sub_grid(grid, scene.bbox)
            if part is None:  # footprint misses the box entirely
                cache[scene.id] = []
                return
            sub, row0, col0 = part
            size = sub.pixel_size(pixel_m)
            async with sem:
                try:
                    stack = await fetch_crop(http, sub, size, code, scene, swir)
                except Exception as exc:  # keep going; one bad scene should not stop the run
                    failed += 1
                    print(f"  skip {scene.id}: {type(exc).__name__}", file=sys.stderr)
                    return
            local = summarize(stack, cell_index_map(sub, *size), sub.rows * sub.cols, swir)
            per_cell = []
            for rec in local:
                if rec["covered"] == 0:  # tile does not reach this cell
                    continue
                row, col = divmod(rec["cell"], sub.cols)
                per_cell.append({**rec, "cell": (row + row0) * grid.cols + (col + col0)})
            cache[scene.id] = per_cell
            with cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"scene": scene.id, "cells": per_cell}) + "\n")
            done += 1
            if done % 20 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} fetched")

        await asyncio.gather(*(one(c, s) for c, s in todo))
    if failed:
        print(f"  {failed} scenes failed (rerun to retry them)")

    records = []
    for code, scene in scenes:
        for rec in cache.get(scene.id, []):
            records.append(
                {**rec, "scene": scene.id, "sensor": code, "date": scene.datetime.date().isoformat()}
            )
    return records


def pick_observations(records: list[dict], min_clear: float) -> list[dict]:
    """Keep usable passes; when two tiles saw the same cell on the same day, keep the clearer one."""
    best: dict[tuple[int, str], dict] = {}
    for rec in records:
        if rec["pixels"] == 0 or rec["clear"] / rec["pixels"] < min_clear:
            continue
        key = (rec["cell"], rec["date"])
        if key not in best or rec["clear"] > best[key]["clear"]:
            best[key] = rec
    return sorted(best.values(), key=lambda r: (r["cell"], r["date"]))


def monthly_rows(grid: Grid, obs: list[dict], start: str, end: str, swir: bool = False) -> list[dict]:
    extra = ("ndmi", "ndwi", "swir1", "swir2") if swir else ()
    by_key: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for rec in obs:
        by_key[(rec["cell"], rec["date"][:7])].append(rec)
    rows = []
    for cell in range(grid.rows * grid.cols):
        lon, lat = grid.cell_center(cell)
        for first, _ in months(start, end):
            month = f"{first:%Y-%m}"
            group = by_key.get((cell, month), [])
            row: dict[str, object] = {
                "cell_longitude": lon,
                "cell_latitude": lat,
                "month": month,
                "n_obs": len(group),
                "n_s30": sum(r["sensor"] == "S30" for r in group),
                "n_l30": sum(r["sensor"] == "L30" for r in group),
            }
            if group:
                row["ndvi_median"] = round(statistics.median(r["ndvi"] for r in group), 4)
                row["ndvi_max"] = round(max(r["ndvi"] for r in group), 4)
                row["red_median"] = round(statistics.median(r["red"] for r in group), 4)
                row["nir_median"] = round(statistics.median(r["nir"] for r in group), 4)
                row["clear_fraction_mean"] = round(
                    statistics.mean(r["clear"] / r["pixels"] for r in group), 3
                )
                for name in extra:
                    row[f"{name}_median"] = round(statistics.median(r[name] for r in group), 4)
            else:  # no usable pass this month: leave blank, do not invent a value
                row.update(ndvi_median="", ndvi_max="", red_median="", nir_median="", clear_fraction_mean="")
                row.update({f"{name}_median": "" for name in extra})
            rows.append(row)
    return rows


def join_fldas(rows: list[dict], fldas_path: Path) -> tuple[list[dict], int]:
    """Attach FLDAS columns by (cell center, month)."""
    with fldas_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        lon_key = "cell_longitude" if "cell_longitude" in (reader.fieldnames or []) else "longitude"
        lat_key = "cell_latitude" if "cell_latitude" in (reader.fieldnames or []) else "latitude"
        skip = {"longitude", "latitude", "cell_longitude", "cell_latitude", "time"}
        variables = [c for c in reader.fieldnames or [] if c not in skip]
        table = {
            (round(float(r[lon_key]), 2), round(float(r[lat_key]), 2), r["time"][:7]): r
            for r in reader
        }
    matched = 0
    for row in rows:
        hit = table.get((row["cell_longitude"], row["cell_latitude"], row["month"]))
        matched += hit is not None
        for var in variables:
            row[var] = hit[var] if hit else ""
    return rows, matched


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coords", default="-100.0,38.0,-99.8,38.2", help="west,south,east,north")
    ap.add_argument("--start", default="2022-01", help="first month, YYYY-MM")
    ap.add_argument("--end", default="2023-12", help="last month, YYYY-MM")
    ap.add_argument("--fldas", type=Path, help="CSV from `atlas data fldas landsurface` to join")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--max-scene-cloud", type=float, default=80, help="skip scenes cloudier than this (%%)")
    ap.add_argument("--min-clear", type=float, default=0.3, help="min clear fraction of a cell to use a pass")
    ap.add_argument("--workers", type=int, default=6, help="parallel downloads")
    ap.add_argument(
        "--pixel-m", type=float, default=PIXEL_M, help="pixel size to request in meters (30 = native)"
    )
    ap.add_argument("--swir", action="store_true",
                    help="also fetch SWIR1/SWIR2 and add NDMI / NDWI columns (own cache: cache_swir.jsonl)")
    ap.add_argument("--search-cache", type=Path, default=None,
                    help="scene search cache to reuse (default: <out>/search_cache.jsonl)")
    args = ap.parse_args()

    grid = snap([float(v) for v in args.coords.split(",")])
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"Grid: {grid.west:.1f},{grid.south:.1f},{grid.east:.1f},{grid.north:.1f} "
        f"= {grid.cols}x{grid.rows} FLDAS cells, {args.start} to {args.end}"
    )

    print("Searching HLS...")
    scenes = asyncio.run(
        search_scenes(grid, args.start, args.end, args.max_scene_cloud,
                      args.search_cache or args.out / "search_cache.jsonl")
    )
    cache = args.out / ("cache_swir.jsonl" if args.swir else "cache.jsonl")  # SWIR records have more fields
    records = asyncio.run(collect_observations(grid, scenes, cache, args.workers, args.pixel_m, args.swir))

    obs = pick_observations(records, args.min_clear)
    obs_rows = [
        {
            "cell_longitude": grid.cell_center(r["cell"])[0],
            "cell_latitude": grid.cell_center(r["cell"])[1],
            "date": r["date"],
            "sensor": r["sensor"],
            "scene": r["scene"],
            "clear_fraction": round(r["clear"] / r["pixels"], 3),
            "ndvi": round(r["ndvi"], 4),
            "red": round(r["red"], 4),
            "nir": round(r["nir"], 4),
            **({k: round(r[k], 4) for k in ("ndmi", "ndwi", "swir1", "swir2")} if args.swir else {}),
        }
        for r in obs
    ]
    if obs_rows:
        write_csv(args.out / "hls_observations.csv", obs_rows)

    rows = monthly_rows(grid, obs, args.start, args.end, args.swir)
    write_csv(args.out / "hls_monthly.csv", rows)
    filled = sum(r["n_obs"] > 0 for r in rows)
    print(f"\nhls_observations.csv: {len(obs_rows)} clear passes")
    print(f"hls_monthly.csv: {len(rows)} cell-months, {filled} with data, {len(rows) - filled} empty")

    if args.fldas:
        rows, matched = join_fldas(rows, args.fldas)
        write_csv(args.out / "training_table.csv", rows)
        print(f"training_table.csv: {matched}/{len(rows)} rows matched an FLDAS row")
    else:
        print("No --fldas given; skipped the join (get an EARTHDATA_TOKEN, run atlas data fldas, rerun).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
