"""HLS 2016-2019 straight from NASA (LP DAAC), for the years Planetary Computer does not have.

Planetary Computer's HLS v2 collections start 2020-01-01. NASA's own archive has the same HLS v2.0 product
from late 2015 (Sentinel-2, S30) and 2013 (Landsat, L30). NASA has no "crop this box for me" service, so
this script reads each scene's files itself, fetching only the part over the box:

  1. Search NASA's catalog (CMR) per sensor per month; scenes cloudier than 80% are skipped (as build_table.py).
  2. Per scene, open Fmask, Red, NIR, SWIR1, SWIR2 (Cloud Optimized GeoTIFFs, UTM) at an overview level:
     --pixel-m 120 reads the files' built-in 4x overview (made with nearest neighbour, so Fmask bits stay exact).
     If Fmask shows no clear pixel in the box, the other four bands are not downloaded.
  3. Each pixel goes to the 0.1-degree FLDAS cell its center falls in; per cell the same clear-pixel test and
     NDVI / NDMI / NDWI means as build_table.py (its summarize / pick_observations / monthly_rows are reused).

Output in --out: hls_monthly.csv and hls_observations.csv (same columns as build_table.py --swir),
cache_nasa.jsonl (finished scenes) and search_nasa.jsonl (finished searches); a rerun resumes.

Needs the Earthdata token (EARTHDATA_TOKEN in ../Atlas/.env). Run from this folder with the practice venv:
    .venv/bin/python nasa_backfill.py --coords=-105.0,34.0,-104.5,34.5 --start 2016-01 --end 2019-12 --out X
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_table import CELL, Grid, months, monthly_rows, pick_observations, snap, summarize, write_csv  # noqa: E402

CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
PRODUCTS = {"S30": "HLSS30", "L30": "HLSL30"}
BANDS = {"S30": ("B04", "B8A", "B11", "B12"), "L30": ("B04", "B05", "B06", "B07")}  # red, nir, swir1, swir2
NODATA = -9999  # reflectance fill; Fmask fill is 255
OVERVIEW = {30: None, 60: 0, 120: 1, 240: 2, 480: 3}  # pixel size -> overview level (2x, 4x, 8x, 16x)
SEARCH_WAITS = (5, 15, 30, 60, 120, 300)


def earthdata_token() -> str:
    token = os.environ.get("EARTHDATA_TOKEN")
    env = HERE.parent / "Atlas" / ".env"
    if not token and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("EARTHDATA_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("'\"")
    if not token:
        sys.exit("EARTHDATA_TOKEN not set (env var or ../Atlas/.env)")
    return token


def gdal_env(token: str) -> None:
    """GDAL reads the files over HTTPS with the Earthdata token; NASA redirects to its file host."""
    os.environ.update({
        "GDAL_HTTP_AUTH": "BEARER",
        "GDAL_HTTP_BEARER": token,
        "CPL_VSIL_CURL_USE_HEAD": "NO",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "3",
        # Each piece of a file is read once, so a big block cache only costs RAM (default 5% of RAM per process;
        # five regions at once used ~0.6 GB each and squeezed the 7 GB server).
        "GDAL_CACHEMAX": "64",
    })


def search_month(http: httpx.Client, code: str, grid: Grid, first, last, max_cloud: float) -> list[dict]:
    """All scenes of one sensor in one month over the grid, as {id, date, cloud, urls}."""
    params = {
        "short_name": PRODUCTS[code], "version": "2.0", "page_size": 2000,
        "bounding_box": f"{grid.west},{grid.south},{grid.east},{grid.north}",
        "temporal": f"{first:%Y-%m-%d}T00:00:00Z,{last:%Y-%m-%d}T23:59:59Z",
    }
    for attempt, wait in enumerate((*SEARCH_WAITS, None)):
        try:
            resp = http.get(CMR, params=params)
            resp.raise_for_status()
            break
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if wait is None or (status is not None and status not in (403, 429, 500, 502, 503, 504)):
                raise
            print(f"  search {code} {first:%Y-%m}: {status or type(exc).__name__}, retry {attempt + 1} in {wait}s",
                  flush=True)
            time.sleep(wait)
    scenes = []
    for item in resp.json()["items"]:
        umm = item["umm"]
        attrs = {a["Name"]: a["Values"][0] for a in umm.get("AdditionalAttributes", [])}
        cloud = float(attrs.get("CLOUD_COVERAGE", 0))
        if cloud > max_cloud:
            continue
        urls = {u["URL"].rsplit(".", 2)[-2]: u["URL"] for u in umm.get("RelatedUrls", [])
                if u["URL"].startswith("https://") and u["URL"].endswith(".tif")}
        if not all(b in urls for b in (*BANDS[code], "Fmask")):
            continue
        scenes.append({"id": umm["GranuleUR"], "date": umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:10],
                       "cloud": cloud, "urls": {b: urls[b] for b in (*BANDS[code], "Fmask")}})
    return scenes


def search_scenes(grid: Grid, start: str, end: str, max_cloud: float, cache: Path) -> list[tuple[str, dict]]:
    done: dict[tuple[str, str], list[dict]] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            done[(e["code"], e["month"])] = e["scenes"]
    found = []
    with httpx.Client(timeout=120) as http:
        for code in PRODUCTS:
            for first, last in months(start, end):
                key = (code, f"{first:%Y-%m}")
                if key not in done:
                    done[key] = search_month(http, code, grid, first, last, max_cloud)
                    print(f"  {code} {key[1]}: {len(done[key])} scenes", flush=True)
                    with cache.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"code": code, "month": key[1], "scenes": done[key]}) + "\n")
                found.extend((code, s) for s in done[key])
    return found


def cell_pixels(grid: Grid, pixel_m: float) -> np.ndarray:
    """Pixels a whole cell holds at this size (the denominator of the clear fraction, like a full crop)."""
    out = np.zeros(grid.rows * grid.cols)
    for i in range(grid.rows * grid.cols):
        lat = grid.cell_center(i)[1]
        out[i] = (CELL * 111_320 * math.cos(math.radians(lat))) * (CELL * 110_574) / pixel_m**2
    return out


def read_scene(code: str, scene: dict, grid: Grid, level: int | None, full_cell: np.ndarray) -> list[dict]:
    """Per-cell records for one scene (same fields as build_table.collect_observations)."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from pyproj import Transformer

    kw = {} if level is None else {"overview_level": level}
    with rasterio.open(scene["urls"]["Fmask"], **kw) as ds:
        left, bottom, right, top = transform_bounds("EPSG:4326", ds.crs, grid.west, grid.south, grid.east,
                                                    grid.north, densify_pts=21)
        win = from_bounds(left, bottom, right, top, ds.transform).round_offsets().round_lengths()
        full = rasterio.windows.Window(0, 0, ds.width, ds.height)
        if not rasterio.windows.intersect(win, full):  # catalog outline touches the box, pixels do not
            return []
        win = win.intersection(full)
        fm = ds.read(1, window=win)
        transform, crs = ds.window_transform(win), ds.crs
    h, w = fm.shape
    # pixel centers -> lon/lat -> FLDAS cell (outside the grid = dropped)
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    xs, ys = transform * (cols, rows)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xs, ys)
    c = np.floor((lon - grid.west) / CELL + 1e-9).astype(int)
    r = np.floor((grid.north - lat) / CELL + 1e-9).astype(int)
    inside = (c >= 0) & (c < grid.cols) & (r >= 0) & (r < grid.rows)
    if not inside.any():
        return []
    cells = (r * grid.cols + c)[inside][None, :]
    fm_in = fm[inside]
    valid = fm_in != 255
    reject = (fm_in & 0b0011_1110) != 0
    aerosol_high = (fm_in & 0b1100_0000) == 0b1100_0000
    zeros = np.zeros_like(fm_in, dtype=np.int16)
    if (valid & ~reject & ~aerosol_high).any():  # some clear pixel: fetch the reflectance bands
        bands = []
        for b in BANDS[code]:
            with rasterio.open(scene["urls"][b], **kw) as ds:
                bands.append(ds.read(1, window=win)[inside])
        valid &= bands[0] != NODATA
    else:  # all cloud / fill: no need to download reflectance
        bands = [zeros] * 4
    red, nir, s1, s2 = (np.where(b == NODATA, 0, b) for b in bands)  # 0 = not clear in summarize()
    stack = np.stack([red, nir, s1, s2, fm_in, valid.astype(np.uint8)])[:, None, :]
    n_cells = grid.rows * grid.cols
    out = []
    for rec in summarize(stack, cells, n_cells, swir=True):
        if rec["covered"] == 0:
            continue
        rec["pixels"] = int(round(full_cell[rec["cell"]]))  # clear fraction of the whole cell, as with a crop
        out.append(rec)
    return out


def collect(grid: Grid, scenes: list[tuple[str, dict]], cache: Path, workers: int, pixel_m: float
            ) -> tuple[list[dict], int]:
    done: dict[str, list[dict]] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            done[e["scene"]] = e["cells"]
    todo = [(c, s) for c, s in scenes if s["id"] not in done]
    print(f"Reading {len(todo)} scenes from NASA ({len(scenes) - len(todo)} cached), ~{pixel_m:.0f} m pixels", flush=True)
    level, full_cell = OVERVIEW[int(pixel_m)], cell_pixels(grid, pixel_m)
    failed = n = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool, cache.open("a", encoding="utf-8") as fh:
        futures = {pool.submit(read_scene, c, s, grid, level, full_cell): s for c, s in todo}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                done[s["id"]] = fut.result()
            except Exception as exc:  # keep going; rerun to retry
                failed += 1
                print(f"  retry crop {type(exc).__name__} {s['id']}: {str(exc)[:120]}", file=sys.stderr, flush=True)
                continue
            fh.write(json.dumps({"scene": s["id"], "cells": done[s["id"]]}) + "\n")
            fh.flush()
            n += 1
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} read ({n / (time.time() - t0) * 60:.0f}/min)", flush=True)
    if failed:
        print(f"  {failed} scenes failed (rerun to retry them)")
    records = [{**rec, "scene": s["id"], "sensor": c, "date": s["date"]} for c, s in scenes for rec in done.get(s["id"], [])]
    return records, failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coords", required=True, help="west,south,east,north")
    ap.add_argument("--start", default="2016-01")
    ap.add_argument("--end", default="2019-12")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pixel-m", type=float, default=120, choices=sorted(OVERVIEW))
    ap.add_argument("--max-scene-cloud", type=float, default=80)
    ap.add_argument("--min-clear", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=16, help="scenes read at the same time")
    args = ap.parse_args()

    gdal_env(earthdata_token())
    grid = snap([float(v) for v in args.coords.split(",")])
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Grid {grid.cols}x{grid.rows} cells, {args.start} to {args.end}; searching NASA CMR...", flush=True)
    scenes = search_scenes(grid, args.start, args.end, args.max_scene_cloud, args.out / "search_nasa.jsonl")
    records, failed = collect(grid, scenes, args.out / "cache_nasa.jsonl", args.workers, args.pixel_m)
    obs = pick_observations(records, args.min_clear)
    obs_rows = [{
        "cell_longitude": grid.cell_center(r["cell"])[0], "cell_latitude": grid.cell_center(r["cell"])[1],
        "date": r["date"], "sensor": r["sensor"], "scene": r["scene"],
        "clear_fraction": round(r["clear"] / r["pixels"], 3),
        **{k: round(r[k], 4) for k in ("ndvi", "red", "nir", "ndmi", "ndwi", "swir1", "swir2")},
    } for r in obs]
    if obs_rows:
        write_csv(args.out / "hls_observations.csv", obs_rows)
    rows = monthly_rows(grid, obs, args.start, args.end, swir=True)
    write_csv(args.out / "hls_monthly.csv", rows)
    filled = sum(r["n_obs"] > 0 for r in rows)
    print(f"hls_monthly.csv: {len(rows)} cell-months, {filled} with data; {len(obs_rows)} clear passes")
    # Non-zero exit when scenes failed, so pull_regions.py retries the step (finished scenes are cached).
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
