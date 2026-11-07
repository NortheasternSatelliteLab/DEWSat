"""The month loop shared by the two HLS pullers: search (cached), read every scene, composite, save.

Structure as in the paper run's code/paper_hls/pull_common.py (commit f17e700; after Austin Schlather's
build_table.py / nasa_backfill.py): one region per process, a month-by-month scene search cached in a jsonl file,
scenes read in a thread pool, and a month's file written only when every one of its scenes was read, so a rerun
resumes.

Additions: only HLS v2.0 granules are accepted (anything else is counted and skipped); granules sharing a
(sensor, tile, date) are merged into one view for the per-view record; SWIR1 and NDMI; the per-view, per-cell
values (hls_common.view_cell_stats) are saved with the month.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from cog_read import read_scene
from grid import load_regions, month_list, region_grid
from hls_common import (VIEW_FIELDS, MonthComposite, ViewAccumulator, clear_mask, clear_ndvi, ndmi_of, save_month,
                        view_cell_stats)

SENSOR_CODE = {"L30": 0, "S30": 1}


def granule_tile(gid: str) -> str:
    return gid.split(".")[2]            # HLS.S30.T15TVG.2021190T170849.v2.0 -> T15TVG


def pull_months(region_name: str, start: str, end: str, out_root: Path, workers: int, source: str,
                sensors: tuple[str, ...], search: Callable, urls: Callable, cache_name: str, meta: dict) -> int:
    """search(grid, month, sensor) -> [scene dict with id, date]; urls(scene, sensor) -> (fmask, red, nir, swir1)."""
    region = next(r for r in load_regions(include_unbatched=True) if r.name == region_name)
    grid = region_grid(region)
    d = out_root / region.name / "hls"
    d.mkdir(parents=True, exist_ok=True)
    (d / "grid.json").write_text(json.dumps({"region": region.name, **grid.as_dict()}, indent=1))
    cache_path = d / cache_name
    searched: dict[tuple[str, str], list[dict]] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            e = json.loads(line)
            searched[(e["sensor"], e["month"])] = e["scenes"]
    todo = [m for m in month_list(start, end) if not (d / f"{m}.npz").exists()]
    print(f"{region.name}: {source}, grid {grid.cols}x{grid.rows} cells = {grid.width}x{grid.height} px "
          f"({grid.px_x}x{grid.px_y}/cell); {len(todo)} months to do", flush=True)
    failed = []
    shape = (grid.height, grid.width)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for month in todo:
            t0 = time.time()
            scenes = []
            for sensor in sensors:
                if (sensor, month) not in searched:
                    searched[(sensor, month)] = search(grid, month, sensor)
                    with cache_path.open("a") as fh:
                        fh.write(json.dumps({"sensor": sensor, "month": month,
                                             "scenes": searched[(sensor, month)]}) + "\n")
                scenes += [(sensor, s) for s in searched[(sensor, month)]]
            not_v20 = [s["id"] for _, s in scenes if not s["id"].endswith(".v2.0")]
            scenes = [(sen, s) for sen, s in scenes if s["id"].endswith(".v2.0")]
            comp = MonthComposite(*shape)
            comp_ndmi = MonthComposite(*shape)
            views: dict[tuple[str, str, str], ViewAccumulator] = {}
            errors = 0
            futures = {pool.submit(read_scene, grid, *urls(s, sensor)): (sensor, s) for sensor, s in scenes}
            for fut in as_completed(futures):
                sensor, s = futures[fut]
                try:
                    got = fut.result()
                except Exception as exc:  # noqa: BLE001 - the month is retried on the next run
                    errors += 1
                    print(f"  FAILED {s['id']}: {type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr, flush=True)
                    continue
                if got is None:          # the file does not reach the box
                    continue
                key = (sensor, granule_tile(s["id"]), s["date"])
                acc = views.get(key)
                if acc is None:
                    acc = views[key] = ViewAccumulator(shape)
                if "red" not in got:     # covered, but no possibly-clear pixel
                    acc.add(s["id"], s.get("cloud"), got["covered"])
                    continue
                red, nir, swir, fmask, valid = got["red"], got["nir"], got["swir"], got["fmask"], got["valid"]
                # paper composite rule (per granule, cell >= 30% clear), unchanged
                ndvi_c, _ = clear_ndvi(red, nir, fmask, valid, grid.px_x, grid.px_y, grid.rows, grid.cols)
                comp.add(s["date"], s["id"], ndvi_c, 0, 0)
                comp_ndmi.add(s["date"], s["id"], ndmi_of(nir, swir, np.isfinite(ndvi_c) & got["swir_valid"]), 0, 0)
                # per-view record: every clear pixel, no 30% rule
                clear = clear_mask(red, nir, fmask, valid)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ndvi_v = np.where(clear, (nir.astype(np.int32) - red) / (nir.astype(np.int32) + red),
                                      np.nan).astype(np.float32)
                acc.add(s["id"], s.get("cloud"), got["covered"], ndvi_v, ndmi_of(nir, swir, clear & got["swir_valid"]))
            if errors:
                failed.append(month)
                print(f"  {month}: {errors} scene(s) failed; month left for a rerun", flush=True)
                continue
            ndvi, nobs = comp.finish()
            ndmi, ndmi_nobs = comp_ndmi.finish()
            keys = sorted(views)
            stats = {f: [] for f in VIEW_FIELDS}
            view_meta = []
            for k in keys:
                st = view_cell_stats(views[k], grid.rows, grid.cols, grid.px_y, grid.px_x)
                for f in VIEW_FIELDS:
                    stats[f].append(st[f])
                view_meta.append({"sensor": k[0], "tile": k[1], "date": k[2], "granules": views[k].granules,
                                  "cloud": views[k].clouds})
            n_cells = grid.n_cells
            arrays = {"ndvi": ndvi, "nobs": nobs, "ndmi": ndmi, "ndmi_nobs": ndmi_nobs,
                      "view_sensor": np.array([SENSOR_CODE[k[0]] for k in keys], np.uint8),
                      "view_date": np.array([int(k[2].replace("-", "")) for k in keys], np.int32),
                      "view_tile": np.array([k[1] for k in keys], dtype="U6")}
            for f in VIEW_FIELDS:
                dt = np.int32 if f.startswith("n_") else np.float32
                arrays[f"view_{f}"] = (np.stack(stats[f]).astype(dt) if keys
                                       else np.zeros((0, n_cells), dt))
            n_merged = sum(len(views[k].granules) > 1 for k in keys)
            save_month(d / f"{month}.npz", arrays,
                       {"source": source, "month": month, "scenes_found": len(scenes) + len(not_v20),
                        "scenes_v20": len(scenes), "scenes_not_v20": not_v20,
                        "scenes_used": len(comp.scenes_used), "dates": sorted(comp.by_date),
                        "views": view_meta, "views_merged_from_2plus_granules": n_merged, **meta})
            print(f"  {month}: {len(scenes)} scenes ({len(not_v20)} not v2.0), {len(keys)} views ({n_merged} merged), "
                  f"{len(comp.scenes_used)} with clear cells, {len(comp.by_date)} dates, "
                  f"{float((nobs > 0).mean()):.0%} of pixels composited, {time.time() - t0:.0f}s", flush=True)
    if failed:
        print(f"{region.name}: {len(failed)} month(s) failed: {', '.join(failed)}", flush=True)
        return 1
    print(f"{region.name}: done", flush=True)
    return 0
