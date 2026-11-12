"""The month loop shared by the two HLS pullers: search (cached), read every scene, composite, save.

Structure after Austin Schlather's build_table.py / nasa_backfill.py (dewstat-northeastern@7adb8d6): one region per
process, a month-by-month scene search cached in a jsonl file, scenes read in a thread pool, and finished work kept
so a rerun resumes. A month's file is written only when every one of its scenes was read.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cog_read import read_scene
from grid import load_regions, month_list, region_grid
from hls_common import MonthComposite, clear_ndvi, save_month


def pull_months(region_name: str, start: str, end: str, out_root: Path, workers: int, source: str,
                sensors: tuple[str, ...], search: Callable, urls: Callable, cache_name: str, meta: dict) -> int:
    """search(grid, month, sensor) -> [scene dict with id, date]; urls(scene, sensor) -> (fmask, red, nir)."""
    region = next(r for r in load_regions() if r.name == region_name)
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
            comp = MonthComposite(grid.height, grid.width)
            errors = 0
            futures = {pool.submit(read_scene, grid, *urls(s, sensor)): s for sensor, s in scenes}
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    got = fut.result()
                except Exception as exc:  # noqa: BLE001 - the month is retried on the next run
                    errors += 1
                    print(f"  FAILED {s['id']}: {type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr, flush=True)
                    continue
                if got is None:
                    continue
                red, nir, fmask, valid = got
                ndvi, _ = clear_ndvi(red, nir, fmask, valid, grid.px_x, grid.px_y, grid.rows, grid.cols)
                comp.add(s["date"], s["id"], ndvi, 0, 0)
            if errors:
                failed.append(month)
                print(f"  {month}: {errors} scene(s) failed; month left for a rerun", flush=True)
                continue
            ndvi, nobs = comp.finish()
            save_month(d / f"{month}.npz", ndvi, nobs,
                       {"source": source, "month": month, "scenes_found": len(scenes),
                        "scenes_used": len(comp.scenes_used), "dates": sorted(comp.by_date), **meta})
            print(f"  {month}: {len(scenes)} scenes, {len(comp.scenes_used)} with clear cells, {len(comp.by_date)} "
                  f"dates, {float((nobs > 0).mean()):.0%} of pixels composited, {time.time() - t0:.0f}s", flush=True)
    if failed:
        print(f"{region.name}: {len(failed)} month(s) failed: {', '.join(failed)}", flush=True)
        return 1
    print(f"{region.name}: done", flush=True)
    return 0
