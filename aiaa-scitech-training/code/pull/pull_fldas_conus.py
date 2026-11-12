"""FLDAS Noah monthly 0.1-degree (FLDAS_NOAH01_C_GL_M v001) SoilMoi10_40cm_tavg over the CONUS window, 2001-01..2026-09.

Same product, variable, CMR granule search, OPeNDAP ASCII hyperslab and Earthdata bearer token as the paper run's
code/paper_hls/pull_fldas.py (commit f17e700, after Austin Schlather's Atlas FLDAS pull). The one change: one
hyperslab per month covers the whole planning window (lon -125.0..-66.5, lat 24.0..49.5, 585 x 255 cells), so 309
requests serve every region instead of one request per region and month. The label variable is never a model input.

Outputs
- data/locked/fldas_conus.npz: the full cube (months x 255 x 585, NaN = fill). LOCKED: it holds fresh-test cells.
- data/fldas/dev/<region>.npz: development regions only (the existing 102 and new development regions).
- data/locked/fldas_test/<region>.npz: fresh-test regions; hashed in data/locked/LOCK.json. Nothing reads these
  before Phase 5.
- logs/pull/fldas_check.json: the existing 102 regions' values compared with the paper run's fldas.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import V2, earthdata_token, load_regions, region_grid  # noqa: E402
from paper_pull_fldas import (SHORT_NAME, VARIABLE, VERSION, X_LAST, Y_LAST, ascii_url, axis_endpoints,  # noqa: E402
                              get_text, granules, parse_slab)

W0, S0, E0, N0 = -125.0, 24.0, -66.5, 49.5
NX, NY = round((E0 - W0) * 10), round((N0 - S0) * 10)
FILL = -9999.0
LOCKED = V2 / "data" / "locked"
CACHE = LOCKED / "fldas_conus_cache"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2001-01")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    token = earthdata_token()
    grans = [g for g in granules(a.start, a.end) if a.start <= g[0] <= a.end]
    print(f"{len(grans)} FLDAS granules {grans[0][0]}..{grans[-1][0]} ({SHORT_NAME} v{VERSION})", flush=True)
    with httpx.Client(timeout=httpx.Timeout(600.0), limits=httpx.Limits(max_connections=a.workers + 4)) as http:
        text = get_text(http, ascii_url(grans[0][1], f"X[0:{X_LAST}:{X_LAST}],Y[0:{Y_LAST}:{Y_LAST}]"), token)
        x0, x1 = axis_endpoints(text, "X")
        y0, y1 = axis_endpoints(text, "Y")
        dx, dy = (x1 - x0) / X_LAST, (y1 - y0) / Y_LAST
        if abs(abs(dx) - 0.1) > 1e-4 or abs(abs(dy) - 0.1) > 1e-4:
            raise SystemExit(f"unexpected FLDAS axes X {x0}..{x1}, Y {y0}..{y1}")
        xa = round((W0 + 0.05 - x0) / dx)
        ya = round((S0 + 0.05 - y0) / dy)
        xb, yb = xa + NX - 1, ya + NY - 1
        print(f"axes X {x0}..{x1} step {dx:.4f}, Y {y0}..{y1} step {dy:.4f}; window X[{xa}:{xb}] Y[{ya}:{yb}]",
              flush=True)
        todo = [(m, u) for m, u in grans if not (CACHE / f"{m}.npy").exists()]
        print(f"{len(todo)} months to read", flush=True)

        def one(month, url):
            t = get_text(http, ascii_url(url, f"{VARIABLE}[0:0][{ya}:{yb}][{xa}:{xb}]"), token)
            rows = np.array(parse_slab(t, NY, NX), np.float64)     # south to north
            rows = np.where(rows <= FILL + 1, np.nan, rows).astype(np.float32)
            tmp = CACHE / f"{month}.tmp.npy"
            np.save(tmp, rows)
            tmp.rename(CACHE / f"{month}.npy")
            return month

        failed = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futs = [pool.submit(one, m, u) for m, u in todo]
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    f.result()
                except SystemExit:
                    raise
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(f"  FAILED: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                if i % 25 == 0:
                    print(f"  {i}/{len(todo)} months ({(time.time() - t0) / i:.1f} s each)", flush=True)
    if failed:
        print(f"{failed} months failed; rerun to resume", flush=True)
        return 1
    months = [m for m, _ in grans]
    cube = np.stack([np.load(CACHE / f"{m}.npy") for m in months])            # (T, NY, NX), row 0 = south
    lons = np.round(W0 + 0.05 + np.arange(NX) * 0.1, 2)
    lats = np.round(S0 + 0.05 + np.arange(NY) * 0.1, 2)
    np.savez_compressed(LOCKED / "fldas_conus.npz", values=cube, months=np.array(months), lons=lons, lats=lats,
                        x_index=np.arange(xa, xb + 1), y_index=np.arange(ya, yb + 1))
    # per-region extraction
    regions = load_regions()
    (V2 / "data" / "fldas" / "dev").mkdir(parents=True, exist_ok=True)
    (LOCKED / "fldas_test").mkdir(parents=True, exist_ok=True)
    lock = {"note": "Fresh-test regions' FLDAS labels. Not read by any code before Phase 5 (see DECISIONS.md).",
            "files": {}}
    for r in regions:
        g = region_grid(r)
        lon_c = np.array([g.cell_center(i)[0] for i in range(g.n_cells)])
        lat_c = np.array([g.cell_center(i)[1] for i in range(g.n_cells)])
        xi = np.round((lon_c - (W0 + 0.05)) / 0.1).astype(int)
        yi = np.round((lat_c - (S0 + 0.05)) / 0.1).astype(int)
        vals = cube[:, yi, xi].T.astype(np.float32)                       # (cells, months)
        dest = (LOCKED / "fldas_test" if r.role == "test" else V2 / "data" / "fldas" / "dev") / f"{r.name}.npz"
        np.savez_compressed(dest, values=vals, months=np.array(months), cell_lon=lon_c, cell_lat=lat_c)
        if r.role == "test":
            lock["files"][r.name] = sha256(dest)
    lock["cube_sha256"] = sha256(LOCKED / "fldas_conus.npz")
    lock["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (LOCKED / "LOCK.json").write_text(json.dumps(lock, indent=1))
    # check against the paper run's per-region pull for the existing 102
    check = {"regions": 0, "values": 0, "max_abs_diff": 0.0, "mismatch_over_1e-6": 0, "missing_both": 0,
             "missing_one": 0}
    paper = V2 / "inputs" / "paper_data" / "regions"
    for r in regions:
        if r.origin != "existing":
            continue
        z = np.load(V2 / "data" / "fldas" / "dev" / f"{r.name}.npz")
        idx = {(round(float(lo), 2), round(float(la), 2)): i for i, (lo, la) in enumerate(zip(z["cell_lon"], z["cell_lat"]))}
        mi = {m: j for j, m in enumerate(z["months"])}
        check["regions"] += 1
        with (paper / r.name / "fldas.csv").open() as fh:
            for row in csv.DictReader(fh):
                i = idx[(round(float(row["cell_longitude"]), 2), round(float(row["cell_latitude"]), 2))]
                j = mi.get(row["month"])
                if j is None:
                    continue
                new = z["values"][i, j]
                old = row["SoilMoi10_40cm_tavg"]
                if old == "" and not np.isfinite(new):
                    check["missing_both"] += 1
                    continue
                if old == "" or not np.isfinite(new):
                    check["missing_one"] += 1
                    continue
                diff = abs(float(old) - float(new))
                check["values"] += 1
                check["max_abs_diff"] = max(check["max_abs_diff"], diff)
                check["mismatch_over_1e-6"] += int(diff > 1e-6)
    (V2 / "logs" / "pull").mkdir(parents=True, exist_ok=True)
    (V2 / "logs" / "pull" / "fldas_check.json").write_text(json.dumps(check, indent=1))
    print("check vs paper fldas.csv:", check, flush=True)
    print(f"done: {len(months)} months, {len(regions)} regions", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
