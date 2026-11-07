"""FLDAS Noah monthly 0.1-degree (FLDAS_NOAH01_C_GL_M v001) root-zone soil moisture for every region.

Austin Schlather's pull (dewstat-northeastern@7adb8d6) got FLDAS through `atlas data fldas landsurface`, which reads one
OPeNDAP hyperslab per granule and region from NASA's Earthdata OPeNDAP service. This does the same request directly
(Atlas could not be installed in this environment): the same product and version, a CMR granule search, one ASCII
hyperslab per (granule, region), and the Earthdata bearer token. Only the label variable SoilMoi10_40cm_tavg is read;
it is never a model input.

Output: <out>/<region>/fldas.csv (cell_longitude, cell_latitude, month, SoilMoi10_40cm_tavg) once every granule of
the region is in; per-granule results are cached in <out>/<region>/fldas_cache.jsonl so a rerun resumes.

    python pull_fldas.py --out ../../data/regions --workers 16
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import earthdata_token, load_regions, region_grid  # noqa: E402

CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
SHORT_NAME, VERSION = "FLDAS_NOAH01_C_GL_M", "001"
VARIABLE = "SoilMoi10_40cm_tavg"
FILL = -9999.0
X_LAST, Y_LAST = 3599, 1499  # monthly Noah 0.1-degree grid (Atlas fldas.py)
WAITS = (3, 10, 30, 60, 120, 300)
ROW = re.compile(r"\[(\d+)\]\[(\d+)\]\s*,\s*(.*)$")
NUM = re.compile(r"-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?")
_lock = threading.Lock()


def granules(start: str, end: str) -> list[tuple[str, str]]:
    """(YYYY-MM, OPeNDAP base URL) per month, from CMR (metadata only, no credentials)."""
    r = httpx.get(CMR, params={"short_name": SHORT_NAME, "version": VERSION, "page_size": 2000,
                               "temporal": f"{start}-01T00:00:00Z,{end}-28T23:59:59Z", "sort_key": "start_date"},
                  timeout=120)
    r.raise_for_status()
    out = []
    for item in r.json()["items"]:
        umm = item["umm"]
        month = umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:7]
        url = next(u["URL"] for u in umm["RelatedUrls"] if "OPENDAP" in str(u.get("Subtype", "")).upper())
        out.append((month, url))
    return sorted(out)


def get_text(http: httpx.Client, url: str, token: str) -> str:
    for wait in (*WAITS, None):
        try:
            r = http.get(url, headers={"Authorization": f"Bearer {token}"}, follow_redirects=True)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            if r.status_code in (401, 403):
                raise SystemExit(f"Earthdata refused the token ({r.status_code}); check EARTHDATA_TOKEN")
            r.raise_for_status()
            return r.text
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if wait is None:
                raise
            status = getattr(getattr(exc, "response", None), "status_code", None)
            print(f"  retry {status or type(exc).__name__} in {wait}s: {url[-90:]}", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def ascii_url(base: str, constraint: str) -> str:
    return f"{base}.ascii?{quote(constraint, safe=',:')}"


def axis_endpoints(text: str, name: str) -> tuple[float, float]:
    """First and last value of a 1-D axis from either ASCII flavour."""
    for line in text.splitlines():
        head, _, rest = line.partition(",")
        if head.strip().split(".")[-1].split("[")[0] == name and rest:
            vals = [float(v) for v in NUM.findall(rest)]
            if len(vals) >= 2:
                return vals[0], vals[-1]
    raise ValueError(f"no {name} axis in the OPeNDAP response")


def parse_slab(text: str, ny: int, nx: int) -> list[list[float]]:
    """[ny][nx] values of VARIABLE, south to north and west to east, from a [1][ny][nx] ASCII hyperslab.

    The Earthdata cloud gateway labels each row by its latitude (``...[...time=14426][...Y=42.05], v, v``);
    classic DAP2 labels rows by index (``[0][2], v, v``). Both are accepted.
    """
    by_lat: dict[float, list[float]] = {}
    by_index: dict[int, list[float]] = {}
    for line in text.splitlines():
        head, _, rest = line.partition(",")
        if not rest or "[" not in head or (VARIABLE not in head and not head.lstrip().startswith("[")):
            continue
        vals = [float(v) for v in NUM.findall(rest)]
        lat = re.search(r"Y=(-?\d+(?:\.\d+)?)\]", head)
        if lat:
            by_lat[float(lat.group(1))] = vals
            continue
        m = ROW.search(line)
        if m:
            by_index[int(m.group(2))] = vals
    rows = [by_lat[k] for k in sorted(by_lat)] if by_lat else [by_index[k] for k in sorted(by_index)]
    if len(rows) != ny or any(len(r) != nx for r in rows):
        raise ValueError(f"expected {ny} rows of {nx} values, got {len(rows)} rows")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2001-01")
    ap.add_argument("--end", default="2026-12")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    token = earthdata_token()
    regions = load_regions()
    if a.only:
        regions = [r for r in regions if r.name in set(a.only.split(","))]
    grans = granules(a.start, a.end)
    print(f"{len(grans)} FLDAS granules {grans[0][0]}..{grans[-1][0]}; {len(regions)} regions", flush=True)

    with httpx.Client(timeout=httpx.Timeout(300.0), limits=httpx.Limits(max_connections=a.workers + 4)) as http:
        # Axis endpoints once: index i <-> center lon/lat, checked against the expected 0.1-degree spacing.
        text = get_text(http, ascii_url(grans[0][1], f"X[0:{X_LAST}:{X_LAST}],Y[0:{Y_LAST}:{Y_LAST}]"), token)
        x0, x1 = axis_endpoints(text, "X")
        y0, y1 = axis_endpoints(text, "Y")
        dx, dy = (x1 - x0) / X_LAST, (y1 - y0) / Y_LAST
        if abs(abs(dx) - 0.1) > 1e-4 or abs(abs(dy) - 0.1) > 1e-4:
            raise SystemExit(f"unexpected FLDAS axes X {x0}..{x1}, Y {y0}..{y1}")
        print(f"axes: X {x0}..{x1} step {dx:.4f}; Y {y0}..{y1} step {dy:.4f}", flush=True)

        plans = {}
        for r in regions:
            g = region_grid(r)
            lons = [g.cell_center(i)[0] for i in range(g.cols)]
            lats = [g.cell_center(i * g.cols)[1] for i in range(g.rows)]
            xi = sorted(round((lon - x0) / dx) for lon in lons)
            yi = sorted(round((lat - y0) / dy) for lat in lats)
            plans[r.name] = (g, xi[0], xi[-1], yi[0], yi[-1])

        def cache_of(name: str) -> dict[str, list[list[float]]]:
            p = a.out / name / "fldas_cache.jsonl"
            done = {}
            if p.exists():
                for line in p.read_text().splitlines():
                    e = json.loads(line)
                    done[e["month"]] = e["values"]
            return done

        caches = {r.name: cache_of(r.name) for r in regions}
        tasks = [(r.name, m, u) for r in regions for m, u in grans if m not in caches[r.name]]
        print(f"{len(tasks)} (region, month) slabs to read", flush=True)

        def one(name: str, month: str, url: str):
            g, xa, xb, ya, yb = plans[name]
            text = get_text(http, ascii_url(url, f"{VARIABLE}[0:0][{ya}:{yb}][{xa}:{xb}]"), token)
            return name, month, parse_slab(text, yb - ya + 1, xb - xa + 1)

        t0, n, failed = time.time(), 0, 0
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(one, *t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    name, month, values = fut.result()
                except SystemExit:
                    raise
                except Exception as exc:  # noqa: BLE001 - counted, retried on the next run
                    failed += 1
                    print(f"  FAILED slab: {type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr, flush=True)
                    continue
                with _lock:
                    caches[name][month] = values
                    p = a.out / name / "fldas_cache.jsonl"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with p.open("a") as fh:
                        fh.write(json.dumps({"month": month, "values": values}) + "\n")
                n += 1
                if n % 500 == 0:
                    print(f"  {n}/{len(tasks)} slabs ({n / (time.time() - t0) * 60:.0f}/min)", flush=True)

    complete = 0
    for r in regions:
        g, xa, xb, ya, yb = plans[r.name]
        cache = caches[r.name]
        if len(cache) < len(grans):
            continue
        complete += 1
        with (a.out / r.name / "fldas.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cell_longitude", "cell_latitude", "month", VARIABLE])
            for month, _ in grans:
                vals = cache[month]
                for i in range(g.n_cells):
                    lon, lat = g.cell_center(i)
                    row = round((lat - y0) / dy) - ya
                    col = round((lon - x0) / dx) - xa
                    v = vals[row][col]
                    w.writerow([lon, lat, month, "" if v <= FILL + 1 else v])
    print(f"fldas.csv written for {complete}/{len(regions)} regions; {failed} slabs failed", flush=True)
    return 1 if failed or complete < len(regions) else 0


if __name__ == "__main__":
    raise SystemExit(main())
