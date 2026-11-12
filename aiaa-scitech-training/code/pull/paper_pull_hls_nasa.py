"""HLS v2 (S30 + L30) 2016-2019 from NASA LP DAAC -> per-pixel monthly NDVI composites, same grid as 2020+.

Adapted from Austin Schlather's data-pipeline/practice/nasa_backfill.py (dewstat-northeastern@7adb8d6): the same CMR
search (HLSS30 / HLSL30 v2.0, scene cloud <= 80%) and the same Earthdata bearer-token GDAL settings. Changes: only
Red, NIR and Fmask are read, through cog_read.py (the reader the 2020+ pull uses), so both eras share one pixel
sampling; the token comes from an earthaccess login with the user's ~/.netrc (grid.earthdata_token) and is never
printed. Only NASA hosts are read in this process, so the bearer token is never sent anywhere else.

    python pull_hls_nasa.py --region iowa_corn --start 2016-01 --end 2019-12 --out ../../data/regions
"""

from __future__ import annotations

import argparse
import calendar
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import earthdata_token  # noqa: E402
from pull_common import pull_months  # noqa: E402

CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
PRODUCTS = {"S30": ("HLSS30", "B04", "B8A"), "L30": ("HLSL30", "B04", "B05")}
WAITS = (5, 15, 30, 60, 120, 300)
MAX_CLOUD = 80.0


def gdal_env(token: str) -> None:
    os.environ.update({
        "GDAL_HTTP_AUTH": "BEARER", "GDAL_HTTP_BEARER": token, "CPL_VSIL_CURL_USE_HEAD": "NO",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES", "GDAL_HTTP_MULTIPLEX": "YES", "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "3", "GDAL_CACHEMAX": "64",
    })


def search(grid, month: str, sensor: str) -> list[dict]:
    short, red, nir = PRODUCTS[sensor]
    y, m = map(int, month.split("-"))
    params = {"short_name": short, "version": "2.0", "page_size": 2000,
              "bounding_box": f"{grid.west},{grid.south},{grid.east},{grid.north}",
              "temporal": f"{month}-01T00:00:00Z,{month}-{calendar.monthrange(y, m)[1]:02d}T23:59:59Z"}
    for wait in (*WAITS, None):
        try:
            r = httpx.get(CMR, params=params, timeout=120)
            r.raise_for_status()
            break
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if wait is None or (status is not None and status not in (403, 429, 500, 502, 503, 504)):
                raise
            print(f"  retry search {status or type(exc).__name__} {sensor} {month} in {wait}s", file=sys.stderr,
                  flush=True)
            time.sleep(wait)
    out = []
    for item in r.json()["items"]:
        umm = item["umm"]
        attrs = {a["Name"]: a["Values"][0] for a in umm.get("AdditionalAttributes", [])}
        cloud = float(attrs.get("CLOUD_COVERAGE", 0))
        if cloud > MAX_CLOUD:
            continue
        files = {u["URL"].rsplit(".", 2)[-2]: u["URL"] for u in umm.get("RelatedUrls", [])
                 if u["URL"].startswith("https://") and u["URL"].endswith(".tif")}
        if not all(b in files for b in (red, nir, "Fmask")):
            continue
        out.append({"id": umm["GranuleUR"], "date": umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:10],
                    "cloud": cloud, "urls": {"Fmask": files["Fmask"], "red": files[red], "nir": files[nir]}})
    return out


def urls(scene: dict, sensor: str) -> tuple[str, str, str]:
    u = scene["urls"]
    return u["Fmask"], u["red"], u["nir"]


def main() -> int:
    global MAX_CLOUD
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True)
    ap.add_argument("--start", default="2016-01")
    ap.add_argument("--end", default="2019-12")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-scene-cloud", type=float, default=80)
    a = ap.parse_args()
    MAX_CLOUD = a.max_scene_cloud
    gdal_env(earthdata_token())
    return pull_months(a.region, a.start, a.end, a.out, a.workers, "nasa_lpdaac_cog", tuple(PRODUCTS), search, urls,
                       "search_nasa.jsonl", {"max_scene_cloud": MAX_CLOUD, "overview_level": 1})


if __name__ == "__main__":
    raise SystemExit(main())
