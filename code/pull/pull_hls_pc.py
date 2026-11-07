"""HLS v2 (S30 + L30) 2020+ from Microsoft Planetary Computer -> per-pixel monthly NDVI/NDMI composites and
per-view, per-cell values (DEWSat_BiLSTM_v2). Copied from the paper run's code/paper_hls/pull_hls_pc.py (commit
f17e700); the only change is that SWIR1 (S30 B11, L30 B06) is searched and read as well.

Adapted from Austin Schlather's data-pipeline/practice/build_table.py (dewstat-northeastern@7adb8d6): the same
collections (hls2-s30, hls2-l30), scene cloud filter (eo:cloud_cover <= 80), month-by-month search, cloud rule and
~120 m pixels. Changes: the STAC search is a direct HTTP call (not the Atlas client), only Red/NIR/Fmask are read,
and each scene's files are read directly (signed with Planetary Computer's anonymous SAS token) by cog_read.py,
the same reader the NASA 2016-2019 backfill uses, instead of through Planetary Computer's crop endpoint.

No Earthdata credentials are used or present in this process.
Output: <out>/<region>/hls/<YYYY-MM>.npz (see hls_common.py), grid.json, search_pc.jsonl.

    python pull_hls_pc.py --region iowa_corn --start 2020-01 --end 2026-09 --out ../../data/regions
"""

from __future__ import annotations

import argparse
import calendar
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_common import pull_months  # noqa: E402

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
SENSORS = {"S30": ("hls2-s30", "B04", "B8A", "B11"), "L30": ("hls2-l30", "B04", "B05", "B06")}
RETRY = (403, 429, 500, 502, 503, 504)
WAITS = (2, 5, 15, 30, 60, 120)


def _retry(call, label: str):
    for wait in (*WAITS, None):
        try:
            r = call()
            if r.status_code in RETRY:
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if wait is None or (status is not None and status not in RETRY):
                raise
            print(f"  retry {label} {status or type(exc).__name__} in {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


class SasTokens:
    """Planetary Computer's anonymous read token per collection, refreshed ten minutes before it expires."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tokens: dict[str, tuple[str, datetime]] = {}

    def get(self, collection: str) -> str:
        with self._lock:
            tok = self._tokens.get(collection)
            if tok is None or (tok[1] - datetime.now(timezone.utc)).total_seconds() < 600:
                j = _retry(lambda: httpx.get(SAS.format(collection=collection), timeout=60), f"sas {collection}")
                expiry = datetime.fromisoformat(j["msft:expiry"].replace("Z", "+00:00"))
                tok = (j["token"], expiry)
                self._tokens[collection] = tok
            return tok[0]


def search(grid, month: str, sensor: str) -> list[dict]:
    collection, red, nir, swir = SENSORS[sensor]
    y, m = map(int, month.split("-"))
    body = {"collections": [collection], "bbox": [grid.west, grid.south, grid.east, grid.north],
            "datetime": f"{month}-01T00:00:00Z/{month}-{calendar.monthrange(y, m)[1]:02d}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lte": MAX_CLOUD}}, "limit": 500}
    out = []
    while True:
        page = _retry(lambda: httpx.post(STAC, json=body, timeout=120), f"search {sensor} {month}")
        for f in page.get("features", []):
            a = f["assets"]
            out.append({"id": f["id"], "collection": f["collection"], "date": f["properties"]["datetime"][:10],
                        "cloud": f["properties"].get("eo:cloud_cover"),
                        "urls": {"Fmask": a["Fmask"]["href"], "red": a[red]["href"], "nir": a[nir]["href"],
                                 "swir1": a[swir]["href"]}})
        nxt = next((ln for ln in page.get("links", []) if ln.get("rel") == "next"), None)
        if not nxt:
            return out
        body = {**body, **(nxt.get("body") or {})}


MAX_CLOUD = 80.0
TOKENS = SasTokens()


def urls(scene: dict, sensor: str) -> tuple[str, str, str, str]:
    tok = TOKENS.get(scene["collection"])
    u = scene["urls"]
    return f"{u['Fmask']}?{tok}", f"{u['red']}?{tok}", f"{u['nir']}?{tok}", f"{u['swir1']}?{tok}"


def main() -> int:
    global MAX_CLOUD
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True)
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-scene-cloud", type=float, default=80)
    a = ap.parse_args()
    MAX_CLOUD = a.max_scene_cloud
    return pull_months(a.region, a.start, a.end, a.out, a.workers, "planetary_computer_cog", tuple(SENSORS),
                       search, urls, "search_pc.jsonl", {"max_scene_cloud": MAX_CLOUD, "overview_level": 1})


if __name__ == "__main__":
    raise SystemExit(main())
