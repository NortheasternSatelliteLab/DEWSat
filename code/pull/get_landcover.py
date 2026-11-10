"""Download ESA CCI Land Cover 2020 (v2.1.1, 300 m, LCCS classes) over the contiguous U.S. window.

Source: Microsoft Planetary Computer collection `esa-cci-lc` (Cloud Optimized GeoTIFFs, anonymous SAS token).
Output: data/landcover/esa_cci_lc_2020_conus.tif (uint8 LCCS class, EPSG:4326, 1/360 degree) + .json provenance.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import rasterio
from rasterio.windows import from_bounds

V2 = Path(__file__).resolve().parents[2]  # the repository root
OUT = V2 / "data" / "landcover"
BBOX = (-125.0, 24.0, -66.5, 49.5)   # W, S, E, N: CONUS
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    body = {"collections": ["esa-cci-lc"], "bbox": list(BBOX), "datetime": "2020-01-01T00:00:00Z/2020-12-31T23:59:59Z",
            "limit": 50}
    feats = httpx.post(STAC, json=body, timeout=120).json()["features"]
    feats = [f for f in feats if "-2020-" in f["id"]]
    print([f["id"] for f in feats])
    tok = httpx.get("https://planetarycomputer.microsoft.com/api/sas/v1/token/esa-cci-lc", timeout=60).json()["token"]
    res = 1 / 360
    W, S, E, N = BBOX
    width, height = round((E - W) / res), round((N - S) / res)
    mosaic = np.zeros((height, width), np.uint8)
    for f in feats:
        href = f["assets"]["lccs_class"]["href"]
        with rasterio.open(f"/vsicurl/{href}?{tok}") as ds:
            b = ds.bounds
            w, s, e, n = max(W, b.left), max(S, b.bottom), min(E, b.right), min(N, b.top)
            if w >= e or s >= n:
                continue
            win = from_bounds(w, s, e, n, transform=ds.transform).round_offsets().round_lengths()
            arr = ds.read(1, window=win)
            r0, c0 = round((N - n) / res), round((w - W) / res)
            mosaic[r0:r0 + arr.shape[0], c0:c0 + arr.shape[1]] = np.where(arr > 0, arr,
                                                                           mosaic[r0:r0 + arr.shape[0], c0:c0 + arr.shape[1]])
            print(f["id"], arr.shape, ds.res, ds.crs, flush=True)
    profile = {"driver": "GTiff", "height": height, "width": width, "count": 1, "dtype": "uint8", "crs": "EPSG:4326",
               "transform": rasterio.transform.from_bounds(W, S, E, N, width, height), "compress": "deflate",
               "tiled": True, "nodata": 0}
    path = OUT / "esa_cci_lc_2020_conus.tif"
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mosaic, 1)
    (OUT / "esa_cci_lc_2020_conus.json").write_text(json.dumps({
        "product": "ESA CCI Land Cover / C3S, LCCS map, 300 m, annual, version 2.1.1, year 2020",
        "source": "Microsoft Planetary Computer, collection esa-cci-lc", "items": [f["id"] for f in feats],
        "bbox": BBOX, "pixel_deg": res, "downloaded": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes_present": {int(k): int(v) for k, v in zip(*np.unique(mosaic, return_counts=True))}}, indent=1))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
