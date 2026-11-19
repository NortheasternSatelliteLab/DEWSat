"""Daily NLDAS-2 Noah soil moisture 10-40 cm at every region cell (for the weekly/daily persistence check).

FLDAS for the US is monthly only, so a weekly or daily question needs another answer key. NLDAS-2 Noah (0.125°, hourly)
has the same 10-40 cm layer. One file per day (the 12:00 UTC hour, ~6 MB each) is downloaded, the region cells are read
(nearest NLDAS pixel to each 0.1° cell centre), and the file is deleted.

    .venv/bin/python nldas_daily.py --start 2019-01-01 --end 2025-12-31 --workers 8
Writes nldas_daily.npz (days x cells, m³/m³ = kg/m² ÷ 300) and nldas_cells.csv. Resumable: finished days live in
nldas_days/ as small .npy files.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
URL = ("https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_NOAH0125_H.2.0/{y}/{doy:03d}/"
       "NLDAS_NOAH0125_H.A{ymd}.1200.020.nc")


def token() -> str:
    for line in (HERE.parent / "Atlas" / ".env").read_text().splitlines():
        if line.startswith("EARTHDATA_TOKEN="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit("EARTHDATA_TOKEN not found in Atlas/.env")


def region_cells(csvs: list[Path]) -> list[tuple[str, str, float, float]]:
    """(region, split, lon, lat) for every 0.1° cell centre, the same cells the model uses."""
    out = []
    for path in csvs:
        for r in csv.DictReader(path.open()):
            w, s, e, n = (float(r[k]) for k in ("west", "south", "east", "north"))
            for lat in np.arange(s + 0.05, n, 0.1):
                for lon in np.arange(w + 0.05, e, 0.1):
                    out.append((r["name"], r["split"], round(float(lon), 2), round(float(lat), 2)))
    return out


def fetch_day(day: date, rows: np.ndarray, cols: np.ndarray, out_dir: Path, tok: str) -> str:
    dest = out_dir / f"{day:%Y%m%d}.npy"
    if dest.exists():
        return "cached"
    url = URL.format(y=day.year, doy=day.timetuple().tm_yday, ymd=f"{day:%Y%m%d}")
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        nc = Path(tmp) / "f.nc"
        for _ in range(4):
            r = subprocess.run(["curl", "-s", "-L", "-m", "300", "-H", f"Authorization: Bearer {tok}", "-o", str(nc),
                                "-w", "%{http_code}", url], capture_output=True, text=True)
            if r.stdout == "200":
                break
        else:
            return f"FAILED {day} http {r.stdout}"
        with rasterio.open(f"netcdf:{nc}:SoilM_10_40cm") as d:
            grid = d.read(1).astype("float32")
        v = grid[rows, cols]
        v[v <= -9000] = np.nan
        np.save(dest, v / 300.0)  # kg/m² over a 0.3 m layer -> m³/m³
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    cells = region_cells([HERE / "regions.csv", HERE / "regions_new.csv"])
    lon = np.array([c[2] for c in cells]); lat = np.array([c[3] for c in cells])
    # NLDAS grid: 464 x 224, pixel centres -124.9375 + 0.125 i, rows north-up from 52.9375.
    cols = np.rint((lon + 124.9375) / 0.125).astype(int)
    rows = np.rint((52.9375 - lat) / 0.125).astype(int)
    with (HERE / "nldas_cells.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["region", "split", "lon", "lat"]); w.writerows(cells)

    out_dir = HERE / "nldas_days"; out_dir.mkdir(exist_ok=True)
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    days = [d0 + timedelta(k) for k in range((d1 - d0).days + 1)]
    tok = token()
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for day, status in zip(days, ex.map(lambda d: fetch_day(d, rows, cols, out_dir, tok), days)):
            done += 1
            if status.startswith("FAILED"):
                print(status, flush=True)
            if done % 50 == 0:
                print(f"{done}/{len(days)} days", flush=True)
    have = [d for d in days if (out_dir / f"{d:%Y%m%d}.npy").exists()]
    sm = np.stack([np.load(out_dir / f"{d:%Y%m%d}.npy") for d in have])
    np.savez_compressed(HERE / "nldas_daily.npz", sm=sm, days=np.array([d.isoformat() for d in have]))
    print(f"wrote nldas_daily.npz: {sm.shape[0]} days x {sm.shape[1]} cells; missing {len(days) - len(have)} days")
    return 0 if len(have) == len(days) else 1


if __name__ == "__main__":
    raise SystemExit(main())
