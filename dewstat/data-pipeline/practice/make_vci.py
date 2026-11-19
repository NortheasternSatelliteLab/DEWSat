"""Convert monthly NDVI (hls_monthly.csv) into a VCI table (vci.csv).

VCI = Vegetation Condition Index:

    VCI = 100 * (NDVI - NDVI_min) / (NDVI_max - NDVI_min)

where NDVI_min / NDVI_max are the lowest / highest NDVI seen in THAT cell in
THAT calendar month (e.g. every July) during the baseline years.

    0   = as bad as this place has ever looked at this time of year
    100 = as good as it has ever looked
    <35 = common drought threshold

Run:
    uv run --project ../Atlas python make_vci.py --hls regions/iowa_corn/hls_monthly.csv --out regions/iowa_corn/vci.csv
    uv run --project ../Atlas python make_vci.py ... --baseline 2020-01:2023-12   # min/max from these years only
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hls", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--baseline",
        default=None,
        help="START:END months (YYYY-MM) whose NDVI sets the min/max. Default: every month in the file.",
    )
    ap.add_argument(
        "--no-clip",
        action="store_true",
        help="Keep values outside 0..100 (a month that beats the baseline record). Useful for models.",
    )
    args = ap.parse_args()

    with args.hls.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)]

    first, last = min(r["month"] for r in rows), max(r["month"] for r in rows)
    lo, hi = args.baseline.split(":") if args.baseline else (first, last)

    # 1. Lowest and highest NDVI per (cell, calendar month) inside the baseline years.
    extremes: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["ndvi_median"] != "" and lo <= r["month"] <= hi:
            extremes[(r["cell_longitude"], r["cell_latitude"], r["month"][5:])].append(float(r["ndvi_median"]))

    # 2. Rescale every month's NDVI between those two extremes.
    out_rows = []
    for r in rows:
        key = (r["cell_longitude"], r["cell_latitude"], r["month"][5:])
        seen = extremes.get(key, [])
        row = {
            "cell_longitude": r["cell_longitude"],
            "cell_latitude": r["cell_latitude"],
            "month": r["month"],
            "n_obs": r["n_obs"],
            "ndvi": r["ndvi_median"],
            "ndvi_min": "",
            "ndvi_max": "",
            "vci": "",
            "baseline_years": len(seen),
        }
        if r["ndvi_median"] != "" and len(seen) >= 2 and max(seen) > min(seen):
            ndvi, mn, mx = float(r["ndvi_median"]), min(seen), max(seen)
            # Outside the baseline years NDVI can beat the old record; clip to the 0..100 scale.
            vci = 100 * (ndvi - mn) / (mx - mn)
            if not args.no_clip:
                vci = min(max(vci, 0.0), 100.0)
            row.update(ndvi_min=round(mn, 4), ndvi_max=round(mx, 4), vci=round(vci, 1))
        out_rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    filled = [float(r["vci"]) for r in out_rows if r["vci"] != ""]
    drought = sum(v < 35 for v in filled)
    print(f"baseline {lo}..{hi}; wrote {args.out}: {len(out_rows)} rows, {len(filled)} with VCI, "
          f"{drought} ({drought / max(len(filled), 1):.0%}) below 35 (drought)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
