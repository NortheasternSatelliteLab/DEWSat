"""Join the two HLS sources into one table per region, then rebuild VCI on it.

    months <  2020-01   regions/<name>/nasa/hls_monthly.csv            (nasa_backfill.py, NASA's own archive)
    months >= 2020-01   regions/<name>/swir/hls_monthly.csv (original 22) or hls_monthly.csv (combined "bands" pull)

Writes, next to the old files (which are left untouched, so earlier results stay reproducible):
    hls_full.csv    same 15 columns as hls_monthly.csv, 2016-01 .. latest
    vci_full.csv    make_vci.py on hls_full.csv, baseline 2016-01 .. BASELINE_END (training years only)

The two sources matched exactly where they overlap (New Mexico, Jun-Aug 2021: same clear passes, NDVI/NDMI/NDWI
corr 1.0000), so the switch at 2020-01 is a change of server, not of data.

Run from this folder:   python merge_hls_years.py regions.csv regions_new.csv   [--force]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "regions"
SWITCH = "2020-01"  # Planetary Computer's HLS starts here
BASELINE = "2016-01:2022-12"  # VCI min/max from training years only


def pc_table(d: Path) -> Path | None:
    for p in (d / "swir" / "hls_monthly.csv", d / "hls_monthly.csv"):
        if p.exists() and "ndmi_median" in p.open(encoding="utf-8").readline():
            return p
    return None


def merge_one(d: Path, force: bool) -> str:
    nasa, pc = d / "nasa" / "hls_monthly.csv", pc_table(d)
    if not (d / ".nasa.done").exists() or not nasa.exists():
        return "skip: nasa step not done"
    if pc is None or not ((d / ".bands.done").exists() or (d / ".swir.done").exists()):
        return "skip: Planetary Computer pull not done"
    out = d / "hls_full.csv"
    if out.exists() and not force and out.stat().st_mtime > max(nasa.stat().st_mtime, pc.stat().st_mtime):
        return "up to date"
    with nasa.open(encoding="utf-8", newline="") as fh:
        early = [r for r in csv.DictReader(fh) if r["month"] < SWITCH]
    with pc.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames
        late = [r for r in reader if r["month"] >= SWITCH]
    cells_early = {(r["cell_longitude"], r["cell_latitude"]) for r in early}
    cells_late = {(r["cell_longitude"], r["cell_latitude"]) for r in late}
    if cells_early != cells_late:
        raise ValueError(f"{d.name}: NASA and Planetary Computer tables cover different cells")
    rows = sorted(early + late, key=lambda r: (float(r["cell_longitude"]), -float(r["cell_latitude"]), r["month"]))
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)
    subprocess.run([sys.executable, str(HERE / "make_vci.py"), "--hls", str(out), "--out", str(d / "vci_full.csv"),
                    "--baseline", BASELINE, "--no-clip"], check=True, stdout=subprocess.DEVNULL)
    photo = lambda rs: sum(r["n_obs"] not in ("", "0") for r in rs) / max(len(rs), 1)
    return f"ok: {len(rows)} cell-months; with a photo: 2016-19 {photo(early):.0%}, 2020+ {photo(late):.0%}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("regions", nargs="+", type=Path, help="region list CSV(s)")
    ap.add_argument("--force", action="store_true", help="rebuild even if hls_full.csv is newer than its sources")
    ap.add_argument("--vci-baseline", default=None,
                    help="only (re)build an extra VCI from hls_full.csv with this START:END baseline, written as "
                         "vci_full_b<START year>.csv (e.g. 2020-01:2022-12 -> vci_full_b2020.csv)")
    args = ap.parse_args()
    if args.vci_baseline:
        for path in args.regions:
            with path.open(encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    d = OUT / r["name"]
                    out = d / f"vci_full_b{args.vci_baseline[:4]}.csv"
                    subprocess.run([sys.executable, str(HERE / "make_vci.py"), "--hls", str(d / "hls_full.csv"),
                                    "--out", str(out), "--baseline", args.vci_baseline, "--no-clip"],
                                   check=True, stdout=subprocess.DEVNULL)
                    print(f"{r['name']:<28}{out.name}")
        return 0
    bad = 0
    for path in args.regions:
        with path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    msg = merge_one(OUT / r["name"], args.force)
                except Exception as exc:
                    msg, bad = f"ERROR {exc}", bad + 1
                print(f"{r['name']:<28}{msg}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
