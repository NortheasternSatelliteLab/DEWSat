"""Pull FLDAS + HLS + VCI for every region in regions.csv (the multi-region training data).

For each region, into regions/<name>/:
    fldas.csv           atlas data fldas landsurface   (2001 .. latest)
    hls_monthly.csv     step "bands": build_table.py --swir, one pass reading Red, NIR, SWIR1, SWIR2 and Fmask
                        per scene -> NDVI + NDMI + NDWI per cell per month (HLS 2016 .. last full month;
                        plus cache_swir.jsonl)
    vci.csv             make_vci.py --no-clip, baseline = HLS start .. BASELINE_END (training years only)
    nasa/hls_monthly.csv  step "nasa": nasa_backfill.py, 2016 .. 2019 read from NASA's own archive (Planetary
                        Computer's HLS starts 2020). Same columns as hls_monthly.csv; matched it exactly on 2021.

The 22 original regions were pulled with two older steps instead of "bands" and keep that layout:
    hls                 build_table.py: hls_monthly.csv with NDVI only (cache.jsonl)
    swir                build_table.py --swir: swir/hls_monthly.csv (own cache swir/cache_swir.jsonl)
A region that has .hls.done / .swir.done never runs "bands" (so its files are never overwritten), and a
region that has .bands.done never runs "hls" / "swir". The model reads either layout.

Safe to stop and rerun: finished steps are skipped (a .done file marks each), and build_table.py's
cache.jsonl means a half-finished HLS pull resumes instead of starting over.

Run from this folder:
    uv run --project ../Atlas python pull_regions.py                    # every region
    uv run --project ../Atlas python pull_regions.py --only iowa_corn   # one region (comma list ok)
    uv run --project ../Atlas python pull_regions.py --status           # what is done / left
    uv run --project ../Atlas python pull_regions.py --steps swir       # only the SWIR pull

Timing (measured on a 6-core, 7 GB Linux server): FLDAS ~20-25 min per region; "bands" ~15-25 min per region
(Planetary Computer serves ~230 scenes/min per user); "nasa" ~15-25 min per region. 80 regions took ~9 h at
--parallel 6 --workers 6 (more regions at once got HTTP 403 from Planetary Computer's search). Run long pulls in
tmux/screen, and watch them with watch_pull.sh.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATLAS = HERE.parent / "Atlas"
OUT = HERE / "regions"
FLDAS_START = "2001-01-01"
HLS_START = "2016-01"  # HLS Sentinel-2 starts late 2015; Landsat (L30) 2013
BASELINE_END = "2022-12"  # VCI min/max only from years the model trains on (no peeking at val/test)
STEPS = ("fldas", "bands", "vci", "nasa")  # new regions
LEGACY_STEPS = ("fldas", "hls", "vci", "swir", "nasa")  # layout of the 22 original regions
ALL_STEPS = ("fldas", "bands", "hls", "vci", "swir", "nasa")
NASA_END = "2019-12"  # Planetary Computer's HLS starts 2020-01; "nasa" fills 2016-01 .. NASA_END from NASA itself
# nasa_backfill.py needs rasterio, which lives in this folder's own venv (not the Atlas one)
NASA_PY = HERE / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
# atlas's default set leaves out rain
# and air temperature, which the model needs, so always ask for these explicitly.
FLDAS_VARIABLES = (
    "SoilMoi00_10cm_tavg,SoilMoi10_40cm_tavg,SoilMoi40_100cm_tavg,SoilMoi100_200cm_tavg,"
    "Evap_tavg,Qs_tavg,Qsb_tavg,RadT_tavg,Rainf_f_tavg,Tair_f_tavg"
)
RETRY_WAITS = (120, 600)  # seconds to wait before retry 1 and retry 2 of a failed step
# HLS pixel size to request. 120 m matched 30 m on Iowa (per-cell NDVI/NDMI within 0.007, corr 0.999) and on
# New Mexico rangeland (NDVI/NDMI/NDWI corr 0.9999, mean |diff| <= 0.0006, clear-photo months unchanged)
# and downloads ~1.5-2x faster: the server's per-file overhead, not the data size, is the bottleneck.
PIXEL_M = 120


def last_full_month() -> str:
    t = date.today()
    y, m = (t.year, t.month - 1) if t.month > 1 else (t.year - 1, 12)
    return f"{y:04d}-{m:02d}"


def load_regions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def coords(r: dict) -> str:
    return f"{r['west']},{r['south']},{r['east']},{r['north']}"


def run(cmd: list[str], log: Path, cwd: Path = HERE) -> None:
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(map(str, cmd))}\n")
        fh.flush()
        # No stdin: a step that wants to ask something (newer atlas confirms downloads) fails instead of hanging.
        proc = subprocess.run(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}; see {log}")


def atlas_exe() -> str:
    exe = shutil.which("atlas")
    if not exe:
        sys.exit("`atlas` not found. Run this with:  uv run --project ../Atlas python pull_regions.py")
    return exe


def do_fldas(r: dict, d: Path, log: Path) -> None:
    run([atlas_exe(), "data", "fldas", "landsurface",
         "--date", f"{FLDAS_START}:{date.today().isoformat()}",
         f"--coords={coords(r)}", "--limit", "1000", "--variables", FLDAS_VARIABLES, "--yes",
         "--out", str(d / "fldas_pull"), "--table", str(d / "fldas.csv")], log, cwd=ATLAS)


def do_hls(r: dict, d: Path, log: Path, workers: int, pixel_m: float = PIXEL_M) -> None:
    run([sys.executable, "-u", str(HERE / "build_table.py"), f"--coords={coords(r)}",
         "--start", HLS_START, "--end", last_full_month(), "--out", str(d), "--workers", str(workers),
         "--pixel-m", str(pixel_m)], log)


def do_swir(r: dict, d: Path, log: Path, workers: int, pixel_m: float = PIXEL_M) -> None:
    """Red, NIR and both SWIR bands -> NDVI, NDMI, NDWI per cell per month, in regions/<name>/swir/."""
    out = d / "swir"
    out.mkdir(exist_ok=True)
    run([sys.executable, "-u", str(HERE / "build_table.py"), f"--coords={coords(r)}", "--swir",
         "--start", HLS_START, "--end", last_full_month(), "--out", str(out), "--workers", str(workers),
         "--pixel-m", str(pixel_m), "--search-cache", str(d / "search_cache.jsonl")], log)


def do_bands(r: dict, d: Path, log: Path, workers: int, pixel_m: float = PIXEL_M) -> None:
    """Red, NIR, both SWIR bands and Fmask in one request per scene -> NDVI, NDMI, NDWI in hls_monthly.csv."""
    run([sys.executable, "-u", str(HERE / "build_table.py"), f"--coords={coords(r)}", "--swir",
         "--start", HLS_START, "--end", last_full_month(), "--out", str(d), "--workers", str(workers),
         "--pixel-m", str(pixel_m)], log)


def do_nasa(r: dict, d: Path, log: Path, workers: int, pixel_m: float = PIXEL_M) -> None:
    """HLS 2016 .. NASA_END straight from NASA (same bands and columns as "bands") -> nasa/hls_monthly.csv."""
    run([str(NASA_PY), "-u", str(HERE / "nasa_backfill.py"), f"--coords={coords(r)}", "--start", HLS_START,
         "--end", NASA_END, "--out", str(d / "nasa"), "--workers", str(workers), "--pixel-m", str(int(pixel_m))], log)


def region_steps(d: Path) -> tuple[str, ...]:
    """Old layout (separate hls + swir pulls) if the region was started that way, else the combined one."""
    return LEGACY_STEPS if any((d / f".{s}.done").exists() for s in ("hls", "swir")) else STEPS


def do_vci(r: dict, d: Path, log: Path) -> None:
    with (d / "hls_monthly.csv").open(encoding="utf-8") as fh:
        first = min(row["month"] for row in csv.DictReader(fh))
    run([sys.executable, str(HERE / "make_vci.py"), "--hls", str(d / "hls_monthly.csv"),
         "--out", str(d / "vci.csv"), "--baseline", f"{first}:{BASELINE_END}", "--no-clip"], log)


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", type=Path, default=HERE / "regions.csv")
    ap.add_argument("--only", default=None, help="comma-separated region names")
    ap.add_argument("--workers", type=int, default=3, help="parallel HLS downloads per region")
    ap.add_argument("--pixel-m", type=float, default=PIXEL_M, help="HLS pixel size in meters (30 = native)")
    # Requests in flight at once = parallel x workers (12 x 3 = 36; the old 3 x 6 = 18 ran fine on the laptop).
    # RAM grows with --parallel: ~0.2-0.3 GB per region at 120 m (each region is its own build_table.py process).
    # Watch with watch_pull.sh; if RAM gets tight or 403/429/503 retries climb, restart with --parallel 6 to 8.
    ap.add_argument("--parallel", type=int, default=12, help="regions pulled at the same time")
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    ap.add_argument("--steps", default=",".join(ALL_STEPS),
                    help=f"comma list of steps to run (default: all; each region runs those of its layout)")
    ap.add_argument("--out-root", type=Path, default=OUT, help="folder holding the region folders")
    args = ap.parse_args()
    steps = tuple(s for s in ALL_STEPS if s in args.steps.split(","))
    if not steps:
        sys.exit(f"--steps must name at least one of: {', '.join(ALL_STEPS)}")
    OUT = args.out_root

    regions = load_regions(args.regions)
    if args.only:
        keep = set(args.only.split(","))
        regions = [r for r in regions if r["name"] in keep]
        missing = keep - {r["name"] for r in regions}
        if missing:
            sys.exit(f"unknown region(s): {', '.join(sorted(missing))}")

    if args.status:
        for r in regions:
            d = OUT / r["name"]
            todo = [s for s in region_steps(d) if not (d / f".{s}.done").exists()]
            print(f"{r['name']:<26}{r['split']:<7}{'done' if not todo else 'todo: ' + ','.join(todo)}")
        return 0

    print(f"{len(regions)} regions, {args.parallel} at a time; per-region logs in regions/<name>/pull_log.txt",
          flush=True)
    failed: list[str] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:  # each region runs in its own subprocesses
        for result in pool.map(lambda r: pull_one(r, args.workers, steps, args.pixel_m), regions):
            failed.extend(result)
    hours = (time.time() - started) / 3600
    print(f"\nall regions done in {hours:.1f} h" if not failed
          else f"\nfailed after {hours:.1f} h (rerun to retry just these): {', '.join(failed)}")
    return 1 if failed else 0


def pull_one(r: dict, workers: int, steps: tuple[str, ...] = STEPS, pixel_m: float = PIXEL_M) -> list[str]:
    """The chosen steps for one region. Returns the steps that failed."""
    name = r["name"]
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    log = d / "pull_log.txt"
    for step in (s for s in steps if s in region_steps(d)):
        marker = d / f".{step}.done"
        if marker.exists():
            continue
        say(f"{name:<24} {step}: started")
        t0 = time.time()
        for attempt, wait in enumerate((*RETRY_WAITS, None)):
            try:
                if step == "fldas":
                    do_fldas(r, d, log)
                elif step == "bands":
                    do_bands(r, d, log, workers, pixel_m)
                elif step == "nasa":
                    do_nasa(r, d, log, workers, pixel_m)
                elif step == "hls":
                    do_hls(r, d, log, workers, pixel_m)
                elif step == "vci":
                    do_vci(r, d, log)
                else:
                    do_swir(r, d, log, workers, pixel_m)
                break
            except Exception as exc:
                # Planetary Computer answers 403 when it is busy; waiting and retrying usually works.
                # HLS resumes from cache.jsonl / cache_swir.jsonl, so a retry does not redo finished scenes.
                if wait is None:  # out of retries: move on to other regions; rerun later to retry
                    say(f"{name:<24} {step}: FAILED ({exc})")
                    return [f"{name}:{step}"]
                say(f"{name:<24} {step}: failed (attempt {attempt + 1}), retrying in {wait // 60} min")
                time.sleep(wait)
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        say(f"{name:<24} {step}: done in {(time.time() - t0) / 60:.1f} min")
    return []


_print_lock = threading.Lock()


def say(msg: str) -> None:
    with _print_lock:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
