"""Run one pull step over every region, a few regions at a time, with per-region logs and retries.

Modelled on Austin Schlather's data-pipeline/practice/pull_regions.py (dewstat-northeastern@7adb8d6): each region
runs in its own subprocess, finished work is skipped on a rerun, and a failed region is retried after a wait.

    python run_pull.py pc   --end 2026-09 --parallel 8 --workers 6
    python run_pull.py nasa --parallel 6 --workers 12          # Earthdata login from ~/.netrc (earthaccess)
    python run_pull.py pc   --status
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # ~/Desktop/DEWSat_Paper_BiLSTM
sys.path.insert(0, str(HERE))
from grid import earthdata_token, load_regions, month_list  # noqa: E402

STEPS = {"pc": ("pull_hls_pc.py", "2020-01"), "nasa": ("pull_hls_nasa.py", "2016-01")}
RETRY_WAITS = (120, 600)
_lock = threading.Lock()


def say(msg: str) -> None:
    with _lock:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def months_done(out: Path, region: str, months: list[str]) -> int:
    d = out / region / "hls"
    return sum((d / f"{m}.npz").exists() for m in months)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=sorted(STEPS))
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default="2019-12")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "regions")
    ap.add_argument("--logs", type=Path, default=ROOT / "logs" / "pull")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None, help="comma-separated region names")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    script, default_start = STEPS[a.step]
    start = a.start or default_start
    months = month_list(start, a.end)
    regions = load_regions()
    if a.only:
        keep = set(a.only.split(","))
        regions = [r for r in regions if r.name in keep]
    if a.status:
        done = 0
        for r in regions:
            n = months_done(a.out, r.name, months)
            done += n == len(months)
            print(f"{r.name:<30}{r.split:<6}{n:>4}/{len(months)}")
        print(f"{done}/{len(regions)} regions complete for {a.step} {start}..{a.end}")
        return 0

    log_dir = a.logs / a.step
    log_dir.mkdir(parents=True, exist_ok=True)
    say(f"{a.step}: {len(regions)} regions, {start}..{a.end}, {a.parallel} at a time, {a.workers} workers each")

    env = {k: v for k, v in os.environ.items() if k != "EARTHDATA_TOKEN"}
    if a.step == "nasa":  # log in once; children get the token in their environment only (never printed)
        env["EARTHDATA_TOKEN"] = earthdata_token()

    def one(r) -> tuple[str, int]:
        cmd = [sys.executable, "-u", "-I", str(HERE / script), "--region", r.name, "--start", start, "--end", a.end,
               "--out", str(a.out), "--workers", str(a.workers)]
        for attempt, wait in enumerate((*RETRY_WAITS, None)):
            if months_done(a.out, r.name, months) == len(months):
                return r.name, 0
            t0 = time.time()
            with (log_dir / f"{r.name}.log").open("a") as fh:
                fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {attempt + 1}: {' '.join(cmd)}\n")
                fh.flush()
                code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      env=env).returncode
            say(f"{r.name:<30} exit {code} after {(time.time() - t0) / 60:.1f} min "
                f"({months_done(a.out, r.name, months)}/{len(months)} months)")
            if code == 0:
                return r.name, 0
            if wait is not None:
                time.sleep(wait)
        return r.name, 1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        results = list(pool.map(one, regions))
    failed = [n for n, c in results if c]
    summary = {"step": a.step, "start": start, "end": a.end, "regions": len(regions), "failed": failed,
               "hours": round((time.time() - t0) / 3600, 2), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    say(f"{a.step}: finished in {summary['hours']} h; failed: {', '.join(failed) or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
