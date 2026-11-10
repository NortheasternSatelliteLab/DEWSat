"""Phase 1 pull driver: new regions in their seeded batches of 50, then the original 102 re-extracted per scene.

Each region runs two steps as separate processes, as in the paper run's run_pull.py (after Austin Schlather's
pull_regions.py): NASA LP DAAC for 2016-01..2019-12 and Planetary Computer for 2020-01..2026-09. The two archives are
different servers, so their steps run in two pools at the same time. A failed step is retried after 2 and 10
minutes; a region that still fails is retried once more after the last batch, then logged as incomplete.

Stop rules (checked before each batch; STOP_PULL and the disk rule also between steps):
- 72 hours since the first launch (data/pull_state.json keeps the start time across restarts);
- free disk below 100 GB;
- a file named STOP_PULL in the repository root;
- no batch left.
A batch cut short by STOP_PULL or the disk rule is left out of the frozen dataset, so every frozen batch is whole and
balanced. When the driver stops it writes data/FROZEN.json, the frozen list of regions.

    caffeinate -dimsu nohup .venv/bin/python -u -I code/pull/pull_driver.py > logs/pull/driver.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grid import V2, earthdata_token, load_regions, month_list  # noqa: E402

PLAN = V2 / "data" / "plan" / "batches.json"
CONTROL = V2 / "data" / "pull_control.json"   # optional {"par_nasa": n, "par_pc": n}, read before each batch
STATE = V2 / "data" / "pull_state.json"
STOP = V2 / "STOP_PULL"
OUT_NEW = V2 / "data" / "regions"
OUT_ORIG = V2 / "data" / "regions_reextract"
LOGS = V2 / "logs" / "pull"
LIMIT_H = 72.0
MIN_FREE_GB = 100.0
STEPS = {"nasa": ("pull_hls_nasa.py", "2016-01", "2019-12"), "pc": ("pull_hls_pc.py", "2020-01", "2026-09")}
RETRY_WAITS = (120, 600)
_lock = threading.Lock()


def say(msg: str) -> None:
    with _lock:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "started_epoch": time.time(), "batches": {},
            "originals": {}, "events": []}


def save_state(st: dict) -> None:
    tmp = STATE.with_name(STATE.name + ".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, STATE)


def free_gb() -> float:
    return shutil.disk_usage(V2).free / 1e9


def stop_reason(st: dict, include_time: bool = True) -> str | None:
    if STOP.exists():
        return "STOP_PULL file"
    if free_gb() < MIN_FREE_GB:
        return f"free disk {free_gb():.0f} GB < {MIN_FREE_GB:.0f} GB"
    if include_time and (time.time() - st["started_epoch"]) / 3600 >= LIMIT_H:
        return f"{LIMIT_H:.0f} hours of pulling"
    return None


def months_done(out: Path, region: str, step: str) -> int:
    _, a, b = STEPS[step]
    d = out / region / "hls"
    return sum((d / f"{m}.npz").exists() for m in month_list(a, b))


def step_complete(out: Path, region: str, step: str) -> bool:
    _, a, b = STEPS[step]
    return months_done(out, region, step) == len(month_list(a, b))


class Runner:
    def __init__(self, workers: int, env: dict, st: dict, token: str):
        self.workers = workers
        self.env = env                                   # Planetary Computer steps: no Earthdata credentials
        self.env_nasa = {**env, "EARTHDATA_TOKEN": token}  # NASA steps only; never printed or written
        self.st = st
        self.aborted = False

    def step(self, region: str, step: str, out: Path) -> bool:
        script, a, b = STEPS[step]
        cmd = [sys.executable, "-u", "-I", str(HERE / script), "--region", region, "--start", a, "--end", b,
               "--out", str(out), "--workers", str(self.workers)]
        log_dir = LOGS / ("orig_" + step if out == OUT_ORIG else step)
        log_dir.mkdir(parents=True, exist_ok=True)
        for attempt, wait in enumerate((*RETRY_WAITS, None)):
            if step_complete(out, region, step):
                return True
            r = stop_reason(self.st, include_time=False)
            if r:
                self.aborted = True
                say(f"{region} {step}: not started ({r})")
                return False
            t0 = time.time()
            with (log_dir / f"{region}.log").open("a") as fh:
                fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {attempt + 1}: {' '.join(cmd)}\n")
                fh.flush()
                code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      env=self.env_nasa if step == "nasa" else self.env).returncode
            say(f"{region:<12} {step:<4} exit {code} after {(time.time() - t0) / 60:.1f} min "
                f"({months_done(out, region, step)}/{len(month_list(a, b))} months)")
            if code == 0 and step_complete(out, region, step):
                return True
            if wait is not None:
                time.sleep(wait)
        return False

    def regions(self, names: list[str], out: Path, par_nasa: int, par_pc: int) -> dict[str, bool]:
        res: dict[str, dict[str, bool]] = {n: {} for n in names}
        with ThreadPoolExecutor(max_workers=par_nasa) as pn, ThreadPoolExecutor(max_workers=par_pc) as pp:
            futs = []
            for n in names:
                futs.append((n, "nasa", pn.submit(self.step, n, "nasa", out)))
                futs.append((n, "pc", pp.submit(self.step, n, "pc", out)))
            for n, s, f in futs:
                try:
                    res[n][s] = bool(f.result())
                except Exception as exc:  # noqa: BLE001
                    say(f"{n} {s}: driver error {type(exc).__name__}: {exc}")
                    res[n][s] = False
        return {n: all(v.values()) for n, v in res.items()}


def counts(st: dict) -> dict:
    """Data counts for STATUS.md: complete regions by role and side, cells, months on disk."""
    regs = {r.name: r for r in load_regions()}
    out = {"regions_complete": {}, "cells_complete": 0, "month_files": 0}
    for b, info in st["batches"].items():
        for n in info.get("ok", []):
            r = regs[n]
            key = f"{r.role}_{r.side}"
            out["regions_complete"][key] = out["regions_complete"].get(key, 0) + 1
            out["cells_complete"] += 25
    for d in (OUT_NEW, OUT_ORIG):
        if d.exists():
            for reg in d.iterdir():
                h = reg / "hls"
                if h.is_dir():
                    out["month_files"] += sum(1 for p in h.iterdir() if p.suffix == ".npz" and ".tmp" not in p.name)
    out["originals_complete"] = sum(len(v.get("ok", [])) for v in st["originals"].values())
    out["free_disk_gb"] = round(free_gb(), 1)
    out["elapsed_h"] = round((time.time() - st["started_epoch"]) / 3600, 2)
    return out


def write_status(st: dict, note: str) -> None:
    c = counts(st)
    st["counts"] = c
    st["note"] = note
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(st)
    try:
        sys.path.insert(0, str(HERE.parent))
        import status  # code/status.py renders STATUS.md from every stage's state
        status.render()
    except Exception as exc:  # noqa: BLE001
        say(f"STATUS.md not rendered: {type(exc).__name__}: {exc}")


def freeze(st: dict, reason: str) -> None:
    regs = {r.name: r for r in load_regions()}
    plan = json.loads(PLAN.read_text())
    whole = [int(b) for b, v in st["batches"].items() if v.get("state") == "complete"]
    new_ok = sorted(n for b in whole for n in st["batches"][str(b)]["ok"])
    failed = sorted(n for b in whole for n in st["batches"][str(b)].get("failed", []))
    orig_ok = sorted(n for v in st["originals"].values() for n in v.get("ok", []))
    frozen = {
        "frozen": time.strftime("%Y-%m-%d %H:%M:%S"), "stop_reason": reason,
        "elapsed_h": round((time.time() - st["started_epoch"]) / 3600, 2),
        "batches_planned": len(plan["batches"]), "batches_frozen": sorted(whole),
        "batches_left_out": sorted(int(b) for b, v in st["batches"].items() if v.get("state") != "complete"),
        "new_regions": {n: {"role": regs[n].role, "side": regs[n].side, "batch": regs[n].batch} for n in new_ok},
        "new_regions_failed": failed,
        "original_regions_reextracted": orig_ok,
        "counts": {
            "new_dev": sum(regs[n].role == "dev" for n in new_ok),
            "new_test": sum(regs[n].role == "test" for n in new_ok),
            "new_dev_west": sum(regs[n].role == "dev" and regs[n].side == "west" for n in new_ok),
            "new_dev_east": sum(regs[n].role == "dev" and regs[n].side == "east" for n in new_ok),
            "new_test_west": sum(regs[n].role == "test" and regs[n].side == "west" for n in new_ok),
            "new_test_east": sum(regs[n].role == "test" and regs[n].side == "east" for n in new_ok),
        },
    }
    (V2 / "data" / "FROZEN.json").write_text(json.dumps(frozen, indent=1))
    st["frozen"] = frozen["frozen"]
    st["events"].append({"time": frozen["frozen"], "event": f"frozen ({reason})"})
    write_status(st, f"frozen: {reason}")
    say(f"FROZEN: {frozen['counts']}, originals re-extracted {len(orig_ok)}; reason: {reason}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--par-nasa", type=int, default=10)
    ap.add_argument("--par-pc", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-batches", type=int, default=None, help="testing only")
    ap.add_argument("--no-originals", action="store_true")
    a = ap.parse_args()
    st = load_state()
    st["events"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": f"driver start pid {os.getpid()}",
                         "args": vars(a)})
    save_state(st)
    plan = json.loads(PLAN.read_text())
    env = {k: v for k, v in os.environ.items() if k != "EARTHDATA_TOKEN"}
    env["GDAL_CACHEMAX"] = "64"
    run = Runner(a.workers, env, st, earthdata_token())
    reason = None
    for b, names in enumerate(plan["batches"], 1):
        if a.max_batches is not None and b > a.max_batches:
            reason = f"--max-batches {a.max_batches}"
            break
        info = st["batches"].get(str(b), {})
        if info.get("state") == "complete":
            continue
        reason = stop_reason(st)
        if reason:
            break
        try:
            ctl = json.loads(CONTROL.read_text()) if CONTROL.exists() else {}
        except Exception:  # noqa: BLE001
            ctl = {}
        a.par_nasa = int(ctl.get("par_nasa", a.par_nasa))
        a.par_pc = int(ctl.get("par_pc", a.par_pc))
        st["batches"][str(b)] = {"state": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "regions": names, "par_nasa": a.par_nasa, "par_pc": a.par_pc}
        write_status(st, f"pulling batch {b}")
        say(f"batch {b}/{len(plan['batches'])}: {len(names)} regions")
        t0 = time.time()
        res = run.regions(names, OUT_NEW, a.par_nasa, a.par_pc)
        ok = [n for n, v in res.items() if v]
        bad = [n for n, v in res.items() if not v]
        if run.aborted:
            st["batches"][str(b)].update(state="cut short", ok=ok, failed=bad,
                                         finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            reason = stop_reason(st, include_time=False) or "stopped"
            write_status(st, f"batch {b} cut short: {reason}")
            break
        st["batches"][str(b)].update(state="complete", ok=ok, failed=bad, hours=round((time.time() - t0) / 3600, 2),
                                     finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_status(st, f"batch {b} complete")
        say(f"batch {b} complete in {(time.time() - t0) / 3600:.2f} h: {len(ok)} ok, {len(bad)} failed {bad}")
    else:
        reason = "no batch left"
    # last retry for regions of whole batches that failed
    if not stop_reason(st, include_time=False) and not run.aborted:
        for b, info in st["batches"].items():
            if info.get("state") == "complete" and info.get("failed"):
                say(f"final retry of batch {b}'s failed regions: {info['failed']}")
                res = run.regions(info["failed"], OUT_NEW, a.par_nasa, a.par_pc)
                info["ok"] = sorted(info["ok"] + [n for n, v in res.items() if v])
                info["failed"] = [n for n, v in res.items() if not v]
                write_status(st, f"final retry of batch {b}")
    # original 102, re-extracted per scene, if the budget allows
    if not a.no_originals and not run.aborted and reason in ("no batch left",):
        orig = sorted(r.name for r in load_regions() if r.origin == "existing")
        halves = [orig[: len(orig) // 2], orig[len(orig) // 2:]]
        for i, names in enumerate(halves, 1):
            if st["originals"].get(str(i), {}).get("state") == "complete":
                continue
            r = stop_reason(st)
            if r:
                reason = f"{reason}; originals stopped: {r}"
                break
            st["originals"][str(i)] = {"state": "running", "regions": names}
            write_status(st, f"re-extracting originals part {i}")
            res = run.regions(names, OUT_ORIG, a.par_nasa, a.par_pc)
            st["originals"][str(i)].update(state="complete" if not run.aborted else "cut short",
                                           ok=[n for n, v in res.items() if v],
                                           failed=[n for n, v in res.items() if not v])
            write_status(st, f"originals part {i} done")
            if run.aborted:
                break
    freeze(st, reason or "unknown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
