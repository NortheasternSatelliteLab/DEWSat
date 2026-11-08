"""DEWSat_BiLSTM_v2 pipeline: one resumable job (DESIGN.md 12).

    caffeinate -dimsu nohup .venv/bin/python -u code/dewsat2/run.py --stages all > logs/run.log 2>&1 &

Stages: features, simple, search, final, score, checks, whatif, weekly, phase5, phase6. Every unit saves its
predictions, metadata and a DONE marker; a restart skips finished units; a failed unit is logged and skipped (and
retried once at the end of its stage). Sandboxes set DEWSAT_V2_OUT (and DEWSAT_SMOKE=1 for tiny settings).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# PyTorch's Apple-GPU (MPS) allocator keeps freed memory cached until a process passes 1.4x the GPU's recommended
# working set (its default low watermark). On 2026-11-08 the two MPS workers grew to ~10 GB each (8.9 GB of it cached
# GPU buffers) and pushed ~20 GB into swap, slowing every worker (DECISIONS.md #48). Recycle cached memory above 25% of
# the working set instead. Set here, before torch loads, so the spawned pool workers inherit it. Results are unchanged:
# only when freed memory is reused differs.
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")

STAGES = ("features", "simple", "search", "final", "score", "checks", "whatif", "weekly", "phase5", "phase6")
_W: dict = {}


def worker_init(device: str, threads: int):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(threads)
    _W["device"], _W["threads"] = device, threads
    import units
    units.ctx()


def worker_task(spec: dict) -> dict:
    import units
    from common import record_failure
    name = spec_name(spec)
    t0 = time.time()
    try:
        meta = units.unit(spec, _W["device"], _W["threads"])
        return {"name": name, "ok": True, "seconds": time.time() - t0, "device": _W["device"],
                "metric": (meta or {}).get("metric", {}).get("metric") if isinstance(meta, dict) else None}
    except Exception as exc:  # noqa: BLE001
        record_failure(name, exc)
        return {"name": name, "ok": False, "seconds": time.time() - t0, "device": _W["device"], "error": str(exc)[:300]}
    finally:
        free_gpu_cache()


def free_gpu_cache() -> None:
    """Hand the MPS allocator's cached blocks back after every unit (DECISIONS.md #48); a no-op on CPU workers."""
    if _W.get("device") != "mps":
        return
    try:
        import torch
        torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def spec_name(spec: dict) -> str:
    if spec["type"] == "simple":
        return f"simple/{spec['job']}/fold{spec['k'] + 1}"
    c = spec["cfg"]
    if spec["type"] in ("whatif", "weekly", "p5"):
        return f"{spec['type']}/{spec.get('variant', spec.get('job', ''))}/{c['family']}/fold{spec.get('k', 0) + 1}/seed{spec['seed']}"
    tail = f"cfg{c['id']:02d}" if spec["type"] == "search" else f"seed{spec['seed']}"
    return f"{spec['type']}/{spec['job']}/{c['family']}/fold{spec['k'] + 1}/{tail}"


def spec_done(spec: dict) -> bool:
    import units
    if spec["type"] == "simple":
        return (units.simple_dir(spec["job"], spec["k"]) / "DONE").exists()
    if spec["type"] == "whatif":
        return (units.whatif_dir(spec["variant"], spec["cfg"]["family"], spec["k"], spec["seed"]) / "DONE").exists()
    if spec["type"] == "weekly":
        import weekly_stage
        return (weekly_stage.unit_dir(spec) / "DONE").exists()
    if spec["type"] == "p5":
        import phase5_stage
        return (phase5_stage.unit_dir(spec) / "DONE").exists()
    c = spec["cfg"]
    if spec["type"] == "search":
        return (units.search_dir(spec["job"], c["family"], spec["k"], c["id"]) / "DONE").exists()
    return (units.final_dir(spec["job"], c["family"], spec["k"], spec["seed"]) / "DONE").exists()


def run_specs(stage: str, specs: list[dict], n_cpu: int, n_mps: int, threads: int, status) -> dict:
    """Run units on a CPU pool and an MPS pool; recurrent units go to whichever pool frees up first by estimate."""
    import search
    from common import log
    todo = [s for s in specs if not spec_done(s)]
    log(f"{stage}: {len(specs)} units, {len(todo)} to run ({n_cpu} CPU workers x {threads} threads, {n_mps} MPS)")
    status.stage(stage, total=len(specs), done=len(specs) - len(todo), state="running")
    if not todo:
        status.stage(stage, state="done")
        return {"failed": []}
    # static greedy assignment by estimated finish time (temporal CNNs only on MPS)
    n_win = status.n_windows_estimate
    loads = {"cpu": [0.0] * max(n_cpu, 1), "mps": [0.0] * max(n_mps, 1)}
    plan = []
    for s in sorted(todo, key=lambda s: -(search.cost(s["cfg"], n_win) if "cfg" in s else 600)):
        if s["type"] == "simple":
            dev = "cpu"
        else:
            pref = search.device_for(s["cfg"])
            c_pref = search.cost(s["cfg"], n_win)
            if n_mps == 0:
                dev = "cpu"
            elif s["cfg"]["family"] == "tcn" or n_cpu == 0:
                dev = "mps"
            else:
                alt = "cpu" if pref == "mps" else "mps"
                c_alt = c_pref * (3.0 if pref == "mps" else 1.6)
                f_pref = min(loads[pref]) + c_pref
                f_alt = min(loads[alt]) + c_alt
                dev = pref if f_pref <= f_alt else alt
        cst = search.cost(s["cfg"], n_win) if "cfg" in s else 600
        i = loads[dev].index(min(loads[dev]))
        loads[dev][i] += cst
        plan.append((dev, s))
    order = lambda s: (s.get("job", ""), s.get("variant", ""), s.get("k", 0), s.get("cfg", {}).get("family", ""),  # noqa: E731
                       s.get("cfg", {}).get("id", 0), s.get("seed", 0))
    by_dev = {d: sorted([s for dd, s in plan if dd == d], key=order) for d in ("cpu", "mps")}
    pools = {}
    if by_dev["cpu"]:
        pools["cpu"] = ProcessPoolExecutor(max_workers=max(n_cpu, 1), initializer=worker_init, initargs=("cpu", threads))
    if by_dev["mps"]:
        pools["mps"] = ProcessPoolExecutor(max_workers=max(n_mps, 1), initializer=worker_init, initargs=("mps", 2))
    futs = {}
    for d, items in by_dev.items():
        for s in items:
            futs[pools[d].submit(worker_task, s)] = s
    failed, n_done, t0 = [], len(specs) - len(todo), time.time()
    for f in as_completed(futs):
        r = f.result()
        n_done += 1
        if not r["ok"]:
            failed.append(futs[f])
        status.unit_done(stage, r, n_done, len(specs), t0)
    for p in pools.values():
        p.shutdown()
    status.stage(stage, done=n_done, failed=len(failed), state="done" if not failed else "done with failures")
    return {"failed": failed}


def stage_specs(stage: str) -> list[dict]:
    import search
    import units
    from common import FAMILIES, FINAL_SEEDS, N_FOLDS
    specs = []
    if stage == "simple":
        specs = [{"type": "simple", "job": j, "k": k} for j in units.JOBS for k in range(N_FOLDS)]
    elif stage == "search":
        for j in units.JOBS:
            for k in range(N_FOLDS):
                for fam in FAMILIES:
                    for cfg in search.net_configs(j, fam):
                        specs.append({"type": "search", "job": j, "k": k, "cfg": cfg})
    elif stage == "final":
        for j in units.JOBS:
            for k in range(N_FOLDS):
                for fam in FAMILIES:
                    cfg = units.chosen_config(j, fam, k)
                    if cfg is None:
                        continue
                    for s in FINAL_SEEDS:
                        specs.append({"type": "final", "job": j, "k": k, "cfg": cfg, "seed": s})
    return specs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="all")
    ap.add_argument("--cpu-workers", type=int, default=3)
    ap.add_argument("--mps-workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--sandbox-regions", default=None, help="comma-separated names (sandbox only)")
    a = ap.parse_args()
    import pstatus as st_mod
    from common import log
    stages = STAGES if a.stages == "all" else tuple(a.stages.split(","))
    status = st_mod.PipelineStatus()
    for stage in stages:
        t0 = time.time()
        status.stage(stage, state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
        if stage == "features":
            import dataset
            if a.sandbox_regions:
                dataset.build(frozen=False, names=a.sandbox_regions.split(","), workers=a.feature_workers)
            else:
                dataset.build(frozen=True, workers=a.feature_workers)
            status.refresh_counts()
            try:
                import datacount
                datacount.run(frozen=not a.sandbox_regions)
            except Exception as exc:  # noqa: BLE001
                log(f"data counts not written: {type(exc).__name__}: {exc}")
            status.stage(stage, state="done")
        elif stage in ("simple", "search", "final"):
            specs = stage_specs(stage)
            n_cpu = a.cpu_workers + (a.mps_workers if stage == "simple" else 0)
            res = run_specs(stage, specs, n_cpu, 0 if stage == "simple" else a.mps_workers, a.threads, status)
            if res["failed"]:
                log(f"{stage}: retrying {len(res['failed'])} failed units once")
                run_specs(stage + "_retry", res["failed"], n_cpu, 0 if stage == "simple" else a.mps_workers,
                          a.threads, status)
        else:
            import stages_late
            getattr(stages_late, stage)(status)
        status.stage(stage, finished=time.strftime("%Y-%m-%d %H:%M:%S"), minutes=round((time.time() - t0) / 60, 1))
        try:
            import report
            report.write_all()
        except Exception as exc:  # noqa: BLE001
            log(f"report not written after {stage}: {type(exc).__name__}: {exc}")
    log("pipeline finished")
    status.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
