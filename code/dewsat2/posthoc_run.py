"""Runner for the post-hoc job-1 models (DESIGN.md amendment 4): search, final, score.

    cd ~/Desktop/DEWSat_BiLSTM_v2 && source env.sh && mkdir -p logs/posthoc
    caffeinate -dimsu nohup .venv/bin/python -u code/dewsat2/posthoc_run.py --stages search,final,score \
        --cpu-workers 4 --mps-workers 2 --threads 2 >> logs/posthoc/run.log 2>&1 &

bilstm_vci runs where the pipeline runs a bilstm (GPU from 128 hidden units in total, else CPU), the CNN models on the GPU (MPS). Every run saves predictions, metadata and a DONE
marker under runs/posthoc/; a restart skips finished runs; a failed run is written to runs/posthoc/failures/ and
retried once at the end of its stage. Progress: logs/posthoc/status.json. Sandboxes set DEWSAT_V2_OUT.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")     # as run.py (DECISIONS.md #48)

_W: dict = {}


def say(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] posthoc: {msg}", flush=True)


def worker_init(device: str, threads: int):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(threads)
    _W["device"], _W["threads"] = device, threads
    import units
    units.ctx()


def record_failure(name: str, exc: Exception) -> None:
    from common import RUNS, write_json
    write_json(RUNS / "posthoc" / "failures" / (name.replace("/", "__") + ".json"),
               {"unit": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-3000:]})


def worker_task(spec: dict) -> dict:
    import posthoc
    name = posthoc.spec_name(spec)
    t0 = time.time()
    try:
        meta = posthoc.run_unit(spec, _W["device"], _W["threads"])
        return {"name": name, "ok": True, "seconds": time.time() - t0,
                "metric": ((meta or {}).get("metric") or {}).get("metric")}
    except Exception as exc:  # noqa: BLE001
        record_failure(name, exc)
        return {"name": name, "ok": False, "seconds": time.time() - t0, "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        if _W.get("device") == "mps":
            try:
                import torch
                torch.mps.empty_cache()
            except Exception:  # noqa: BLE001
                pass


def _cost(s: dict) -> float:
    c = s["cfg"]
    if c["family"] == "bilstm_vci":
        return c["epochs"] * c["hidden"] * c.get("layers", 1)
    return c["epochs"] * c["width"] * (12 if c["family"] == "cnn_bilstm" else 1)


def run_stage(stage: str, specs: list[dict], a, status: dict) -> list[dict]:
    import posthoc
    from common import LOGS, write_json
    todo = [s for s in specs if not (posthoc.spec_dir(s) / "DONE").exists()]
    say(f"{stage}: {len(specs)} runs, {len(todo)} to do ({a.cpu_workers} CPU workers x {a.threads} threads, "
        f"{a.mps_workers} MPS)")
    st = status["stages"].setdefault(stage, {})
    st.update({"total": len(specs), "done": len(specs) - len(todo), "failed": 0, "state": "running",
               "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    write_json(LOGS / "posthoc" / "status.json", status)
    if not todo:
        st["state"] = "done"
        write_json(LOGS / "posthoc" / "status.json", status)
        return []
    by_dev = {"cpu": [], "mps": []}
    for s in todo:
        by_dev[posthoc.device_for(s["cfg"])].append(s)
    defer = set(filter(None, a.defer.split(",")))
    for d in by_dev:              # deferred runs last; then fold by fold (prep and tiles cached per worker), longest first
        by_dev[d].sort(key=lambda s: (posthoc.spec_name(s) in defer, s["k"], -_cost(s)))
    pools = {}
    if by_dev["cpu"]:
        pools["cpu"] = ProcessPoolExecutor(max_workers=a.cpu_workers, initializer=worker_init,
                                           initargs=("cpu", a.threads))
    if by_dev["mps"]:
        pools["mps"] = ProcessPoolExecutor(max_workers=a.mps_workers, initializer=worker_init, initargs=("mps", 2))
    futs = {}
    for d, items in by_dev.items():
        for s in items:
            futs[pools[d].submit(worker_task, s)] = s
    failed, n_done, t0, n_new = [], len(specs) - len(todo), time.time(), 0
    for f in as_completed(futs):
        r = f.result()
        n_done += 1
        n_new += 1
        if not r["ok"]:
            failed.append(futs[f])
            say(f"FAILED {r['name']}: {r['error']}")
        rate = n_new / max(time.time() - t0, 1.0)
        left = len(specs) - n_done
        st.update({"done": n_done, "failed": len(failed), "elapsed_h": round((time.time() - t0) / 3600, 2),
                   "eta": time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + left / rate)) if left else None,
                   "last": r["name"]})
        write_json(LOGS / "posthoc" / "status.json", status)
        if n_new % 10 == 0 or left == 0 or not r["ok"]:
            say(f"{stage}: {n_done}/{len(specs)} ({r['name']} {r['seconds']:.0f}s, metric {r.get('metric')})")
    for p in pools.values():
        p.shutdown()
    st["state"] = "done" if not failed else "done with failures"
    st["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(LOGS / "posthoc" / "status.json", status)
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="search,final,score")
    ap.add_argument("--cpu-workers", type=int, default=4)
    ap.add_argument("--mps-workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--models", default=",".join(("bilstm_vci", "cnn_lr", "cnn_bilstm")),
                    help="comma-separated subset of the post-hoc models (search and final only)")
    ap.add_argument("--defer", default="", help="comma-separated run names to queue last (e.g. a run another "
                    "process is still finishing)")
    a = ap.parse_args()
    import posthoc
    from common import LOGS, read_json, write_json
    status = read_json(LOGS / "posthoc" / "status.json") or {}
    status.update({"pid": os.getpid(), "process_started": time.strftime("%Y-%m-%d %H:%M:%S"), "state": "running"})
    status.setdefault("stages", {})
    for stage in a.stages.split(","):
        if stage in ("search", "final"):
            want = set(a.models.split(","))
            failed = run_stage(stage, [s for s in posthoc.stage_specs(stage) if s["cfg"]["family"] in want], a,
                               status)
            if failed:
                say(f"{stage}: retrying {len(failed)} failed runs once")
                run_stage(stage + "_retry", failed, a, status)
        elif stage == "score":
            t0 = time.time()
            res = posthoc.score_dev()
            for job, e in res["jobs"].items():
                lr = "lr_latest_age" if job == "job2" else "lr_vci_t"
                for m in dict.fromkeys([lr, e["best_simple"], "bilstm", *e["new_models"]]):
                    if m in e["models"]:
                        say(f"  {job} {m}: {e['metric']} {e['models'][m]['all']}")
                for k, v in e["verdicts"].items():
                    say(f"  {job} {k}: {e['paired'][k]['all']} -> {v}")
            status["stages"]["score"] = {"state": "done", "minutes": round((time.time() - t0) / 60, 1),
                                         "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        else:
            raise ValueError(f"unknown stage {stage!r}")
        write_json(LOGS / "posthoc" / "status.json", status)
    status["state"] = "finished"
    write_json(LOGS / "posthoc" / "status.json", status)
    say("finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
