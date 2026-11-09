"""Runner for the post-hoc models (DESIGN.md amendments 4 and 4b): search, final, score.

    cd ~/Desktop/DEWSat_BiLSTM_v2 && source env.sh && mkdir -p logs/posthoc && rm -f logs/posthoc/STOP
    caffeinate -dimsu nohup .venv/bin/python -u code/dewsat2/posthoc_run.py --stages search,final,score \
        --cpu-workers 4 --mps-workers 2 --threads 2 >> logs/posthoc/run.log 2>&1 &

Runs are handed out one at a time as workers free up (DECISIONS.md #59). The CNN models go to GPU (MPS) workers only.
bilstm_vci goes where the pipeline places a bilstm (GPU from 128 recurrent units in total, else CPU); an idle CPU
worker also takes a GPU-placed bilstm_vci run (the cheapest on the CPU) while the GPU backlog would outlast it there,
as run.py's scheduler moves recurrent units between pools. Every run saves predictions, metadata and a DONE marker
under runs/posthoc/; a restart skips finished runs; a failed run is written to runs/posthoc/failures/ and retried once
at the end of its stage. Graceful stop: create logs/posthoc/STOP; running runs finish and are saved, nothing new
starts, and the process exits (delete the file before restarting). Progress: logs/posthoc/status.json. Sandboxes set
DEWSAT_V2_OUT.

The learning curve (DESIGN.md amendment 5, lcurve.py) runs through the same dispatcher with its own status, STOP file
and failures (logs/lcurve/, runs/lcurve/failures/):

    cd ~/Desktop/DEWSat_BiLSTM_v2 && source env.sh && mkdir -p logs/lcurve && rm -f logs/lcurve/STOP
    caffeinate -dimsu nohup nice -n 10 .venv/bin/python -u code/dewsat2/posthoc_run.py --stages lcurve,lcurve_score \
        --cpu-workers 2 --mps-workers 1 --threads 2 >> logs/lcurve/run.log 2>&1 &

The ReLU Bi-LSTM (DESIGN.md amendment 7, relu.py) does too (logs/relu/, runs/relu/failures/): stages relu_pilot and
relu_pilot_score first, then relu and relu_score (commands in relu.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")     # as run.py (DECISIONS.md #48)

_W: dict = {}


def say(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {_ns()}: {msg}", flush=True)


def _ns() -> str:
    """'posthoc', 'lcurve' or 'relu': the folder of this runner's status, STOP file and failures (inherited by the
    workers)."""
    return os.environ.get("DEWSAT_RUNNER_NS", "posthoc")


GROUPS = {"lcurve": ("lcurve", "lcurve_score"), "relu": ("relu_pilot", "relu_pilot_score", "relu", "relu_score")}
HAND_FAMILIES = ("bilstm_relu", "bilstm_hand")            # amendment 7: the hand-written cell (slower than nn.LSTM)


def _mod(spec: dict):
    """The module that owns a run: lcurve for learning-curve runs, relu for the ReLU Bi-LSTM, posthoc otherwise."""
    if spec["type"] == "lc":
        import lcurve
        return lcurve
    if spec["type"] == "relu":
        import relu
        return relu
    import posthoc
    return posthoc


def status_file() -> Path:
    from common import LOGS
    return LOGS / _ns() / "status.json"


def stop_file() -> Path:
    from common import LOGS
    return LOGS / _ns() / "STOP"


def worker_init(device: str, threads: int):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(threads)
    _W["device"], _W["threads"] = device, threads
    import units
    units.ctx()


def record_failure(name: str, exc: BaseException) -> None:
    from common import RUNS, write_json
    write_json(RUNS / _ns() / "failures" / (name.replace("/", "__") + ".json"),
               {"unit": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-3000:]})


def worker_task(spec: dict) -> dict:
    mod = _mod(spec)
    name = mod.spec_name(spec)
    t0 = time.time()
    try:
        meta = mod.run_unit(spec, _W["device"], _W["threads"])
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


def _est(s: dict, dev: str) -> float:
    """Rough seconds for one run on a device, for scheduling and the ETA only: search.cost's per-epoch rates for a
    bilstm (CPU doubled: the CPU workers share the machine at lower priority), CNN rates from the timing test and the
    finished runs; ~200k fitting windows for a search run, ~500k for a final run (a learning-curve run: its slice's
    share of a final run's)."""
    c = s["cfg"]
    if s["type"] == "lc":
        import lcurve
        n = 5.0 * lcurve.data_fraction(s["slice"])
    else:
        n = 5.0 if s["type"] in ("ph_final", "relu") else 2.0
    if c["family"] in ("bilstm_vci", "bilstm", *HAND_FAMILIES):
        h, layers = c["hidden"], c.get("layers", 1)
        per = 2.0 + 0.012 * h * layers if dev == "mps" else 2 * 1.95 * (h / 32) ** 0.95 * layers
        if c["family"] in HAND_FAMILIES:                    # timing check (DECISIONS.md #83): x1.1-1.3 CPU, x3-4 MPS
            per *= 3.5 if dev == "mps" else 1.25
    elif c["family"] == "cnn_lr":
        per = 1.5 + 0.15 * c["width"]
    else:
        per = 20.0 + 0.6 * c["width"]
    return per * c["epochs"] * n + 15


def _pick(dev: str, pending: list[dict], pools: dict, n_mps: int) -> dict | None:
    """The next run for a free worker on dev, or None (the worker waits)."""
    if dev == "mps":
        for i, s in enumerate(pending):                     # GPU-placed runs first, in queue order
            if _mod(s).device_for(s["cfg"]) == "mps":
                return pending.pop(i)
        return pending.pop(0) if pending else None          # then anything (CPU-placed recurrent runs)
    flex = [i for i, s in enumerate(pending) if s["cfg"]["family"] in ("bilstm_vci", "bilstm", *HAND_FAMILIES)]
    for i in flex:                                          # CPU-placed runs first, in queue order
        if _mod(pending[i]).device_for(pending[i]["cfg"]) == "cpu":
            return pending.pop(i)
    if not flex:
        return None
    i = min(flex, key=lambda j: _est(pending[j], "cpu"))    # the GPU-placed run that is cheapest on the CPU
    if "mps" not in pools:
        return pending.pop(i)
    backlog = sum(_est(s, "mps") for s in pending) / max(n_mps, 1)
    return pending.pop(i) if _est(pending[i], "cpu") < backlog else None


def _eta(pending: list[dict], ratio: float, a) -> str | None:
    if not pending:
        return None
    gpu = sum(_est(s, "mps") for s in pending if _mod(s).device_for(s["cfg"]) == "mps")
    cpu = sum(_est(s, "cpu") for s in pending if _mod(s).device_for(s["cfg"]) == "cpu")
    left = ratio * max(gpu / max(a.mps_workers, 1), cpu / max(a.cpu_workers, 1))
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + left))


def run_stage(stage: str, specs: list[dict], a, status: dict) -> tuple[list[dict], bool]:
    """Run one stage's unfinished runs; returns (failed runs, stopped by the STOP file)."""
    from common import write_json
    todo = [s for s in specs if not (_mod(s).spec_dir(s) / "DONE").exists()]
    say(f"{stage}: {len(specs)} runs, {len(todo)} to do ({a.cpu_workers} CPU workers x {a.threads} threads, "
        f"{a.mps_workers} MPS)")
    st = status["stages"].setdefault(stage, {})
    st.update({"total": len(specs), "done": len(specs) - len(todo), "failed": 0, "state": "running",
               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "workers": {"cpu": a.cpu_workers, "mps": a.mps_workers}})
    write_json(status_file(), status)
    if not todo:
        st["state"] = "done"
        write_json(status_file(), status)
        return [], False
    defer = set(filter(None, a.defer.split(",")))
    # deferred runs last; then fold by fold and slice by slice (prep and tiles cached per worker), longest first
    pending = sorted(todo, key=lambda s: (_mod(s).spec_name(s) in defer, s["k"], s.get("slice", {}).get("name", ""),
                                          -_est(s, "mps")))
    n_workers = {"cpu": a.cpu_workers, "mps": a.mps_workers}
    make = {d: (lambda d=d: ProcessPoolExecutor(max_workers=n_workers[d], initializer=worker_init,
                                                initargs=(d, a.threads if d == "cpu" else 2))) for d in n_workers}
    pools = {d: make[d]() for d in n_workers if n_workers[d] > 0}
    free = {d: n_workers[d] for d in pools}
    inflight: dict = {}
    failed, n_done, t0, n_new, stopping = [], len(specs) - len(todo), time.time(), 0, False
    spent, guessed = 0.0, 0.0
    while True:
        if not stopping and stop_file().exists():
            stopping = True
            say(f"{stage}: STOP file found; letting {len(inflight)} running runs finish, starting no new ones")
        if not stopping:
            for d in pools:
                while free[d] > 0:
                    s = _pick(d, pending, pools, a.mps_workers)
                    if s is None:
                        break
                    inflight[pools[d].submit(worker_task, s)] = (s, d)
                    free[d] -= 1
        if not inflight:
            break
        done_set, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
        for f in done_set:
            s, d = inflight.pop(f)
            free[d] += 1
            try:
                r = f.result()
            except Exception as exc:  # noqa: BLE001  (a worker process died: BrokenProcessPool)
                record_failure(_mod(s).spec_name(s), exc)
                r = {"name": _mod(s).spec_name(s), "ok": False, "seconds": 0.0,
                     "error": f"{type(exc).__name__}: {exc}"[:300]}
            n_done += 1
            n_new += 1
            if r["ok"]:
                spent += r["seconds"]
                guessed += _est(s, d)
            else:
                failed.append(s)
                say(f"FAILED {r['name']} ({d}): {r['error']}")
            st.update({"done": n_done, "failed": len(failed), "elapsed_h": round((time.time() - t0) / 3600, 2),
                       "eta": _eta(pending, spent / guessed if guessed else 1.0, a), "last": r["name"],
                       "running": {dd: n_workers[dd] - free[dd] for dd in pools}, "queued": len(pending)})
            write_json(status_file(), status)
            if n_new % 10 == 0 or not r["ok"] or (not pending and not inflight):
                say(f"{stage}: {n_done}/{len(specs)} ({r['name']} {d} {r['seconds']:.0f}s, metric {r.get('metric')})")
        for d in list(pools):                               # replace a broken pool once its runs have drained
            if getattr(pools[d], "_broken", False) and not any(dd == d for _, dd in inflight.values()):
                pools[d].shutdown(wait=False, cancel_futures=True)
                pools[d], free[d] = make[d](), n_workers[d]
                say(f"{stage}: {d} pool restarted after a worker process died")
    for p in pools.values():
        p.shutdown()
    if stopping:
        st["state"] = "stopped"
    elif pending:
        st["state"] = "incomplete"
        say(f"{stage}: {len(pending)} runs need a GPU worker and were not run")
    else:
        st["state"] = "done" if not failed else "done with failures"
    st["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(status_file(), status)
    return failed, stopping


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
    from common import read_json, write_json
    stages = a.stages.split(",")
    ns = "posthoc"
    for group, names in GROUPS.items():
        hit = [s in names for s in stages]
        if any(hit) and not all(hit):
            raise SystemExit(f"run the {group} stages ({', '.join(names)}) on their own")
        if all(hit):
            ns = group
    os.environ["DEWSAT_RUNNER_NS"] = ns                     # before any worker starts
    if stop_file().exists():
        say(f"{stop_file()} exists; delete it to run. Nothing started.")
        return 0
    status = read_json(status_file()) or {}
    status.update({"pid": os.getpid(), "process_started": time.strftime("%Y-%m-%d %H:%M:%S"), "state": "running"})
    status.setdefault("stages", {})
    for stage in stages:
        if stage in ("search", "final", "lcurve", "relu_pilot", "relu"):
            if stage == "lcurve":
                import lcurve
                specs = lcurve.stage_specs()
                say(f"lcurve: networks {lcurve.models()}, seeds {list(lcurve.SEEDS)}")
            elif stage in ("relu_pilot", "relu"):
                import relu
                specs = relu.stage_specs(stage)
                say(f"{stage}: {len(specs)} runs, families {sorted({s['cfg']['family'] for s in specs})}")
            else:
                want = set(a.models.split(","))
                specs = [s for s in posthoc.stage_specs(stage) if s["cfg"]["family"] in want]
            failed, stopped = run_stage(stage, specs, a, status)
            if failed and not stopped:
                say(f"{stage}: retrying {len(failed)} failed runs once")
                _, stopped = run_stage(stage + "_retry", failed, a, status)
            if stopped:
                status["state"] = "stopped"
                write_json(status_file(), status)
                say("stopped (STOP file); a restart resumes from the finished runs")
                return 0
        elif stage == "relu_pilot_score":
            import relu
            res = relu.pilot_check()
            for r in res["tanh_reproduction"]:
                say(f"  pilot seed {r['seed']}: AUC hand {r['auc_hand']:.5f} original {r['auc_original']:.5f} "
                    f"(difference {r['auc_difference']:+.5f}), max |logit difference| {r['max_logit_difference']}")
            if res["relu_run"]:
                say(f"  pilot ReLU run: loss {['%.4f' % x for x in res['relu_run']['loss']]}")
            status["stages"]["relu_pilot_score"] = {"state": "passed" if res["passed"] else "failed",
                                                    "failures": res["failures"],
                                                    "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
            if not res["passed"]:
                status["state"] = "pilot failed"
                write_json(status_file(), status)
                say(f"pilot FAILED: {res['failures']}; nothing more is trained")
                return 1
            say("pilot passed")
        elif stage == "relu_score":
            import relu
            t0 = time.time()
            res = relu.score()
            for job, e in res["jobs"].items():
                for m, v in e["models"].items():
                    say(f"  {job} {m}: within-fold AUC {v['all']}")
                for k, v in e["verdicts"].items():
                    say(f"  {job} {k}: {e['paired'][k]['all']} -> {v}")
            say(f"  losses fell {res['all_losses_fell']}, finite {res['all_losses_finite']}; reproduce "
                f"{res['reproduce_max_logit_difference']}")
            status["stages"]["relu_score"] = {"state": "done", "minutes": round((time.time() - t0) / 60, 1),
                                              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        elif stage == "lcurve_score":
            import lcurve
            t0 = time.time()
            res = lcurve.score()
            for axis, e in res["axes"].items():
                for row in e["sizes"]:
                    say(f"  {axis} {row['size']}: " + ", ".join(
                        f"{m} {row['auc_within'][m]['all'][0]:.4f}" for m in row["auc_within"]) + " | " + ", ".join(
                        f"{m} - LR {row['network_minus_lr'][m]['all']}" for m in row["network_minus_lr"]))
                for m, ch in e["change_in_difference"].items():
                    say(f"  {axis} {m}: change smallest->full {ch['smallest_to_full']}, "
                        f"second largest->full {ch['second_largest_to_full']}")
            status["stages"]["lcurve_score"] = {"state": "done", "minutes": round((time.time() - t0) / 60, 1),
                                                "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
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
        write_json(status_file(), status)
    status["state"] = "finished"
    write_json(status_file(), status)
    say("finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
