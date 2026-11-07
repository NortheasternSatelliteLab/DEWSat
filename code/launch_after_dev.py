"""Chain launcher (DECISIONS.md #45): after the development stages finish cleanly, score the fresh test once (Phase 5),
run Phase 6, then regenerate the reports. It never starts the pull or the development stages: those belong to
`code/pull/pull_driver.py` and `code/launch_when_frozen.sh`, so no stage can be started twice.

    cd ~/Desktop/DEWSat_BiLSTM_v2 && source env.sh && mkdir -p logs/chain
    caffeinate -dimsu nohup .venv/bin/python -u code/launch_after_dev.py >> logs/chain/launcher.log 2>&1 &
    .venv/bin/python code/launch_after_dev.py --check     # print what it would do now; changes nothing

Every 2 minutes it re-renders HANDOFF.md, then:
 1. waits for data/FROZEN_REGIONS.json (the freeze);
 2. waits while the pull driver, the training launcher or any `code/dewsat2/run.py --stages` process runs;
 3. checks the finished development run: all eight stages finished, no failed unit without a later success, gate
    scores for all four jobs, the pull finished (data/FROZEN.json), and three sanity checks on development numbers
    only: job-1 LR-on-VCI(T) within-fold AUC within 0.55-0.95, a job-1 AUC for every network family, and the positive
    control (best network at least 0.05 above LR). If a check fails it waits; after reviewing, a person can approve
    the sanity part with `touch logs/chain/GO_PHASE5`;
 4. starts `run.py --stages phase5,phase6` once (marker logs/chain/phase56_launched.json; Phase 5 itself refuses a
    second scoring through data/locked/FRESH_TEST_SCORED.lock). Phase 5 is never restarted. If only Phase 6 failed
    (the lock exists), Phase 6 alone is retried at most twice, 30 minutes apart;
 5. writes an automatic closing paragraph (scores/conclusion_paragraph.json, marked automatic) only if none exists,
    regenerates the reports, renders HANDOFF.md and exits.
A file STOP_CHAIN in the folder root makes it exit without starting anything further. It never kills a process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

V2 = Path.home() / "Desktop" / "DEWSat_BiLSTM_v2"
sys.path.insert(0, str(V2 / "code"))
import handoff  # noqa: E402

CHAIN = V2 / "logs" / "chain"
STATE = CHAIN / "state.json"
LOCKDIR = CHAIN / "LOCK"
MARK = CHAIN / "phase56_launched.json"
GO = CHAIN / "GO_PHASE5"
STOP = V2 / "STOP_CHAIN"
PY = str(V2 / ".venv" / "bin" / "python")
POLL = 120
PHASE6_RETRIES = 2
PHASE6_WAIT = 1800
FAMILIES = ("bilstm", "resbilstm", "gru", "lstm", "tcn")
BUSY = ("pull driver", "training launcher", "pipeline run.py")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, p)


def set_state(phase: str, note: str = "", **kw) -> None:
    s = handoff.read_json(STATE, {}) or {}
    s["problems"] = kw.pop("problems", [])
    s.update(pid=os.getpid(), phase=phase, note=note, updated=now(), **kw)
    write_json(STATE, s)


def render() -> None:
    try:
        handoff.render()
    except Exception as exc:  # noqa: BLE001
        log(f"HANDOFF.md not rendered: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- checks (development outputs only)

def _point(v):
    try:
        x = float(v[0])
        return x if math.isfinite(x) else None
    except Exception:  # noqa: BLE001
        return None


def dev_problems(root: Path) -> list[str]:
    st = handoff.read_json(root / "logs" / "status_pipeline.json")
    if not st:
        return ["the development stages have not started (no logs/status_pipeline.json)"]
    stages = st.get("stages", {})
    out = [f"stage {s} not finished (state: {stages.get(s, {}).get('state', 'not started')})"
           for s in handoff.DEV_STAGES if not stages.get(s, {}).get("finished")]
    left = handoff.outstanding_failures(root / "runs")
    if left:
        out.append(f"{len(left)} failed unit(s) without a later success, e.g. {', '.join(left[:3])}")
    sc = handoff.read_json(root / "scores" / "scores.json") or {}
    if sorted(g.get("job") for g in sc.get("gate", [])) != ["job1", "job2", "job3", "job4"]:
        out.append("scores/scores.json lacks gate rows for all four jobs")
    return out


def sanity_problems(root: Path) -> list[str]:
    out = []
    sc = handoff.read_json(root / "scores" / "scores.json") or {}
    models = (sc.get("subsets", {}).get("job1") or {}).get("models", {})

    def auc(m):
        return _point(((models.get(m) or {}).get("all") or {}).get("auc_within"))
    lr = auc("lr_vci_t")
    if lr is None or not 0.55 <= lr <= 0.95:
        out.append(f"job-1 LR on VCI(T) within-fold AUC is {lr}, outside 0.55-0.95")
    miss = [f for f in FAMILIES if auc(f) is None]
    if miss:
        out.append(f"no job-1 within-fold AUC for {', '.join(miss)}")
    pc = ((handoff.read_json(root / "scores" / "checks.json") or {}).get("positive_control") or {}).get("scores", {})
    lrp = _point(pc.get("lr_vci_t"))
    nets = [x for x in (_point(v) for k, v in pc.items() if k in FAMILIES) if x is not None]
    if lrp is None or not nets or max(nets) < lrp + 0.05:
        out.append(f"positive control not passed (LR {lrp}, best network {max(nets) if nets else None})")
    return out


def assess(root: Path = V2, sandbox: bool = False) -> dict:
    """What to do now. With sandbox=True (testing on a sandbox output folder) the freeze, process and pull checks
    are skipped and nothing is read from the real chain bookkeeping."""
    lock = (root / "locked" if sandbox else V2 / "data" / "locked") / "FRESH_TEST_SCORED.lock"
    if not sandbox:
        if not (V2 / "data" / "FROZEN_REGIONS.json").exists():
            return {"phase": "waiting for the freeze", "action": "wait",
                    "note": "data/FROZEN_REGIONS.json not written yet; the training launcher starts the development "
                            "stages at the freeze"}
        pr = handoff.processes()
        busy = {k: v for k in BUSY if (v := handoff.work_pids(pr[k]))}
        if busy:
            return {"phase": "waiting while pipeline processes run", "action": "wait",
                    "note": "; ".join(f"{k} pid {', '.join(map(str, v))}" for k, v in busy.items())}
    mark = None if sandbox else handoff.read_json(MARK)
    p6 = handoff.read_json(root / "scores" / "phase6.json", {}) or {}
    if lock.exists():
        if "nldas" in p6 and "spei" in p6:
            return {"phase": "finalizing", "action": "finalize", "note": "fresh test scored and Phase 6 written"}
        tries = [x for x in (mark or {}).get("attempts", []) if x.get("stages") == "phase6"]
        if len(tries) >= PHASE6_RETRIES:
            return {"phase": "blocked: Phase 6 failed", "action": "blocked",
                    "note": f"Phase 6 failed after {len(tries)} retries; a person must look at logs/run_phase56.log"}
        last = ((mark or {}).get("attempts") or [{}])[-1].get("ended_epoch", 0)
        if time.time() - last < PHASE6_WAIT:
            return {"phase": "waiting to retry Phase 6", "action": "wait",
                    "note": f"Phase 6 did not finish; retry {len(tries) + 1} of {PHASE6_RETRIES} after a 30-minute wait"}
        return {"phase": "retrying Phase 6", "action": "retry_phase6", "note": f"retry {len(tries) + 1} of {PHASE6_RETRIES}"}
    if mark:
        return {"phase": "blocked: Phase 5 did not finish", "action": "blocked",
                "note": "Phase 5 was started once and wrote no lock; it is never restarted automatically. A person must "
                        "read logs/run_phase56.log and logs/failures/ before resuming it"}
    problems = dev_problems(root)
    if not sandbox and not (V2 / "data" / "FROZEN.json").exists():
        problems.append("the pull has not finished (no data/FROZEN.json) and its driver is not running")
    if problems:
        return {"phase": "blocked: development run incomplete", "action": "blocked", "note": problems[0],
                "problems": problems}
    sanity = sanity_problems(root)
    if sanity and not (GO.exists() and not sandbox):
        return {"phase": "blocked: sanity check", "action": "blocked",
                "note": sanity[0] + " (review; `touch logs/chain/GO_PHASE5` approves)", "problems": sanity}
    return {"phase": "ready", "action": "launch",
            "note": "development run complete and checked" + ("; sanity stop approved by GO_PHASE5" if sanity else "")}


# ---------------------------------------------------------------- closing paragraph (fallback only)

def automatic_paragraph(scores: Path):
    sys.path.insert(0, str(V2 / "code" / "dewsat2"))
    from report import ci, lab
    sc = handoff.read_json(scores / "scores.json")
    p5 = handoff.read_json(scores / "phase5.json")
    if not sc:
        return None
    sel = sc.get("selection", {}).get("job1", {})
    b1 = sel.get("best_bilstm")

    def pair(sub, a, b):
        p = (sub or {}).get("paired", {}).get(f"{a} - {b}")
        return None if not p else p["all"]["dauc_within"]

    def word(d):
        if not d or d[1] is None:
            return "not scored"
        return "better" if d[1] > 0 else "worse" if d[2] < 0 else "no detectable difference"
    dev = pair(sc.get("subsets", {}).get("job1"), b1, "lr_vci_t")
    fresh = pair(((p5 or {}).get("fresh") or {}).get("job1"), b1, "lr_vci_t")
    if word(dev) == "not scored" or word(fresh) == "not scored":
        head, verb = "Not decided automatically", "could not be compared on both sets with"
    else:
        yes = word(dev) == "better" and word(fresh) == "better"
        head, verb = ("Yes", "beat") if yes else ("No", "did not beat")
    t = (f"{head}. For estimating the current month's drought (job 1), the best Bi-LSTM variant ({lab(b1)}) {verb} "
         f"logistic regression on VCI(T). Its within-fold AUC minus LR's was {ci(dev, signed=True)} in development "
         f"({word(dev)}) and {ci(fresh, signed=True)} on the fresh test ({word(fresh)}).")
    for g in sc.get("gate", []):
        for r in g.get("rows", []):
            if r.get("kind") != "gate":
                continue
            d = r["dauc_within"]
            t += (f" Gate, {g.get('label', g.get('job'))}: {lab(r['model'])} minus {lab(r['comparator'])} "
                  f"{ci(d['all'], signed=True)} (West {ci(d['west'], signed=True)}, East {ci(d['east'], signed=True)}): "
                  f"{r['verdict']}.")
    t += (" Why, in the terms fixed before training: a model counts as better only when the 95% interval of its paired "
          "within-fold AUC difference (2,000 whole-region resamples) lies above zero and it is not significantly worse "
          "on either side of 98 W. This paragraph was written automatically by code/launch_after_dev.py from "
          "scores/scores.json and scores/phase5.json because the run finished without a hand-written one; it does "
          "not analyse the reasons behind the result (report.md has every table).")
    return t


# ---------------------------------------------------------------- actions

def run_stages(stages: str) -> int:
    mark = handoff.read_json(MARK, {}) or {"launched": now(), "attempts": []}
    att = {"stages": stages, "started": now()}
    mark.setdefault("attempts", []).append(att)
    write_json(MARK, mark)
    set_state(f"running {stages}", f"run.py --stages {stages}; log logs/run_phase56.log")
    log(f"starting run.py --stages {stages}")
    with open(V2 / "logs" / "run_phase56.log", "a") as f:
        p = subprocess.Popen([PY, "-u", "code/dewsat2/run.py", "--stages", stages], cwd=V2, stdout=f,
                             stderr=subprocess.STDOUT, env=os.environ.copy())
        att["pid"] = p.pid
        write_json(MARK, mark)
        while p.poll() is None:
            render()
            set_state(f"running {stages}", f"run.py --stages {stages} (pid {p.pid}); log logs/run_phase56.log")
            time.sleep(POLL)
    att.update(exit=p.returncode, ended=now(), ended_epoch=time.time())
    write_json(MARK, mark)
    log(f"run.py --stages {stages} exited with {p.returncode}")
    return p.returncode


def finalize() -> None:
    para = V2 / "scores" / "conclusion_paragraph.json"
    if not para.exists():
        try:
            t = automatic_paragraph(V2 / "scores")
            if t:
                write_json(para, {"paragraph": t, "automatic": True, "written": now(), "by": "code/launch_after_dev.py"})
                log("automatic closing paragraph written")
        except Exception as exc:  # noqa: BLE001
            log(f"automatic closing paragraph not written: {type(exc).__name__}: {exc}")
    r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0, 'code/dewsat2'); import report; report.write_all()"],
                       cwd=V2, capture_output=True, text=True, env=os.environ.copy())
    log(f"reports regenerated (exit {r.returncode}) {r.stderr[-800:]}")
    set_state("done", "Phase 5 and Phase 6 finished and the reports were regenerated; the manual wrap-up "
                      "(HANDOFF.md step 10) is left")
    render()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _cmd(pid: int) -> str:
    return subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout


def acquire() -> bool:
    CHAIN.mkdir(parents=True, exist_ok=True)
    try:
        LOCKDIR.mkdir()
    except FileExistsError:
        old = handoff.read_json(LOCKDIR / "owner.json")
        if old is None:
            time.sleep(5)
            old = handoff.read_json(LOCKDIR / "owner.json") or {}
        pid = old.get("pid")
        if pid and _alive(pid) and "launch_after_dev.py" in _cmd(pid):
            log(f"another chain launcher is running (pid {pid}); exiting")
            return False
        log(f"stale lock (pid {pid}); taking it over")
    write_json(LOCKDIR / "owner.json", {"pid": os.getpid(), "started": now()})
    return True


def release() -> None:
    own = handoff.read_json(LOCKDIR / "owner.json") or {}
    if own.get("pid") == os.getpid():
        (LOCKDIR / "owner.json").unlink(missing_ok=True)
        try:
            LOCKDIR.rmdir()
        except OSError:
            pass


def loop() -> int:
    last = None
    while True:
        render()
        if STOP.exists():
            set_state("stopped", "STOP_CHAIN found; exited without starting anything further")
            log("STOP_CHAIN found; exiting")
            return 0
        rep = assess()
        if rep["note"] != last:
            log(f"{rep['phase']}: {rep['note']}")
            last = rep["note"]
        act = rep["action"]
        if act in ("wait", "blocked"):
            set_state(rep["phase"], rep["note"], problems=rep.get("problems", []))
            time.sleep(POLL)
        elif act == "launch":
            run_stages("phase5,phase6")
        elif act == "retry_phase6":
            run_stages("phase6")
        elif act == "finalize":
            finalize()
            return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="print the current assessment and exit; changes nothing")
    ap.add_argument("--sandbox", default=None, help="with --check: assess a sandbox output folder (e.g. smoke5)")
    a = ap.parse_args()
    if a.check:
        root = Path(a.sandbox).resolve() if a.sandbox else V2
        print(json.dumps(assess(root, sandbox=bool(a.sandbox)), indent=1))
        if a.sandbox:
            print(automatic_paragraph(root / "scores"))
        return 0
    if not os.environ.get("TMPDIR", "").startswith(str(V2)):
        log("TMPDIR is not inside ~/Desktop/DEWSat_BiLSTM_v2: source env.sh first")
        return 2
    if not acquire():
        return 1
    try:
        os.setsid()                     # own session: unaffected by whatever started it
    except OSError:
        pass
    set_state("started", "chain launcher started", started=now())
    log(f"chain launcher started, pid {os.getpid()}")
    try:
        return loop()
    except Exception as exc:  # noqa: BLE001
        import traceback
        log("chain launcher crashed:\n" + traceback.format_exc())
        set_state("crashed", f"{type(exc).__name__}: {exc}; restart it with the command in HANDOFF.md section 3")
        render()
        return 3
    finally:
        release()


if __name__ == "__main__":
    sys.exit(main())
