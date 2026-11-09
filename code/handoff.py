"""Render HANDOFF.md: where the run stands, what is left, the exact commands and files for each remaining stage, and
how a fresh session picks the run up. It only reads state files and writes HANDOFF.md (atomically).

    .venv/bin/python code/handoff.py

code/launch_after_dev.py re-renders it every 2 minutes while it runs (DECISIONS.md #45).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

V2 = Path.home() / "Desktop" / "DEWSat_BiLSTM_v2"
DATA, LOGS, RUNS, SCORES = V2 / "data", V2 / "logs", V2 / "runs", V2 / "scores"
CHAIN = LOGS / "chain"
FRESH_LOCK = DATA / "locked" / "FRESH_TEST_SCORED.lock"
N_BATCHES = 12
DEV_STAGES = ("features", "simple", "search", "final", "score", "checks", "whatif", "weekly")
LATE_STAGES = ("phase5", "phase5_composite", "phase5_thinned", "phase5_composite_retry", "phase5_thinned_retry", "phase6")
UNIT_KINDS = ("simple", "search", "final", "whatif", "weekly")
# (role, pgrep -f pattern). Patterns match the real processes' arguments, not shells that merely mention the names.
PROCS = (
    ("pull driver", r"pull_driver\.py --par-nasa"),
    ("training launcher", r"code/launch_when_frozen\.sh"),
    ("pipeline run.py", r"code/dewsat2/run\.py --stages"),
    ("chain launcher", r"code/launch_after_dev\.py$"),
    ("post-hoc runner", r"posthoc_run\.py --stages (search|final|score)"),
    ("learning-curve runner", r"posthoc_run\.py --stages lcurve"),
)
# runners of the models added after the development results (DESIGN.md amendments 4-6): status file, stages
RUNNERS = (("post-hoc runner", LOGS / "posthoc" / "status.json", "amendments 4 and 4b: bilstm_vci, cnn_lr, cnn_bilstm"),
           ("learning-curve runner", LOGS / "lcurve" / "status.json", "amendment 5: job 1, years and regions axes"))


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return default


def pids(pattern: str) -> list[int]:
    try:
        # -a: macOS pgrep otherwise leaves out the caller's ancestors, so the chain launcher could not list itself
        r = subprocess.run(["pgrep", "-a", "-f", pattern], capture_output=True, text=True, timeout=20)
        return [int(x) for x in r.stdout.split()]
    except Exception:  # noqa: BLE001
        return []


def processes() -> dict[str, list[int]]:
    return {name: pids(pat) for name, pat in PROCS}


def work_pids(found: list[int]) -> list[int]:
    """Drop `caffeinate` helpers: they only hold the sleep assertion and do no work of their own."""
    if not found:
        return []
    try:
        r = subprocess.run(["ps", "-o", "pid=,comm=", "-p", ",".join(map(str, found))], capture_output=True, text=True,
                           timeout=20)
    except Exception:  # noqa: BLE001
        return found
    out = []
    for line in r.stdout.splitlines():
        bits = line.split(None, 1)
        if len(bits) == 2 and not bits[1].strip().endswith("caffeinate"):
            out.append(int(bits[0]))
    return out


def unit_dir(runs: Path, name: str):
    """Run folder of a failed unit's name as logged in runs/failures.json (None for non-development units)."""
    parts = name.split("/")
    if not parts or parts[0] not in UNIT_KINDS:
        return None
    if parts[0] == "weekly":                      # logged as weekly/weekly/<family>/fold/seed; stored without the 2nd
        parts = [parts[0]] + parts[2:]
    parts = [p for p in parts if p]
    return runs.joinpath(*parts) if len(parts) > 1 else None


def outstanding_failures(runs: Path = RUNS) -> list[str]:
    """Development units that failed and still have no DONE marker (a later success clears them)."""
    out = []
    for it in read_json(runs / "failures.json", []) or []:
        name = it.get("run", "")
        d = unit_dir(runs, name)
        if d is not None and not (d / "DONE").exists() and name not in out:
            out.append(name)
    return out


def _v(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and not math.isfinite(x):
        return ""
    return str(x)


def pull_lines(st) -> list[str]:
    if not st:
        return ["- **Pull (Phase 1):** no state file yet."]
    b = st.get("batches", {})
    done = sorted(int(k) for k, v in b.items() if v.get("state") == "complete")
    run = sorted((int(k), v) for k, v in b.items() if v.get("state") == "running")
    hours = [v["hours"] for v in b.values() if v.get("hours")]
    failed = sorted(n for v in b.values() for n in v.get("failed", []))
    c = st.get("counts", {})
    rc = c.get("regions_complete", {})
    L = [f"- **Pull (Phase 1):** {len(done)} of {N_BATCHES} batches complete"
         + (f"; batch {run[0][0]} running since {run[0][1].get('started')}" if run else "")
         + (f"; {min(hours):.2f}-{max(hours):.2f} h per batch (mean {sum(hours) / len(hours):.2f} h)" if hours else "")
         + "."]
    L.append(f"- New regions complete (counted at batch ends): development {rc.get('dev_west', 0) + rc.get('dev_east', 0)} "
             f"(West {rc.get('dev_west', 0)}, East {rc.get('dev_east', 0)}), fresh test "
             f"{rc.get('test_west', 0) + rc.get('test_east', 0)} (West {rc.get('test_west', 0)}, East "
             f"{rc.get('test_east', 0)}); failed regions: {', '.join(failed) if failed else 'none'}.")
    orig = st.get("originals", {})
    L.append(f"- Original 102 re-extracted: {c.get('originals_complete', 0)}/102"
             + (" (" + ", ".join(f"part {k} {v.get('state')}" for k, v in sorted(orig.items())) + ")" if orig
                else " (starts after batch 12)") + ".")
    L.append(f"- Free disk {c.get('free_disk_gb', '?')} GB (the pull stops below 100 GB); {c.get('elapsed_h', '?')} h of "
             f"the 72 h limit used (pull state updated {st.get('updated', '?')}).")
    return L


def pipeline_lines(s) -> list[str]:
    if not s:
        return ["- **Development stages:** not started (the training launcher starts them at the freeze)."]
    L = []
    d = s.get("data")
    if d:
        L.append(f"- Data: {d['dev_regions']} development regions ({d['dev_cells']:,} cells), {d['test_regions']} "
                 f"fresh-test regions ({d['test_cells']:,} cells); development cells with SWIR "
                 f"{d['dev_cells_with_ndmi']:,}, with per-scene values {d['dev_cells_with_views']:,}.")
    L.append(f"- Pipeline process {s.get('pid')}: {s.get('state')} (started {s.get('process_started')}"
             + (f", finished {s['finished']}" if s.get("finished") else "") + "). Source: `logs/status_pipeline.json`.")
    L += ["", "| Stage | State | Done / total | Failed units | ETA | Finished | Minutes |", "|---|---|---|---|---|---|---|"]
    for k, v in s.get("stages", {}).items():
        L.append(f"| {k} | {_v(v.get('state'))} | {_v(v.get('done'))} / {_v(v.get('total'))} | {_v(v.get('failed_units'))} "
                 f"| {_v(v.get('eta'))} | {_v(v.get('finished'))} | {_v(v.get('minutes'))} |")
    L.append("")
    return L


def runner_lines(pr) -> list[str]:
    """One line per post-hoc runner: its stages from its own status file (DECISIONS.md #52-#66)."""
    L = []
    for name, f, what in RUNNERS:
        r = read_json(f)
        if not r:
            L.append(f"- **{name}** ({what}): no status file yet (`{f.relative_to(V2)}`).")
            continue
        alive = bool(work_pids(pr.get(name, [])))
        parts = []
        for k, v in r.get("stages", {}).items():
            parts.append(f"{k} {v.get('state')}" + (f" {v.get('done')}/{v.get('total')}" if v.get("total") else "")
                         + (f", {v.get('failed')} failed" if v.get("failed") else "")
                         + (f", ETA {v['eta']}" if v.get("eta") and v.get("state") == "running" else "")
                         + (f", finished {v['finished']}" if v.get("finished") else ""))
        L.append(f"- **{name}** ({what}): process {r.get('pid')} " + ("running" if alive else "not running")
                 + f", state {r.get('state')}; " + "; ".join(parts) + f". Status: `{f.relative_to(V2)}`.")
    stop = V2 / "STOP_CHAIN"
    if stop.exists():
        L.append("- `STOP_CHAIN` exists: the chain launcher will not start Phase 5 (DECISIONS.md #51; removed when the "
                 "post-hoc models are part of Phase 5, #66).")
    return L


def now_line(pr, st, s, fz, cs) -> str:
    if cs and cs.get("phase") == "done":
        return "Finished: Phases 1-6 done and the reports regenerated. Only the manual wrap-up (step 10) is left."
    if pr["pipeline run.py"] and s:
        running = [k for k, v in s.get("stages", {}).items() if v.get("state") == "running"]
        cur = running[-1] if running else "?"
        v = s["stages"].get(cur, {})
        prog = f" ({_v(v.get('done'))}/{_v(v.get('total'))}, ETA {_v(v.get('eta'))})" if v.get("total") else ""
        what = "Phase 5/6" if cur in LATE_STAGES else "Development stages"
        return f"{what} running: stage `{cur}`{prog}." + (" The pull is still re-extracting the original 102."
                                                          if pr["pull driver"] else "")
    if not fz:
        b = (st or {}).get("batches", {})
        run = [k for k, v in b.items() if v.get("state") == "running"]
        n = sum(1 for v in b.values() if v.get("state") == "complete")
        return (f"Phase 1 pull: {n} of {N_BATCHES} batches complete" + (f", batch {run[0]} running" if run else "")
                + ". Training starts by itself at the freeze.")
    if pr["training launcher"]:
        return "Region set frozen; the training launcher is starting the development stages."
    busy = [n for n, _, _ in RUNNERS if work_pids(pr.get(n, []))]
    if busy:
        return ("Development stages finished; " + " and ".join(busy) + " running (section 1 below). "
                + ("Phase 5 waits for `STOP_CHAIN` to be removed." if (V2 / "STOP_CHAIN").exists() else
                   f"Chain launcher: {(cs or {}).get('phase')}."))
    if cs:
        return f"Chain launcher: {cs.get('phase')}: {cs.get('note')}"
    return "Nothing is running and the chain launcher has no state: see sections 2 and 5."


def step_lines(st, fz, fr, s, cs, mark, p6, para, lim) -> list[str]:
    b = (st or {}).get("batches", {})
    n_done = sum(1 for v in b.values() if v.get("state") == "complete")
    stages = (s or {}).get("stages", {})
    dev_done = bool(s) and all(stages.get(k, {}).get("finished") for k in DEV_STAGES)
    steps = [
        (n_done == N_BATCHES, "Pull batches 1-12 of new regions", "automatic: pull driver (`code/pull/pull_driver.py`)"),
        (bool((st or {}).get("originals")) or bool(fr), "Final retry of any failed region", "automatic: pull driver"),
        (bool(fz), "Freeze the region set (`data/FROZEN_REGIONS.json`)",
         "automatic: training launcher (`code/launch_when_frozen.sh` runs `code/pull/freeze_regions.py` every 2 min)"),
        (bool(fr), "Re-extract the original 102 per scene, then `data/FROZEN.json`", "automatic: pull driver (two halves)"),
        (dev_done, "Development stages features, simple, search, final, score, checks, whatif, weekly",
         "automatic: the training launcher becomes `code/dewsat2/run.py` (log `logs/run.log`)"),
        (bool(read_json(SCORES / "posthoc.json")), "Post-hoc models (DESIGN.md amendments 4, 4b): search, final, "
         "development scores (`scores/posthoc.json`)", "automatic: post-hoc runner (`code_posthoc/`, log "
         "`logs/posthoc/run.log`)"),
        (not (V2 / "STOP_CHAIN").exists() and bool(mark), "Amendment 6: the post-hoc models and the plain Bi-LSTM in "
         "Phase 5; `STOP_CHAIN` removed, chain launcher restarted", "By hand, after reviewing `scores/posthoc.json` "
         "(DECISIONS.md #66)"),
        (bool(mark), "Failure and sanity check of the development run",
         "automatic: chain launcher (`code/launch_after_dev.py`)"),
        (FRESH_LOCK.exists(), "Phase 5: fresh test scored once, realistic sampling; writes the lock",
         "automatic: chain launcher runs `run.py --stages phase5,phase6` (log `logs/run_phase56.log`)"),
        ("nldas" in p6 and "spei" in p6, "Phase 6: NLDAS-2 agreement and gridMET SPEI benchmark",
         "automatic: same run; if only Phase 6 fails it is retried at most twice"),
        (bool(read_json(SCORES / "lcurve.json")), "Learning curve (amendment 5; secondary)", "learning-curve runner "
         "(`code_lcurve/`, log `logs/lcurve/run.log`); it yields to Phase 5 (`touch logs/lcurve/STOP`) and resumes "
         "afterwards"),
        (bool(cs) and cs.get("phase") == "done", "Reports regenerated (report.md, audit.md, "
         "fatemeh_numbers.md, data_card.md)", "automatic: run.py after every stage; the chain launcher at the end, with "
         "an automatic closing paragraph only if none was written by hand"),
        (bool(para) and not para.get("automatic") and bool(lim), "Manual wrap-up",
         "by hand: a hand-written closing paragraph (`scores/conclusion_paragraph.json`, replacing any "
         "automatic one) and `scores/limitations.json`, regenerate the reports, final DECISIONS.md / FOLDERS.md "
         "entries, commit `code/`, and ask before removing the empty MPS temp folders (section 7)"),
    ]
    return [f"{i}. [{'x' if ok else ' '}] **{title}**: {who}." for i, (ok, title, who) in enumerate(steps, 1)]


COMMANDS = """Run every command from the folder with its environment (it points TMPDIR and the caches inside the folder):

```bash
cd ~/Desktop/DEWSat_BiLSTM_v2 && source env.sh
```

See what is running (roles in section 1):

```bash
pgrep -fl 'pull_driver.py --par-nasa|code/launch_when_frozen.sh|code/dewsat2/run.py --stages|code/launch_after_dev.py'
```

See what the chain launcher would do now (prints only, changes nothing):

```bash
.venv/bin/python code/launch_after_dev.py --check
```

Re-render this file by hand:

```bash
.venv/bin/python code/handoff.py
```

**Only if the process for that stage is not running** (check first; never start a second copy):

- Pull, if the driver died before writing `data/FROZEN.json` (finished batches and months are kept):

```bash
caffeinate -dimsu nohup .venv/bin/python -u -I code/pull/pull_driver.py --par-nasa 10 --par-pc 12 --workers 12 >> logs/pull/driver.log 2>&1 &
```

- Stop the pull cleanly (the batch in progress is left out of the dataset): `touch STOP_PULL`.
- Training launcher, if it died before the freeze and the development stages never started:

```bash
caffeinate -dimsu nohup code/launch_when_frozen.sh >> logs/launcher.log 2>&1 &
```

- Development stages, if `run.py` crashed after the freeze (finished units are skipped; list only the stages that
  have no `finished` time in section 1, in this order):

```bash
caffeinate -dimsu nohup .venv/bin/python -u code/dewsat2/run.py --stages search,final,score,checks,whatif,weekly --cpu-workers 4 --mps-workers 2 --threads 2 --feature-workers 6 >> logs/run.log 2>&1 &
```

- Chain launcher (Phase 5, Phase 6, final reports), if it is not running:

```bash
mkdir -p logs/chain && caffeinate -dimsu nohup .venv/bin/python -u code/launch_after_dev.py >> logs/chain/launcher.log 2>&1 &
```

- Approve Phase 5 after the chain launcher stopped on a sanity check, only after reviewing why: `touch logs/chain/GO_PHASE5`.
- Stop the chain launcher before it starts anything further: `touch STOP_CHAIN` (it never kills a running stage).
- Phase 6 alone, after the fresh-test lock exists and nothing else runs (Phase 6 keeps what it finished; if
  `scores/phase6.json` says SPEI was "Not completed", remove its `spei` key first so SPEI is tried again):

```bash
.venv/bin/python -u code/dewsat2/run.py --stages phase6 >> logs/run_phase56.log 2>&1
```

- Post-hoc runner (only if `logs/posthoc/status.json` shows unfinished stages and it is not running; finished runs
  are skipped). Graceful stop: `touch logs/posthoc/STOP` (running runs finish, no new ones start):

```bash
rm -f logs/posthoc/STOP && caffeinate -dimsu nohup nice -n 10 .venv/bin/python -u code_posthoc/dewsat2/posthoc_run.py --stages search,final,score --cpu-workers 2 --mps-workers 2 --threads 2 >> logs/posthoc/run.log 2>&1 &
```

- Learning-curve runner (after Phase 5, or alongside Phase 6; finished runs are skipped; each runner keeps its own
  worktree so every run of an experiment has the same commit). Graceful stop: `touch logs/lcurve/STOP`:

```bash
rm -f logs/lcurve/STOP && caffeinate -dimsu nohup nice -n 10 .venv/bin/python -u code_lcurve/dewsat2/posthoc_run.py --stages lcurve,lcurve_score --cpu-workers 2 --mps-workers 1 --threads 2 >> logs/lcurve/run.log 2>&1 &
```

- Regenerate the reports:

```bash
.venv/bin/python -c "import sys; sys.path.insert(0, 'code/dewsat2'); import report; report.write_all()"
```
"""

FILES = """- State: `STATUS.md`, this file, `data/pull_state.json`, `logs/status_pipeline.json`, `logs/chain/state.json`.
- Logs: `logs/pull/driver.log`, `logs/launcher.log` and `logs/freeze_check.log` (training launcher), `logs/run.log`
  (development stages), `logs/run_phase56.log` and `logs/chain/launcher.log` (chain launcher), `runs/failures.json`
  and `logs/failures/` (failed units with tracebacks).
- Freeze: `data/FROZEN_REGIONS.json` (the region set; training starts on it), `data/FROZEN.json` (the pull finished).
- Fresh test: `data/locked/LOCK.json` (hashes of the locked label files), `data/locked/FRESH_TEST_SCORED.lock`
  (written once, after scoring; Phase 5 refuses to run if it exists; Phase 6 refuses to run without it).
- Chain launcher bookkeeping: `logs/chain/LOCK/` (one instance), `logs/chain/phase56_launched.json` (Phase 5 started
  once), `logs/chain/GO_PHASE5` (approval after a sanity stop), `STOP_CHAIN`.
- Post-hoc models (amendments 4-6): `runs/posthoc/`, `scores/posthoc.json`, `logs/posthoc/` (status, STOP); tiles
  `cache/tiles/` (`tiles.json`); learning curve `runs/lcurve/`, `scores/lcurve.json`, `logs/lcurve/`. Worktrees:
  `code_posthoc/` (branch `posthoc`) and `code_lcurve/` (branch `lcurve`) are what the running runners use; `main`
  contains both since `31b382e`.
- Results: `scores/*.json`; reports `report.md`, `audit.md`, `fatemeh_numbers.md`, `data_card.md`.
- Design and gate, fixed before training: `code/design/DESIGN.md`, `code/design/gate.json` (commit `f5e92aa`).
- Code: `code/pull/` (pull), `code/dewsat2/` (pipeline; `run.py` is the entry point), `code/launch_when_frozen.sh`,
  `code/launch_after_dev.py`, `code/handoff.py`, `code/status.py`.
"""

RESUME = """1. Read, in this order: this file, `STATUS.md`, `DECISIONS.md`, `code/design/DESIGN.md` and `gate.json`,
   `audit/phase0_audit.md`, and `audit.md` once it has results.
2. The user's standing rules for this task:
   - Everything the task creates stays inside `~/Desktop/DEWSat_BiLSTM_v2/` (code, venv, downloads, caches, temp
     files, runs, logs, reports). Source `env.sh` before every command.
   - The older folders (`DEW ML`, `DEWSat_Paper_BiLSTM`, `DEWSat_Capability_Tests`, `DEWSat_Fatemeh_package`,
     `DEWSat_transfer`) are read-only: change nothing there, permissions included. Copy what you need.
   - The Earthdata login in `~/.netrc` is read only by the download code. Never copy, print or commit it.
   - The fresh test (fresh-test regions, 2025-01 to 2026-09) is locked: no model, setting or decision may look at its
     labels before Phase 5, and it is scored once (`data/locked/FRESH_TEST_SCORED.lock`).
   - The design and gate committed in `f5e92aa` are binding; log every departure or decision in `DECISIONS.md`.
   - The user is away: make reasonable decisions, log them, and do not ask questions.
3. Check the processes (section 3). If every process that should be running is running, change nothing: watch with
   `STATUS.md`, this file and the logs. Never start `run.py` by hand while the training launcher, a `run.py` or the
   chain launcher is alive.
4. If a process died, use the matching command in section 3, after reading its log tail, and log what you did.
5. When step 9 is ticked, do step 10: replace the automatic closing paragraph with a hand-written yes/no paragraph
   that says why, using only logged numbers; write `scores/limitations.json` (`{"items": [...]}`); regenerate the
   reports; update `DECISIONS.md`, `STATUS.md` and `FOLDERS.md`; commit `code/`. Ask before deleting anything outside
   the folder.
"""

ESTIMATES = """Updated 2026-11-09 03:45 (from the finished runs' seconds per epoch per 100k windows; Phase 5 trains on
about 1.0 M job-1 windows, 2.4 times a final run's):

| Stage | Work | Duration | Expected end |
|---|---|---|---|
| Post-hoc finals and scores | last CNN finals on the GPU | about 30 min | Nov 9 04:15 |
| Phase 5 composite | 140 refits; the 20 tile-CNN refits on 2 GPU workers set the length | 3.5-5 h | Nov 9 09:00 |
| Phase 5 thinned | 18 refits of each job's best network on 2 workers | 2.5-4 h | Nov 9 12:30 |
| Phase 5 scoring | bootstrap of every row | 0.3-1 h | Nov 9 13:00 |
| Phase 6 | NLDAS-2, SPEI | 1-3 h | Nov 9 15:00 |
| Learning curve | 180 runs (90 done early); runs again after Phase 5 | 1-2 h | Nov 9 evening |

Original estimate, kept for the record. Made 2026-11-07 11:30 from measured timings: pull batches 1.13-1.26 h (mean 1.18 h); network seconds per epoch
per 100k windows on real windows (`logs/bench_real.json`); window counts of the 102-region sandbox (`smokeT`) scaled by
cells (x5.35). Network stages use the pipeline's own cost model, calibrated on the benchmark (measured/model 0.83 CPU,
0.88 MPS); the range allows 1.0-2.0x for six workers sharing 12 cores and one GPU. Script and output:
`logs/status_check_2026-11-07/`. Stages marked "rough" have no full-scale measurement.

| Stage | Work | Duration | Expected end (range) |
|---|---|---|---|
| Pull batches 7-12 | 6 x 1.18 h | about 7 h from 10:16 | Nov 7 17:20 (17:05-17:45) |
| Freeze, training starts | launcher polls every 2 min | minutes | Nov 7 17:25 |
| Originals re-extracted (alongside training) | 102 regions, two halves | 2.4-3.5 h | Nov 7 20:30 (19:45-21:15) |
| Features (rough) | 702 regions; sandbox 102 in 70 s | 0.3-0.8 h | Nov 7 18:00 |
| Simple models (rough) | 12 units, 30 tree settings each | 0.5-1.5 h | Nov 7 19:00 |
| Search | 1,800 units, 55 worker-hours at benchmark speed | 9-18 h (central 13.7) | Nov 8 08:40 (Nov 8 04:00-13:00) |
| Final refits | 600 units, 37 worker-hours | 6-12 h (central 9.3) | Nov 8 18:00 (Nov 8 10:00 - Nov 9 01:00) |
| Score, checks, what-ifs, weekly (rough) | positive control 15 trainings, 48 what-if and 12 weekly units | 3-6 h | Nov 8 22:30 (Nov 8 13:00 - Nov 9 07:00) |
| Phase 5 (partly rough) | 70 refits on all development data (2.0-2.9 h), 18 thinned refits on 2 workers (2.5-5 h), scoring | 5-8.5 h | Nov 9 05:00 (Nov 8 18:00 - Nov 9 15:30) |
| Phase 6 (rough) | NLDAS-2 309 months; SPEI point series for every region | 1-3 h | Nov 9 07:00 (Nov 8 19:00 - Nov 9 18:30) |
| Reports | regenerated after every stage | minutes | with Phase 6 |

The task's 6-day limit (pull included) ends Nov 13 03:03, so the latest estimate leaves more than 3 days.
"""

KNOWN = """- `STATUS.md` shows an empty "Phases 3-4 pipeline" section: `code/status.py` reads `logs/status_pipeline.json`,
  but the pipeline writes its table to `logs/status_pipeline_lines.json`. Nothing is lost; section 1 of this file
  shows the table. The one-line fix in `code/status.py` waits for the user's OK (DECISIONS.md #47).
- PyTorch's Apple-GPU (MPS) backend creates an empty folder per compiled graph in the macOS per-user temp folder
  (`/var/folders/hz/_y4bwsp114bgsxv6j56hq81c0000gn/T/com.apple.MetalPerformanceShadersGraph/mpsgraph-<pid>-...`),
  ignoring TMPDIR. 4,708 empty folders (0 bytes) came from the benchmarks and sandboxes on Nov 7 03:19-04:55, and the
  MPS workers will add more. Remove them only with the user's OK, after the run (DECISIONS.md #46).
"""


def render(path: Path = V2 / "HANDOFF.md") -> str:
    st = read_json(DATA / "pull_state.json")
    fz = read_json(DATA / "FROZEN_REGIONS.json")
    fr = read_json(DATA / "FROZEN.json")
    s = read_json(LOGS / "status_pipeline.json")
    cs = read_json(CHAIN / "state.json")
    mark = read_json(CHAIN / "phase56_launched.json")
    p6 = read_json(SCORES / "phase6.json", {}) or {}
    para = read_json(SCORES / "conclusion_paragraph.json")
    lim = read_json(SCORES / "limitations.json")
    pr = processes()
    L = ["# HANDOFF: DEWSat_BiLSTM_v2", "",
         f"_Rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by `code/handoff.py`. The chain launcher re-renders it every 2 "
         "minutes while it runs; if this time is old, that launcher has stopped (section 3)._", "",
         "## 1. Where things stand", "", f"**Now:** {now_line(pr, st, s, fz, cs)}", ""]
    L += pull_lines(st)
    L.append(f"- Freeze: " + (f"region set frozen {fz.get('frozen')}: {json.dumps(fz.get('counts'))}, failed regions "
                              f"{fz.get('failed_regions') or 'none'}." if fz else "not yet (`data/FROZEN_REGIONS.json`)."))
    L.append("- Pull finished: " + (f"yes, `data/FROZEN.json` ({fr.get('stop_reason', fr.get('reason', ''))})." if fr
                                    else "not yet (`data/FROZEN.json`)."))
    L += pipeline_lines(s)
    L += runner_lines(pr)
    left = outstanding_failures()
    L.append(f"- Failed units without a later success: {len(left)}" + (f" ({', '.join(left[:8])})" if left else "") + ".")
    L.append("- Chain launcher: " + (f"{cs.get('phase')}: {cs.get('note')} (checked {cs.get('updated')})" if cs
                                     else "no state yet.")
             + (" Problems: " + "; ".join(cs.get("problems", [])) if cs and cs.get("problems") else ""))
    lock = read_json(FRESH_LOCK)
    L.append("- Fresh test: " + (f"scored once at {lock.get('scored')} (lock written)." if lock else
                                 "locked and not scored (no model or decision has seen its labels)."))
    L.append("- Phase 6: " + ("NLDAS-2 done; " if "nldas" in p6 else "NLDAS-2 not done; ")
             + ("SPEI done." if "spei" in p6 else "SPEI not done."))
    L.append("- Closing paragraph: " + ("none yet." if not para else
                                        "automatic (to be replaced by hand)." if para.get("automatic") else "hand-written."))
    L += ["", "| Role | PIDs | Pattern |", "|---|---|---|"]
    for name, pat in PROCS:
        L.append(f"| {name} | {', '.join(map(str, pr[name])) or 'not running'} | `{pat}` |")
    L += ["", "## 2. What is left, in order, and what starts it", ""]
    L += step_lines(st, fz, fr, s, cs, mark, p6, para, lim)
    L += ["", "## 3. Exact commands", "", COMMANDS, "## 4. Files", "", FILES, "## 5. Resuming in a fresh session",
          "", RESUME, "## 6. Timeline estimate", "", ESTIMATES, "## 7. Known issues", "", KNOWN]
    text = "\n".join(L).rstrip() + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return text


if __name__ == "__main__":
    render()
    print(f"wrote {V2 / 'HANDOFF.md'}")
