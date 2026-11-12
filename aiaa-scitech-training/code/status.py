"""Render STATUS.md from every stage's state file (pull, audit, pipeline, fresh test)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]  # the repository root


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def render() -> None:
    lines = ["# Status", "", f"_Updated {time.strftime('%Y-%m-%d %H:%M:%S')} by `code/status.py`._", ""]
    # ---- pull
    st = _read(V2 / "data" / "pull_state.json")
    plan = _read(V2 / "data" / "plan" / "plan.json") or {}
    lines += ["## Phase 1: data pull", ""]
    if not st:
        lines += ["Not started.", ""]
    else:
        c = st.get("counts", {})
        rc = c.get("regions_complete", {})
        lines += [f"- Started {st['started']}; elapsed {c.get('elapsed_h', '?')} h of the 72 h limit; free disk "
                  f"{c.get('free_disk_gb', '?')} GB (limit 100 GB). Last note: {st.get('note', '')} ({st.get('updated', '')}).",
                  f"- Plan: {plan.get('n_batches', '?')} balanced batches of 50 ({plan.get('in_batches', '?')} new regions: "
                  f"{plan.get('n_dev', '?')} development and {plan.get('n_test', '?')} fresh-test planned in total, "
                  f"{plan.get('not_in_any_batch', '?')} of them outside any balanced batch), then the original 102 "
                  "re-extracted per scene.",
                  f"- New regions complete: development West {rc.get('dev_west', 0)}, East {rc.get('dev_east', 0)}; "
                  f"fresh test West {rc.get('test_west', 0)}, East {rc.get('test_east', 0)} "
                  f"({c.get('cells_complete', 0)} cells). Originals re-extracted: {c.get('originals_complete', 0)}/102. "
                  f"Monthly files on disk: {c.get('month_files', 0)}.", ""]
        lines += ["| Batch | State | Regions ok | Failed | Hours | Finished |", "|---|---|---|---|---|---|"]
        for b, v in sorted(st.get("batches", {}).items(), key=lambda kv: int(kv[0])):
            lines.append(f"| {b} | {v.get('state')} | {len(v.get('ok', []))} | {len(v.get('failed', []))} | "
                         f"{v.get('hours', '')} | {v.get('finished', '')} |")
        for b, v in sorted(st.get("originals", {}).items()):
            lines.append(f"| originals {b} | {v.get('state')} | {len(v.get('ok', []))} | {len(v.get('failed', []))} | | |")
        lines.append("")
    fz = _read(V2 / "data" / "FROZEN.json")
    if fz:
        lines += [f"**Dataset frozen** {fz['frozen']} ({fz['stop_reason']}): {fz['counts']}; originals re-extracted "
                  f"{len(fz['original_regions_reextracted'])}.", ""]
    fl = _read(V2 / "logs" / "pull" / "fldas_check.json")
    if fl:
        lines += [f"- FLDAS labels pulled (CONUS window, 2001-01..2026-09). Check against the paper run's files for the "
                  f"102: {fl['values']:,} values, max abs difference {fl['max_abs_diff']:.2e}, "
                  f"{fl['mismatch_over_1e-6']} over 1e-6.", ""]
    # ---- other stages write their own status json; listed here if present
    for name, title in (("audit", "Phase 0 audit"), ("pipeline", "Phases 3-4 pipeline"),
                        ("weekly", "Weekly-input experiment"), ("fresh", "Phase 5 fresh test"),
                        ("extras", "Phase 6 extras")):
        if name == "pipeline":
            # the pipeline writes its rendered section (header, data line, stage table) to its own lines file
            s = _read(V2 / "logs" / "status_pipeline_lines.json")
            if s:
                lines += list(s.get("lines", [])) + [""]
            continue
        s = _read(V2 / "logs" / f"status_{name}.json")
        if s:
            lines += [f"## {title}", ""] + [f"- {x}" for x in s.get("lines", [])] + [""]
    # ---- the post-hoc models' and the learning curve's runners (DESIGN.md amendments 4-6)
    try:
        import handoff
        rl = handoff.runner_lines(handoff.processes())
        if rl:
            lines += ["## Post-hoc models and learning curve", ""] + rl + [""]
    except Exception:  # noqa: BLE001
        pass
    path = Path(os.environ.get("DEWSAT_STATUS_OUT", str(V2 / "STATUS.md")))
    tmp = path.with_name("STATUS.md.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    render()
