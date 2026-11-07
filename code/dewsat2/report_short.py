"""audit.md, fatemeh_numbers.md and data_card.md, from the same logged files as report.md."""

from __future__ import annotations

import time

from common import OUT, SCORES, read_json, write_text
from report import ci, lab


def conclusion() -> dict:
    """Email-ready lines from the logged scores, plus the hand-written closing paragraph once it exists."""
    try:
        import conclusion as C
        return C.build()
    except Exception:  # noqa: BLE001
        return read_json(SCORES / "conclusion.json", {}) or {}


def write(d: dict) -> None:
    write_text(OUT / "audit.md", audit(d))
    write_text(OUT / "fatemeh_numbers.md", fatemeh(d))
    import data_card
    write_text(OUT / "data_card.md", data_card.card(d))


def gate_lines(d) -> list[str]:
    sc = d["scores"]
    out = []
    if not sc:
        return ["Development results not in yet."]
    for g in sc["gate"]:
        for r in g.get("rows", []):
            dd = r["dauc_within"]
            out.append(f"- {g['job']} [{r['kind']}] {lab(r['model'])} − {lab(r['comparator'])}: "
                       f"{ci(dd['all'], signed=True)} (W {ci(dd['west'], signed=True)}, E {ci(dd['east'], signed=True)}) "
                       f"→ {r['verdict']}")
    return out


def audit(d: dict) -> str:
    import report as R
    L = ["# DEWSat_BiLSTM_v2: self-contained summary", "",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M')} from the run's logged files; full detail in `report.md`._", ""]
    for sec in (R.sec_phase0(d), R.sec_data(d), R.sec_design()):
        L += sec
    L += ["## Gate outcomes (development, within-fold AUC)", ""] + gate_lines(d) + [""]
    sc = d["scores"]
    if sc:
        for key in ("job1", "job2", "job4_either"):
            s = sc["subsets"].get(key)
            if s:
                L += [f"### {key} ({s['n']:,} windows)", ""] + R.model_table(s, R.all_models(s)) + [""]
        L += R.sec_results(d)[-0:0]
        j3 = sc.get("job3_mean", {}).get("onset")
        if j3:
            L += ["### job3 mean onset within-fold AUC", ""] + [f"- {lab(m)}: {ci(v['all'])}" for m, v in j3["within"].items()] + [""]
    L += R.sec_checks(d) + R.sec_whatif(d) + R.sec_weekly(d) + R.sec_fresh(d) + R.sec_limits(d)
    c = conclusion()
    L += ["## Did the Bi-LSTM beat logistic regression?", "", c.get("paragraph", "Not decided yet: the run is not finished."), ""]
    text = "\n".join(L) + "\n"
    lines = text.splitlines()
    if len(lines) > 400:
        text = "\n".join(lines[:395] + ["", "_(trimmed to stay under ~400 lines; see report.md)_"] +
                         ["", "## Did the Bi-LSTM beat logistic regression?", "", c.get("paragraph", "")]) + "\n"
    return text


def fatemeh(d: dict) -> str:
    c = conclusion()
    L = ["# DEWSat: numbers for Prof. Ghoreishi", "", c.get("bottom_line", "Bottom line: pending (the run is still going)."), ""]
    p5 = d["phase5"]
    if p5:
        L += c.get("fresh_lines", []) + [""]
        L += c.get("realistic_lines", []) + [""]
    if c.get("dev_lines"):
        L += ["**Development** (4 outer folds, regions and years held out together; within-fold AUC, 95% intervals from "
              "2,000 whole-region resamples):"] + c["dev_lines"]
    return "\n".join(L) + "\n"
