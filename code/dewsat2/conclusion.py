"""Short, email-ready lines for fatemeh_numbers.md, built only from logged scores (scores.json, phase5.json).

The closing yes/no paragraph for audit.md is written by hand once the results are in (it needs a reason,
not only numbers) and stored in scores/conclusion_paragraph.json; every number it quotes comes from these files.
"""

from __future__ import annotations

from common import SCORES, read_json, write_json
from report import ci, lab


def _pair(sub: dict, a: str, b: str, key: str = "dauc_within"):
    p = (sub or {}).get("paired", {}).get(f"{a} - {b}")
    return None if not p else p["all"][key]


def _auc(sub: dict, m: str):
    r = (sub or {}).get("models", {}).get(m)
    return None if not r else r["all"]["auc_within"]


def build() -> dict:
    sc = read_json(SCORES / "scores.json")
    p5 = read_json(SCORES / "phase5.json")
    st = read_json(SCORES / "phase5_settings.json") or {}
    out = {"dev_lines": [], "fresh_lines": [], "realistic_lines": []}
    if not sc:
        return out
    sel = sc["selection"]
    j1 = sel.get("job1", {})
    b1, n1 = j1.get("best_bilstm"), j1.get("best_network")
    dev_b = _pair(sc["subsets"].get("job1"), b1, "lr_vci_t") if b1 else None
    dev_n = _pair(sc["subsets"].get("job1"), n1, "lr_vci_t") if n1 else None
    fr_b = _pair(p5["fresh"].get("job1"), b1, "lr_vci_t") if (p5 and b1) else None
    fr_n = _pair(p5["fresh"].get("job1"), n1, "lr_vci_t") if (p5 and n1) else None

    def verdict(d):
        if not d or d[1] is None:
            return "not scored"
        if d[1] > 0:
            return "better"
        if d[2] < 0:
            return "worse"
        return "no detectable difference"

    if dev_b:
        line = (f"**Bottom line:** for estimating the current month's drought (job 1), the best Bi-LSTM variant "
                f"({lab(b1)}) did not beat logistic regression on VCI(T)" if verdict(dev_b) != "better" or
                (fr_b and verdict(fr_b) != "better") else
                f"**Bottom line:** for estimating the current month's drought (job 1), the best Bi-LSTM variant "
                f"({lab(b1)}) beat logistic regression on VCI(T)")
        line += f": within-fold AUC difference {ci(dev_b, signed=True)} in development ({verdict(dev_b)})"
        if fr_b:
            line += f" and {ci(fr_b, signed=True)} on the fresh test ({verdict(fr_b)})"
        line += "."
        if n1 and n1 != b1 and dev_n:
            line += (f" The best temporal network overall ({lab(n1)}): {ci(dev_n, signed=True)} in development"
                     + (f", {ci(fr_n, signed=True)} on the fresh test" if fr_n else "") + ".")
        out["bottom_line"] = line
    # development, one line per job (gate rows)
    for g in sc["gate"]:
        for r in g.get("rows", []):
            if r["kind"] == "gate":
                d = r["dauc_within"]
                out["dev_lines"].append(f"- Development, {g['label']}: {lab(r['model'])} − {lab(r['comparator'])} "
                                        f"within-fold ΔAUC {ci(d['all'], signed=True)} (West {ci(d['west'], signed=True)}, "
                                        f"East {ci(d['east'], signed=True)}): {r['verdict']}.")
    if p5:
        fr = p5["fresh"]
        out["fresh_lines"].append("**Fresh test** (regions never used in development, 2025-01 to 2026-09, scored once):")
        mods = st.get("models", {})
        for job, title, comp in (("job1", "Job 1, estimation now", "lr_vci_t"), ("job2", "Job 2, missing month", "lr_latest_age"),
                                 ("job4", "Job 4, where VCI changed most", "lr_vci_t")):
            s = fr.get(job)
            if not s:
                continue
            nets = mods.get("job1" if job == "job4" else job, {}).get("nets", [])
            parts = [f"{lab(comp)} AUC {ci(_auc(s, comp))}"]
            for m in nets:
                if m in s["models"]:
                    parts.append(f"{lab(m)} {ci(_auc(s, m))} (difference {ci(_pair(s, m, comp), signed=True)})")
            out["fresh_lines"].append(f"- {title} ({s['n']:,} windows, {s['sides']['all']['regions']} regions): " +
                                      "; ".join(parts) + ".")
        m3 = fr.get("job3_mean_onset")
        if m3:
            nets = mods.get("job3", {}).get("nets", [])
            bs = mods.get("job3", {}).get("best_simple")
            parts = [f"{lab(bs)} {ci(m3['models'][bs]['all'])}" if bs in m3["models"] else ""]
            for m in nets:
                if m in m3["models"]:
                    d = m3["paired"].get(f"{m} - {bs}")
                    parts.append(f"{lab(m)} {ci(m3['models'][m]['all'])}" + (f" (difference {ci(d['all'], signed=True)})" if d else ""))
            out["fresh_lines"].append("- Job 3, forecast, mean onset AUC over leads 1-3: " + "; ".join(p for p in parts if p) + ".")
        cov = p5.get("coverage")
        if cov:
            out["realistic_lines"].append(
                "**Realistic sampling** (fresh test, single clear views, one pass every N days, 3 start dates): share of "
                f"cell-months with a fresh view: full HLS record {cov['full_record_composite']:.0%}, " +
                ", ".join(f"{N} days {cov[f'N{N}']['share_mean']:.0%}" for N in (7, 16, 32)) + ".")
        for job, comp in (("job1", "lr_vci_t"), ("job2", "lr_latest_age")):
            r = p5.get("realistic", {}).get(job, {})
            for ver, title in (("composite_trained", "trained on composites"), ("thin_trained", "trained on thinned development data")):
                v = r.get(ver)
                if not v:
                    continue
                net = (mods.get(job, {}).get("nets") or [None])[0]
                bits = []
                for N in (7, 16, 32):
                    keys = [k for k in v if k.startswith(f"thin{N}_")]
                    if not keys:
                        continue
                    a = [v[k]["models"].get(comp, {}).get("all", {}).get("auc_within", [None])[0] for k in keys]
                    b = [v[k]["models"].get(net, {}).get("all", {}).get("auc_within", [None])[0] for k in keys] if net else []
                    a = [x for x in a if x is not None]
                    b = [x for x in b if x is not None]
                    if a:
                        bits.append(f"{N} days: {lab(comp)} {sum(a) / len(a):.3f}" + (f", {lab(net)} {sum(b) / len(b):.3f}" if b else ""))
                if bits:
                    out["realistic_lines"].append(f"- {job}, models {title} (mean AUC over the 3 starts): " + "; ".join(bits) + ".")
    para = read_json(SCORES / "conclusion_paragraph.json")
    if para:
        out["paragraph"] = para["paragraph"]
    write_json(SCORES / "conclusion.json", out)
    return out
