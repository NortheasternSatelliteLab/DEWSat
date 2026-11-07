"""Score stage: pooled outer-fold predictions per job, subsets (jobs 1-4), every metric, and the gate (gate.json)."""

from __future__ import annotations

import json
import time

import numpy as np

import scoring
from common import (CODE, FAMILIES, FINAL_SEEDS, LEADS, N_FOLDS, RUNS, SCORES, SIMPLE, git_state, log, read_json,
                    write_json)
from windows import Base, raw_windows

GATE = json.loads((CODE / "design" / "gate.json").read_text())


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def load_preds(job: str, k: int, n_heads: int):
    """(wid, {model: prob (n, heads)}, seeds used per family) for one job and fold."""
    sd = RUNS / "simple" / job / f"fold{k + 1}"
    preds, used = {}, {}
    wid = None
    if (sd / "DONE").exists():
        with np.load(sd / "pred.npz") as z:
            wid = z["wid"]
            prob = z["prob"]                                    # (n, heads, n_simple)
            meta = read_json(sd / "meta.json")
            for j, m in enumerate(meta["models"]):
                preds[m] = prob[:, :, j].astype(float)
    for fam in FAMILIES:
        ps = []
        for s in FINAL_SEEDS:
            d = RUNS / "final" / job / fam / f"fold{k + 1}" / f"seed{s}"
            if not (d / "DONE").exists():
                continue
            with np.load(d / "pred.npz") as z:
                if wid is not None and not np.array_equal(z["wid"], wid):
                    raise RuntimeError(f"{job}/{fam}/fold{k + 1}/seed{s}: window ids differ from the simple models'")
                wid = z["wid"] if wid is None else wid
                ps.append(_sig(z["logit"].astype(float)))
        used[fam] = len(ps)
        if len(ps) == len(FINAL_SEEDS):
            preds[fam] = np.mean(ps, axis=0)
    return wid, preds, used


def collect(base: Base, arrays: dict) -> dict:
    subsets: dict[str, list] = {}
    status = {"seeds_used": {}}

    def add(name, wid, y, preds, extra=None):
        subsets.setdefault(name, []).append({"wid": wid, "y": y, "preds": preds, "extra": extra or {}})

    for job, nh in (("job1", 1), ("job2", 1), ("job3", len(LEADS))):
        for k in range(N_FOLDS):
            wid, preds, used = load_preds(job, k, nh)
            status["seeds_used"][f"{job}/fold{k + 1}"] = used
            if wid is None:
                continue
            if job in ("job1", "job2"):
                y = base.label0[wid]
                pr = {m: p[:, 0] for m, p in preds.items()}
                extra = {}
                if job == "job1":
                    V, O = raw_windows(arrays["feats"][k], arrays["obs"][k], base, wid)
                    extra = {"d1": np.where(O[:, -2], np.abs(V[:, -1] - V[:, -2]), np.nan),
                             "d3": np.where(O[:, -4], np.abs(V[:, -1] - V[:, -4]), np.nan)}
                add(job, wid, y, pr, extra)
            else:
                lab0 = base.label0[wid]
                for j, h in enumerate(LEADS):
                    yh = base.label_h[wid, j]
                    ok = yh >= 0
                    pr = {m: p[:, j] for m, p in preds.items()}
                    pr["persistence"] = lab0.astype(float)
                    add(f"job3_lead{h}_all", wid[ok], yh[ok], {m: v[ok] for m, v in pr.items()})
                    on = ok & (lab0 == 0)
                    add(f"job3_lead{h}_onset", wid[on], yh[on], {m: v[on] for m, v in pr.items()})
    out = {}
    for name, plist in subsets.items():
        models = set().union(*(p["preds"].keys() for p in plist))
        keep = sorted(m for m in models if all(m in p["preds"] for p in plist))
        out[name] = {"wid": np.concatenate([p["wid"] for p in plist]),
                     "y": np.concatenate([p["y"] for p in plist]).astype(np.int64),
                     "preds": {m: np.concatenate([p["preds"][m] for p in plist]) for m in keep},
                     "extra": {e: np.concatenate([p["extra"][e] for p in plist]) for e in plist[0]["extra"]},
                     "folds_present": len(plist)}
    thr = {}
    if "job1" in out:
        j1 = out["job1"]
        d1, d3 = j1["extra"]["d1"], j1["extra"]["d3"]
        t1, t3 = float(np.nanpercentile(d1, 75)), float(np.nanpercentile(d3, 75))
        thr = {"d1_q75": t1, "d3_q75": t3}
        top1, top3 = np.nan_to_num(d1, nan=-1) >= t1, np.nan_to_num(d3, nan=-1) >= t3
        for nm, m in (("job4_either", top1 | top3), ("job4_top_d1", top1), ("job4_top_d3", top3),
                      ("job4_neither", ~(top1 | top3))):
            out[nm] = {"wid": j1["wid"][m], "y": j1["y"][m], "preds": {kk: v[m] for kk, v in j1["preds"].items()},
                       "extra": {}, "folds_present": j1["folds_present"]}
    for s in out.values():
        s["region"] = base.w_region[s["wid"]]
        s["west"] = base.region_west[s["region"]]
    return {"subsets": out, "status": status, "job4_thresholds": thr}


def choose(values: dict, cands: list[str]):
    c = [m for m in cands if m in values and values[m] is not None]
    return max(c, key=lambda m: values[m]) if c else None


def evaluate(base: Base, arrays: dict) -> dict:
    t0 = time.time()
    col = collect(base, arrays)
    subs = col["subsets"]
    R = base.n_regions
    fold_r = base.region_group
    nets_all = list(FAMILIES)
    result = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state(), "status": col["status"],
              "job4_thresholds": col["job4_thresholds"], "subsets": {}, "gate": [], "job3_mean": {}}
    # point within-fold AUCs for selection (cheap: full sample only)
    def point(name, model):
        s = subs[name]
        if model not in s["preds"]:
            return None
        W, P, N = scoring.win_matrix(s["preds"][model], s["y"], s["region"], R)
        C = np.zeros((1, R)); C[0, np.unique(s["region"])] = 1
        return float(scoring.auc_within_from(W, P, N, C, fold_r)[0])

    job3_names = [f"job3_lead{h}_onset" for h in LEADS]
    job3_all = [f"job3_lead{h}_all" for h in LEADS]
    sel = {}
    for g in GATE["jobs"]:
        simple_c = [m for m in GATE["simple_models"] if m not in g["simple_exclude"]]
        if g["job"] == "job3":
            if not all(n in subs for n in job3_names):
                continue
            mv = scoring.mean_over_leads("job3_mean_onset", [subs[n] for n in job3_names],
                                         simple_c + nets_all + ["persistence"], fold_r, R)
            vals = {m: float(v["all"][0]) for m, v in mv.items()}
        else:
            if g["subset"] not in subs:
                continue
            vals = {m: point(g["subset"], m) for m in simple_c + nets_all}
        sel[g["job"]] = {"values": vals, "best_network": choose(vals, nets_all), "best_simple": choose(vals, simple_c),
                         "best_bilstm": choose(vals, GATE["bilstm_variants"])}
    result["selection"] = sel
    # full scoring with the needed pairs
    for name, s in subs.items():
        job = "job1" if name.startswith(("job1", "job4")) else name[:4]
        j = sel.get("job3" if job.startswith("job3") else job, {})
        comps = [c for c in ("lr_vci_t" if job != "job2" else "lr_latest_age", j.get("best_simple")) if c]
        pairs = [(m, c) for c in dict.fromkeys(comps) for m in s["preds"] if m != c]
        result["subsets"][name] = scoring.score_subset(name, s["y"], s["region"], s["west"], s["preds"], fold_r, R,
                                                       N_FOLDS, pairs)
        result["subsets"][name]["folds_present"] = s["folds_present"]
    # job 3 means over leads (onset: the gate metric; all months: secondary), with paired differences
    for kind, names in (("onset", job3_names), ("all", job3_all)):
        if not all(n in subs for n in names):
            continue
        mv = scoring.mean_over_leads(f"job3_mean_{kind}", [subs[n] for n in names],
                                     list(SIMPLE) + nets_all + ["persistence"], fold_r, R)
        mvp = scoring.mean_over_leads(f"job3_mean_{kind}", [subs[n] for n in names],
                                      list(SIMPLE) + nets_all + ["persistence"], fold_r, R, within=False)
        ent = {"within": {m: {sd: scoring.ci(v[sd]) for sd in scoring.SIDES} for m, v in mv.items()},
               "pooled": {m: {sd: scoring.ci(v[sd]) for sd in scoring.SIDES} for m, v in mvp.items()}, "paired": {}}
        bs = sel.get("job3", {}).get("best_simple")
        for comp in dict.fromkeys([c for c in (bs, "lr_vci_t") if c]):
            if comp in mv:
                for m in mv:
                    if m != comp:
                        ent["paired"][f"{m} - {comp}"] = {sd: scoring.ci(mv[m][sd] - mv[comp][sd]) for sd in scoring.SIDES}
        result["job3_mean"][kind] = ent
    # gate
    for g in GATE["jobs"]:
        s = sel.get(g["job"])
        entry = {"job": g["job"], "label": g["label"], "status": "not scored"}
        if not s or not s["best_network"] or not s["best_simple"]:
            result["gate"].append(entry)
            continue
        rows = [("gate", s["best_network"], s["best_simple"]), ("best_bilstm", s["best_bilstm"], s["best_simple"])]
        if g.get("also_compare"):
            rows += [("vs_lr_vci_t_network", s["best_network"], g["also_compare"]),
                     ("vs_lr_vci_t_bilstm", s["best_bilstm"], g["also_compare"])]
        entry["rows"] = []
        for kind, a, b in rows:
            if not a or not b or a == b:
                continue
            if g["job"] == "job3":
                d = result["job3_mean"]["onset"]["paired"].get(f"{a} - {b}")
            else:
                d = result["subsets"][g["subset"]]["paired"].get(f"{a} - {b}")
                d = {sd: d[sd]["dauc_within"] for sd in scoring.SIDES} if d else None
            if not d:
                continue
            entry["rows"].append({"kind": kind, "model": a, "comparator": b, "dauc_within": d,
                                  "verdict": scoring.verdict(d["all"], d["west"], d["east"], GATE["wording"])})
        entry["status"] = "scored"
        entry["best_network"], entry["best_simple"], entry["best_bilstm"] = (s["best_network"], s["best_simple"],
                                                                          s["best_bilstm"])
        result["gate"].append(entry)
    result["seconds"] = round(time.time() - t0, 1)
    return result


def run(status=None) -> dict:
    import dataset
    meta, arrays = dataset.load("dev")
    base = Base(meta, arrays, "dev")
    res = evaluate(base, arrays)
    write_json(SCORES / "scores.json", res)
    for g in res["gate"]:
        for r in g.get("rows", []):
            log(f"  {g['job']} {r['kind']}: {r['model']} - {r['comparator']} {r['dauc_within']['all']} -> {r['verdict']}")
    return res
