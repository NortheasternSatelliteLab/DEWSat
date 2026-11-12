"""Post-hoc analyses of the one fresh-test scoring, run after the lock: results/scores/fresh_posthoc.json.

Not a second test: the same saved predictions (Phase 5 networks' pred.npz, the simple models refit as Phase 5 did),
sliced to explain the result. Written for the report (section 8b) and the audit.

- network_pairs: paired differences between the networks themselves (job 1 and job 4), which the Phase 5 scorer did not
  compute; scoring.score_subset with the scorer's subset names, so the region resamples are the official ones.
- by_year: drought rate, mean prediction and pooled AUC per year (2025, 2026-01..09) and side.
- within_region ("over time"): AUC from pairs inside the same region, pair-weighted, network minus LR with 1,000
  whole-region resamples. within_month ("across places"): AUC from pairs inside the same month, pair-weighted.
- dev_folds: per development fold, the training-set size, the validation drought rate and each model's AUC, beside the
  Phase 5 training-set size and the fresh-test drought rate.
- reload_check: summary of logs/status_check_2026-11-09/reload_check_phase5.json.

    source env.sh && .venv/bin/python -u code/dewsat2/fresh_posthoc.py
"""

from __future__ import annotations

import time

import numpy as np

import phase5_stage as P5
import scoring
from common import FINAL_SEEDS, LOGS, MONTHS, N_FOLDS, RUNS, SCORES, git_state, log, read_json, write_json

NETS = ("bilstm", "resbilstm", "gru", "bilstm_vci", "cnn_lr", "cnn_bilstm")
SIMPLE = ("lr_vci_t", "gbt_mono")
PAIRS = [("bilstm", "resbilstm"), ("bilstm", "gru"), ("resbilstm", "gru"), ("bilstm", "bilstm_vci"),
         ("resbilstm", "bilstm_vci"), ("gru", "bilstm_vci"), ("bilstm", "cnn_lr"), ("bilstm", "cnn_bilstm"),
         ("bilstm_vci", "cnn_bilstm")]
N_RES = 1000
SEED = 20261109


def _u_pairs(p, y):
    """(Mann-Whitney U with average ranks for ties, number of positive x negative pairs)."""
    from scipy.stats import rankdata
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0, 0
    r = rankdata(np.concatenate([neg, pos]))
    return float(r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2), len(pos) * len(neg)


def _per_group(p, y, g):
    """{group: (U, pairs)} for pairs inside each group."""
    return {k: _u_pairs(p[g == k], y[g == k]) for k in np.unique(g)}


def run() -> dict:
    if not P5.LOCK.exists():
        raise SystemExit("the fresh test has not been scored; nothing to analyse")
    t0 = time.time()
    st = read_json(SCORES / "phase5_settings.json")
    sp, evs = P5.simple_preds("job1", "composite", st["settings"]["job1"]["simple"])
    e = dict(evs)["fresh"]
    tb = P5.ctx5()["base_test"]
    y = e["y0"].astype(int)
    T = tb.w_T[e["idx"]]
    reg = e["region"]
    west = np.asarray(e["west"], bool)
    year = np.array([int(MONTHS[t][:4]) for t in T])
    preds = {m: (sp["fresh"][m][:, 0] if sp["fresh"][m].ndim == 2 else sp["fresh"][m]) for m in SIMPLE}
    for fam in NETS:
        p, idx = P5._net_probs("job1", "composite", fam, "fresh", FINAL_SEEDS)
        if not np.array_equal(idx, e["idx"]):
            raise RuntimeError(f"{fam}: fresh-test window ids differ from the scorer's")
        preds[fam] = p[:, 0] if p.ndim == 2 else p
    R = tb.n_regions
    fold0 = np.zeros(R, np.int64)
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state(),
           "note": "post-hoc, after the lock: the same saved predictions sliced; not a second test", "job": "job1"}

    # network vs network (job 1 and job 4)
    sc_dev = read_json(SCORES / "scores.json")
    thr = sc_dev["job4_thresholds"]
    d1 = np.where(e["O"][:, -2], np.abs(e["V"][:, -1] - e["V"][:, -2]), np.nan)
    d3 = np.where(e["O"][:, -4], np.abs(e["V"][:, -1] - e["V"][:, -4]), np.nan)
    top = (np.nan_to_num(d1, nan=-1) >= thr["d1_q75"]) | (np.nan_to_num(d3, nan=-1) >= thr["d3_q75"])
    nets_only = {m: preds[m] for m in NETS}
    out["network_pairs"] = {}
    for name, sel in (("fresh_job1", np.ones(len(y), bool)), ("fresh_job4", top)):
        s = scoring.score_subset(name, y[sel], reg[sel], west[sel], {m: p[sel] for m, p in nets_only.items()},
                                 fold0, R, 1, PAIRS)
        out["network_pairs"][name] = {"n": s["n"], "auc": {m: s["models"][m]["all"]["auc_within"] for m in NETS},
                                      "paired": {k: {sd: v[sd]["dauc_within"] for sd in scoring.SIDES}
                                                 for k, v in s["paired"].items()}}

    sides = {"all": np.ones(len(y), bool), "west": west, "east": ~west}
    models = list(preds)
    # by year and side
    out["by_year"] = {}
    for sd, s in sides.items():
        for yr in (2025, 2026):
            m = s & (year == yr)
            auc = {}
            for k in models:
                u, n = _u_pairs(preds[k][m], y[m])
                auc[k] = u / n if n else None
            out["by_year"][f"{sd}_{yr}"] = {"n": int(m.sum()), "drought_rate": float(y[m].mean()),
                                            "mean_prediction": {k: float(preds[k][m].mean()) for k in models},
                                            "auc": auc}
    # within region (over time), with region-resampled intervals for network - LR
    rng = np.random.default_rng(SEED)
    out["within_region"] = {}
    for sd, s in sides.items():
        regs = np.unique(reg[s])
        stats = {k: _per_group(preds[k][s], y[s], reg[s]) for k in models}
        u = {k: np.array([stats[k][r][0] for r in regs]) for k in models}
        n = np.array([stats["lr_vci_t"][r][1] for r in regs], float)
        res = {}
        draws = [rng.integers(0, len(regs), len(regs)) for _ in range(N_RES)]
        for k in models:
            pt = u[k].sum() / n.sum()
            dif = [(u[k][b].sum() - u["lr_vci_t"][b].sum()) / n[b].sum() for b in draws]
            lo, hi = np.percentile(dif, [2.5, 97.5])
            res[k] = {"auc": float(pt), "minus_lr": [float(pt - u["lr_vci_t"].sum() / n.sum()), float(lo), float(hi)]}
        out["within_region"][sd] = {"regions": int(len(regs)), "pairs": int(n.sum()), "models": res}
    # within month (across places)
    out["within_month"] = {}
    for sd, s in sides.items():
        res = {}
        for k in models:
            g = _per_group(preds[k][s], y[s], T[s])
            res[k] = sum(v[0] for v in g.values()) / max(sum(v[1] for v in g.values()), 1)
        out["within_month"][sd] = res

    # development folds beside Phase 5
    import units
    c = units.ctx()
    base = c["base"]
    ph = read_json(SCORES / "posthoc.json")["jobs"]["job1"]["models"]
    planned = (sc_dev.get("subsets", {}).get("job1") or {}).get("models", {})
    folds = []
    for k in range(N_FOLDS):
        dd = units.prep("job1", k, "outer")
        ev = dd["ev"]
        yy = base.label0[ev]
        ww = base.region_west[base.w_region[ev]]
        yrs = sorted({MONTHS[t][:4] for t in base.w_T[ev]})
        auc = {}
        for m in ("lr_vci_t", "gbt_mono", "bilstm", "resbilstm", "gru", "bilstm_vci", "cnn_lr", "cnn_bilstm"):
            src = ph.get(m) or planned.get(m)
            if src and src.get("per_fold_auc"):
                auc[m] = float(src["per_fold_auc"][k])
        folds.append({"fold": k + 1, "validation_years": f"{yrs[0]}-{yrs[-1]}", "training_windows": int(len(dd["fit"])),
                      "validation_windows": int(len(ev)), "drought_rate": float(yy.mean()),
                      "drought_west": float(yy[ww].mean()), "drought_east": float(yy[~ww].mean()), "auc": auc})
    n_fit5 = read_json(RUNS / "phase5" / "composite" / "job1" / "bilstm" / "seed42" / "meta.json")["n_fit"]
    out["dev_folds"] = {"folds": folds, "phase5_training_windows": int(n_fit5),
                        "fresh_drought_rate": float(y.mean()),
                        "fresh_auc": {m: float(_u_pairs(preds[m], y)[0] / _u_pairs(preds[m], y)[1]) for m in models}}
    rc = read_json(LOGS / "status_check_2026-11-09" / "reload_check_phase5.json")
    if rc:
        rows = rc["rows"]
        out["reload_check"] = {"models": len(rows), "max_abs_logit_diff": max(r["max_abs_diff"] for r in rows),
                               "params_match": all(r["n_params"] == r["n_params_meta"] for r in rows),
                               "losses_fell": all("fell" in r["history"] for r in rows)}
    out["seconds"] = round(time.time() - t0, 1)
    write_json(SCORES / "fresh_posthoc.json", out)
    log(f"fresh_posthoc: wrote {SCORES / 'fresh_posthoc.json'} in {out['seconds']:.0f}s")
    return out


if __name__ == "__main__":
    run()
