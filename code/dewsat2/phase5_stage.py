"""Phase 5 (DESIGN.md 11): refit on all development data, score the fresh test once, realistic sampling.

The fresh-test labels are read only here, after `check_lock()`; the scorer refuses to run if
data/locked/FRESH_TEST_SCORED.lock exists and writes it after scoring.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np

import nets
import scoring
import search
import simple
import units
import views
from common import (DEV_END, FRESH_END, FRESH_START, LEADS, LOCKED, MI, N_FOLDS, N_MONTHS, RUNS, SCORES, SEQ,
                    FINAL_SEEDS, git_state, log, read_json, save_npz, write_json)
from features import labels_for, reference_mask
from windows import (Base, baseline_features, gather, input_cube, obs_stats, raw_windows, scaler_for)

LOCK = LOCKED / "FRESH_TEST_SCORED.lock"
REF_ALL = 4                       # REFS index of the all-development-months table
REVISITS = (7, 16, 32)
MIN_TRAIN = 200                    # a thinned training set smaller than this (or one-class) is skipped and logged
STARTS = 3
THIN_SEEDS = (42, 43, 44)
THIN_JOBS = ("job1", "job2")
_P5: dict = {}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_lock() -> dict:
    if LOCK.exists():
        raise SystemExit(f"{LOCK} exists: the fresh test was already scored; refusing to score it again")
    lock = json.loads((LOCKED / "LOCK.json").read_text())
    bad = [n for n, h in lock["files"].items() if sha256(LOCKED / "fldas_test" / f"{n}.npz") != h]
    if bad:
        raise SystemExit(f"fresh-test label files changed since they were locked: {bad[:5]}")
    return lock


# ------------------------------------------------------------------------------------------- chosen settings
def chosen_settings() -> dict:
    """Per job and family, the configuration with the highest mean inner-validation metric over the 4 folds; per
    job the boosted trees' configuration and smoothing the same way (DESIGN.md 11)."""
    out = {}
    for job in units.JOBS:
        out[job] = {"nets": {}, "simple": {}}
        from common import FAMILIES
        for fam in FAMILIES:
            cfgs = search.net_configs(job, fam)
            scores = []
            for cfg in cfgs:
                v = []
                for k in range(N_FOLDS):
                    m = read_json(units.search_dir(job, fam, k, cfg["id"]) / "meta.json")
                    v.append(np.nan if m is None or m["metric"]["metric"] is None else m["metric"]["metric"])
                scores.append(float(np.nanmean(v)) if np.isfinite(v).any() else -np.inf)
            i = int(np.argmax(scores))
            out[job]["nets"][fam] = {"cfg": cfgs[i], "mean_inner": scores[i], "all_means": scores}
        g, a = [], []
        for k in range(N_FOLDS):
            m = read_json(units.simple_dir(job, k) / "meta.json")
            g.append(np.nanmean(np.array(m["inner"]["gbt_auc"], float), axis=0))
            a.append(np.nanmean(np.array(m["inner"]["alpha_auc"], float), axis=0))
        gi = int(np.nanargmax(np.nanmean(g, axis=0)))
        ai = int(np.nanargmax(np.nanmean(a, axis=0)))
        out[job]["simple"] = {"gbt": search.gbt_configs(job)[gi], "alpha": simple.ALPHAS[ai],
                              "gbt_mean_inner": float(np.nanmean(g, axis=0)[gi]),
                              "alpha_mean_inner": float(np.nanmean(a, axis=0)[ai])}
    return out


def phase5_models() -> dict:
    """Per job: the gate's best network and best simple model, the best Bi-LSTM variant, and the job's LR."""
    sc = read_json(SCORES / "scores.json")
    out = {}
    for job, key in (("job1", "job1"), ("job2", "job2"), ("job3", "job3")):
        s = sc["selection"][key]
        lr = "lr_latest_age" if job == "job2" else "lr_vci_t"
        out[job] = {"nets": list(dict.fromkeys([s["best_network"], s["best_bilstm"]])), "best_simple": s["best_simple"],
                    "lr": lr}
    s4 = sc["selection"].get("job4")
    out["job4"] = {"nets": list(dict.fromkeys([s4["best_network"], s4["best_bilstm"]])) if s4 else [],
                   "best_simple": s4["best_simple"] if s4 else None, "lr": "lr_vci_t"}
    # job 4 is scored with job-1 models, so its chosen networks are refit with job 1's
    out["job1"]["nets"] = list(dict.fromkeys(out["job1"]["nets"] + out["job4"]["nets"]))
    return out


# ------------------------------------------------------------------------------------------- data
def ctx5() -> dict:
    if "dev" in _P5:
        return _P5
    import dataset
    meta, dev = dataset.load("dev")
    _, test = dataset.load("test")
    _P5["meta"], _P5["dev"], _P5["test"] = meta, dev, test
    _P5["base_dev"] = Base(meta, dev, "dev")
    tb = Base(meta, test, "test", t_last=N_MONTHS - 1)
    _P5["base_test"] = tb
    return _P5


def test_labels() -> tuple[np.ndarray, np.ndarray]:
    """Fresh-test drought flags (cells, months), the paper's rule, from the locked files (Phase 5 only)."""
    c = ctx5()
    if "drought_test" in c:
        return c["drought_test"], c["keep_test"]
    import features as F
    meta, test = c["meta"], c["test"]
    names = meta["test"]["regions"]
    ids = test["cell_id"]
    region = test["region"]
    drought = np.full((len(ids), N_MONTHS), -1, np.int8)
    keep = np.zeros(len(ids), bool)
    old = F.FLDAS_DEV
    try:
        F.FLDAS_DEV = LOCKED / "fldas_test"
        for r, name in enumerate(names):
            rows = np.where(region == r)[0]
            centers = [(float(s.split(":")[1]), float(s.split(":")[2])) for s in ids[rows]]
            d, kp = labels_for(name, len(rows), centers)
            drought[rows], keep[rows] = d, kp
    finally:
        F.FLDAS_DEV = old
    c["drought_test"], c["keep_test"] = drought, keep
    return drought, keep


def dev_all(job: str, feats=None, obs=None, tag="composite", with_X: bool = True) -> dict:
    """All development training windows for one job, with the all-development VCI table (or given inputs)."""
    c = ctx5()
    key = ("dev", job, tag)
    if key in _P5:
        return _P5[key]
    base, dev = c["base_dev"], c["dev"]
    feats = dev["feats"][REF_ALL] if feats is None else feats
    obs = dev["obs"][REF_ALL] if obs is None else obs
    t_obs, n_prev = obs_stats(obs, base)
    T = base.w_T
    has0 = base.label0 >= 0
    if job == "job1":
        ok = t_obs & has0
    elif job == "job2":
        ok = (n_prev >= 8) & has0
    else:
        ok = t_obs & has0 & (base.label_h >= 0).all(axis=1) & (T <= DEV_END - max(LEADS))
    fit = np.where(ok)[0]
    hide = job == "job2"
    scaler = scaler_for(base, feats, obs, fit)
    F = input_cube(feats, obs, scaler)
    d = {"fit": fit, "scaler": scaler, "hide": hide}
    if with_X:
        d["X_fit"] = gather(F, base, fit, hide)
    d["V_fit"], d["O_fit"] = raw_windows(feats, obs, base, fit, hide)
    d["Y_fit"], d["M_fit"] = units.labels(base, job, fit)
    d["f_fit"] = baseline_features(d["V_fit"], d["O_fit"], base.w_T[fit])
    _P5[key] = d
    return d


def test_windows(job: str, feats, obs, scaler: dict, drought, keep, thresholds=None) -> dict:
    """Fresh-test windows (T in 2025-01..2026-09) for one job with the given inputs and the training scaler."""
    c = ctx5()
    tb = c["base_test"]
    t_obs, n_prev = obs_stats(obs, tb)
    T = tb.w_T
    lab0 = drought[tb.w_cell, T]
    in_t = (T >= FRESH_START) & (T <= FRESH_END) & keep[tb.w_cell] & (lab0 >= 0)
    if job in ("job1", "job4"):
        ok = in_t & t_obs
    elif job == "job2":
        ok = in_t & ~t_obs & (n_prev >= 8)
    else:
        ok = in_t & t_obs & (T + max(LEADS) <= FRESH_END)
    idx = np.where(ok)[0]
    hide = job == "job2"
    F = input_cube(feats, obs, scaler)
    d = {"idx": idx, "X": gather(F, tb, idx, hide)}
    d["V"], d["O"] = raw_windows(feats, obs, tb, idx, hide)
    d["f"] = baseline_features(d["V"], d["O"], T[idx])
    d["y0"] = lab0[idx].astype(np.int64)
    if job == "job3":
        d["yh"] = np.stack([drought[tb.w_cell[idx], T[idx] + h] for h in LEADS], axis=1).astype(np.int64)
    if job == "job4" and thresholds:
        d1 = np.where(d["O"][:, -2], np.abs(d["V"][:, -1] - d["V"][:, -2]), np.nan)
        d3 = np.where(d["O"][:, -4], np.abs(d["V"][:, -1] - d["V"][:, -4]), np.nan)
        top = (np.nan_to_num(d1, nan=-1) >= thresholds["d1_q75"]) | (np.nan_to_num(d3, nan=-1) >= thresholds["d3_q75"])
        d["sel"] = top
    d["region"] = tb.w_region[idx]
    d["west"] = tb.region_west[d["region"]]
    return d


# ------------------------------------------------------------------------------------------- thinning
def pass_days(n_regions: int, N: int, rep: int, kind: str) -> list[np.ndarray]:
    """One pass every N days from a random start in [0, N) days after 2016-01-01, per region (seeded)."""
    rng = np.random.default_rng([20261109, N, rep, 0 if kind == "test" else 1])
    starts = rng.integers(0, N, size=n_regions)
    first = int(views.MONTH_START_DAY[0])
    last = int(views.MONTH_END_DAY[-1])
    return [np.arange(first + s, last + 1, N) for s in starts]


def thinned_inputs(part: str, N: int, rep: int):
    """Monthly single-view features (cells, months, 5) and observed (cells, months) for one revisit interval and
    start: the latest pass-day view with >= 30% of the cell clear in each month; VCI_mean and VCI_p10 against the
    per-cell single-view table from development years (leave-one-year-out for development months)."""
    c = ctx5()
    key = ("thin", part, N, rep)
    if key in _P5:
        return _P5[key]
    meta, arr = c["meta"], c[part]
    names = meta[part]["regions"]
    n = len(arr["cell_id"])
    feats = np.full((n, N_MONTHS, 5), np.nan, np.float32)
    obs = np.zeros((n, N_MONTHS), bool)
    days = pass_days(len(names), N, rep, part)
    ref = reference_mask("all")
    region = arr["region"]
    for r, name in enumerate(names):
        rows = np.where(region == r)[0]
        if not arr["has_views"][rows].any():
            continue
        rv = views.region_views(name)
        pos = {s: i for i, s in enumerate(rv["ids"])}
        for row in rows:
            cv_all = views.cell_views(rv, pos[str(arr["cell_id"][row])])
            if cv_all is None:
                continue
            vci_all = views.single_view_vci(cv_all, ref)           # table from every clear view in dev years
            p10_all = views.single_view_vci(cv_all, ref, "p10")
            seen = np.isin(cv_all["day"], days[r])
            if not seen.any():
                continue
            m = cv_all["month"][seen]
            # latest seen view per month (views are sorted by day)
            last_idx = {}
            for j, mm in zip(np.where(seen)[0], m):
                last_idx[int(mm)] = j
            for mm, j in last_idx.items():
                if not np.isfinite(vci_all[j]):
                    continue
                feats[row, mm] = (cv_all["ndvi"][j], cv_all["std"][j], vci_all[j],
                                  p10_all[j] if np.isfinite(p10_all[j]) else vci_all[j], cv_all["clear"][j])
                obs[row, mm] = True
    _P5[key] = (feats, obs)
    return feats, obs


# ------------------------------------------------------------------------------------------- units
def unit_dir(spec):
    return RUNS / "phase5" / spec["train"] / spec["job"] / spec["cfg"]["family"] / f"seed{spec['seed']}"


def eval_sets(job: str, scaler: dict, train: str) -> list[tuple[str, dict]]:
    """Evaluation windows for a Phase 5 model: the fresh test with composites (composite-trained models) and the
    thinned fresh test for each revisit interval and start (jobs 1-2)."""
    c = ctx5()
    drought, keep = test_labels()
    out = []
    if train == "composite":
        out.append(("fresh", test_windows(job, c["test"]["feats"][REF_ALL], c["test"]["obs"][REF_ALL], scaler,
                                          drought, keep)))
    if job in THIN_JOBS:
        Ns = REVISITS if train == "composite" else (int(train[4:]),)
        for N in Ns:
            for rep in range(STARTS):
                f, o = thinned_inputs("test", N, rep)
                out.append((f"thin{N}_s{rep}", test_windows(job, f, o, scaler, drought, keep)))
    return out


def train_set(job: str, train: str) -> dict:
    if train == "composite":
        return dev_all(job)
    N = int(train[4:])
    c = ctx5()
    parts = []
    for rep in range(STARTS):
        f, o = thinned_inputs("dev", N, rep)
        parts.append((f, o))
    # pool the 3 starts: stack as three copies of the cells (same labels)
    key = ("devthin", job, N)
    if key in _P5:
        return _P5[key]
    base = c["base_dev"]
    ds = [dev_all(job, f, o, tag=f"thin{N}_s{r}", with_X=False) for r, (f, o) in enumerate(parts)]
    # one scaler for the pooled set: refit on the pooled training windows' observed cell-months
    vals = []
    for (f, o), dd in zip(parts, ds):
        diff = np.zeros((base.n_cells, N_MONTHS + 1), np.int64)
        np.add.at(diff, (base.w_cell[dd["fit"]], np.maximum(base.w_T[dd["fit"]] - 11, 0)), 1)
        np.add.at(diff, (base.w_cell[dd["fit"]], base.w_T[dd["fit"]] + 1), -1)
        used = (np.cumsum(diff, axis=1)[:, :N_MONTHS] > 0) & o
        vals.append(f[used][:, :4].astype(np.float64))
    v = np.concatenate(vals)
    scaler = {"center": v.mean(axis=0).tolist(), "spread": np.maximum(v.std(axis=0), 1e-6).tolist(), "rows": len(v)}
    X, Vs, Os, Ys, Ms, fs = [], [], [], [], [], []
    for (f, o), dd in zip(parts, ds):
        F = input_cube(f, o, scaler)
        X.append(gather(F, base, dd["fit"], dd["hide"]))
        Vs.append(dd["V_fit"]); Os.append(dd["O_fit"]); Ys.append(dd["Y_fit"]); Ms.append(dd["M_fit"])
    d = {"X_fit": np.concatenate(X), "V_fit": np.concatenate(Vs), "O_fit": np.concatenate(Os),
         "Y_fit": np.concatenate(Ys), "M_fit": np.concatenate(Ms), "scaler": scaler,
         "fit": np.concatenate([dd["fit"] for dd in ds])}
    d["f_fit"] = baseline_features(d["V_fit"], d["O_fit"], base.w_T[d["fit"]])
    _P5[key] = d
    return d


def run_unit(spec: dict, device: str, threads: int) -> dict:
    d_out = unit_dir(spec)
    if (d_out / "DONE").exists():
        return read_json(d_out / "meta.json")
    job, train, cfg = spec["job"], spec["train"], spec["cfg"]
    d = train_set(job, train)
    y0 = d["Y_fit"][d["M_fit"][:, 0], 0] if len(d["Y_fit"]) else np.zeros(0)
    if len(y0) < MIN_TRAIN or y0.min() == y0.max():
        d_out.mkdir(parents=True, exist_ok=True)
        meta = {"spec": spec, "skipped": f"too few training windows ({len(y0)}) or one class", "git": git_state()}
        write_json(d_out / "meta.json", meta)
        (d_out / "DONE").write_text("skipped\n")
        return meta
    evs = eval_sets(job, d["scaler"], train)
    data = {"X_fit": d["X_fit"], "Y_fit": d["Y_fit"], "M_fit": d["M_fit"], "hide_ok": np.r_[np.ones(11, bool), False]}
    ev_list = []
    if cfg["family"] == "resbilstm":
        heads = len(LEADS) if job == "job3" else 1
        if job == "job2":
            co = simple.lr_coefs(d["f_fit"], d["Y_fit"][:, 0], False)["lr_latest_age"]
            data["base_coef"] = {"a": [co["a"]], "b": [co["b"]], "c": [co["c"]], "a0": [co["a0"]]}
            data["V_fit"] = np.nan_to_num(d["V_fit"])
            for nm, e in evs:
                z = np.where(e["f"]["has"], co["a"] + co["b"] * np.nan_to_num(e["f"]["vci_latest"]) +
                             co["c"] * np.nan_to_num(e["f"]["age"]), co["a0"])
                ev_list.append((nm, e["X"], z[:, None].astype(np.float32)))
        else:
            ab = [simple.lr_coefs(d["f_fit"], d["Y_fit"][:, j], True)["lr_vci_t"] for j in range(heads)]
            a = np.array([x["a"] for x in ab], np.float32)
            b = np.array([x["b"] for x in ab], np.float32)
            data["base_fit"] = (a[None] + b[None] * np.nan_to_num(d["V_fit"][:, -1])[:, None]).astype(np.float32)
            for nm, e in evs:
                ev_list.append((nm, e["X"], (a[None] + b[None] * np.nan_to_num(e["V"][:, -1])[:, None]).astype(np.float32)))
    else:
        ev_list = [(nm, e["X"], None) for nm, e in evs]
    data["X_eval"] = ev_list
    res = nets.train(cfg, data, spec["seed"], device=device, threads=threads)
    d_out.mkdir(parents=True, exist_ok=True)
    save_npz(d_out / "pred.npz", **{f"logit_{nm}": z.astype(np.float32) for nm, z in res["logits"].items()},
             **{f"idx_{nm}": e["idx"].astype(np.int64) for nm, e in evs})
    meta = {"spec": spec, "device": device, "git": git_state(), "seconds": res["seconds"], "n_fit": int(len(d["X_fit"])),
            "evals": {nm: int(len(e["idx"])) for nm, e in evs}}
    write_json(d_out / "meta.json", meta)
    (d_out / "DONE").write_text(meta["git"]["commit"] + "\n")
    return meta


# ------------------------------------------------------------------------------------------- run and score
def simple_preds(job: str, train: str, settings: dict) -> dict:
    """Every simple model, trained on the given set, predicted on each evaluation set."""
    d = train_set(job, train)
    y0 = d["Y_fit"][d["M_fit"][:, 0], 0] if len(d["Y_fit"]) else np.zeros(0)
    if len(y0) < MIN_TRAIN or y0.min() == y0.max():
        return None, []
    evs = eval_sets(job, d["scaler"], train)
    heads = range(len(LEADS)) if job == "job3" else range(1)
    out = {nm: {} for nm, _ in evs}
    for j in heads:
        for nm, e in evs:
            o, _ = simple.fit_predict(d["f_fit"], d["V_fit"], d["O_fit"], d["Y_fit"][:, j], e["f"], e["V"], e["O"],
                                      settings["gbt"], settings["alpha"], job != "job2")
            for m, p in o.items():
                out[nm].setdefault(m, []).append(p)
    return {nm: {m: np.stack(v, axis=1) for m, v in mm.items()} for nm, mm in out.items()}, evs


def run(status=None) -> dict:
    import run as runner
    lock = check_lock()
    t0 = time.time()
    st = chosen_settings()
    models = phase5_models()
    write_json(SCORES / "phase5_settings.json", {"settings": st, "models": models})
    specs = []
    for job in units.JOBS:
        for fam in models[job]["nets"]:
            cfg = st[job]["nets"][fam]["cfg"]
            for s in FINAL_SEEDS:
                specs.append({"type": "p5", "job": job, "train": "composite", "cfg": cfg, "seed": s})
        if job in THIN_JOBS:
            fam = models[job]["nets"][0]
            cfg = st[job]["nets"][fam]["cfg"]
            for N in REVISITS:
                for s in THIN_SEEDS:
                    specs.append({"type": "p5", "job": job, "train": f"thin{N}", "cfg": cfg, "seed": s})
    # build the shared inputs once in this process (workers rebuild their own copies)
    runner.run_specs("phase5", specs, n_cpu=2, n_mps=2, threads=4, status=status)
    res = score_all(st, models, lock)
    res["seconds"] = round(time.time() - t0, 1)
    write_json(SCORES / "phase5.json", res)
    LOCK.write_text(json.dumps({"scored": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state()}) + "\n")
    log(f"fresh test scored once; lock written: {LOCK}")
    return res


def _net_probs(job, train, fam, name, seeds):
    ps, idx = [], None
    for s in seeds:
        p = unit_dir({"job": job, "train": train, "cfg": {"family": fam}, "seed": s}) / "pred.npz"
        if not p.exists():
            return None, None
        with np.load(p) as z:
            ps.append(1 / (1 + np.exp(-z[f"logit_{name}"].astype(float))))
            idx = z[f"idx_{name}"]
    return np.mean(ps, axis=0), idx


def score_all(st: dict, models: dict, lock: dict) -> dict:
    c = ctx5()
    tb = c["base_test"]
    R = tb.n_regions
    fold0 = np.zeros(R, np.int64)
    sc_dev = read_json(SCORES / "scores.json")
    out = {"git": git_state(), "lock": {"files": len(lock["files"])}, "fresh": {}, "realistic": {}}
    for job in units.JOBS:
        sp, evs = simple_preds(job, "composite", st[job]["simple"])
        for nm, e in evs:
            preds = {m: p for m, p in sp[nm].items()}
            for fam in models[job]["nets"]:
                p, idx = _net_probs(job, "composite", fam, nm, FINAL_SEEDS)
                assert np.array_equal(idx, e["idx"])
                preds[fam] = p
            if nm == "fresh":
                out["fresh"].update(score_fresh(job, e, preds, models, R, fold0, sc_dev))
            else:
                out["realistic"].setdefault(job, {}).setdefault("composite_trained", {})[nm] = \
                    score_simple_set(f"{job}_{nm}", e, preds, R, fold0, models[job])
        if job in THIN_JOBS:
            for N in REVISITS:
                sp2, evs2 = simple_preds(job, f"thin{N}", st[job]["simple"])
                if sp2 is None:
                    out["realistic"].setdefault(job, {}).setdefault("thin_trained_skipped", []).append(
                        f"thin{N}: too few training windows")
                    continue
                for nm, e in evs2:
                    preds = {m: p for m, p in sp2[nm].items()}
                    fam = models[job]["nets"][0]
                    p, idx = _net_probs(job, f"thin{N}", fam, nm, THIN_SEEDS)
                    if p is not None:
                        preds[fam] = p
                    out["realistic"].setdefault(job, {}).setdefault("thin_trained", {})[nm] = \
                        score_simple_set(f"{job}_{nm}_thin", e, preds, R, fold0, models[job])
    out["coverage"] = coverage()
    return out


def score_simple_set(name, e, preds, R, fold0, mj) -> dict:
    y = e["y0"]
    pr = {m: (p[:, 0] if p.ndim == 2 else p) for m, p in preds.items()}
    pr = {m: p for m, p in pr.items() if np.isfinite(p).all()}
    comps = [c for c in dict.fromkeys([mj["best_simple"], mj["lr"]]) if c in pr]
    pairs = [(m, cc) for cc in comps for m in pr if m != cc]
    s = scoring.score_subset(name, y, e["region"], e["west"], pr, fold0, R, 1, pairs)
    s["n_windows"] = int(len(y))
    return s


def score_fresh(job, e, preds, models, R, fold0, sc_dev) -> dict:
    out = {}
    mj = models[job]
    if job in ("job1", "job2"):
        out[job] = score_simple_set(f"fresh_{job}", e, preds, R, fold0, mj)
        if job == "job1":
            e4 = dict(e)
            thr = sc_dev["job4_thresholds"]
            d1 = np.where(e["O"][:, -2], np.abs(e["V"][:, -1] - e["V"][:, -2]), np.nan)
            d3 = np.where(e["O"][:, -4], np.abs(e["V"][:, -1] - e["V"][:, -4]), np.nan)
            top = (np.nan_to_num(d1, nan=-1) >= thr["d1_q75"]) | (np.nan_to_num(d3, nan=-1) >= thr["d3_q75"])
            e4 = {"y0": e["y0"][top], "region": e["region"][top], "west": e["west"][top]}
            p4 = {m: (p[top, 0] if p.ndim == 2 else p[top]) for m, p in preds.items()}
            out["job4"] = score_simple_set("fresh_job4", e4, p4, R, fold0, models["job4"])
        return out
    # job 3: per lead, all months and onsets, persistence reference; mean onset metric
    lab0 = e["y0"]
    subsets = []
    for j, h in enumerate(LEADS):
        yh = e["yh"][:, j]
        for kind, m in (("all", yh >= 0), ("onset", (yh >= 0) & (lab0 == 0))):
            pr = {mm: p[m, j] for mm, p in preds.items() if p.ndim == 2}
            pr["persistence"] = lab0[m].astype(float)
            pr = {mm: p for mm, p in pr.items() if np.isfinite(p).all()}
            sub = {"y": yh[m], "region": e["region"][m], "west": e["west"][m], "preds": pr}
            comps = [cc for cc in dict.fromkeys([mj["best_simple"], mj["lr"]]) if cc in pr]
            out[f"job3_lead{h}_{kind}"] = scoring.score_subset(f"fresh_job3_{h}_{kind}", sub["y"], sub["region"],
                                                               sub["west"], pr, fold0, R, 1,
                                                               [(a, b) for b in comps for a in pr if a != b])
            if kind == "onset":
                subsets.append(sub)
    mods = sorted(set.intersection(*[set(s["preds"]) for s in subsets]))
    mv = scoring.mean_over_leads("fresh_job3_mean_onset", subsets, mods, fold0, R)
    ent = {"models": {m: {sd: scoring.ci(v[sd]) for sd in scoring.SIDES} for m, v in mv.items()}, "paired": {}}
    for b in dict.fromkeys([mj["best_simple"], mj["lr"]]):
        if b in mv:
            for a in mv:
                if a != b:
                    ent["paired"][f"{a} - {b}"] = {sd: scoring.ci(mv[a][sd] - mv[b][sd]) for sd in scoring.SIDES}
    out["job3_mean_onset"] = ent
    return out


def coverage() -> dict:
    """Share of fresh-test cell-months (2025-01..2026-09, labelled) with a fresh view, full record and per interval."""
    c = ctx5()
    drought, keep = test_labels()
    tm = np.arange(FRESH_START, FRESH_END + 1)
    lab = (drought[:, tm] >= 0) & keep[:, None]
    hv = c["test"]["has_views"]
    full = c["test"]["obs"][REF_ALL][:, tm]
    out = {"total_cell_months": int(lab.sum()), "full_record_composite": float(full[lab].mean())}
    for N in REVISITS:
        sh = []
        for rep in range(STARTS):
            _, o = thinned_inputs("test", N, rep)
            sh.append(float(o[:, tm][lab & hv[:, None]].mean()))
        out[f"N{N}"] = {"share_by_start": sh, "share_mean": float(np.mean(sh))}
    return out


def _unused():
    return MI, SEQ
