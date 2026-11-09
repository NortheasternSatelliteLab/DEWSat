"""Learning curve for job 1 (DESIGN.md amendment 5): does the Bi-LSTM's gap to LR on VCI(T) close as the labelled
training data grow, and does it matter whether the extra data are more regions or more years?

Development data only: the outer folds' scored windows, labels, within-fold AUC and whole-region resamples as every
job-1 model. Each network keeps its fold's chosen setting (no re-tuning) and trains with SEEDS on a slice of the
fold's outer training windows; its prediction is the seeds' mean probability. LR on VCI(T) is refit on every slice.
The full-data point is the existing final runs' SEEDS (nothing retrained).

    years    folds 1 and 4 (five training years each): targets T in k of the five years, k = 2, 3, 4; three subsets
             per k from numpy default_rng(LC_SEED + fold index); all training regions
    regions  all four folds: a random 1/8, 1/4 or 1/2 of the fold's training regions, nested, drawn within West and
             within East (default_rng(LC_SEED + 100 + fold index)), at least one per side; all training years

The scaler comes from each slice's own training windows (DESIGN.md 1); the fold's VCI table and inputs are unchanged.
Runs: runs/lcurve/<model>/fold<k>/<slice>/seed<s>/ (pred.npz, meta.json, DONE). Scores: scores/lcurve.json.
DEWSAT_LC_MODELS (comma-separated) overrides the two networks; sandboxes use it to exercise every model's code path.
"""

from __future__ import annotations

import itertools
import os
import time

import numpy as np

from common import FINAL_SEEDS, MONTHS, N_FOLDS, REFS, RUNS, SCORES, git_state, read_json, save_npz, write_json

SEEDS = FINAL_SEEDS[:3]                       # 42, 43, 44 (sandbox: 42, 43)
LC_SEED = 20261109
YEAR_FOLDS = (0, 3)                           # folds 1 and 4: five training years each
YEAR_SIZES = (2, 3, 4)
N_YEARS_FULL = 5
N_YEAR_SUBSETS = 3
REGION_FRACS = (0.125, 0.25, 0.5)
FRAC_NAME = {0.125: "1of8", 0.25: "1of4", 0.5: "1of2"}
NEW_CANDIDATES = ("bilstm_vci", "cnn_bilstm")
YEAR = np.array([int(m[:4]) for m in MONTHS])
_SETS: dict = {}


def _sets(k: int):
    """(base, windows.Sets) for job 1, fold k (cached per process)."""
    import units
    from windows import Sets
    c = units.ctx()
    if k not in _SETS:
        _SETS[k] = Sets(c["base"], "job1", k, c["arrays"]["obs"][k])
    return c["base"], _SETS[k]


def train_years(k: int) -> list[int]:
    base, sets = _sets(k)
    return sorted(set(YEAR[base.w_T[sets.outer_train]].tolist()))


def slices_for(k: int) -> list[dict]:
    """Every slice of fold k, deterministic: axis, size, subset index j, kept years or regions, and a name."""
    base, sets = _sets(k)
    out = []
    if k in YEAR_FOLDS:
        years = train_years(k)
        if len(years) != N_YEARS_FULL:
            raise RuntimeError(f"fold {k + 1}: {len(years)} training years, expected {N_YEARS_FULL}")
        rng = np.random.default_rng(LC_SEED + k)
        for size in YEAR_SIZES:
            combos = list(itertools.combinations(years, size))
            pick = rng.choice(len(combos), N_YEAR_SUBSETS, replace=False)
            for j, ci in enumerate(pick.tolist()):
                ys = [int(y) for y in combos[ci]]
                out.append({"axis": "years", "k": k, "size": size, "j": j, "years": ys,
                            "name": f"years{size}_" + "-".join(str(y) for y in ys)})
    regs = np.unique(base.w_region[sets.outer_train])
    west = base.region_west[regs]
    rng = np.random.default_rng(LC_SEED + 100 + k)
    perm = {side: rng.permutation(regs[west if side == "west" else ~west]) for side in ("west", "east")}
    for frac in REGION_FRACS:
        keep = []
        for side in ("west", "east"):
            p = perm[side]
            if len(p):
                keep += p[:max(1, int(round(frac * len(p))))].tolist()
        out.append({"axis": "regions", "k": k, "size": frac, "j": 0, "regions": sorted(int(r) for r in keep),
                    "name": f"regions_{FRAC_NAME[frac]}"})
    return out


def fit_windows(sl: dict | None, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(training windows, scored windows) of a slice; sl None = the full outer training set."""
    base, sets = _sets(k)
    fit = sets.outer_train
    if sl is not None:
        if sl["axis"] == "years":
            fit = fit[np.isin(YEAR[base.w_T[fit]], sl["years"])]
        else:
            fit = fit[np.isin(base.w_region[fit], sl["regions"])]
    return fit, sets.outer_val


def prep_slice(sl: dict) -> dict:
    """units.prep('job1', k, 'outer') restricted to the slice's training windows (scaler from those windows)."""
    import units
    from windows import gather, input_cube, scaler_for
    c = units.ctx()
    key = ("lcurve", sl["k"], sl["name"])
    if key in c["prep"]:
        return c["prep"][key]
    c["prep"].clear()
    t0 = time.time()
    base, arrays = c["base"], c["arrays"]
    k = sl["k"]
    feats, obs = arrays["feats"][k], arrays["obs"][k]
    fit, ev = fit_windows(sl, k)
    if len(fit) == 0:
        raise RuntimeError(f"fold {k + 1} {sl['name']}: no training windows")
    scaler = scaler_for(base, feats, obs, fit)
    F = input_cube(feats, obs, scaler)
    d = {"k": k, "slice": sl["name"], "fit": fit, "ev": ev, "scaler": scaler}
    d["X_fit"], d["X_ev"] = gather(F, base, fit), gather(F, base, ev)
    d["Y_fit"], d["M_fit"] = units.labels(base, "job1", fit)
    d["Y_ev"], d["M_ev"] = units.labels(base, "job1", ev)
    d["hide_ok"] = np.r_[np.ones(11, bool), False]
    d["seconds"] = round(time.time() - t0, 1)
    c["prep"][key] = d
    return d


# --------------------------------------------------------------------------------------------- models and runs
def best_new_model() -> str:
    """The amendment-4 Bi-LSTM with the higher job-1 development within-fold AUC (scores/posthoc.json)."""
    sc = read_json(SCORES / "posthoc.json")
    if not sc or "job1" not in sc.get("jobs", {}):
        raise RuntimeError("scores/posthoc.json has no job-1 scores: the learning curve picks its second Bi-LSTM there")
    m = sc["jobs"]["job1"]["models"]
    cands = [x for x in NEW_CANDIDATES if x in m and m[x]["all"][0] is not None]
    if not cands:
        raise RuntimeError(f"none of {NEW_CANDIDATES} is scored in scores/posthoc.json")
    return max(cands, key=lambda x: m[x]["all"][0])


def models() -> list[str]:
    env = os.environ.get("DEWSAT_LC_MODELS")
    if env:
        return [x for x in env.split(",") if x]
    return ["bilstm", best_new_model()]


def cfg_for(model: str, k: int) -> dict | None:
    if model == "bilstm":
        import units
        return units.chosen_config("job1", "bilstm", k)
    import posthoc
    return posthoc.chosen_config(model, "job1", k)


def stage_specs() -> list[dict]:
    specs = []
    for model in models():
        for k in range(N_FOLDS):
            cfg = cfg_for(model, k)
            if cfg is None:
                raise RuntimeError(f"{model} fold {k + 1}: no chosen setting")
            for sl in slices_for(k):
                specs += [{"type": "lc", "job": "job1", "k": k, "cfg": cfg, "seed": s, "slice": sl} for s in SEEDS]
    return specs


def lc_dir(model: str, k: int, sl_name: str, seed: int):
    return RUNS / "lcurve" / model / f"fold{k + 1}" / sl_name / f"seed{seed}"


def full_dir(model: str, k: int, seed: int):
    import units
    if model == "bilstm":
        return units.final_dir("job1", "bilstm", k, seed)
    import posthoc
    return posthoc.final_dir(model, "job1", k, seed)


def spec_dir(spec: dict):
    return lc_dir(spec["cfg"]["family"], spec["k"], spec["slice"]["name"], spec["seed"])


def spec_name(spec: dict) -> str:
    return f"lcurve/{spec['cfg']['family']}/fold{spec['k'] + 1}/{spec['slice']['name']}/seed{spec['seed']}"


def device_for(cfg: dict) -> str:
    """Recurrent models where the pipeline places a bilstm (GPU from 128 recurrent units in total); CNN on the GPU."""
    if cfg["family"] in ("bilstm", "bilstm_vci"):
        import search
        return search.device_for({**cfg, "family": "bilstm"})
    return "mps"


def data_fraction(sl: dict) -> float:
    return sl["size"] / N_YEARS_FULL if sl["axis"] == "years" else float(sl["size"])


def run_unit(spec: dict, device: str, threads: int) -> dict:
    """One network on one slice and seed, trained exactly as its final runs (units.run_net / posthoc.run_unit)."""
    import posthoc
    import simple
    import units
    out_dir = spec_dir(spec)
    if (out_dir / "DONE").exists():
        return read_json(out_dir / "meta.json")
    sl, cfg, seed, k = spec["slice"], spec["cfg"], spec["seed"], spec["k"]
    d = prep_slice(sl)
    t0 = time.time()
    fam = cfg["family"]
    if fam in ("bilstm", "bilstm_vci"):
        import nets
        Xf, Xe, net_cfg = d["X_fit"], d["X_ev"], dict(cfg)
        if fam == "bilstm_vci":
            Xf, Xe = Xf.copy(), Xe.copy()
            Xf[:, :, posthoc.ZERO_CH] = 0.0
            Xe[:, :, posthoc.ZERO_CH] = 0.0
            net_cfg["family"] = "bilstm"
        data = {"X_fit": Xf, "Y_fit": d["Y_fit"], "M_fit": d["M_fit"], "hide_ok": d["hide_ok"],
                "X_eval": [("ev", Xe, None)]}
        res = nets.train(net_cfg, data, seed, device=device, threads=threads)
    elif fam == "cnn_bilstm":
        import cnn
        base = units.ctx()["base"]
        tiles = posthoc.tiles_for(REFS[k])
        fit = cnn.Batches(tiles, base.w_cell[d["fit"]], base.w_T[d["fit"]], d["X_fit"], 12)
        ev = cnn.Batches(tiles, base.w_cell[d["ev"]], base.w_T[d["ev"]], d["X_ev"], 12)
        res = cnn.train(cfg, fit, d["Y_fit"], d["M_fit"], seed, device=device, threads=threads, evals=[("ev", ev)])
    else:
        raise ValueError(f"learning curve: no runner for {fam}")
    z = res["logits"]["ev"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npz(out_dir / "pred.npz", wid=d["ev"].astype(np.int64), logit=z.astype(np.float32))
    meta = {"job": "job1", "family": fam, "fold": k + 1, "slice": sl, "data_fraction": data_fraction(sl), "cfg": cfg,
            "seed": seed, "device": device, "threads": threads, "git": git_state(), "n_params": res["n_params"],
            "seconds": res["seconds"], "unit_seconds": round(time.time() - t0, 1), "prep_seconds": d["seconds"],
            "history": res["history"], "metric": {"metric": simple.auc(z[:, 0], d["Y_ev"][:, 0])},
            "scaler": d["scaler"], "n_fit": int(len(d["fit"])), "n_ev": int(len(d["ev"])),
            "fit_regions": int(len(np.unique(units.ctx()["base"].w_region[d["fit"]]))), "pid": os.getpid()}
    write_json(out_dir / "meta.json", meta)
    (out_dir / "DONE").write_text(meta["git"]["commit"] + "\n")
    return meta


# --------------------------------------------------------------------------------------------- scoring
def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def seed_mean(model: str, k: int, sl: dict | None, wid: np.ndarray) -> np.ndarray:
    ps = []
    for s in SEEDS:
        d = full_dir(model, k, s) if sl is None else lc_dir(model, k, sl["name"], s)
        if not (d / "DONE").exists():
            raise RuntimeError(f"{d} is not finished")
        with np.load(d / "pred.npz") as z:
            if not np.array_equal(z["wid"], wid):
                raise RuntimeError(f"{d}: window ids differ from fold {k + 1}'s scored windows")
            ps.append(_sig(z["logit"][:, 0].astype(np.float64)))
    return np.mean(ps, axis=0)


def lr_fit_predict(sl: dict | None, k: int) -> tuple[np.ndarray, float, int]:
    """LR on VCI(T) fit on the slice's training windows: (scored-window probabilities, slope per VCI point, n fit)."""
    import simple
    import units
    c = units.ctx()
    base, feats = c["base"], c["arrays"]["feats"][k]
    fit, ev = fit_windows(sl, k)

    def x(idx):
        return feats[base.w_cell[idx], base.w_T[idx], 2][:, None].astype(np.float64)
    m = simple.LR(["vci_t"]).fit(x(fit), base.label0[fit])
    return m.predict(x(ev)), float(m.raw_coefficients()[1][0]), int(len(fit))


def _slice_of(k: int, axis: str, size, j: int) -> dict | None:
    full = (axis == "years" and size == N_YEARS_FULL) or (axis == "regions" and size == 1.0)
    if full:
        return None
    hits = [s for s in slices_for(k) if s["axis"] == axis and s["size"] == size and s["j"] == j]
    if len(hits) != 1:
        raise RuntimeError(f"fold {k + 1}: no unique {axis} slice of size {size}, subset {j}")
    return hits[0]


def score() -> dict:
    """scores/lcurve.json: per axis and size, each model's within-fold AUC and network minus LR, with intervals."""
    import scoring
    import units
    c = units.ctx()
    base = c["base"]
    R, fold_r = base.n_regions, base.region_group
    nets_ = models()
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state(), "networks": nets_,
           "seeds": list(SEEDS), "note": "DESIGN.md amendment 5; development data only; no pass/fail", "axes": {},
           "slices": {f"fold{k + 1}": slices_for(k) for k in range(N_FOLDS)}}
    for axis, folds in (("years", YEAR_FOLDS), ("regions", tuple(range(N_FOLDS)))):
        evs = {k: _sets(k)[1].outer_val for k in folds}
        y = np.concatenate([base.label0[evs[k]] for k in folds]).astype(np.int64)
        reg = np.concatenate([base.w_region[evs[k]] for k in folds])
        west = base.region_west[reg]
        Cs = {sd: scoring.side_counts(reg, west, sd, R, f"lcurve_{axis}|{sd}") for sd in scoring.SIDES}
        sizes = list(YEAR_SIZES) + [N_YEARS_FULL] if axis == "years" else list(REGION_FRACS) + [1.0]
        vecs, rows = {}, []
        for size in sizes:
            n_j = N_YEAR_SUBSETS if (axis == "years" and size != N_YEARS_FULL) else 1
            acc = {m: {sd: [] for sd in scoring.SIDES} for m in nets_ + ["lr_vci_t"]}
            per_fold, lr_info, n_fit = {m: [] for m in nets_ + ["lr_vci_t"]}, [], []
            for j in range(n_j):
                preds = {m: [] for m in nets_ + ["lr_vci_t"]}
                for k in folds:
                    sl = _slice_of(k, axis, size, j)
                    wid = evs[k]
                    for m in nets_:
                        preds[m].append(seed_mean(m, k, sl, wid))
                    p, slope, nf = lr_fit_predict(sl, k)
                    preds["lr_vci_t"].append(p)
                    lr_info.append({"fold": k + 1, "slice": None if sl is None else sl["name"], "slope": slope,
                                    "n_fit": nf})
                    n_fit.append(nf)
                for m, ps in preds.items():
                    pm = np.concatenate(ps)
                    W, P, N = scoring.win_matrix(pm, y, reg, R)
                    per_fold[m].append(scoring.per_fold_auc(W, P, N, fold_r, N_FOLDS))
                    for sd in scoring.SIDES:
                        acc[m][sd].append(scoring.auc_within_from(W, P, N, Cs[sd], fold_r))
            mean = {m: {sd: np.mean(v[sd], axis=0) for sd in scoring.SIDES} for m, v in acc.items()}
            vecs[size] = mean
            row = {"size": size, "subsets": n_j, "data_fraction": size / N_YEARS_FULL if axis == "years" else size,
                   "mean_training_windows": float(np.mean(n_fit)),
                   "auc_within": {m: {sd: scoring.ci(mean[m][sd]) for sd in scoring.SIDES} for m in mean},
                   "network_minus_lr": {m: {sd: scoring.ci(mean[m][sd] - mean["lr_vci_t"][sd])
                                            for sd in scoring.SIDES} for m in nets_},
                   "per_fold_auc": per_fold, "lr_fits": lr_info}
            rows.append(row)
        change = {}
        for m in nets_:
            dif = {s: vecs[s][m]["all"] - vecs[s]["lr_vci_t"]["all"] for s in sizes}
            change[m] = {"smallest_to_full": scoring.ci(dif[sizes[-1]] - dif[sizes[0]]),
                         "second_largest_to_full": scoring.ci(dif[sizes[-1]] - dif[sizes[-2]])}
        out["axes"][axis] = {"folds": [k + 1 for k in folds], "n": int(len(y)), "n_pos": int(y.sum()),
                             "regions": int(len(np.unique(reg))), "sizes": rows, "change_in_difference": change}
    write_json(SCORES / "lcurve.json", out)
    return out
