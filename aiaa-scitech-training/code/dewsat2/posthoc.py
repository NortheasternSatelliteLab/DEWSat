"""Units, configurations and development scoring for the models added after the development results
(DESIGN.md amendments 4 and 4b).

    bilstm_vci  each job's bilstm (same 30 configurations and seeds) with NDVI_mean, NDVI_std, valid_frac and the
                season channels zeroed, so only VCI_mean, VCI_p10, observed and months-since carry information;
                jobs 1, 2 and 3 (job 4 is scored with the job-1 models)
    cnn_lr      tile encoder on month T's VCI tile, dropout, one linear layer; job 1
    cnn_bilstm  tile encoder on each of the 12 monthly tiles + observed + months-since -> bilstm encoder and head;
                job 1

Runs live under runs/posthoc/{search,final}/<job>/<model>/ (search: inner split, one seed 1000 + id; final: outer
split, FINAL_SEEDS, model.pt saved). Development scores in results/scores/posthoc.json, from the score stage's own pooling
(score_stage.collect), subset names and region draws, so the new rows line up with the pre-registered ones.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from common import (CACHE, FINAL_SEEDS, LEADS, N_FOLDS, REFS, RUNS, SCORES, SMOKE, git_state, read_json, save_npz,
                    write_json)

MODEL_JOBS = {"bilstm_vci": ("job1", "job2", "job3"), "cnn_lr": ("job1",), "cnn_bilstm": ("job1",)}
MODELS = tuple(MODEL_JOBS)
ZERO_CH = [0, 1, 4, 5, 6]                    # NDVI_mean, NDVI_std, valid_frac, season sin, season cos
N_CFG = {"cnn_lr": 20, "cnn_bilstm": 12}
LABELS = {"bilstm_vci": "VCI-only Bi-LSTM", "cnn_lr": "CNN + LR", "cnn_bilstm": "CNN + Bi-LSTM"}
_TILES: dict = {}


def configs(model: str, job: str = "job1") -> list[dict]:
    """Search configurations, drawn once with a fixed seed (amendments 4 and 4b)."""
    import search
    if model == "bilstm_vci":
        out = []
        for c in search.net_configs(job, "bilstm"):          # the job's own bilstm configurations
            c = dict(c)
            c["family"], c["net_family"] = "bilstm_vci", "bilstm"
            out.append(c)
        return out
    if job != "job1":
        raise ValueError(f"{model} is a job-1 model")
    rng = np.random.default_rng(search.seed_for("posthoc", "job1", model))
    n = 2 if SMOKE else N_CFG[model]
    out = []
    for i in range(n):
        if model == "cnn_lr":
            cfg = {"id": i, "family": model, "width": int(rng.choice([8, 16, 32])), "depth": int(rng.choice([2, 3])),
                   "lr": float(math.exp(rng.uniform(math.log(3e-4), math.log(3e-3)))),
                   "weight_decay": float(math.exp(rng.uniform(math.log(1e-6), math.log(1e-2)))),
                   "dropout": float(rng.uniform(0.0, 0.5)), "epochs": int(rng.choice([6, 10, 15, 20])), "batch": 512}
        else:
            mask_on = bool(rng.random() < 0.5)
            cfg = {"id": i, "family": model, "width": int(rng.choice([8, 16])), "depth": int(rng.choice([2, 3])),
                   "hidden": int(rng.choice([16, 24, 32, 48, 64])), "layers": 1,
                   "lr": float(math.exp(rng.uniform(math.log(3e-4), math.log(3e-3)))),
                   "weight_decay": float(math.exp(rng.uniform(math.log(1e-6), math.log(1e-2)))),
                   "dropout": float(rng.uniform(0.0, 0.5)),
                   "mask_p": float(rng.uniform(0.1, 0.4)) if mask_on else 0.0,
                   "epochs": int(rng.choice([6, 10, 15])), "batch": 512}
        if SMOKE:
            cfg["epochs"] = 1
        out.append(cfg)
    return out


def device_for(cfg: dict) -> str:
    """bilstm_vci as the pipeline places a bilstm (search.device_for: GPU from 128 hidden units in total), the CNN
    models on the GPU."""
    if cfg["family"] == "bilstm_vci":
        import search
        return search.device_for({**cfg, "family": "bilstm"})
    return "mps"


def search_dir(model: str, job: str, k: int, i: int):
    return RUNS / "posthoc" / "search" / job / model / f"fold{k + 1}" / f"cfg{i:02d}"


def final_dir(model: str, job: str, k: int, seed: int):
    return RUNS / "posthoc" / "final" / job / model / f"fold{k + 1}" / f"seed{seed}"


def spec_dir(spec: dict):
    c = spec["cfg"]
    if spec["type"] == "ph_search":
        return search_dir(c["family"], spec["job"], spec["k"], c["id"])
    return final_dir(c["family"], spec["job"], spec["k"], spec["seed"])


def spec_name(spec: dict) -> str:
    c = spec["cfg"]
    tail = f"cfg{c['id']:02d}" if spec["type"] == "ph_search" else f"seed{spec['seed']}"
    return f"posthoc/{spec['type'][3:]}/{spec['job']}/{c['family']}/fold{spec['k'] + 1}/{tail}"


def tiles_for(ref: str, part: str = "dev") -> np.ndarray:
    """The tile array for one VCI table, loaded once per process (one kept)."""
    key = (part, ref)
    if key not in _TILES:
        _TILES.clear()
        _TILES[key] = np.load(CACHE / "tiles" / f"{part}_{ref}.npy")
    return _TILES[key]


def lstm_macs(hidden: int, layers: int, n_in: int, n_heads: int, steps: int = 12) -> int:
    """Multiply-adds of one bilstm prediction (both directions, four gates, the SeqNet head)."""
    macs, d_in = 0, n_in
    for _ in range(layers):
        macs += steps * 2 * 4 * hidden * (d_in + hidden)
        d_in = 2 * hidden
    return int(macs + 4 * hidden * 2 * hidden + 2 * hidden * n_heads)


def run_unit(spec: dict, device: str, threads: int) -> dict:
    """One search or final run of a post-hoc model (resumable: a finished run is read back)."""
    import units
    out_dir = spec_dir(spec)
    if (out_dir / "DONE").exists():
        return read_json(out_dir / "meta.json")
    cfg, k, job = spec["cfg"], spec["k"], spec["job"]
    split = "inner" if spec["type"] == "ph_search" else "outer"
    seed = 1000 + cfg["id"] if spec["type"] == "ph_search" else spec["seed"]
    d = units.prep(job, k, split)
    t0 = time.time()
    if cfg["family"] == "bilstm_vci":
        import nets
        Xf, Xe = d["X_fit"].copy(), d["X_ev"].copy()
        Xf[:, :, ZERO_CH] = 0.0
        Xe[:, :, ZERO_CH] = 0.0
        net_cfg = dict(cfg)
        net_cfg["family"] = "bilstm"
        data = {"X_fit": Xf, "Y_fit": d["Y_fit"], "M_fit": d["M_fit"], "hide_ok": d["hide_ok"],
                "X_eval": [("ev", Xe, None)]}
        res = nets.train(net_cfg, data, seed, device=device, threads=threads)
        macs = lstm_macs(cfg["hidden"], cfg.get("layers", 1), 9 - len(ZERO_CH), d["Y_fit"].shape[1])
    else:
        import cnn
        base = units.ctx()["base"]
        tiles = tiles_for(REFS[k])
        steps = 1 if cfg["family"] == "cnn_lr" else 12
        fit = cnn.Batches(tiles, base.w_cell[d["fit"]], base.w_T[d["fit"]], d["X_fit"], steps)
        ev = cnn.Batches(tiles, base.w_cell[d["ev"]], base.w_T[d["ev"]], d["X_ev"], steps)
        res = cnn.train(cfg, fit, d["Y_fit"], d["M_fit"], seed, device=device, threads=threads, evals=[("ev", ev)])
        macs = cnn.multiply_adds(cfg)
    z = res["logits"]["ev"]
    met = units.eval_metric(job, z, d)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npz(out_dir / "pred.npz", wid=d["ev"].astype(np.int64), logit=z.astype(np.float32))
    if spec["type"] == "ph_final":
        import torch
        torch.save(res["state"], out_dir / "model.pt")
    meta = {"job": job, "family": cfg["family"], "fold": k + 1, "split": split, "cfg": cfg, "seed": seed,
            "device": device, "threads": threads, "git": git_state(), "n_params": res["n_params"],
            "multiply_adds": macs, "seconds": res["seconds"], "unit_seconds": round(time.time() - t0, 1),
            "prep_seconds": d["seconds"], "history": res["history"], "metric": met, "scaler": d["scaler"],
            "tiles": None if cfg["family"] == "bilstm_vci" else f"cache/tiles/dev_{REFS[k]}.npy",
            "n_fit": int(len(d["fit"])), "n_ev": int(len(d["ev"])), "pid": os.getpid()}
    write_json(out_dir / "meta.json", meta)
    (out_dir / "DONE").write_text(meta["git"]["commit"] + "\n")
    return meta


def chosen_config(model: str, job: str, k: int) -> dict | None:
    """Highest inner-validation metric among the finished search runs of this model, job and fold."""
    best = None
    for c in configs(model, job):
        d = search_dir(model, job, k, c["id"])
        m = read_json(d / "meta.json")
        if m is None or not (d / "DONE").exists():
            continue
        v = m["metric"]["metric"]
        if v is not None and np.isfinite(v) and (best is None or v > best[0]):
            best = (v, m["cfg"])
    return None if best is None else best[1]


def stage_specs(stage: str) -> list[dict]:
    specs = []
    for model, jobs in MODEL_JOBS.items():
        for job in jobs:
            for k in range(N_FOLDS):
                if stage == "search":
                    specs += [{"type": "ph_search", "job": job, "k": k, "cfg": c} for c in configs(model, job)]
                elif stage == "final":
                    cfg = chosen_config(model, job, k)
                    if cfg is not None:
                        specs += [{"type": "ph_final", "job": job, "k": k, "cfg": cfg, "seed": s} for s in FINAL_SEEDS]
    return specs


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def final_probs(model: str, job: str, k: int, wid: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Mean probability (n, heads) over the finished final seeds; None unless all FINAL_SEEDS are done."""
    ps = []
    for s in FINAL_SEEDS:
        d = final_dir(model, job, k, s)
        if not (d / "DONE").exists():
            continue
        with np.load(d / "pred.npz") as z:
            if not np.array_equal(z["wid"], wid):
                raise RuntimeError(f"{job}/{model}/fold{k + 1}/seed{s}: window ids differ from the simple models'")
            ps.append(_sig(z["logit"].astype(float)))
    return (np.mean(ps, axis=0) if len(ps) == len(FINAL_SEEDS) else None), len(ps)


def score_dev() -> dict:
    """Every job with every pre-registered and post-hoc model, pooled and resampled exactly as the score stage:
    score_stage.collect builds the subsets (the post-hoc models' seed means added to each fold's predictions), and the
    official subset names give the same region draws. Writes results/scores/posthoc.json."""
    import dataset
    import score_stage
    import scoring
    from common import FAMILIES, SIMPLE
    from windows import Base
    meta_ds, arrays = dataset.load("dev")
    base = Base(meta_ds, arrays, "dev")
    R, fold_r = base.n_regions, base.region_group
    used = {}
    orig = score_stage.load_preds

    def load_plus(job, k, nh):
        wid, preds, u = orig(job, k, nh)
        if wid is not None:
            for model, jobs in MODEL_JOBS.items():
                if job in jobs:
                    p, n = final_probs(model, job, k, wid)
                    u[model] = n
                    if p is not None:
                        preds[model] = p
        used[f"{job}/fold{k + 1}"] = u
        return wid, preds, u

    score_stage.load_preds = load_plus
    try:
        col = score_stage.collect(base, arrays)
    finally:
        score_stage.load_preds = orig
    subs = col["subsets"]
    gate = score_stage.GATE
    wording = gate["wording"]
    sel = (read_json(SCORES / "scores.json") or {}).get("selection") or {}
    out_jobs = {}
    for g in gate["jobs"]:
        job = g["job"]
        model_job = "job1" if job == "job4" else job
        new = [m for m, js in MODEL_JOBS.items() if model_job in js]
        simple_c = [m for m in gate["simple_models"] if m not in g["simple_exclude"]]
        lr = "lr_latest_age" if job == "job2" else "lr_vci_t"
        entry = {"label": g["label"], "new_models": new}
        if job == "job3":
            names = [f"job3_lead{h}_onset" for h in LEADS]
            if not all(n in subs for n in names):
                continue
            mv = scoring.mean_over_leads("job3_mean_onset", [subs[n] for n in names],
                                         list(SIMPLE) + list(FAMILIES) + new + ["persistence"], fold_r, R)
            best_simple = (sel.get(job) or {}).get("best_simple") or max(
                (m for m in simple_c if m in mv), key=lambda m: mv[m]["all"][0], default=None)
            comps = [c for c in dict.fromkeys([lr, best_simple]) if c and c in mv]
            pairs = [(m, c) for m in new if m in mv for c in comps] + [(a, b) for a, b in (("bilstm_vci", "bilstm"),)
                                                                      if a in mv and b in mv]
            entry.update({"metric": "mean onset within-fold AUC over leads 1-3",
                          "models": {m: {sd: scoring.ci(v[sd]) for sd in scoring.SIDES} for m, v in mv.items()},
                          "paired": {f"{a} - {b}": {sd: scoring.ci(mv[a][sd] - mv[b][sd]) for sd in scoring.SIDES}
                                     for a, b in pairs}})
        else:
            name = g["subset"]
            if name not in subs:
                continue
            s = subs[name]

            def point(m):
                W, P, N = scoring.win_matrix(s["preds"][m], s["y"], s["region"], R)
                C = np.zeros((1, R))
                C[0, np.unique(s["region"])] = 1
                return float(scoring.auc_within_from(W, P, N, C, fold_r)[0])
            best_simple = (sel.get(job) or {}).get("best_simple") or max(
                (m for m in simple_c if m in s["preds"]), key=point, default=None)
            comps = [c for c in dict.fromkeys([lr, best_simple]) if c and c in s["preds"]]
            pairs = [(m, c) for m in new if m in s["preds"] for c in comps]
            pairs += [(a, b) for a, b in (("bilstm_vci", "bilstm"), ("cnn_bilstm", "bilstm_vci"), ("cnn_bilstm", "cnn_lr"))
                      if a in s["preds"] and b in s["preds"]]
            res = scoring.score_subset(name, s["y"], s["region"], s["west"], s["preds"], fold_r, R, N_FOLDS, pairs)
            entry.update({"metric": "within-fold AUC", "subset": name, "n": res["n"], "n_pos": res["n_pos"],
                          "models": {m: {sd: v[sd]["auc_within"] for sd in scoring.SIDES} | {
                              "per_fold_auc": v["per_fold_auc"], "brier": v["all"]["brier"]}
                              for m, v in res["models"].items()},
                          "paired": {k: {sd: v[sd]["dauc_within"] for sd in scoring.SIDES}
                                     for k, v in res["paired"].items()}})
        entry["best_simple"] = best_simple
        entry["verdicts"] = {k: scoring.verdict(v["all"], v["west"], v["east"], wording)
                             for k, v in entry["paired"].items()}
        out_jobs[job] = entry
    sizes = {}
    for model, jobs in MODEL_JOBS.items():
        for job in jobs:
            per = []
            for k in range(N_FOLDS):
                mm = read_json(final_dir(model, job, k, FINAL_SEEDS[0]) / "meta.json") or {}
                per.append({"fold": k + 1, "cfg": mm.get("cfg"), "n_params": mm.get("n_params"),
                            "multiply_adds": mm.get("multiply_adds")})
            sizes[f"{job}/{model}"] = per
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state(),
           "note": "Added after the development results (DESIGN.md amendments 4 and 4b); not part of the gate.",
           "labels": LABELS, "model_jobs": {m: list(j) for m, j in MODEL_JOBS.items()}, "seeds_used": used,
           "jobs": out_jobs, "chosen": sizes,
           "two_by_two_job1": {"cell average, month T": "lr_vci_t", "cell image, month T": "cnn_lr",
                               "cell average, 12 months": "bilstm_vci", "cell image, 12 months": "cnn_bilstm"}}
    write_json(SCORES / "posthoc.json", out)
    return out
