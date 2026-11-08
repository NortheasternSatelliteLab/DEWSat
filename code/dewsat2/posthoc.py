"""Units, configurations and development scoring for the three job-1 models of DESIGN.md amendment 4.

    bilstm_vci  the bilstm (same 30 job-1 configurations and seeds) with NDVI_mean, NDVI_std, valid_frac and the
                season channels zeroed, so only VCI_mean, VCI_p10, observed and months-since carry information
    cnn_lr      tile encoder on month T's VCI tile, dropout, one linear layer
    cnn_bilstm  tile encoder on each of the 12 monthly tiles + observed + months-since -> bilstm encoder and head

Runs live under runs/posthoc/ (search: inner split, one seed 1000 + id; final: outer split, FINAL_SEEDS, model.pt
saved); development scores in scores/posthoc.json, scored like every job-1 model (score stage's pooling and draws).
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from common import (CACHE, FINAL_SEEDS, N_FOLDS, REFS, RUNS, SCORES, SMOKE, git_state, read_json, save_npz,
                    write_json)

JOB = "job1"
MODELS = ("bilstm_vci", "cnn_lr", "cnn_bilstm")
ZERO_CH = [0, 1, 4, 5, 6]                    # NDVI_mean, NDVI_std, valid_frac, season sin, season cos
N_CFG = {"bilstm_vci": 30, "cnn_lr": 20, "cnn_bilstm": 12}
LABELS = {"bilstm_vci": "VCI-only Bi-LSTM", "cnn_lr": "CNN + LR", "cnn_bilstm": "CNN + Bi-LSTM"}
_TILES: dict = {}


def configs(model: str) -> list[dict]:
    """Search configurations, drawn once with a fixed seed (amendment 4)."""
    import search
    if model == "bilstm_vci":
        out = []
        for c in search.net_configs(JOB, "bilstm"):          # the bilstm's own 30 job-1 configurations
            c = dict(c)
            c["family"], c["net_family"] = "bilstm_vci", "bilstm"
            out.append(c)
        return out
    rng = np.random.default_rng(search.seed_for("posthoc", JOB, model))
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
    return "cpu" if cfg["family"] == "bilstm_vci" else "mps"


def search_dir(model: str, k: int, i: int):
    return RUNS / "posthoc" / "search" / JOB / model / f"fold{k + 1}" / f"cfg{i:02d}"


def final_dir(model: str, k: int, seed: int):
    return RUNS / "posthoc" / "final" / JOB / model / f"fold{k + 1}" / f"seed{seed}"


def spec_dir(spec: dict):
    c = spec["cfg"]
    if spec["type"] == "ph_search":
        return search_dir(c["family"], spec["k"], c["id"])
    return final_dir(c["family"], spec["k"], spec["seed"])


def spec_name(spec: dict) -> str:
    c = spec["cfg"]
    tail = f"cfg{c['id']:02d}" if spec["type"] == "ph_search" else f"seed{spec['seed']}"
    return f"posthoc/{spec['type'][3:]}/{c['family']}/fold{spec['k'] + 1}/{tail}"


def tiles_for(ref: str, part: str = "dev") -> np.ndarray:
    """The tile array for one VCI table, loaded once per process (one kept)."""
    key = (part, ref)
    if key not in _TILES:
        _TILES.clear()
        _TILES[key] = np.load(CACHE / "tiles" / f"{part}_{ref}.npy")
    return _TILES[key]


def run_unit(spec: dict, device: str, threads: int) -> dict:
    """One search or final run of a post-hoc model (resumable: a finished run is read back)."""
    import units
    out_dir = spec_dir(spec)
    if (out_dir / "DONE").exists():
        return read_json(out_dir / "meta.json")
    cfg, k = spec["cfg"], spec["k"]
    split = "inner" if spec["type"] == "ph_search" else "outer"
    seed = 1000 + cfg["id"] if spec["type"] == "ph_search" else spec["seed"]
    d = units.prep(JOB, k, split)
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
        macs = None
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
    met = units.eval_metric(JOB, z, d)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npz(out_dir / "pred.npz", wid=d["ev"].astype(np.int64), logit=z.astype(np.float32))
    if spec["type"] == "ph_final":
        import torch
        torch.save(res["state"], out_dir / "model.pt")
    meta = {"job": JOB, "family": cfg["family"], "fold": k + 1, "split": split, "cfg": cfg, "seed": seed,
            "device": device, "threads": threads, "git": git_state(), "n_params": res["n_params"],
            "multiply_adds": macs, "seconds": res["seconds"], "unit_seconds": round(time.time() - t0, 1),
            "prep_seconds": d["seconds"], "history": res["history"], "metric": met, "scaler": d["scaler"],
            "tiles": None if cfg["family"] == "bilstm_vci" else f"cache/tiles/dev_{REFS[k]}.npy",
            "n_fit": int(len(d["fit"])), "n_ev": int(len(d["ev"])), "pid": os.getpid()}
    write_json(out_dir / "meta.json", meta)
    (out_dir / "DONE").write_text(meta["git"]["commit"] + "\n")
    return meta


def chosen_config(model: str, k: int) -> dict | None:
    """Highest inner-validation AUC among the finished search runs of this model and fold."""
    best = None
    for c in configs(model):
        d = search_dir(model, k, c["id"])
        m = read_json(d / "meta.json")
        if m is None or not (d / "DONE").exists():
            continue
        v = m["metric"]["metric"]
        if v is not None and np.isfinite(v) and (best is None or v > best[0]):
            best = (v, m["cfg"])
    return None if best is None else best[1]


def stage_specs(stage: str) -> list[dict]:
    specs = []
    if stage == "search":
        for model in MODELS:
            for k in range(N_FOLDS):
                for cfg in configs(model):
                    specs.append({"type": "ph_search", "k": k, "cfg": cfg})
    elif stage == "final":
        for model in MODELS:
            for k in range(N_FOLDS):
                cfg = chosen_config(model, k)
                if cfg is None:
                    continue
                for s in FINAL_SEEDS:
                    specs.append({"type": "ph_final", "k": k, "cfg": cfg, "seed": s})
    return specs


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def final_preds(model: str, k: int, wid: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Mean probability over the finished final seeds (None unless all FINAL_SEEDS are done)."""
    ps = []
    for s in FINAL_SEEDS:
        d = final_dir(model, k, s)
        if not (d / "DONE").exists():
            continue
        with np.load(d / "pred.npz") as z:
            if not np.array_equal(z["wid"], wid):
                raise RuntimeError(f"{model}/fold{k + 1}/seed{s}: window ids differ from the simple models'")
            ps.append(_sig(z["logit"][:, 0].astype(float)))
    return (np.mean(ps, axis=0) if len(ps) == len(FINAL_SEEDS) else None), len(ps)


def score_dev() -> dict:
    """Job 1 on the four outer folds with every pre-registered and post-hoc model, pooled and resampled exactly as
    the score stage (subset name "job1", so the same region draws). Writes scores/posthoc.json."""
    import json

    import dataset
    import score_stage
    import scoring
    from common import FAMILIES, SIMPLE
    from windows import Base
    meta_ds, arrays = dataset.load("dev")
    base = Base(meta_ds, arrays, "dev")
    R, fold_r = base.n_regions, base.region_group
    parts, used = [], {}
    for k in range(N_FOLDS):
        wid, preds, u = score_stage.load_preds(JOB, k, 1)
        pr = {m: p[:, 0] for m, p in preds.items()}
        for model in MODELS:
            p, n = final_preds(model, k, wid)
            u[model] = n
            if p is not None:
                pr[model] = p
        used[f"fold{k + 1}"] = u
        parts.append((wid, pr))
    keep = sorted(set.intersection(*(set(p) for _, p in parts)))
    wid = np.concatenate([w for w, _ in parts])
    y = base.label0[wid].astype(np.int64)
    pr = {m: np.concatenate([p[m] for _, p in parts]) for m in keep}
    region = base.w_region[wid]
    west = base.region_west[region]
    gate = next(j for j in score_stage.GATE["jobs"] if j["job"] == JOB)
    simple_c = [m for m in score_stage.GATE["simple_models"] if m not in gate["simple_exclude"]]
    sc = read_json(SCORES / "scores.json") or {}
    best_simple = ((sc.get("selection") or {}).get(JOB) or {}).get("best_simple")
    if best_simple is None:                              # before the score stage: the gate's rule on the same data
        def point(m):
            W, P, N = scoring.win_matrix(pr[m], y, region, R)
            C = np.zeros((1, R))
            C[0, np.unique(region)] = 1
            return float(scoring.auc_within_from(W, P, N, C, fold_r)[0])
        vals = {m: point(m) for m in simple_c if m in pr}
        best_simple = max(vals, key=vals.get) if vals else None
    new = [m for m in MODELS if m in pr]
    pairs = []
    for m in new:
        for c in dict.fromkeys(["lr_vci_t", best_simple]):
            if c and c in pr:
                pairs.append((m, c))
    for a, b in (("bilstm_vci", "bilstm"), ("cnn_bilstm", "bilstm_vci"), ("cnn_bilstm", "cnn_lr")):
        if a in pr and b in pr:
            pairs.append((a, b))
    res = scoring.score_subset(JOB, y, region, west, pr, fold_r, R, N_FOLDS, pairs)
    wording = score_stage.GATE["wording"]
    verdicts = {}
    for key, v in res["paired"].items():
        d = {s: v[s]["dauc_within"] for s in scoring.SIDES}
        verdicts[key] = scoring.verdict(d["all"], d["west"], d["east"], wording)
    sizes = {}
    for m in new:
        per = []
        for k in range(N_FOLDS):
            mm = read_json(final_dir(m, k, FINAL_SEEDS[0]) / "meta.json") or {}
            per.append({"fold": k + 1, "cfg": mm.get("cfg"), "n_params": mm.get("n_params"),
                        "multiply_adds": mm.get("multiply_adds")})
        sizes[m] = per
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state(), "job": JOB,
           "note": "Added after the development results (DESIGN.md amendment 4); not part of the gate.",
           "models_new": new, "labels": LABELS, "best_simple": best_simple, "seeds_used": used,
           "subset": res, "verdicts": verdicts, "chosen": sizes,
           "two_by_two": {"cell average, month T": "lr_vci_t", "cell image, month T": "cnn_lr",
                          "cell average, 12 months": "bilstm_vci", "cell image, 12 months": "cnn_bilstm"}}
    write_json(SCORES / "posthoc.json", out)
    return out
