"""Work units: simple models per (job, fold); one search run per (job, family, fold, configuration); one final run
per (job, family, fold, seed). Each saves predictions, metadata and a DONE marker; finished units are skipped."""

from __future__ import annotations

import json
import os
import time

import numpy as np

import nets
import search
import simple
from common import (FINAL_SEEDS, LEADS, N_CONFIGS, RUNS, git_state, log, read_json, save_npz, write_json)
from windows import Base, Sets, baseline_features, gather, input_cube, raw_windows, scaler_for

JOBS = ("job1", "job2", "job3")
_CTX: dict = {}


def ctx() -> dict:
    """Per-process dataset and window base (loaded once)."""
    if "base" not in _CTX:
        import dataset
        meta, arrays = dataset.load("dev")
        _CTX["meta"], _CTX["arrays"] = meta, arrays
        _CTX["base"] = Base(meta, arrays, "dev")
        _CTX["prep"] = {}
    return _CTX


def labels(base: Base, job: str, idx: np.ndarray):
    if job == "job3":
        Y = base.label_h[idx]
    else:
        Y = base.label0[idx][:, None]
    return np.maximum(Y, 0).astype(np.float32), Y >= 0


def prep(job: str, k: int, split: str) -> dict:
    """Inputs, labels and baseline features for one job, fold and split (cached; one entry kept)."""
    c = ctx()
    key = (job, k, split)
    if key in c["prep"]:
        return c["prep"][key]
    c["prep"].clear()
    t0 = time.time()
    base, arrays = c["base"], c["arrays"]
    feats, obs = arrays["feats"][k], arrays["obs"][k]
    sets = Sets(base, job, k, obs)
    fit, ev = sets.get(split)
    hide = job == "job2"
    scaler = scaler_for(base, feats, obs, fit)
    F = input_cube(feats, obs, scaler)
    d = {"job": job, "k": k, "split": split, "fit": fit, "ev": ev, "scaler": scaler, "inner_year": sets.inner_year}
    d["X_fit"], d["X_ev"] = gather(F, base, fit, hide), gather(F, base, ev, hide)
    d["V_fit"], d["O_fit"] = raw_windows(feats, obs, base, fit, hide)
    d["V_ev"], d["O_ev"] = raw_windows(feats, obs, base, ev, hide)
    d["Y_fit"], d["M_fit"] = labels(base, job, fit)
    d["Y_ev"], d["M_ev"] = labels(base, job, ev)
    d["f_fit"] = baseline_features(d["V_fit"], d["O_fit"], base.w_T[fit])
    d["f_ev"] = baseline_features(d["V_ev"], d["O_ev"], base.w_T[ev])
    d["onset_ev"] = base.label0[ev] == 0
    d["hide_ok"] = np.r_[np.ones(11, bool), False]
    d["seconds"] = round(time.time() - t0, 1)
    c["prep"][key] = d
    return d


def eval_metric(job: str, logits: np.ndarray, d: dict) -> dict:
    """Inner-split selection metric: AUC (job 3: mean over leads of the onset AUC); log loss for reference."""
    out = {}
    if job == "job3":
        per = []
        for j in range(len(LEADS)):
            m = d["M_ev"][:, j] & d["onset_ev"]
            per.append(simple.auc(logits[m, j], d["Y_ev"][m, j]))
        out["auc_leads_onset"] = per
        out["metric"] = float(np.nanmean(per))
        per_all = [simple.auc(logits[d["M_ev"][:, j], j], d["Y_ev"][d["M_ev"][:, j], j]) for j in range(len(LEADS))]
        out["auc_leads_all"] = per_all
    else:
        out["metric"] = simple.auc(logits[:, 0], d["Y_ev"][:, 0])
    return out


# --------------------------------------------------------------------------------------------- simple models
def simple_dir(job, k):
    return RUNS / "simple" / job / f"fold{k + 1}"


def run_simple(job: str, k: int) -> dict:
    """Inner split: boosted-tree search and smoothing choice; outer split: refit all with the choices, predict."""
    d_out = simple_dir(job, k)
    if (d_out / "DONE").exists():
        return read_json(d_out / "meta.json")
    t0 = time.time()
    with_t = job != "job2"
    meta = {"job": job, "fold": k + 1, "git": git_state(), "inner": {}, "chosen": {}, "lr": {}}
    heads = range(len(LEADS)) if job == "job3" else range(1)
    di = prep(job, k, "inner")
    gbt_auc = np.zeros((len(list(heads)), N_CONFIGS))
    alpha_auc = np.zeros((len(list(heads)), len(simple.ALPHAS)))
    lr_inner = []
    for j in heads:
        mf = di["M_fit"][:, j]
        me = di["M_ev"][:, j] & (di["onset_ev"] if job == "job3" else True)
        sub = lambda dd, m: {kk: vv[m] for kk, vv in dd.items()}  # noqa: E731
        res = simple.search_gbt(sub(di["f_fit"], mf), di["Y_fit"][mf, j], sub(di["f_ev"], me), di["Y_ev"][me, j], job)
        gbt_auc[j] = [r["auc"] for r in res]
        ra = simple.search_alpha(di["V_fit"][mf], di["O_fit"][mf], di["Y_fit"][mf, j], sub(di["f_fit"], mf),
                                 di["V_ev"][me], di["O_ev"][me], di["Y_ev"][me, j], sub(di["f_ev"], me))
        alpha_auc[j] = [r["auc"] for r in ra]
        # inner-split LR fits for the residual networks' base
        lr_inner.append(simple.lr_coefs(sub(di["f_fit"], mf), di["Y_fit"][mf, j], with_t))
        log(f"  simple {job} fold {k + 1} head {j}: best GBT inner AUC {np.nanmax(gbt_auc[j]):.4f}")
    g_best = int(np.nanargmax(np.nanmean(gbt_auc, axis=0)))
    a_best = int(np.nanargmax(np.nanmean(alpha_auc, axis=0)))
    meta["inner"] = {"gbt_auc": gbt_auc.tolist(), "alpha_auc": alpha_auc.tolist(), "n_fit": int(len(di["fit"])),
                     "n_ev": int(len(di["ev"])), "inner_year": di["inner_year"]}
    meta["chosen"] = {"gbt": search.gbt_configs(job)[g_best], "alpha": simple.ALPHAS[a_best]}
    meta["lr"]["inner"] = lr_inner
    do = prep(job, k, "outer")
    probs, lr_outer = [], []
    for j in heads:
        mf = do["M_fit"][:, j]
        sub = lambda dd, m: {kk: vv[m] for kk, vv in dd.items()}  # noqa: E731
        out, info = simple.fit_predict(sub(do["f_fit"], mf), do["V_fit"][mf], do["O_fit"][mf], do["Y_fit"][mf, j],
                                       do["f_ev"], do["V_ev"], do["O_ev"], meta["chosen"]["gbt"],
                                       meta["chosen"]["alpha"], with_t)
        probs.append(np.stack([out.get(m, np.full(len(do["ev"]), np.nan)) for m in simple_models()], axis=1))
        lr_outer.append({k2: info[k2] for k2 in ("lr_vci_t", "lr_latest_age") if k2 in info})
    meta["lr"]["outer"] = lr_outer
    meta["outer"] = {"n_fit": int(len(do["fit"])), "n_ev": int(len(do["ev"])),
                     "rate_fit": [float(do["Y_fit"][do["M_fit"][:, j], j].mean()) for j in heads]}
    meta["models"] = list(simple_models())
    meta["seconds"] = round(time.time() - t0, 1)
    save_npz(d_out / "pred.npz", wid=do["ev"].astype(np.int64), prob=np.stack(probs, axis=1).astype(np.float32))
    write_json(d_out / "meta.json", meta)
    (d_out / "DONE").write_text(meta["git"]["commit"] + "\n")
    return meta


def simple_models():
    from common import SIMPLE
    return SIMPLE


# --------------------------------------------------------------------------------------------- networks
def base_for(job: str, k: int, split: str, family: str, d: dict) -> dict:
    """Residual base logits from the same split's LR fits (simple stage)."""
    if family != "resbilstm":
        return {}
    m = read_json(simple_dir(job, k) / "meta.json")
    if m is None:
        raise RuntimeError(f"simple/{job}/fold{k + 1} missing: the residual network needs its LR base")
    lr = m["lr"][split]
    if job == "job2":
        c = lr[0]["lr_latest_age"]
        coef = {"a": [c["a"]], "b": [c["b"]], "c": [c["c"]], "a0": [c["a0"]]}
        f = d["f_ev"]
        z = np.where(f["has"], c["a"] + c["b"] * np.nan_to_num(f["vci_latest"]) + c["c"] * np.nan_to_num(f["age"]),
                     c["a0"])
        return {"base_coef": coef, "V_fit": np.nan_to_num(d["V_fit"]), "base_ev": z[:, None].astype(np.float32)}
    a = np.array([h["lr_vci_t"]["a"] for h in lr], np.float32)
    b = np.array([h["lr_vci_t"]["b"] for h in lr], np.float32)
    bf = a[None, :] + b[None, :] * np.nan_to_num(d["V_fit"][:, -1])[:, None]
    be = a[None, :] + b[None, :] * np.nan_to_num(d["V_ev"][:, -1])[:, None]
    return {"base_fit": bf.astype(np.float32), "base_ev": be.astype(np.float32)}


def search_dir(job, family, k, i):
    return RUNS / "search" / job / family / f"fold{k + 1}" / f"cfg{i:02d}"


def final_dir(job, family, k, seed):
    return RUNS / "final" / job / family / f"fold{k + 1}" / f"seed{seed}"


def run_net(job: str, family: str, k: int, split: str, cfg: dict, seed: int, out_dir, device: str,
            threads: int, save_model: bool) -> dict:
    if (out_dir / "DONE").exists():
        return read_json(out_dir / "meta.json")
    d = prep(job, k, split)
    extra = base_for(job, k, split, family, d)
    data = {"X_fit": d["X_fit"], "Y_fit": d["Y_fit"], "M_fit": d["M_fit"], "hide_ok": d["hide_ok"],
            "base_fit": extra.get("base_fit"), "V_fit": extra.get("V_fit"), "base_coef": extra.get("base_coef"),
            "X_eval": [("ev", d["X_ev"], extra.get("base_ev"))]}
    res = nets.train(cfg, data, seed, device=device, threads=threads)
    z = res["logits"]["ev"]
    met = eval_metric(job, z, d)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npz(out_dir / "pred.npz", wid=d["ev"].astype(np.int64), logit=z.astype(np.float32))
    if save_model:
        import torch
        torch.save(res["state"], out_dir / "model.pt")
    meta = {"job": job, "family": family, "fold": k + 1, "split": split, "cfg": cfg, "seed": seed, "device": device,
            "threads": threads, "git": git_state(), "n_params": res["n_params"], "seconds": res["seconds"],
            "prep_seconds": d["seconds"], "history": res["history"], "metric": met, "scaler": d["scaler"],
            "n_fit": int(len(d["fit"])), "n_ev": int(len(d["ev"])), "pid": os.getpid()}
    write_json(out_dir / "meta.json", meta)
    (out_dir / "DONE").write_text(meta["git"]["commit"] + "\n")
    return meta


def chosen_config(job: str, family: str, k: int) -> dict | None:
    """The configuration with the highest inner-split metric among the 30 (None until all 30 are done)."""
    best = None
    for i in range(N_CONFIGS):
        m = read_json(search_dir(job, family, k, i) / "meta.json")
        if m is None or not (search_dir(job, family, k, i) / "DONE").exists():
            return None
        v = m["metric"]["metric"]
        if v is not None and np.isfinite(v) and (best is None or v > best[0]):
            best = (v, m["cfg"])
    return None if best is None else best[1]


# --------------------------------------------------------------------------------------------- what-ifs
WHATIF_VARIANTS = ("ndmi_off", "ndmi_on", "pixel", "cell")


def whatif_dir(variant, family, k, seed):
    return RUNS / "whatif" / variant / family / f"fold{k + 1}" / f"seed{seed}"


def prep_variant(variant: str, k: int) -> dict:
    """Job-1 outer split for a what-if (DESIGN.md 10). ndmi_off/ndmi_on: cells with SWIR and NDMI at T, without or
    with the two NDMI channels; pixel/cell: windows with T observed under both definitions, per-pixel or per-cell
    VCI inputs."""
    c = ctx()
    key = ("whatif", variant, k)
    if key in c["prep"]:
        return c["prep"][key]
    c["prep"].clear()
    base, arrays = c["base"], c["arrays"]
    feats, obs = arrays["feats"][k], arrays["obs"][k]
    sets = Sets(base, "job1", k, obs)
    fit, ev = sets.outer_train, sets.outer_val
    if variant.startswith("ndmi"):
        nd = arrays["ndmi"][k]                                     # (cells, months, 2)
        okT = arrays["has_ndmi"][base.w_cell] & np.isfinite(nd[base.w_cell, base.w_T, 0]) & \
            np.isfinite(nd[base.w_cell, base.w_T, 1])
        fit, ev = fit[okT[fit]], ev[okT[ev]]
        f_in, o_in = feats, obs
    else:
        cellf, fixed, obs_c = arrays["cellf"][k], arrays["fixed"], arrays["obs_c"]
        both = obs[base.w_cell, base.w_T] & obs_c[base.w_cell, base.w_T] & np.isfinite(
            cellf[base.w_cell, base.w_T, 0])
        fit, ev = fit[both[fit]], ev[both[ev]]
        if variant == "cell":
            f_in = np.concatenate([fixed[:, :, 0:2], cellf[:, :, 0:2], fixed[:, :, 3:4]], axis=2)
            o_in = obs_c & np.isfinite(cellf[:, :, 0])
        else:
            f_in, o_in = feats, obs
    scaler = scaler_for(base, f_in, o_in, fit)
    F = input_cube(f_in, o_in, scaler)
    d = {"fit": fit, "ev": ev, "scaler": scaler, "variant": variant}
    d["X_fit"], d["X_ev"] = gather(F, base, fit), gather(F, base, ev)
    if variant == "ndmi_on":
        nd = arrays["ndmi"][k]
        o = obs & np.isfinite(nd[:, :, 0]) & np.isfinite(nd[:, :, 1])
        diff = np.zeros((base.n_cells, o.shape[1] + 1), np.int64)        # training windows' cell-months only
        np.add.at(diff, (base.w_cell[fit], np.maximum(base.w_T[fit] - 11, 0)), 1)
        np.add.at(diff, (base.w_cell[fit], base.w_T[fit] + 1), -1)
        used = (np.cumsum(diff, axis=1)[:, :o.shape[1]] > 0) & o
        vals = nd[:, :, 0][used].astype(np.float64)
        mu, sd = float(vals.mean()), float(max(vals.std(), 1e-6))
        G = np.zeros((nd.shape[0], F.shape[1], 2), np.float32)
        from windows import PAD
        G[:, PAD:, 0] = np.where(o, (np.nan_to_num(nd[:, :, 0]) - mu) / sd, 0.0)
        G[:, PAD:, 1] = np.where(o, np.nan_to_num(nd[:, :, 1]) / 100.0, 0.0)
        from numpy.lib.stride_tricks import sliding_window_view
        def g2(idx):
            return np.ascontiguousarray(sliding_window_view(G, 12, axis=1)[base.w_cell[idx], base.w_T[idx]]
                                        .transpose(0, 2, 1))
        d["X_fit"] = np.concatenate([d["X_fit"], g2(fit)], axis=2)
        d["X_ev"] = np.concatenate([d["X_ev"], g2(ev)], axis=2)
        d["ndmi_scaler"] = {"mu": mu, "sd": sd}
    V_fit, _ = raw_windows(f_in, o_in, base, fit)
    V_ev, _ = raw_windows(f_in, o_in, base, ev)
    d["vT_fit"], d["vT_ev"] = V_fit[:, -1], V_ev[:, -1]
    d["Y_fit"], d["M_fit"] = labels(base, "job1", fit)
    d["Y_ev"], d["M_ev"] = labels(base, "job1", ev)
    d["hide_ok"] = np.r_[np.ones(11, bool), False]
    d["onset_ev"] = base.label0[ev] == 0
    c["prep"][key] = d
    return d


def run_whatif(variant: str, family: str, k: int, cfg: dict, seed: int, device: str, threads: int) -> dict:
    out_dir = whatif_dir(variant, family, k, seed)
    if (out_dir / "DONE").exists():
        return read_json(out_dir / "meta.json")
    d = prep_variant(variant, k)
    data = {"X_fit": d["X_fit"], "Y_fit": d["Y_fit"], "M_fit": d["M_fit"], "hide_ok": d["hide_ok"],
            "X_eval": [("ev", d["X_ev"], None)]}
    if d["X_fit"].shape[2] > 9:
        data["zero_on_hide"] = np.r_[nets.ZERO_ON_HIDE, np.ones(d["X_fit"].shape[2] - 9, np.float32)]
    if family == "resbilstm":
        t = np.isfinite(d["vT_fit"])
        lr = simple.LR(["vci_t"]).fit(d["vT_fit"][t][:, None].astype(np.float64), d["Y_fit"][t, 0])
        a, b = lr.raw_coefficients()
        data["base_fit"] = (a + b[0] * np.nan_to_num(d["vT_fit"]))[:, None].astype(np.float32)
        data["X_eval"] = [("ev", d["X_ev"], (a + b[0] * np.nan_to_num(d["vT_ev"]))[:, None].astype(np.float32))]
    res = nets.train(cfg, data, seed, device=device, threads=threads)
    z = res["logits"]["ev"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npz(out_dir / "pred.npz", wid=d["ev"].astype(np.int64), logit=z.astype(np.float32))
    meta = {"variant": variant, "family": family, "fold": k + 1, "cfg": cfg, "seed": seed, "device": device,
            "git": git_state(), "seconds": res["seconds"], "n_fit": int(len(d["fit"])), "n_ev": int(len(d["ev"])),
            "metric": {"metric": simple.auc(z[:, 0], d["Y_ev"][:, 0])}, "history": res["history"]}
    write_json(out_dir / "meta.json", meta)
    (out_dir / "DONE").write_text(meta["git"]["commit"] + "\n")
    return meta


def unit(spec: dict, device: str, threads: int) -> dict:
    """Entry point used by the worker processes."""
    t = spec["type"]
    if t == "simple":
        return run_simple(spec["job"], spec["k"])
    cfg = spec["cfg"]
    if t == "search":
        return run_net(spec["job"], cfg["family"], spec["k"], "inner", cfg, 1000 + cfg["id"],
                       search_dir(spec["job"], cfg["family"], spec["k"], cfg["id"]), device, threads, False)
    if t == "final":
        return run_net(spec["job"], cfg["family"], spec["k"], "outer", cfg, spec["seed"],
                       final_dir(spec["job"], cfg["family"], spec["k"], spec["seed"]), device, threads, True)
    if t == "whatif":
        return run_whatif(spec["variant"], cfg["family"], spec["k"], cfg, spec["seed"], device, threads)
    if t == "weekly":
        import weekly_stage
        return weekly_stage.run_unit(spec, device, threads)
    if t == "p5":
        import phase5_stage
        return phase5_stage.run_unit(spec, device, threads)
    raise ValueError(t)


def _unused():
    return json, FINAL_SEEDS
