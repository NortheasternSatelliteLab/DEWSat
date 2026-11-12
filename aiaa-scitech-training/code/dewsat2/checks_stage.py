"""Checks on the new models (DESIGN.md 8): order test and positive control, job 1."""

from __future__ import annotations

import numpy as np
import torch

import nets
import scoring
import search
import simple
import units
from common import FAMILIES, FINAL_SEEDS, N_FOLDS, RUNS, SCORES, git_state, log, read_json, save_npz, write_json
from windows import Base, gather, input_cube, raw_windows, scaler_for

N_PERM = 5
PC_SEEDS = (42, 43, 44)


def _load_net(meta: dict, path, n_in: int, n_heads: int, device="cpu"):
    c = meta["cfg"]
    m = nets.SeqNet(c["family"], n_in, n_heads, c["hidden"], c.get("layers", 1), c.get("dropout", 0.2), c.get("bidir"))
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return m.to(device).eval()


def order_test(status=None) -> dict:
    out = {"families": {}, "n_perm": N_PERM}
    c = units.ctx()
    base = c["base"]
    ys, regs, perms_auc = [], [], {}
    store = {f: {"orig": [], "perm": [[] for _ in range(N_PERM)]} for f in FAMILIES}
    for k in range(N_FOLDS):
        d = units.prep("job1", k, "outer")
        ys.append(d["Y_ev"][:, 0].astype(np.int64))
        regs.append(base.w_region[d["ev"]])
        rng = np.random.default_rng([20261108, k])
        perms = [np.argsort(rng.random((len(d["X_ev"]), 11)), axis=1) for _ in range(N_PERM)]
        for fam in FAMILIES:
            extra = units.base_for("job1", k, "outer", fam, d)
            be = extra.get("base_ev")
            po, pp = [], [[] for _ in range(N_PERM)]
            for s in FINAL_SEEDS:
                rd = units.final_dir("job1", fam, k, s)
                if not (rd / "DONE").exists():
                    continue
                meta = read_json(rd / "meta.json")
                net = _load_net(meta, rd / "model.pt", d["X_ev"].shape[2], 1)
                po.append(1 / (1 + np.exp(-nets.predict_logits(net, d["X_ev"], be)[:, 0])))
                for r, P in enumerate(perms):
                    Xp = d["X_ev"].copy()
                    Xp[:, :11] = d["X_ev"][np.arange(len(Xp))[:, None], P]
                    pp[r].append(1 / (1 + np.exp(-nets.predict_logits(net, Xp, be)[:, 0])))
            if po:
                store[fam]["orig"].append(np.mean(po, axis=0))
                for r in range(N_PERM):
                    store[fam]["perm"][r].append(np.mean(pp[r], axis=0))
        log(f"  order test fold {k + 1} done")
    y = np.concatenate(ys)
    reg = np.concatenate(regs)
    west = base.region_west[reg]
    fold_r = base.region_group
    R = base.n_regions
    for fam in FAMILIES:
        if len(store[fam]["orig"]) != N_FOLDS:
            continue
        preds = {"original": np.concatenate(store[fam]["orig"])}
        for r in range(N_PERM):
            preds[f"shuffled_{r}"] = np.concatenate(store[fam]["perm"][r])
        sc = scoring.score_subset(f"order_{fam}", y, reg, west, preds, fold_r, R, N_FOLDS,
                                  [(f"shuffled_{r}", "original") for r in range(N_PERM)])
        out["families"][fam] = {
            "original": sc["models"]["original"]["all"]["auc_within"],
            "shuffled": [sc["models"][f"shuffled_{r}"]["all"]["auc_within"] for r in range(N_PERM)],
            "shuffled_minus_original": [sc["paired"][f"shuffled_{r} - original"]["all"]["dauc_within"]
                                        for r in range(N_PERM)],
            "mean_abs_prob_change": float(np.mean([np.mean(np.abs(preds[f"shuffled_{r}"] - preds["original"]))
                                                   for r in range(N_PERM)]))}
        log(f"  order test {fam}: original {out['families'][fam]['original'][0]:.4f}, shuffled "
            f"{np.mean([v[0] for v in out['families'][fam]['shuffled']]):.4f}")
    out["n_windows"] = int(len(y))
    return out


def positive_control(status=None) -> dict:
    """Synthetic label VCI(T-3) - VCI(T) > 20 (both observed) on outer fold 1, same code (DESIGN.md 8)."""
    k = 0
    c = units.ctx()
    base, arrays = c["base"], c["arrays"]
    feats, obs = arrays["feats"][k], arrays["obs"][k]
    d = units.prep("job1", k, "outer")
    res = {"label": "VCI_mean(T-3) - VCI_mean(T) > 20, both observed", "fold": 1, "families": {}}

    def syn(idx, V, O):
        ok = O[:, -1] & O[:, -4]
        return ok, ((V[:, -4] - V[:, -1]) > 20).astype(np.float32)

    okf, yf = syn(d["fit"], d["V_fit"], d["O_fit"])
    oke, ye = syn(d["ev"], d["V_ev"], d["O_ev"])
    fit, ev = d["fit"][okf], d["ev"][oke]
    scaler = scaler_for(base, feats, obs, fit)
    F = input_cube(feats, obs, scaler)
    Xf, Xe = gather(F, base, fit), gather(F, base, ev)
    Vf, _ = raw_windows(feats, obs, base, fit)
    Ve, _ = raw_windows(feats, obs, base, ev)
    yf, ye = yf[okf], ye[oke]
    lr = simple.LR(["vci_t"]).fit(Vf[:, -1:].astype(np.float64), yf)
    p_lr = lr.predict(Ve[:, -1:].astype(np.float64))
    a, b = lr.raw_coefficients()
    reg = base.w_region[ev]
    west = base.region_west[reg]
    preds = {"lr_vci_t": p_lr}
    for fam in FAMILIES:
        cfg = units.chosen_config("job1", fam, k)
        if cfg is None:
            continue
        ps = []
        for s in PC_SEEDS:
            data = {"X_fit": Xf, "Y_fit": yf[:, None], "M_fit": np.ones((len(yf), 1), bool),
                    "hide_ok": np.r_[np.ones(11, bool), False], "X_eval": [("ev", Xe, None)]}
            if fam == "resbilstm":
                data["base_fit"] = (a + b[0] * Vf[:, -1:]).astype(np.float32)
                data["X_eval"] = [("ev", Xe, (a + b[0] * Ve[:, -1:]).astype(np.float32))]
            r = nets.train(cfg, data, s, device=search.device_for(cfg), threads=4)
            ps.append(1 / (1 + np.exp(-r["logits"]["ev"][:, 0])))
        preds[fam] = np.mean(ps, axis=0)
        log(f"  positive control {fam}: AUC {simple.auc(preds[fam], ye):.4f} (LR {simple.auc(p_lr, ye):.4f})")
    sc = scoring.score_subset("positive_control", ye.astype(np.int64), reg, west, preds,
                              np.zeros(base.n_regions, np.int64), base.n_regions, 1,
                              [(f, "lr_vci_t") for f in preds if f != "lr_vci_t"])
    res["scores"] = {m: v["all"]["auc_within"] for m, v in sc["models"].items()}
    res["paired"] = {p: v["all"]["dauc_within"] for p, v in sc["paired"].items()}
    res["n_fit"], res["n_eval"] = int(len(fit)), int(len(ev))
    res["rate_fit"], res["rate_eval"] = float(yf.mean()), float(ye.mean())
    return res


def run(status=None) -> dict:
    out = {"git": git_state()}
    path = SCORES / "checks.json"
    prev = read_json(path, {}) or {}
    out["order_test"] = prev.get("order_test") or order_test(status)
    write_json(path, {**prev, **out})
    out["positive_control"] = prev.get("positive_control") or positive_control(status)
    write_json(path, {**prev, **out})
    return out


def _unused():
    return save_npz, RUNS
