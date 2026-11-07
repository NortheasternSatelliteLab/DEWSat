"""Within-fold AUC (primary), pooled AUC, Brier, reliability/ECE; whole-region bootstrap; paired differences.

The exact region-bootstrap AUC is the capability run's (scoring.py, commit e13e7a4): with W[r, s] the number of
(positive in region r, negative in region s) pairs ranked correctly (ties 1/2) and c_r the times region r is drawn,
    AUC = sum_{r,s} c_r c_s W[r, s] / ((sum_r c_r P_r)(sum_s c_s N_s)),
and the within-fold AUC keeps only pairs whose two regions belong to the same outer fold.
"""

from __future__ import annotations

import zlib

import numpy as np

from common import BOOT_SEED, N_BOOT

SIDES = ("all", "west", "east")


def win_matrix(score, y, reg, R):
    pos = y == 1
    P = np.bincount(reg[pos], minlength=R).astype(float)
    N = np.bincount(reg[~pos], minlength=R).astype(float)
    W = np.zeros((R, R))
    sp, rp = score[pos], reg[pos]
    sn, rn = score[~pos], reg[~pos]
    order = np.argsort(rn, kind="stable")
    rn, sn = rn[order], sn[order]
    bounds = np.searchsorted(rn, np.arange(R + 1))
    for s in range(R):
        a, b = bounds[s], bounds[s + 1]
        if a == b or len(sp) == 0:
            continue
        neg = np.sort(sn[a:b])
        lt = np.searchsorted(neg, sp, "left")
        le = np.searchsorted(neg, sp, "right")
        W[:, s] = np.bincount(rp, weights=lt + 0.5 * (le - lt), minlength=R)
    return W, P, N


def auc_from(W, P, N, C):
    num = ((C @ W) * C).sum(axis=1)
    den = (C @ P) * (C @ N)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def auc_within_from(W, P, N, C, fold_r):
    same = fold_r[:, None] == fold_r[None, :]
    num = ((C @ (W * same)) * C).sum(axis=1)
    den = np.zeros(C.shape[0])
    for f in np.unique(fold_r):
        m = fold_r == f
        den += (C[:, m] @ P[m]) * (C[:, m] @ N[m])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def per_fold_auc(W, P, N, fold_r, n_folds):
    out = []
    for f in range(n_folds):
        m = fold_r == f
        den = P[m].sum() * N[m].sum()
        out.append(float(W[np.ix_(m, m)].sum() / den) if den > 0 else None)
    return out


def boot_counts(regions: np.ndarray, R: int, key: str, n_boot: int = N_BOOT) -> np.ndarray:
    """(n_boot + 1, R) region draw counts; row 0 = every present region once."""
    seed = (BOOT_SEED + zlib.crc32(key.encode())) % (2 ** 32)
    rng = np.random.default_rng(seed)
    m = len(regions)
    C = np.zeros((n_boot + 1, R))
    C[0, regions] = 1
    if m:
        draw = regions[rng.integers(0, m, size=(n_boot, m))]
        rows = np.repeat(np.arange(1, n_boot + 1), m)
        np.add.at(C, (rows, draw.ravel()), 1)
    return C


def ci(values: np.ndarray) -> list:
    pt = values[0]
    rest = values[1:][np.isfinite(values[1:])]
    if not np.isfinite(pt) or len(rest) < 0.5 * (len(values) - 1):
        return [None if not np.isfinite(pt) else float(pt), None, None]
    return [float(pt), float(np.percentile(rest, 2.5)), float(np.percentile(rest, 97.5))]


def reliability(p, y, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            mp, of = float(p[m].mean()), float(y[m].mean())
            out.append([round(edges[b], 1), round(edges[b + 1], 1), int(m.sum()), mp, of])
            ece += m.sum() / len(p) * abs(mp - of)
    return out, float(ece)


def side_counts(region, west, side, R, key):
    reg = region if side == "all" else region[west if side == "west" else ~west]
    return boot_counts(np.unique(reg), R, key)


def score_subset(name: str, y, region, west, preds: dict, fold_r, R: int, n_folds: int,
                 pairs: list[tuple[str, str]]) -> dict:
    """Every model's metrics on one subset, and the requested paired differences (a - b)."""
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "models": {}, "paired": {}, "sides": {}}
    if len(y) == 0:
        return out
    Cs = {s: side_counts(region, west, s, R, f"{name}|{s}") for s in SIDES}
    for s in SIDES:
        m = np.ones(len(y), bool) if s == "all" else (west if s == "west" else ~west)
        out["sides"][s] = {"n": int(m.sum()), "n_pos": int(y[m].sum()), "regions": int(len(np.unique(region[m])))}
    mats = {}
    for model, p in preds.items():
        ok = np.isfinite(p)
        if not ok.all():
            continue                                  # every scored model must cover the whole subset
        W, P, N = win_matrix(p, y, region, R)
        sse = np.bincount(region, weights=(p - y) ** 2, minlength=R)
        cnt = np.bincount(region, minlength=R).astype(float)
        mats[model] = (W, P, N, sse, cnt)
        res = {"per_fold_auc": per_fold_auc(W, P, N, fold_r, n_folds)}
        for s in SIDES:
            C = Cs[s]
            with np.errstate(invalid="ignore", divide="ignore"):
                brier = (C @ sse) / (C @ cnt)
            sm = np.ones(len(y), bool) if s == "all" else (west if s == "west" else ~west)
            rel, ece = reliability(p[sm], y[sm]) if sm.any() else ([], None)
            res[s] = {"auc_within": ci(auc_within_from(W, P, N, C, fold_r)), "auc_pooled": ci(auc_from(W, P, N, C)),
                      "brier": ci(brier), "ece": ece, "reliability": rel, "mean_pred": float(p[sm].mean())}
        out["models"][model] = res
    for a, b in pairs:
        if a not in mats or b not in mats or a == b:
            continue
        Wa, Pa, Na, sa, cnt = mats[a]
        Wb, Pb, Nb, sb, _ = mats[b]
        res = {}
        for s in SIDES:
            C = Cs[s]
            with np.errstate(invalid="ignore", divide="ignore"):
                dB = (C @ (sa - sb)) / (C @ cnt)
            res[s] = {"dauc_within": ci(auc_within_from(Wa, Pa, Na, C, fold_r) - auc_within_from(Wb, Pb, Nb, C, fold_r)),
                      "dauc_pooled": ci(auc_from(Wa, Pa, Na, C) - auc_from(Wb, Pb, Nb, C)), "dbrier": ci(dB)}
        out["paired"][f"{a} - {b}"] = res
    return out


def mean_over_leads(name: str, subsets: list[dict], models: list[str], fold_r, R: int, within: bool = True):
    """Mean over leads of each model's (within-fold) AUC, with shared region resamples; values (n_boot+1,) per side."""
    regs = np.concatenate([s["region"] for s in subsets])
    west = np.concatenate([s["west"] for s in subsets])
    Cs = {s: side_counts(regs, west, s, R, f"{name}|{s}") for s in SIDES}
    vals = {}
    for m in models:
        if not all(m in s["preds"] and np.isfinite(s["preds"][m]).all() for s in subsets):
            continue
        mats = [win_matrix(s["preds"][m], s["y"], s["region"], R) for s in subsets]
        f = (lambda W, P, N, C: auc_within_from(W, P, N, C, fold_r)) if within else auc_from
        vals[m] = {side: np.mean([f(W, P, N, Cs[side]) for W, P, N in mats], axis=0) for side in SIDES}
    return vals


def verdict(d_all, d_west, d_east, wording: dict) -> str:
    lo, hi = d_all[1], d_all[2]
    if lo is None:
        return "not scored"
    if lo > 0:
        for side, d in (("West", d_west), ("East", d_east)):
            if d[2] is not None and d[2] < 0:
                return wording["above_zero_but_side_worse"].format(side=side)
        return wording["pass"]
    if hi < 0:
        return wording["below_zero"]
    return wording["includes_zero"]
