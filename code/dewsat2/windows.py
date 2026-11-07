"""Windows, outer folds, inner splits, per-job training and scored sets, scalers and inputs (DESIGN.md 2-4)."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from common import (BLOCK_IDX, BUFFER_FORECAST, BUFFER_NOWCAST, DEV_END, LEADS, MI, MONTHS, N_FOLDS, N_MONTHS, SEQ)

CALS = np.array([int(m[5:]) for m in MONTHS])
SIN = np.sin(2 * np.pi * (CALS - 1) / 12).astype(np.float32)
COS = np.cos(2 * np.pi * (CALS - 1) / 12).astype(np.float32)
N_CH = 9          # 0-3 scaled NDVI_mean, NDVI_std, VCI_mean, VCI_p10; 4 valid_frac; 5-6 season; 7 observed; 8 age/12
PAD = SEQ - 1     # months padded before 2016-01 (always unobserved)
NOW_INNER_YEAR = {0: 2023, 1: 2023, 2: 2023, 3: 2020}      # DESIGN.md 3 (nowcast jobs)
FC_INNER_YEAR = {0: 2023, 1: 2023, 2: 2023, 3: 2020}       # DESIGN.md 3 (forecast job)


def ref_of_fold(k: int) -> int:
    """Index into REFS of outer fold k's VCI table (f1..f4)."""
    return k


class Base:
    """Fold-independent arrays for development cells: regions, groups, sides, labels and candidate windows."""

    def __init__(self, meta: dict, arrays: dict, part: str = "dev", t_last: int = DEV_END):
        self.meta = meta
        self.arrays = arrays
        m = meta[part]
        self.region = arrays["region"].astype(np.int64)
        self.n_cells = len(self.region)
        self.n_regions = len(m["regions"])
        self.region_west = np.array([s == "west" for s in m["sides"]])
        if part == "dev":
            self.region_group = np.array(m["groups"], np.int64)
            self.group = self.region_group[self.region]
            self.drought = arrays["drought"]
        self.t_last = t_last
        n_t = t_last + 1
        self.n_t = n_t
        self.w_cell = np.repeat(np.arange(self.n_cells), n_t)
        self.w_T = np.tile(np.arange(n_t), self.n_cells)
        self.w_region = self.region[self.w_cell]
        if part == "dev":
            self.w_group = self.group[self.w_cell]
            self.label0 = self.drought[self.w_cell, self.w_T].astype(np.int64)
            lab = np.full((len(self.w_T), len(LEADS)), -1, np.int64)
            for j, h in enumerate(LEADS):
                ok = self.w_T + h <= t_last
                lab[ok, j] = self.drought[self.w_cell[ok], self.w_T[ok] + h]
            self.label_h = lab

    def wid(self, cell, T):
        return cell * self.n_t + T


def obs_stats(obs: np.ndarray, base: Base):
    """Per candidate window: T observed, and observed count among T-11..T-1 (months before 2016-01 unobserved)."""
    padded = np.concatenate([np.zeros((obs.shape[0], PAD), bool), obs[:, :base.n_t]], axis=1)
    cs = np.concatenate([np.zeros((obs.shape[0], 1), np.int64), np.cumsum(padded, axis=1)], axis=1)
    c, T = base.w_cell, base.w_T + PAD
    n_prev = cs[c, T] - cs[c, T - (SEQ - 1)]
    t_obs = obs[base.w_cell, base.w_T]
    return t_obs, n_prev


def _months_ok(T, lo, hi, buf):
    return (T < lo - buf) | (T > hi + buf)


class Sets:
    """Index arrays (into the candidate windows) for one job and outer fold k: inner train/val, outer train/val."""

    def __init__(self, base: Base, job: str, k: int, obs: np.ndarray):
        self.job, self.k = job, k
        t_obs, n_prev = obs_stats(obs, base)
        T, grp = base.w_T, base.w_group
        bs, be = BLOCK_IDX[k]
        fc = job == "job3"
        buf = BUFFER_FORECAST if fc else BUFFER_NOWCAST
        has0 = base.label0 >= 0
        if job in ("job1", "job4"):
            train_ok = t_obs & has0
            score_ok = t_obs & has0
            t_max = DEV_END
        elif job == "job2":
            train_ok = (n_prev >= 8) & has0
            score_ok = (~t_obs) & (n_prev >= 8) & has0
            t_max = DEV_END
        elif job == "job3":
            t_max = DEV_END - max(LEADS)
            has_all = (base.label_h >= 0).all(axis=1)
            train_ok = t_obs & has0 & has_all & (T <= t_max)
            score_ok = t_obs & has0 & (base.label_h >= 0).any(axis=1)
        else:
            raise ValueError(job)
        tr_months = _months_ok(T, bs, be, buf) & (T <= t_max)
        inner_g = (k + 1) % N_FOLDS
        y = (FC_INNER_YEAR if fc else NOW_INNER_YEAR)[k]
        ys, ye = MI[f"{y}-01"], MI[f"{y}-12"]
        in_year = (T >= ys) & (T <= ye)
        self.outer_train = np.where((grp != k) & tr_months & train_ok)[0]
        self.outer_val = np.where((grp == k) & (T >= bs) & (T <= min(be, t_max if fc else be)) & score_ok)[0]
        self.inner_train = np.where((grp != k) & (grp != inner_g) & tr_months & _months_ok(T, ys, ye, buf) & train_ok)[0]
        self.inner_val = np.where((grp == inner_g) & tr_months & in_year & score_ok)[0]
        self.inner_year = y
        self.t_obs, self.n_prev = t_obs, n_prev

    def get(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """(fit windows, evaluation windows) for split 'inner' or 'outer'."""
        if split == "inner":
            return self.inner_train, self.inner_val
        return self.outer_train, self.outer_val


def scaler_for(base: Base, feats: np.ndarray, obs: np.ndarray, fit_idx: np.ndarray) -> dict:
    """z-score of the four scaled features over observed cell-months inside the training windows only."""
    diff = np.zeros((base.n_cells, N_MONTHS + 1), np.int64)
    c, T = base.w_cell[fit_idx], base.w_T[fit_idx]
    np.add.at(diff, (c, np.maximum(T + 1 - SEQ, 0)), 1)
    np.add.at(diff, (c, T + 1), -1)
    used = (np.cumsum(diff, axis=1)[:, :N_MONTHS] > 0) & obs
    vals = feats[used][:, :4].astype(np.float64)
    center, spread = vals.mean(axis=0), np.maximum(vals.std(axis=0), 1e-6)
    return {"center": center.tolist(), "spread": spread.tolist(), "rows": int(used.sum())}


def input_cube(feats: np.ndarray, obs: np.ndarray, scaler: dict) -> np.ndarray:
    """(cells, PAD + months, 9) inputs with PAD unobserved months in front; age is filled per window in gather()."""
    n = feats.shape[0]
    F = np.zeros((n, PAD + N_MONTHS, N_CH), np.float32)
    center = np.asarray(scaler["center"], np.float32)
    spread = np.asarray(scaler["spread"], np.float32)
    o = obs[:, :, None]
    sl = slice(PAD, PAD + N_MONTHS)
    F[:, sl, 0:4] = np.where(o, (np.nan_to_num(feats[:, :, 0:4]) - center) / spread, 0.0)
    F[:, sl, 4] = np.where(obs, np.nan_to_num(feats[:, :, 4]), 0.0)
    cal = (np.arange(PAD + N_MONTHS) - PAD) % 12 + 1
    F[:, :, 5] = np.sin(2 * np.pi * (cal - 1) / 12)[None, :]
    F[:, :, 6] = np.cos(2 * np.pi * (cal - 1) / 12)[None, :]
    F[:, sl, 7] = obs
    return F


def fill_age(X: np.ndarray) -> np.ndarray:
    """Channel 8 = months since the last observation (0 when observed; from the window start if none yet; /12)."""
    obs = X[:, :, 7] > 0.5
    n, steps = obs.shape
    age = np.zeros((n, steps), np.float32)
    cur = np.full(n, -1.0, np.float32)
    for t in range(steps):
        seen = cur >= 0
        age[:, t] = np.where(obs[:, t], 0.0, np.where(seen, cur + 1, float(min(12, t + 1))))
        cur = np.where(obs[:, t], 0.0, np.where(seen, cur + 1, cur))
    X[:, :, 8] = age / 12.0
    return X


def gather(F: np.ndarray, base: Base, idx: np.ndarray, hide_T: bool = False) -> np.ndarray:
    """(n, 12, 9) windows for candidate indices idx; hide_T clears month T (job 2)."""
    view = sliding_window_view(F, SEQ, axis=1)               # (cells, PAD+months-11, ch, 12)
    w = view[base.w_cell[idx], base.w_T[idx]]                 # window starting at padded index T -> months T-11..T
    X = np.ascontiguousarray(w.transpose(0, 2, 1))
    if hide_T:
        X[:, -1, [0, 1, 2, 3, 4, 7]] = 0.0
    return fill_age(X)


def raw_windows(feats: np.ndarray, obs: np.ndarray, base: Base, idx: np.ndarray, hide_T: bool = False):
    """Raw VCI_mean (NaN when unobserved) and the observed mask, (n, 12) each."""
    vci = np.where(obs, feats[:, :, 2], np.nan).astype(np.float32)
    vp = np.concatenate([np.full((vci.shape[0], PAD), np.nan, np.float32), vci], axis=1)
    op = np.concatenate([np.zeros((obs.shape[0], PAD), bool), obs], axis=1)
    V = sliding_window_view(vp, SEQ, axis=1)[base.w_cell[idx], base.w_T[idx]].copy()
    O = sliding_window_view(op, SEQ, axis=1)[base.w_cell[idx], base.w_T[idx]].copy()
    if hide_T:
        V[:, -1] = np.nan
        O[:, -1] = False
    return V, O


def baseline_features(V: np.ndarray, O: np.ndarray, T: np.ndarray) -> dict:
    """Hand-built inputs for the simple models, all at or before month T (as in the capability run)."""
    n = len(V)
    ar = np.arange(n)
    pos = np.where(O, np.arange(SEQ)[None, :], -1).max(axis=1)
    has = pos >= 0
    posc = np.clip(pos, 0, None)
    v_latest = np.where(has, V[ar, posc], np.nan)
    age = np.where(has, SEQ - 1 - posc, np.nan).astype(np.float64)
    out = {"vci_t": V[:, -1].astype(np.float64), "t_obs": O[:, -1], "has": has,
           "vci_latest": v_latest.astype(np.float64), "age": age}
    for lag in (1, 3):
        p = posc - lag
        ok = has & (p >= 0) & O[ar, np.clip(p, 0, None)]
        d = np.where(ok, v_latest - V[ar, np.clip(p, 0, None)], np.nan)
        out[f"d{lag}"] = d.astype(np.float64)
        out[f"d{lag}_ok"] = ok
    last6 = O[:, SEQ - 6:]
    out["below35_6"] = (last6 & (np.nan_to_num(V[:, SEQ - 6:], nan=100.0) < 35)).sum(axis=1).astype(np.float64)
    out["nobs_6"] = last6.sum(axis=1).astype(np.float64)
    out["sin"] = SIN[T].astype(np.float64)
    out["cos"] = COS[T].astype(np.float64)
    return out


def ewma_state(V: np.ndarray, O: np.ndarray, alpha: float) -> np.ndarray:
    s = np.full(len(V), np.nan)
    for j in range(SEQ):
        o = O[:, j]
        v = V[:, j].astype(np.float64)
        s = np.where(o, np.where(np.isnan(s), v, alpha * v + (1 - alpha) * s), s)
    return s


def trend_matrix(b: dict, nan_for_missing: bool) -> np.ndarray:
    if nan_for_missing:
        cols = [b["vci_latest"], b["age"], b["d1"], b["d3"], b["below35_6"], b["nobs_6"], b["sin"], b["cos"]]
    else:
        cols = [b["vci_latest"], b["age"], np.nan_to_num(b["d1"]), b["d1_ok"].astype(float),
                np.nan_to_num(b["d3"]), b["d3_ok"].astype(float), b["below35_6"], b["nobs_6"], b["sin"], b["cos"]]
    return np.stack(cols, axis=1)


TREND_NAMES = ["vci_latest", "age", "d1", "d1_ok", "d3", "d3_ok", "below35_6", "nobs_6", "sin", "cos"]
GBT_NAMES = ["vci_latest", "age", "d1", "d3", "below35_6", "nobs_6", "sin", "cos"]
