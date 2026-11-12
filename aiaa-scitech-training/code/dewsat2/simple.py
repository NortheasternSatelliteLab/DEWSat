"""Simple models (DESIGN.md 5-6): climatology, LR on VCI(T), LR on latest VCI + age, LR + trend, boosted trees
monotone in VCI (30-configuration search on the inner split), smoothed-VCI state model. Ported from the capability
run's baselines.py (commit e13e7a4) with the search added."""

from __future__ import annotations

import math
import warnings

import numpy as np

from search import gbt_configs
from windows import GBT_NAMES, TREND_NAMES, ewma_state, trend_matrix

ALPHAS = (0.2, 0.35, 0.5, 0.65, 0.8, 1.0)


def auc(score, y):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    ok = np.isfinite(score)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], score[ok]))


class LR:
    """Standardized-input logistic regression, C = 1e6 (the paper's setting)."""

    def __init__(self, names):
        self.names = list(names)

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m = LogisticRegression(C=1e6, max_iter=5000).fit((X - self.mu) / self.sd, y)
        return self

    def logit(self, X):
        if len(X) == 0:
            return np.zeros(0)
        return self.m.decision_function((X - self.mu) / self.sd)

    def predict(self, X):
        return 1 / (1 + np.exp(-self.logit(X)))

    def raw_coefficients(self):
        b = self.m.coef_[0] / self.sd
        a = float(self.m.intercept_[0] - np.sum(self.m.coef_[0] * self.mu / self.sd))
        return a, b.tolist()


class GBT:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        cst = [-1 if n == "vci_latest" else 0 for n in GBT_NAMES]
        p = {k: v for k, v in self.cfg.items() if k != "id"}
        self.m = HistGradientBoostingClassifier(monotonic_cst=cst, early_stopping=False, random_state=0, **p).fit(X, y)
        return self

    def predict(self, X):
        if len(X) == 0:
            return np.zeros(0)
        return self.m.predict_proba(X)[:, 1]


class State:
    def __init__(self, alpha: float):
        self.alpha = alpha

    @staticmethod
    def _x(V, O, a):
        s = ewma_state(V, O, a)
        pos = np.where(O, np.arange(O.shape[1])[None, :], -1).max(axis=1)
        return np.stack([s, (O.shape[1] - 1 - pos).astype(float)], axis=1)

    def fit(self, V, O, y):
        self.lr = LR(["state", "age"]).fit(self._x(V, O, self.alpha), y)
        return self

    def predict(self, V, O):
        if len(V) == 0:
            return np.zeros(0)
        return self.lr.predict(self._x(V, O, self.alpha))


def _sub(d, m):
    return {k: v[m] for k, v in d.items()}


def fit_predict(f_fit, V_fit, O_fit, y_fit, f_ev, V_ev, O_ev, gbt_cfg: dict, alpha: float, with_vci_t: bool):
    """Fit every simple model on one target; predictions (n_ev,) per model (NaN where a model does not apply)."""
    out, info = {}, {}
    rate = float(y_fit.mean())
    out["climatology"] = np.full(len(V_ev), rate)
    info["rate"] = rate
    if with_vci_t:
        t = f_fit["t_obs"]
        m = LR(["vci_t"]).fit(f_fit["vci_t"][t][:, None], y_fit[t])
        p = np.full(len(V_ev), np.nan)
        te = f_ev["t_obs"]
        p[te] = m.predict(f_ev["vci_t"][te][:, None])
        out["lr_vci_t"] = p
        a, b = m.raw_coefficients()
        info["lr_vci_t"] = {"a": a, "b": b[0]}
    h = f_fit["has"]
    he = f_ev["has"]
    xa = np.stack([f_fit["vci_latest"], f_fit["age"]], axis=1)
    m2 = LR(["vci_latest", "age"]).fit(xa[h], y_fit[h])
    a2, b2 = m2.raw_coefficients()
    info["lr_latest_age"] = {"a": a2, "b": b2[0], "c": b2[1], "a0": math.log(max(rate, 1e-6) / max(1 - rate, 1e-6))}
    xe = np.stack([f_ev["vci_latest"], f_ev["age"]], axis=1)
    p = np.full(len(V_ev), np.nan)
    p[he] = m2.predict(xe[he])
    out["lr_latest_age"] = p
    m3 = LR(TREND_NAMES).fit(trend_matrix(f_fit, False)[h], y_fit[h])
    p = np.full(len(V_ev), np.nan)
    p[he] = m3.predict(trend_matrix(f_ev, False)[he])
    out["lr_trend"] = p
    g = GBT(gbt_cfg).fit(trend_matrix(f_fit, True)[h], y_fit[h])
    p = np.full(len(V_ev), np.nan)
    p[he] = g.predict(trend_matrix(f_ev, True)[he])
    out["gbt_mono"] = p
    s = State(alpha).fit(V_fit[h], O_fit[h], y_fit[h])
    p = np.full(len(V_ev), np.nan)
    p[he] = s.predict(V_ev[he], O_ev[he])
    out["state_ewma"] = p
    return out, info


def search_gbt(f_fit, y_fit, f_ev, y_ev, job: str) -> list[dict]:
    """Inner-split AUC of every boosted-tree configuration (one target)."""
    h, he = f_fit["has"], f_ev["has"]
    Xf, Xe = trend_matrix(f_fit, True)[h], trend_matrix(f_ev, True)[he]
    res = []
    for cfg in gbt_configs(job):
        g = GBT(cfg).fit(Xf, y_fit[h])
        p = np.full(len(y_ev), np.nan)
        p[he] = g.predict(Xe)
        res.append({"cfg": cfg, "auc": auc(p, y_ev), "pred": p})
    return res


def search_alpha(V_fit, O_fit, y_fit, f_fit, V_ev, O_ev, y_ev, f_ev) -> list[dict]:
    h, he = f_fit["has"], f_ev["has"]
    res = []
    for a in ALPHAS:
        s = State(a).fit(V_fit[h], O_fit[h], y_fit[h])
        p = np.full(len(y_ev), np.nan)
        p[he] = s.predict(V_ev[he], O_ev[he])
        res.append({"alpha": a, "auc": auc(p, y_ev), "pred": p})
    return res


def lr_coefs(f_fit, y_fit, with_vci_t: bool) -> dict:
    """Raw-unit coefficients of LR on VCI(T) and of LR on latest VCI + age (the residual networks' bases)."""
    info = {}
    rate = float(y_fit.mean())
    if with_vci_t:
        t = f_fit["t_obs"]
        a, b = LR(["vci_t"]).fit(f_fit["vci_t"][t][:, None], y_fit[t]).raw_coefficients()
        info["lr_vci_t"] = {"a": a, "b": b[0]}
    h = f_fit["has"]
    xa = np.stack([f_fit["vci_latest"], f_fit["age"]], axis=1)
    a2, b2 = LR(["vci_latest", "age"]).fit(xa[h], y_fit[h]).raw_coefficients()
    info["lr_latest_age"] = {"a": a2, "b": b2[0], "c": b2[1], "a0": math.log(max(rate, 1e-6) / max(1 - rate, 1e-6))}
    return info
