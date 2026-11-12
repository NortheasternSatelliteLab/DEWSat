"""Metric blocks shared by the pilot and the full run (PLAN.md 5). Every interval resamples whole stations."""

from __future__ import annotations

import numpy as np

from common import auc, boot_stations, ci, kappa, pearson

LABELS = ("stn", "nldas", "fldas")


def label_agreement(df: dict, sel: np.ndarray) -> dict:
    """FLDAS vs station and NLDAS vs station: anomaly r, kappa, agreement; paired NLDAS - FLDAS differences."""
    g = df["site"][sel]
    a = {k: df[f"{k}_anom"][sel] for k in LABELS}
    d = {k: df[f"{k}_dr"][sel] for k in LABELS}

    def stat(i):
        return (pearson(a["fldas"][i], a["stn"][i]), pearson(a["nldas"][i], a["stn"][i]),
                kappa(d["fldas"][i], d["stn"][i]), kappa(d["nldas"][i], d["stn"][i]))

    full = stat(np.arange(len(g)))
    bs = boot_stations(g, stat)
    out = {"n_months": int(len(g)), "n_stations": int(len(np.unique(g)))}
    names = ("r_fldas_stn", "r_nldas_stn", "kappa_fldas_stn", "kappa_nldas_stn")
    for j, nm in enumerate(names):
        out[nm] = [full[j], *ci(bs[:, j])]
    out["r_nldas_minus_fldas"] = [full[1] - full[0], *ci(bs[:, 1] - bs[:, 0])]
    out["kappa_nldas_minus_fldas"] = [full[3] - full[2], *ci(bs[:, 3] - bs[:, 2])]
    out["agree_fldas_stn"] = float(np.mean(d["fldas"] == d["stn"]))
    out["agree_nldas_stn"] = float(np.mean(d["nldas"] == d["stn"]))
    out["agree_fldas_nldas"] = float(np.mean(d["fldas"] == d["nldas"]))
    out["drought_rate"] = {k: float(d[k].mean()) for k in LABELS}
    return out


def satellite_vs_labels(df: dict, sel: np.ndarray, vci: np.ndarray, extra: dict | None = None) -> dict:
    """AUC of -VCI for each label's drought flag and r of VCI with each anomaly; paired station - FLDAS differences."""
    g = df["site"][sel]
    v = vci[sel]
    labs = {k: df[f"{k}_dr"][sel] for k in LABELS}
    if extra:
        labs.update({k: x[sel] for k, x in extra.items()})
    anoms = {k: df[f"{k}_anom"][sel] for k in LABELS}
    keys = list(labs)

    def stat(i):
        return tuple(auc(-v[i], labs[k][i]) for k in keys) + tuple(pearson(v[i], anoms[k][i]) for k in LABELS)

    full = stat(np.arange(len(g)))
    bs = boot_stations(g, stat)
    out = {"n_months": int(len(g)), "n_stations": int(len(np.unique(g)))}
    for j, k in enumerate(keys):
        out[f"auc_{k}"] = [full[j], *ci(bs[:, j])]
    for j, k in enumerate(LABELS):
        out[f"r_vci_{k}_anom"] = [full[len(keys) + j], *ci(bs[:, len(keys) + j])]
    js, jf, jn = keys.index("stn"), keys.index("fldas"), keys.index("nldas")
    out["auc_stn_minus_fldas"] = [full[js] - full[jf], *ci(bs[:, js] - bs[:, jf])]
    out["auc_nldas_minus_fldas"] = [full[jn] - full[jf], *ci(bs[:, jn] - bs[:, jf])]
    rs, rf = len(keys) + 0, len(keys) + 2
    out["r_stn_minus_fldas"] = [full[rs] - full[rf], *ci(bs[:, rs] - bs[:, rf])]
    return out


def model_vs_labels(df: dict, sel: np.ndarray, probs: dict[str, np.ndarray]) -> dict:
    """For each model's probability: AUC against station and FLDAS drought on the same windows, paired difference."""
    g = df["site"][sel]
    ys, yf = df["stn_dr"][sel], df["fldas_dr"][sel]
    names = list(probs)

    def stat(i):
        vals = []
        for nm in names:
            p = probs[nm][sel][i]
            vals += [auc(p, ys[i]), auc(p, yf[i])]
        return tuple(vals)

    full = stat(np.arange(len(g)))
    bs = boot_stations(g, stat)
    out = {"n_months": int(len(g)), "n_stations": int(len(np.unique(g)))}
    for j, nm in enumerate(names):
        a, b = 2 * j, 2 * j + 1
        out[nm] = {"auc_stn": [full[a], *ci(bs[:, a])], "auc_fldas": [full[b], *ci(bs[:, b])],
                   "stn_minus_fldas": [full[a] - full[b], *ci(bs[:, a] - bs[:, b])]}
    return out
