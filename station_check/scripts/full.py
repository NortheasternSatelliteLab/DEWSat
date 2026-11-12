"""Full run (PLAN.md 5, amendment 1): all stations, 1 km chip VCI (R7, R10, R7 wide), label agreement, scale check.

Reads results/labels_monthly.csv, results/chip_vci.csv, results/sites_check.csv and v2's dev cache (read-only).
Writes results/full.json and prints a summary.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from common import GROWING, RES, SCORE_MONTHS, V2, auc, boot_stations, ci, save_json
from pilot import BLOCKS, load_labels, table
from scoring import label_agreement, satellite_vs_labels


def paired_auc(df, sel, va, vb, label_key):
    """AUC(-va) - AUC(-vb) for one label, paired station resamples. Returns [a, b, diff, lo, hi]."""
    g, y = df["site"][sel], df[label_key][sel]
    a, b = va[sel], vb[sel]

    def stat(i):
        return auc(-a[i], y[i]), auc(-b[i], y[i])

    fa, fb = stat(np.arange(len(g)))
    bs = boot_stations(g, stat)
    d = bs[:, 0] - bs[:, 1]
    return {"n_months": int(len(g)), "n_stations": int(len(np.unique(g))), "auc_a": fa, "auc_b": fb,
            "diff": [fa - fb, *ci(d)]}


def main() -> None:
    sites = {r["site_id"]: r for r in csv.DictReader(open(RES / "sites_check.csv"))}
    rows = load_labels()
    df = table(rows, set(SCORE_MONTHS))
    chip = {}
    with open(RES / "chip_vci.csv") as fh:
        for r in csv.DictReader(fh):
            chip[(r["site_id"], r["month"])] = r

    def col(name):
        out = np.full(len(df["site"]), np.nan)
        for i, (s, m) in enumerate(zip(df["site"], df["month"])):
            r = chip.get((s, m))
            if r and r.get(name, "") not in ("", None):
                out[i] = float(r[name])
        return out

    v7, v10, v7w = col("vci_r7"), col("vci_r10"), col("vci_r7wide")
    grow = np.isin(df["cal"], list(GROWING))
    tr = df["fldas_tr"]
    side = np.array([sites[s]["side"] for s in df["site"]])
    out = {}
    for season, ss in (("growing", grow), ("all", np.ones(len(grow), bool))):
        res = {"label_agreement": label_agreement(df, np.flatnonzero(ss))}
        sel = ss & np.isfinite(v7) & (tr >= 0)
        res["chip_r7"] = satellite_vs_labels(df, np.flatnonzero(sel), v7, extra={"fldas_trained": np.where(tr >= 0, tr, 0)})
        for sd in ("west", "east"):
            s2 = sel & (side == sd)
            res[f"chip_r7_{sd}"] = satellite_vs_labels(df, np.flatnonzero(s2), v7)
        both = ss & np.isfinite(v7) & np.isfinite(v10)
        res["r10_minus_r7"] = {lab: paired_auc(df, np.flatnonzero(both), v10, v7, f"{lab}_dr") for lab in ("stn", "fldas")}
        res["r7wide_minus_r7"] = {lab: paired_auc(df, np.flatnonzero(both & np.isfinite(v7w)), v7w, v7, f"{lab}_dr")
                                  for lab in ("stn", "fldas")}
        out[season] = res
    # amendment 1: 1 km chip vs 11 km cell at the pilot (development-region) stations
    z = np.load(V2 / "cache" / "dev.npz")
    cid = {str(c): i for i, c in enumerate(z["cell_id"])}
    feats, obs = z["feats"], z["obs"]
    months = [f"{y}-{m:02d}" for y in range(2016, 2027) for m in range(1, 13)][:feats.shape[2]]
    mi = {m: i for i, m in enumerate(months)}
    cellv = np.full(len(df["site"]), np.nan)
    for i, (s, m) in enumerate(zip(df["site"], df["month"])):
        if sites[s]["v2_role"] != "dev":
            continue
        c = cid.get(sites[s]["v2_cell_id"], -1)
        if c < 0:
            continue
        y = int(m[:4])
        ref = next((k for k, (a, b) in enumerate(BLOCKS) if a <= y <= b), 4)
        if obs[ref, c, mi[m]]:
            cellv[i] = feats[ref, c, mi[m], 2]
    for season, ss in (("growing", grow), ("all", np.ones(len(grow), bool))):
        both = ss & np.isfinite(v7) & np.isfinite(cellv)
        out[season]["scale_chip_minus_cell"] = {lab: paired_auc(df, np.flatnonzero(both), v7, cellv, f"{lab}_dr")
                                                for lab in ("stn", "fldas", "nldas")}
    save_json(out, RES / "full.json")

    def p(x):
        return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"

    for season in ("growing", "all"):
        r = out[season]
        la, sv = r["label_agreement"], r["chip_r7"]
        print(f"\n===== {season}")
        print(f"label agreement: {la['n_months']} months, {la['n_stations']} stations")
        for k in ("r_fldas_stn", "r_nldas_stn", "r_nldas_minus_fldas", "kappa_fldas_stn", "kappa_nldas_stn",
                  "kappa_nldas_minus_fldas"):
            print(f"  {k}: {p(la[k])}")
        print(f"  agree fldas-stn {la['agree_fldas_stn']:.3f}, nldas-stn {la['agree_nldas_stn']:.3f}, "
              f"fldas-nldas {la['agree_fldas_nldas']:.3f}; drought rates {la['drought_rate']}")
        print(f"1 km chip VCI (R7): {sv['n_months']} months, {sv['n_stations']} stations")
        for k in ("auc_stn", "auc_nldas", "auc_fldas", "auc_fldas_trained", "auc_stn_minus_fldas",
                  "auc_nldas_minus_fldas", "r_vci_stn_anom", "r_vci_nldas_anom", "r_vci_fldas_anom", "r_stn_minus_fldas"):
            print(f"  {k}: {p(sv[k])}")
        for sd in ("west", "east"):
            x = r[f"chip_r7_{sd}"]
            print(f"  {sd}: {x['n_stations']} st, {x['n_months']} mo | auc_stn {p(x['auc_stn'])} auc_fldas {p(x['auc_fldas'])} "
                  f"diff {p(x['auc_stn_minus_fldas'])}")
        for nm in ("r10_minus_r7", "r7wide_minus_r7", "scale_chip_minus_cell"):
            for lab, x in r[nm].items():
                print(f"  {nm} [{lab}]: {x['n_stations']} st, {x['n_months']} mo | a {x['auc_a']:.3f} b {x['auc_b']:.3f} "
                      f"diff {p(x['diff'])}")


if __name__ == "__main__":
    main()
