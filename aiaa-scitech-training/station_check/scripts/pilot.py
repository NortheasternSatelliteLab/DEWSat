"""Pilot (PLAN.md 4-6): development-region stations, v2's 11 km cell VCI, the data checks, then the pilot metrics.

Reads v2's dev cache and held-out job-1 predictions (read-only). Writes results/pilot.json.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from common import GROWING, RES, SCORE_MONTHS, V2, pearson, save_json
from scoring import label_agreement, model_vs_labels, satellite_vs_labels

V2_MONTHS = None  # 2016-01 .. 2026-09, index 0 = 2016-01
BLOCKS = [(2016, 2017), (2018, 2019), (2020, 2021), (2022, 2023)]
NETS = ("gru", "resbilstm", "bilstm")
N_T = 96  # development window index: cell * 96 + T (v2 windows.py, DEV_END = 2023-12)


def load_labels() -> list[dict]:
    with open(RES / "labels_monthly.csv") as fh:
        return list(csv.DictReader(fh))


def table(rows: list[dict], months: set[str]) -> dict:
    keep = [r for r in rows if r["month"] in months and r["stn_dr"] != "" and r["nldas_dr"] != "" and r["fldas_dr"] != ""]
    df = {"site": np.array([r["site_id"] for r in keep]), "month": np.array([r["month"] for r in keep]),
          "cal": np.array([int(r["cal"]) for r in keep])}
    for k in ("stn", "nldas", "fldas"):
        df[f"{k}_dr"] = np.array([int(r[f"{k}_dr"]) for r in keep])
        df[f"{k}_anom"] = np.array([float(r[f"{k}_anom"]) for r in keep])
    df["fldas_tr"] = np.array([int(r["fldas_dr_trained"]) if r["fldas_dr_trained"] != "" else -1 for r in keep])
    return df


def main() -> None:
    sites = {r["site_id"]: r for r in csv.DictReader(open(RES / "sites_check.csv"))}
    dev = {s: r for s, r in sites.items() if r["v2_role"] == "dev"}
    z = np.load(V2 / "cache" / "dev.npz")
    cid = {str(c): i for i, c in enumerate(z["cell_id"])}
    feats, obs = z["feats"], z["obs"]
    months = [f"{y}-{m:02d}" for y in range(2016, 2027) for m in range(1, 13)][:feats.shape[2]]
    mi = {m: i for i, m in enumerate(months)}
    rows = [r for r in load_labels() if r["site_id"] in dev]
    df = table(rows, set(SCORE_MONTHS))
    cell = np.array([cid.get(dev[s]["v2_cell_id"], -1) for s in df["site"]])
    missing = sorted({s for s, c in zip(df["site"], cell) if c < 0})
    vci = np.full(len(cell), np.nan)
    for i, (c, m) in enumerate(zip(cell, df["month"])):
        if c < 0:
            continue
        y = int(m[:4])
        ref = next((k for k, (a, b) in enumerate(BLOCKS) if a <= y <= b), 4)
        t = mi[m]
        if obs[ref, c, t]:
            vci[i] = feats[ref, c, t, 2]
    has = np.isfinite(vci)
    grow = np.isin(df["cal"], list(GROWING))
    # data checks (PLAN.md 6)
    a_ok = has & grow
    n_st_a = len(np.unique(df["site"][a_ok]))
    per_station = []
    all_labels = load_labels()
    for s in np.unique(df["site"]):
        allrows = [r for r in all_labels if r["site_id"] == s and r["stn_anom"] != ""
                   and r["nldas_anom"] != "" and r["fldas_anom"] != ""]
        st = np.array([float(r["stn_anom"]) for r in allrows])
        nl = np.array([float(r["nldas_anom"]) for r in allrows])
        fl = np.array([float(r["fldas_anom"]) for r in allrows])
        per_station.append((s, pearson(st, nl), pearson(st, fl)))
    pos = sum(1 for _, a, b in per_station if (a > 0) or (b > 0))
    checks = {"a_stations": n_st_a, "a_growing_months": int(a_ok.sum()),
              "a_pass": bool(n_st_a >= 30 and a_ok.sum() >= 1000),
              "b_positive_stations": pos, "b_total": len(per_station),
              "b_pass": bool(pos >= 2 / 3 * len(per_station)),
              "stations_without_cell": missing}
    checks["go_full_run"] = checks["a_pass"] and checks["b_pass"]
    out = {"checks": checks, "per_station_r": [{"site": s, "r_stn_nldas": a, "r_stn_fldas": b} for s, a, b in per_station]}
    # metrics
    for season, sel_season in (("growing", grow), ("all", np.ones(len(grow), bool))):
        sel = sel_season & has
        lab_sel = sel_season
        tr = df["fldas_tr"]
        extra = {"fldas_trained": np.where(tr >= 0, tr, 0)}
        sel_tr = sel & (tr >= 0)
        out[season] = {
            "label_agreement": label_agreement(df, np.flatnonzero(lab_sel)),
            "satellite_cell_vci": satellite_vs_labels(df, np.flatnonzero(sel_tr), vci, extra=extra),
        }
    # v2 held-out predictions (job 1), each fold's validation block
    wid_rows = {}
    for i, (c, m) in enumerate(zip(cell, df["month"])):
        if c >= 0 and m <= "2023-12":
            wid_rows[c * N_T + mi[m]] = i
    probs = {nm: np.full(len(cell), np.nan) for nm in ("lr_vci_t", "gbt_mono", *NETS)}
    for k in range(4):
        sp = np.load(V2 / "runs" / "simple" / "job1" / f"fold{k + 1}" / "pred.npz")
        for j, nm in ((1, "lr_vci_t"), (4, "gbt_mono")):
            for w, p in zip(sp["wid"], sp["prob"][:, 0, j]):
                if int(w) in wid_rows:
                    probs[nm][wid_rows[int(w)]] = p
        for nm in NETS:
            acc, cnt, wids = None, 0, None
            for sd in range(42, 52):
                pz = np.load(V2 / "runs" / "final" / "job1" / nm / f"fold{k + 1}" / f"seed{sd}" / "pred.npz")
                p = 1 / (1 + np.exp(-pz["logit"][:, 0].astype(np.float64)))
                if wids is None:
                    wids, acc = pz["wid"], p
                else:
                    assert np.array_equal(wids, pz["wid"])
                    acc = acc + p
                cnt += 1
            pm = acc / cnt
            for w, p in zip(wids, pm):
                if int(w) in wid_rows:
                    probs[nm][wid_rows[int(w)]] = p
    have = np.all([np.isfinite(v) for v in probs.values()], axis=0)
    for season, sel_season in (("growing", grow), ("all", np.ones(len(grow), bool))):
        out[season]["v2_heldout_models"] = model_vs_labels(df, np.flatnonzero(sel_season & have), probs)
    save_json(out, RES / "pilot.json")
    print(json.dumps(out["checks"], indent=1))
    for season in ("growing", "all"):
        la, sv = out[season]["label_agreement"], out[season]["satellite_cell_vci"]
        print(f"\n[{season}] label agreement over {la['n_months']} months, {la['n_stations']} stations")
        for k in ("r_fldas_stn", "r_nldas_stn", "r_nldas_minus_fldas", "kappa_fldas_stn", "kappa_nldas_stn"):
            print(f"  {k}: {la[k][0]:.3f} [{la[k][1]:.3f}, {la[k][2]:.3f}]")
        print(f"  cell VCI over {sv['n_months']} months, {sv['n_stations']} stations")
        for k in ("auc_stn", "auc_nldas", "auc_fldas", "auc_fldas_trained", "auc_stn_minus_fldas",
                  "r_vci_stn_anom", "r_vci_fldas_anom"):
            print(f"  {k}: {sv[k][0]:.3f} [{sv[k][1]:.3f}, {sv[k][2]:.3f}]")
        mv = out[season]["v2_heldout_models"]
        print(f"  v2 held-out models over {mv['n_months']} windows, {mv['n_stations']} stations")
        for nm in ("lr_vci_t", "gbt_mono", *NETS):
            x = mv[nm]
            print(f"   {nm}: stn {x['auc_stn'][0]:.3f}  fldas {x['auc_fldas'][0]:.3f}  diff {x['stn_minus_fldas'][0]:+.3f} "
                  f"[{x['stn_minus_fldas'][1]:+.3f}, {x['stn_minus_fldas'][2]:+.3f}]")


if __name__ == "__main__":
    main()
