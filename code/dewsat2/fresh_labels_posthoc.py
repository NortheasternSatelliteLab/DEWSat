"""Post-hoc, after the lock and Phase 6: are the fresh-test results specific to the FLDAS label? scores/fresh_labels_posthoc.json.

The saved fresh-test predictions of jobs 1 and 2 (Phase 5 networks' pred.npz, the simple models refit as Phase 5 did)
are scored against other drought labels on the same windows, with paired differences against LR and the job's best
simple model (scoring.score_subset, 2,000 whole-region resamples):
  - FLDAS (the label every model was trained on; the official result, recomputed on the common windows);
  - SPEI-180d drought (SPEI <= -0.84), the timescale Phase 6 chose on development cells (DECISIONS.md #77);
  - SPEI-90d and SPEI-270d, the runners-up within 0.01 AUC of it, as a sensitivity check of that choice;
  - NLDAS-2 Noah 10-40 cm soil moisture below its calendar-month 20th percentile (2001-2022), the label check of
    Phase 6, at the NLDAS cell nearest each cell centre.
Every label is compared on the windows where all of them exist. Not a second test of the FLDAS label.

    cd ~/Desktop/DEWSat_BiLSTM_v2 && source env.sh && .venv/bin/python -u code/dewsat2/fresh_labels_posthoc.py
"""

from __future__ import annotations

import time

import numpy as np

import phase5_stage as P5
import phase6_spei as S
import phase6_stage as P6
import scoring
from common import DATA, FINAL_SEEDS, MONTHS, SCORES, git_state, log, read_json, write_json

EXTRA_SPEI = ("90d", "270d")
SPEI_DROUGHT = -0.84


def spei_series(meta) -> dict:
    """Monthly SPEI per (timescale, fresh-test region) for the chosen timescale and the runners-up (cached)."""
    import httpx
    sp = read_json(SCORES / "phase6.json")["spei"]
    paths = dict(map(tuple, sp["datasets"]))
    labs = [sp["chosen_timescale"], *[x for x in EXTRA_SPEI if x != sp["chosen_timescale"]]]
    cen = {n: (la, lo) for n, la, lo in zip(meta["test"]["regions"], meta["test"]["lat"], meta["test"]["lon"])}
    pack = {lab: tuple(sp["packing"][lab]) for lab in labs}
    with httpx.Client(follow_redirects=True, limits=httpx.Limits(max_connections=10)) as http:
        series = S.fetch(http, [(lab, paths[lab], n, *cen[n]) for lab in labs for n in meta["test"]["regions"]], pack)
    return {"labels": labs, "series": series}


def nldas_flags(tb) -> np.ndarray:
    """(fresh-test cells, MONTHS) NLDAS-2 drought flag (-1 where unknown), as Phase 6's label check builds it."""
    months = P6.label_months()
    info = read_json(SCORES / "phase6.json")["nldas"]
    nd = DATA / "phase6" / "nldas"
    cube = np.stack([np.load(nd / f"{m}.npy") for m in months])
    lon0, lon1, lat0, lat1 = -124.938, -67.0625, 25.0625, 52.9375          # grid endpoints logged by Phase 6 (#77)
    ny, nx = cube.shape[1:]
    dx, dy = (lon1 - lon0) / (nx - 1), (lat1 - lat0) / (ny - 1)
    import dataset
    _, arr = dataset.load("test")
    ids = arr["cell_id"]
    lon = np.array([float(s.split(":")[1]) for s in ids])
    lat = np.array([float(s.split(":")[2]) for s in ids])
    xi = np.clip(np.round((lon - lon0) / dx).astype(int), 0, nx - 1)
    yi = np.clip(np.round((lat - lat0) / dy).astype(int), 0, ny - 1)
    _, flag = P6.percentile_flags(cube[:, yi, xi].T, months)
    off = months.index(MONTHS[0])
    if tb.n_cells != len(ids):
        raise RuntimeError("fresh-test cell tables differ")
    del info
    return flag[:, off:off + len(MONTHS)]


def run() -> dict:
    if not P5.LOCK.exists() or not (SCORES / "phase6.json").exists():
        raise SystemExit("needs the scored fresh test and Phase 6")
    t0 = time.time()
    meta = read_json(DATA.parent / "cache" / "dataset.json")
    sp = spei_series(meta)
    log(f"fresh_labels_posthoc: SPEI series for {sp['labels']} ready in {time.time() - t0:.0f}s")
    st = read_json(SCORES / "phase5_settings.json")
    c = P5.ctx5()
    tb = c["base_test"]
    flag_n = nldas_flags(tb)
    names = meta["test"]["regions"]
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state(),
           "note": "post-hoc, after the lock and Phase 6: the same saved predictions scored against other drought labels",
           "spei_chosen": sp["labels"][0], "spei_drought_threshold": SPEI_DROUGHT, "jobs": {}}
    for job in ("job1", "job2"):
        mj = st["models"][job]
        spr, evs = P5.simple_preds(job, "composite", st["settings"][job]["simple"])
        e = dict(evs)["fresh"]
        preds = {m: (p[:, 0] if p.ndim == 2 else p) for m, p in spr["fresh"].items()}
        for fam in mj["nets"] + mj.get("extra", []):
            p, idx = P5._net_probs(job, "composite", fam, "fresh", FINAL_SEEDS)
            if p is not None:
                if not np.array_equal(idx, e["idx"]):
                    raise RuntimeError(f"{job} {fam}: window ids differ")
                preds[fam] = p[:, 0] if p.ndim == 2 else p
        preds = {m: p for m, p in preds.items() if np.isfinite(p).all() and m != "climatology"}
        cell, T = tb.w_cell[e["idx"]], tb.w_T[e["idx"]]
        reg = e["region"]
        labels = {"fldas": e["y0"].astype(float)}
        for lab in sp["labels"]:
            v = np.array([sp["series"].get((lab, names[r]), np.full(len(MONTHS), np.nan))[t] for r, t in zip(reg, T)])
            labels[f"spei_{lab}"] = np.where(np.isfinite(v), (v <= SPEI_DROUGHT).astype(float), np.nan)
        fn = flag_n[cell, T].astype(float)
        labels["nldas"] = np.where(fn >= 0, fn, np.nan)
        common = np.all([np.isfinite(v) for v in labels.values()], axis=0)
        lr, simp = mj["lr"], mj["best_simple"]
        pairs = [(m, cc) for cc in dict.fromkeys([lr, simp]) for m in preds if m != cc]
        R = tb.n_regions
        fold0 = np.zeros(R, np.int64)
        res = {"windows": int(len(common)), "common_windows": int(common.sum()), "lr": lr, "best_simple": simp,
               "labels": {}}
        for name, y in labels.items():
            yy = y[common].astype(int)
            s = scoring.score_subset(f"fresh_{job}", yy, reg[common], np.asarray(e["west"])[common],
                                     {m: p[common] for m, p in preds.items()}, fold0, R, 1, pairs)
            res["labels"][name] = {"drought_rate": float(yy.mean()),
                                   "agree_with_fldas": float((yy == labels["fldas"][common]).mean()),
                                   "auc": {m: s["models"][m]["all"]["auc_within"] for m in preds},
                                   "paired": {k: {sd: v[sd]["dauc_within"] for sd in scoring.SIDES}
                                              for k, v in s["paired"].items()}}
            log(f"  {job} {name}: drought {yy.mean():.3f}; " + ", ".join(
                f"{m} {s['models'][m]['all']['auc_within'][0]:.3f}" for m in preds))
        out["jobs"][job] = res
    out["seconds"] = round(time.time() - t0, 1)
    write_json(SCORES / "fresh_labels_posthoc.json", out)
    log(f"fresh_labels_posthoc: wrote {SCORES / 'fresh_labels_posthoc.json'} in {out['seconds']:.0f}s")
    return out


if __name__ == "__main__":
    run()
