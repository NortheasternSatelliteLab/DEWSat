"""Phase 6 benchmark: gridMET SPEI (Abatzoglou, Northwest Knowledge Network THREDDS), run only after Phase 5.

At each region's centre, every SPEI timescale gridMET serves (discovered from the THREDDS catalog) is read with the
NetCDF Subset Service as a point time series; a month takes the last value on or before its last day. The timescale
whose -SPEI best ranks the FLDAS drought flag on development cells 2016-2023 (AUC) is chosen. Then every Phase 5
model's fresh-test predictions are scored against SPEI drought (SPEI <= -0.84, the standard-normal 20th percentile),
beside their AUC against the FLDAS label, and the two drought calls' agreement is reported.
"""

from __future__ import annotations

import csv
import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from common import DATA, LOCKED, MONTHS, SCORES, log, read_json, write_json

THREDDS = "http://thredds.northwestknowledge.net:8080/thredds"
CATALOGS = (f"{THREDDS}/catalog/MET/catalog.xml", f"{THREDDS}/catalog.xml")
OUT = DATA / "phase6" / "spei"


def discover(http) -> list[tuple[str, str]]:
    """(timescale label, dataset urlPath) for every SPEI dataset found in the catalogs (two levels deep)."""
    found, seen = [], set()
    todo = list(CATALOGS)
    while todo and len(seen) < 60:
        url = todo.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            txt = http.get(url, timeout=60).text
        except Exception:  # noqa: BLE001
            continue
        for path in re.findall(r'urlPath="([^"]*spei[^"]*\.nc)"', txt, flags=re.I):
            lab = re.search(r"spei(\d+[dy]?)", path, flags=re.I)
            found.append((lab.group(1) if lab else path, path))
        base = url.rsplit("/", 1)[0]
        for ref in re.findall(r'xlink:href="([^"]+catalog\.xml)"', txt):
            if any(k in ref.lower() for k in ("spei", "drought", "met")):
                todo.append(ref if ref.startswith("http") else f"{base}/{ref}")
    return sorted(set(found))


def point_series(http, path: str, lat: float, lon: float):
    url = f"{THREDDS}/ncss/{path}"
    r = http.get(url, params={"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "time_start": "2001-01-01",
                              "time_end": "2026-09-30", "accept": "csv"}, timeout=300)
    r.raise_for_status()
    rows = list(csv.reader(io.StringIO(r.text)))
    head = rows[0]
    ti = next(i for i, h in enumerate(head) if h.lower().startswith("time"))
    vi = len(head) - 1
    dates = [x[ti][:10] for x in rows[1:]]
    vals = [float(x[vi]) if x[vi] not in ("", "NaN", "nan") else np.nan for x in rows[1:]]
    return dates, np.array(vals)


def monthly(dates, vals):
    """Last value on or before each month's last day (months 2016-01..2026-09)."""
    out = np.full(len(MONTHS), np.nan)
    d = np.array(dates)
    for j, m in enumerate(MONTHS):
        sel = np.where(d <= f"{m}-31")[0]
        sel = sel[[dd[:7] == m for dd in d[sel]]] if len(sel) else sel
        if len(sel):
            out[j] = vals[sel[-1]]
    return out


def run() -> dict:
    import httpx
    from sklearn.metrics import roc_auc_score
    import dataset
    OUT.mkdir(parents=True, exist_ok=True)
    meta = read_json(DATA.parent / "cache" / "dataset.json")
    with httpx.Client(follow_redirects=True) as http:
        sets = discover(http)
        if not sets:
            raise RuntimeError("no SPEI dataset found in the gridMET THREDDS catalogs")
        log(f"  SPEI datasets: {sets}")
        regions = [(p, n, la, lo) for p in ("dev", "test") for n, la, lo in
                   zip(meta[p]["regions"], meta[p]["lat"], meta[p]["lon"])]
        series = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(point_series, http, path, la, lo): (lab, n) for lab, path in sets for _, n, la, lo in regions}
            for f in as_completed(futs):
                lab, n = futs[f]
                try:
                    series[(lab, n)] = monthly(*f.result())
                except Exception as exc:  # noqa: BLE001
                    log(f"  SPEI {lab} {n}: {type(exc).__name__}: {exc}")
    # choose the timescale on development cells 2016-2023
    _, dev = dataset.load("dev")
    names = meta["dev"]["regions"]
    tm = np.arange(0, 96)
    best, aucs = None, {}
    for lab, _ in sets:
        s = np.stack([series.get((lab, names[r]), np.full(len(MONTHS), np.nan)) for r in dev["region"]])
        y = dev["drought"][:, tm]
        v = s[:, tm]
        ok = (y >= 0) & np.isfinite(v)
        if ok.sum() < 1000:
            continue
        aucs[lab] = float(roc_auc_score(y[ok], -v[ok]))
        if best is None or aucs[lab] > aucs[best]:
            best = lab
    res = {"title": "Benchmark: gridMET SPEI", "datasets": sets, "auc_minus_spei_for_fldas_flag_dev": aucs,
           "chosen_timescale": best, "lines": []}
    res["lines"].append("AUC of -SPEI for the FLDAS drought flag, development cells 2016-2023: " +
                        ", ".join(f"{k} {v:.3f}" for k, v in sorted(aucs.items(), key=lambda kv: -kv[1])) +
                        f"; chosen: {best}.")
    write_json(OUT / "spei_summary.json", res)
    p5 = read_json(SCORES / "phase5.json")
    if best and p5:
        res.update(score_models_against_spei(best, series, meta))
    return res


def score_models_against_spei(lab, series, meta) -> dict:
    """Fresh test: each Phase 5 model's AUC against the SPEI drought flag (SPEI <= -0.84) and against FLDAS."""
    from sklearn.metrics import roc_auc_score
    import phase5_stage as P5
    c = P5.ctx5()
    drought, keep = P5.test_labels()
    tb = c["base_test"]
    names = meta["test"]["regions"]
    out = {"lines": []}
    for job in ("job1", "job2"):
        st = read_json(SCORES / "phase5_settings.json")
        e = P5.test_windows(job, c["test"]["feats"][P5.REF_ALL], c["test"]["obs"][P5.REF_ALL],
                            P5.train_set(job, "composite")["scaler"], drought, keep)
        spei = np.array([series.get((lab, names[r]), np.full(len(MONTHS), np.nan))[t]
                         for r, t in zip(tb.region[tb.w_cell[e["idx"]]], tb.w_T[e["idx"]])])
        ok = np.isfinite(spei)
        ys = (spei <= -0.84).astype(int)
        sp, _ = P5.simple_preds(job, "composite", st["settings"][job]["simple"])
        preds = dict(sp["fresh"])
        for fam in st["models"][job]["nets"]:
            p, _ = P5._net_probs(job, "composite", fam, "fresh", P5.FINAL_SEEDS)
            preds[fam] = p
        rows = []
        for m, p in preds.items():
            pp = p[:, 0] if p.ndim == 2 else p
            g = ok & np.isfinite(pp)
            if g.sum() < 100 or len(np.unique(ys[g])) < 2:
                continue
            rows.append(f"{m}: AUC vs SPEI drought {roc_auc_score(ys[g], pp[g]):.3f}, vs FLDAS "
                        f"{roc_auc_score(e['y0'][g], pp[g]):.3f}")
        agree = float((ys[ok] == e["y0"][ok]).mean())
        out["lines"].append(f"{job} fresh test ({int(ok.sum()):,} windows; SPEI-{lab} drought rate {ys[ok].mean():.3f}, "
                            f"FLDAS {e['y0'][ok].mean():.3f}, the two calls agree on {agree:.1%}): " + "; ".join(rows) + ".")
    return out


def _unused():
    return LOCKED
