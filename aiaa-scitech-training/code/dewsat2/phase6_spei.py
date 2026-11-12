"""Phase 6 benchmark: gridMET SPEI (Abatzoglou, Northwest Knowledge Network THREDDS), run only after Phase 5.

At each region's centre, SPEI is read with the NetCDF Subset Service as a point time series (2016-01..2026-09, the
months used); a month takes the last value on or before its last day. The timescale whose -SPEI best ranks the FLDAS
drought flag on development cells 2016-2023 (AUC) is chosen, on a seeded sample of 35 West and 35 East development
regions: each point request takes about 22 s on the server, so all 558 regions x 8 timescales would be about 4.5
hours and 4,500 requests (DECISIONS.md #77). Then every Phase 5 model's fresh-test predictions are scored against SPEI
drought (SPEI <= -0.84, the standard-normal 20th percentile) on all 144 fresh-test regions with the chosen timescale,
beside their AUC against the FLDAS label, and the two drought calls' agreement is reported. NCSS returns the packed
integers, so each dataset's scale_factor, add_offset and _FillValue are read from its DAS and applied.
"""

from __future__ import annotations

import csv
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from common import DATA, LOCKED, MONTHS, SCORES, log, read_json, write_json

THREDDS = "http://thredds.northwestknowledge.net:8080/thredds"
# MET/spei/catalog.xml lists the SPEI datasets directly. MET/catalog.xml now answers 404 and the root catalog no
# longer links to MET, so neither old entry point reached them (DECISIONS.md #77); they stay as fallbacks.
CATALOGS = (f"{THREDDS}/catalog/MET/spei/catalog.xml", f"{THREDDS}/catalog/MET/catalog.xml", f"{THREDDS}/catalog.xml")
OUT = DATA / "phase6" / "spei"
SAMPLE_PER_SIDE = 35               # development regions per side (West/East of 98 W) for the timescale choice
SAMPLE_SEED = 20261109


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


def packing(http, path: str) -> tuple[float, float, float | None]:
    """(scale_factor, add_offset, _FillValue) of the spei variable, from the dataset's OPeNDAP DAS."""
    das = http.get(f"{THREDDS}/dodsC/{path}.das", timeout=120).text
    blk = re.search(r"\bspei \{(.*?)\}", das, re.S)
    txt = blk.group(1) if blk else ""

    def num(name, default):
        m = re.search(rf"{name}\s+(-?[\d.eE+-]+);", txt)
        return float(m.group(1)) if m else default
    return num("scale_factor", 1.0), num("add_offset", 0.0), num("_FillValue", None)


def point_series(http, path: str, lat: float, lon: float, pack=(1.0, 0.0, None)):
    """Dates and SPEI values at a point, 2016-01..2026-09, unpacked with the dataset's packing."""
    url = f"{THREDDS}/ncss/{path}"
    # NCSS now answers 400 "No variables requested" without var=; the gridMET SPEI grids are named "spei" (#77)
    r = http.get(url, params={"var": "spei", "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
                              "time_start": "2016-01-01", "time_end": "2026-09-30", "accept": "csv"}, timeout=300)
    r.raise_for_status()
    rows = list(csv.reader(io.StringIO(r.text)))
    head = rows[0]
    ti = next(i for i, h in enumerate(head) if h.lower().startswith("time"))
    vi = len(head) - 1
    dates = [x[ti][:10] for x in rows[1:]]
    raw = np.array([float(x[vi]) if x[vi] not in ("", "NaN", "nan") else np.nan for x in rows[1:]])
    scale, offset, fill = pack
    if fill is not None:
        raw[raw == fill] = np.nan
    if np.nanmax(np.abs(raw), initial=0.0) <= 10:     # already unpacked by the server: SPEI is within about +-5
        return dates, raw
    return dates, raw * scale + offset


def fetch(http, todo, pack, workers: int = 8) -> dict:
    """Monthly SPEI per (timescale, region) for todo = [(label, path, region, lat, lon)], cached in OUT/series so a
    retry skips what it already has; each request is tried 3 times. Failures are logged and left out."""
    cache = OUT / "series"
    cache.mkdir(parents=True, exist_ok=True)

    def one(lab, path, n, la, lo):
        f = cache / f"{lab}__{re.sub(r'[^\w.-]', '_', n)}.npy"
        if f.exists():
            return np.load(f)
        for attempt in range(3):
            try:
                d, v = point_series(http, path, la, lo, pack[lab])
                break
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    raise
                time.sleep(15 * (attempt + 1))
        m = monthly(d, v)
        np.save(f, m)
        return m
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, *x): (x[0], x[2]) for x in todo}
        for k, fu in enumerate(as_completed(futs), 1):
            lab, n = futs[fu]
            try:
                out[(lab, n)] = fu.result()
            except Exception as exc:  # noqa: BLE001
                log(f"  SPEI {lab} {n}: {type(exc).__name__}: {exc}")
            if k % 100 == 0:
                log(f"  SPEI: {k}/{len(todo)} point series")
    return out


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
    cen = {(p, n): (la, lo) for p in ("dev", "test") for n, la, lo in
           zip(meta[p]["regions"], meta[p]["lat"], meta[p]["lon"])}
    dev_names = meta["dev"]["regions"]
    rng = np.random.default_rng(SAMPLE_SEED)
    west = [n for n in dev_names if cen[("dev", n)][1] < -98]
    east = [n for n in dev_names if cen[("dev", n)][1] >= -98]
    sample = sorted(rng.choice(west, SAMPLE_PER_SIDE, replace=False).tolist() +
                    rng.choice(east, SAMPLE_PER_SIDE, replace=False).tolist())
    with httpx.Client(follow_redirects=True, limits=httpx.Limits(max_connections=10)) as http:
        sets = discover(http)
        if not sets:
            raise RuntimeError("no SPEI dataset found in the gridMET THREDDS catalogs")
        pack = {lab: packing(http, path) for lab, path in sets}
        log(f"  SPEI datasets: {sets}; packing {pack}")
        series = fetch(http, [(lab, path, n, *cen[("dev", n)]) for lab, path in sets for n in sample], pack)
        # choose the timescale on the sampled development regions' cells 2016-2023
        _, dev = dataset.load("dev")
        insample = np.isin(np.array(dev_names)[dev["region"]], sample)
        tm = np.arange(0, 96)
        best, aucs, n_cm = None, {}, {}
        for lab, _ in sets:
            s = np.stack([series.get((lab, dev_names[r]), np.full(len(MONTHS), np.nan)) for r in dev["region"][insample]])
            y = dev["drought"][insample][:, tm]
            v = s[:, tm]
            ok = (y >= 0) & np.isfinite(v)
            if ok.sum() < 1000:
                continue
            aucs[lab] = float(roc_auc_score(y[ok], -v[ok]))
            n_cm[lab] = int(ok.sum())
            if best is None or aucs[lab] > aucs[best]:
                best = lab
        if best:
            path = dict(sets)[best]
            series.update(fetch(http, [(best, path, n, *cen[("test", n)]) for n in meta["test"]["regions"]], pack))
    vals = np.concatenate([v for v in series.values()]) if series else np.zeros(0)
    res = {"title": "Benchmark: gridMET SPEI", "datasets": sets, "packing": pack,
           "timescale_choice": {"sample_regions": len(sample), "per_side": SAMPLE_PER_SIDE, "seed": SAMPLE_SEED,
                                "dev_cells": int(insample.sum()), "cell_months": n_cm,
                                "note": "chosen on a seeded sample of development regions (DECISIONS.md #77)"},
           "auc_minus_spei_for_fldas_flag_dev": aucs, "chosen_timescale": best,
           "series": {"fetched": len(series), "value_range": [float(np.nanmin(vals)), float(np.nanmax(vals))] if
                      np.isfinite(vals).any() else None}, "lines": []}
    res["lines"].append(f"AUC of -SPEI for the FLDAS drought flag, development cells 2016-2023 of {len(sample)} sampled "
                        f"development regions ({int(insample.sum()):,} cells): " +
                        ", ".join(f"{k} {v:.3f}" for k, v in sorted(aucs.items(), key=lambda kv: -kv[1])) +
                        f"; chosen: {best}.")
    write_json(OUT / "spei_summary.json", res)
    p5 = read_json(SCORES / "phase5.json")
    if best and p5:
        sc = score_models_against_spei(best, series, meta)
        res["lines"] += sc.pop("lines")          # keep the timescale line beside the scoring lines
        res.update(sc)
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
        for fam in st["models"][job]["nets"] + st["models"][job].get("extra", []):   # amendment 6 rows too
            p, _ = P5._net_probs(job, "composite", fam, "fresh", P5.FINAL_SEEDS)
            if p is not None:
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
