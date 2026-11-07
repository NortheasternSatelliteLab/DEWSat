"""Phase 6 extras, run only after Phase 5 (the fresh test is already scored and locked).

1. Label check: NLDAS-2 Noah monthly 10-40 cm soil moisture (NLDAS_NOAH0125_M v2.0) against FLDAS on the same cells:
   per cell and calendar month percentiles against 2001-2022, the same 20th-percentile drought rule; agreement as
   Spearman correlation of percentiles, Cohen's kappa of drought flags and AUC of the NLDAS percentile for the FLDAS
   flag (development cells 2016-2023 and fresh-test cells 2025-2026).
2. Benchmark: gridMET SPEI at each region's centre; the timescale whose -SPEI best ranks the FLDAS drought flag on
   development cells (AUC) is chosen, then each Phase 5 model's predictions are scored against SPEI drought
   (SPEI <= -0.84, the standard-normal 20th percentile) on the fresh test, beside their FLDAS-label AUC.
"""

from __future__ import annotations

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from common import CACHE, DATA, LOCKED, MONTHS, N_MONTHS, SCORES, V2, git_state, log, read_json, write_json

sys.path.insert(0, str(V2 / "code" / "pull"))

NLDAS_SHORT, NLDAS_VER = "NLDAS_NOAH0125_M", "2.0"
NLDAS_DIR = DATA / "phase6" / "nldas"
SPEI_DIR = DATA / "phase6" / "spei"


def label_months():
    from common import month_list
    return month_list("2001-01", "2026-09")


def nldas_pull(workers: int = 8) -> dict:
    """Monthly NLDAS-2 Noah 10-40 cm soil moisture over the NLDAS grid, one OPeNDAP ASCII request per month."""
    import httpx
    from grid import earthdata_token
    from paper_pull_fldas import axis_endpoints, get_text, NUM
    NLDAS_DIR.mkdir(parents=True, exist_ok=True)
    token = earthdata_token()
    CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
    r = httpx.get(CMR, params={"short_name": NLDAS_SHORT, "version": NLDAS_VER, "page_size": 2000,
                               "temporal": "2001-01-01T00:00:00Z,2026-09-28T23:59:59Z", "sort_key": "start_date"},
                  timeout=120)
    grans = []
    for it in r.json()["items"]:
        u = it["umm"]
        m = u["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:7]
        url = next(x["URL"] for x in u["RelatedUrls"] if "OPENDAP" in str(x.get("Subtype", "")).upper())
        grans.append((m, url))
    grans = sorted(g for g in grans if "2001-01" <= g[0] <= "2026-09")
    with httpx.Client(timeout=httpx.Timeout(600.0), limits=httpx.Limits(max_connections=workers + 2)) as http:
        dds = get_text(http, grans[0][1] + ".dds", token)
        var = next(v for v in re.findall(r"(\w*SoilM\w*10\w*40\w*)\[", dds)) if "SoilM" in dds else None
        if var is None:
            raise SystemExit("no 10-40 cm soil moisture variable in the NLDAS DDS")
        dims = re.search(rf"{var}\[(?:time = )?(\d+)\]\[(?:lat = )?(\d+)\]\[(?:lon = )?(\d+)\]", dds)
        ny, nx = int(dims.group(2)), int(dims.group(3))
        ax = get_text(http, grans[0][1] + f".ascii?lon[0:{nx - 1}:{nx - 1}],lat[0:{ny - 1}:{ny - 1}]", token)
        lon0, lon1 = axis_endpoints(ax, "lon")
        lat0, lat1 = axis_endpoints(ax, "lat")
        todo = [(m, u) for m, u in grans if not (NLDAS_DIR / f"{m}.npy").exists()]

        def one(m, u):
            t = get_text(http, u + f".ascii?{var}[0:0][0:{ny - 1}][0:{nx - 1}]", token)
            rows = {}
            for line in t.splitlines():
                head, _, rest = line.partition(",")
                if not rest or "[" not in head:
                    continue
                mm = re.findall(r"\[(\d+)\]", head)
                lat_lab = re.search(r"lat=(-?\d+(?:\.\d+)?)\]", head)
                vals = [float(v) for v in NUM.findall(rest)]
                if len(vals) != nx:
                    continue
                key = float(lat_lab.group(1)) if lat_lab else int(mm[-1])
                rows[key] = vals
            a = np.array([rows[k] for k in sorted(rows)], np.float32)
            if a.shape != (ny, nx):
                raise ValueError(f"{m}: got {a.shape}")
            a[a < -9000] = np.nan
            np.save(NLDAS_DIR / f"{m}.npy", a)
            return m

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(one, m, u) for m, u in todo]):
                f.result()
    return {"variable": var, "ny": ny, "nx": nx, "lon": [lon0, lon1], "lat": [lat0, lat1], "months": len(grans)}


def percentile_flags(series: np.ndarray, months: list[str]):
    """Per calendar month, empirical percentile (mid-rank vs 2001-2022) and the 20th-percentile drought flag."""
    yrs = np.array([int(m[:4]) for m in months])
    cal = np.array([int(m[5:]) for m in months])
    pct = np.full(series.shape, np.nan)
    flag = np.full(series.shape, -1, np.int8)
    for c in range(1, 13):
        sel = cal == c
        ref = series[:, sel & (yrs >= 2001) & (yrs <= 2022)]
        q20 = np.nanquantile(ref, 0.2, axis=1)
        v = series[:, sel]
        okv = np.isfinite(v)
        flag[:, sel] = np.where(okv, (v < q20[:, None]).astype(np.int8), -1)
        srt = np.sort(ref, axis=1)
        n = np.isfinite(srt).sum(axis=1)
        for i in range(series.shape[0]):
            if n[i] < 20:
                continue
            lo = np.searchsorted(srt[i, :n[i]], v[i], "left")
            hi = np.searchsorted(srt[i, :n[i]], v[i], "right")
            pct[i, sel] = np.where(okv[i], (lo + hi) / 2 / n[i], np.nan)
    return pct, flag


def run(status=None) -> dict:
    if not (LOCKED / "FRESH_TEST_SCORED.lock").exists():
        raise SystemExit("Phase 6 runs only after Phase 5")
    out = read_json(SCORES / "phase6.json", {}) or {}
    out["git"] = git_state()
    if "nldas" not in out:
        info = nldas_pull()
        out["nldas"] = nldas_compare(info)
        write_json(SCORES / "phase6.json", out)
    if "spei" not in out:
        try:
            import phase6_spei
            out["spei"] = phase6_spei.run()
        except Exception as exc:  # noqa: BLE001
            out["spei"] = {"title": "gridMET SPEI benchmark", "lines": [f"Not completed: {type(exc).__name__}: {exc}"]}
        write_json(SCORES / "phase6.json", out)
    return out


def nldas_compare(info: dict) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score, roc_auc_score
    import dataset
    months = label_months()
    cube = np.stack([np.load(NLDAS_DIR / f"{m}.npy") for m in months])               # (T, ny, nx)
    lon0, lon1 = info["lon"]
    lat0, lat1 = info["lat"]
    dx = (lon1 - lon0) / (info["nx"] - 1)
    dy = (lat1 - lat0) / (info["ny"] - 1)
    res = {"title": "Label check: NLDAS-2 Noah 10-40 cm vs FLDAS", "variable": info["variable"], "lines": []}
    meta = read_json(CACHE / "dataset.json")
    for part in ("dev", "test"):
        _, arr = dataset.load(part)
        ids = arr["cell_id"]
        lon = np.array([float(s.split(":")[1]) for s in ids])
        lat = np.array([float(s.split(":")[2]) for s in ids])
        xi = np.clip(np.round((lon - lon0) / dx).astype(int), 0, info["nx"] - 1)
        yi = np.clip(np.round((lat - lat0) / dy).astype(int), 0, info["ny"] - 1)
        nl = cube[:, yi, xi].T                                                   # (cells, 309)
        pct_n, flag_n = percentile_flags(nl, months)
        # FLDAS on the same cells and months
        src = DATA / "fldas" / "dev" if part == "dev" else LOCKED / "fldas_test"
        names = meta[part]["regions"]
        fl = np.full_like(nl, np.nan)
        for r, name in enumerate(names):
            z = np.load(src / f"{name}.npz")
            pos = {(round(float(a), 2), round(float(b), 2)): i for i, (a, b) in enumerate(zip(z["cell_lon"], z["cell_lat"]))}
            rows = np.where(arr["region"] == r)[0]
            for row in rows:
                fl[row] = z["values"][pos[(round(lon[row], 2), round(lat[row], 2))]]
        pct_f, flag_f = percentile_flags(fl, months)
        span = (("2016-01", "2023-12") if part == "dev" else ("2025-01", "2026-09"))
        tm = np.array([span[0] <= m <= span[1] for m in months])
        ok = (flag_f[:, tm] >= 0) & (flag_n[:, tm] >= 0) & np.isfinite(pct_n[:, tm]) & np.isfinite(pct_f[:, tm])
        a, b = pct_f[:, tm][ok], pct_n[:, tm][ok]
        fa, fb = flag_f[:, tm][ok], flag_n[:, tm][ok]
        rho = float(spearmanr(a, b).correlation)
        kap = float(cohen_kappa_score(fa, fb))
        auc = float(roc_auc_score(fa, -b))
        agree = float((fa == fb).mean())
        res[part] = {"cell_months": int(ok.sum()), "spearman_percentile": rho, "kappa_drought": kap,
                     "auc_nldas_pct_for_fldas_flag": auc, "flag_agreement": agree,
                     "fldas_rate": float(fa.mean()), "nldas_rate": float(fb.mean())}
        res["lines"].append(f"{'Development cells 2016-2023' if part == 'dev' else 'Fresh-test cells 2025-01..2026-09'}: "
                            f"{int(ok.sum()):,} cell-months; Spearman correlation of percentiles {rho:.3f}; drought flags agree "
                            f"{agree:.1%} of the time, Cohen's kappa {kap:.3f}; AUC of the NLDAS percentile for the FLDAS flag "
                            f"{auc:.3f}; drought rate FLDAS {fa.mean():.3f}, NLDAS {fb.mean():.3f}.")
    return res


def _unused():
    return time, Path, N_MONTHS, MONTHS
