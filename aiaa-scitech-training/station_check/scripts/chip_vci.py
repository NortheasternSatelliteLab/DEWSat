"""1 km VCI around every station from Ryan's HLS chips, using the paper's rules on 30 m pixels (PLAN.md 3).

Per image: clear pixels (Fmask not 255, bits 1-5 zero, aerosol not high, Red > 0, NIR > 0); the image counts only if
>= 30% of the chip is clear. Same-date images are averaged per pixel; the month's NDVI is the per-pixel median over
dates. VCI per pixel and calendar month against R7 (2016-2022) and R10 (2013-2022), leave-one-year-out inside the
reference, >= 3 years and range >= 0.05, clipped 0-100 (and -50..150 for R7 wide). Chip VCI_mean over usable pixels;
observed when usable >= 5% of the chip. Writes results/chip_vci.csv and results/chip_clipping.json.
Run with a Python that has pandas (v2's .venv), read-only on every input.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import HLS_RYAN, RES, read_sites  # noqa: E402

FMASK_REJECT, AEROSOL_HIGH, FILL = 0b0011_1110, 0b1100_0000, 255
MIN_CLEAR, MIN_USABLE, REF_MIN_YEARS, REF_MIN_RANGE, CLIP_TOL = 0.3, 0.05, 3, 0.05, 1e-3
MONTHS = [f"{y}-{m:02d}" for y in range(2013, 2025) for m in range(1, 13) if f"{y}-{m:02d}" >= "2013-04"]
REFS = {"r7": (2016, 2022), "r10": (2013, 2022)}


def composites(site: str):
    d = HLS_RYAN / site
    files = sorted(f for f in os.listdir(d) if f.endswith(".csv.gz"))
    df = pd.concat([pd.read_csv(d / f, dtype={"image_id": str, "time": str}) for f in files], ignore_index=True)
    df = df.dropna(subset=["image_id", "time", "lon", "lat"]).reset_index(drop=True)
    df["qa"] = df["qa"].fillna(FILL)
    pix = df.groupby(["lon", "lat"], sort=True).ngroup().to_numpy()
    img = df.groupby("image_id", sort=True).ngroup().to_numpy()
    n_pix, n_img = pix.max() + 1, img.max() + 1
    red = np.full((n_img, n_pix), np.nan)
    nir = np.full((n_img, n_pix), np.nan)
    qa = np.full((n_img, n_pix), FILL, np.int32)
    red[img, pix] = df["red"].to_numpy()
    nir[img, pix] = df["nir"].to_numpy()
    qa[img, pix] = df["qa"].to_numpy().astype(np.int32)
    img_date = pd.Series(df["time"].str[:10].to_numpy()).groupby(img).first().sort_index().to_numpy()
    clear = ((qa != FILL) & ((qa & FMASK_REJECT) == 0) & ((qa & AEROSOL_HIGH) != AEROSOL_HIGH)
             & (red > 0) & (nir > 0))
    keep = clear.sum(axis=1) / n_pix >= MIN_CLEAR
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(clear & keep[:, None], (nir - red) / (nir + red), np.nan)
    dates = sorted(set(img_date[keep]))
    di = {x: i for i, x in enumerate(dates)}
    s = np.zeros((len(dates), n_pix))
    c = np.zeros((len(dates), n_pix))
    for i in np.flatnonzero(keep):
        ok = np.isfinite(ndvi[i])
        s[di[img_date[i]], ok] += ndvi[i, ok]
        c[di[img_date[i]], ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        by_date = np.where(c > 0, s / c, np.nan)
    comp = np.full((len(MONTHS), n_pix), np.nan)
    n_dates = np.zeros(len(MONTHS), int)
    mi = {m: i for i, m in enumerate(MONTHS)}
    date_month = np.array([x[:7] for x in dates])
    for m, i in mi.items():
        sel = date_month == m
        if sel.any():
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    comp[i] = np.nanmedian(by_date[sel], axis=0)
            n_dates[i] = int(sel.sum())
    return comp, n_pix, n_dates, int(n_img), int(keep.sum())


def vci_table(comp: np.ndarray, ref: tuple[int, int]):
    years = np.array([int(m[:4]) for m in MONTHS])
    cals = np.array([int(m[5:]) for m in MONTHS])
    inref = (years >= ref[0]) & (years <= ref[1])
    import warnings
    out_raw = np.full_like(comp, np.nan)
    usable = np.zeros_like(comp, bool)
    for t in range(len(MONTHS)):
        sel = inref & (cals == cals[t])
        if inref[t]:
            sel = sel & (years != years[t])
        stack = comp[sel]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            n = np.isfinite(stack).sum(axis=0)
            lo, hi = np.nanmin(stack, axis=0) if len(stack) else None, np.nanmax(stack, axis=0) if len(stack) else None
        if lo is None:
            continue
        ok = (n >= REF_MIN_YEARS) & ((hi - lo) >= REF_MIN_RANGE) & np.isfinite(comp[t])
        with np.errstate(invalid="ignore", divide="ignore"):
            out_raw[t] = np.where(ok, 100.0 * (comp[t] - lo) / (hi - lo), np.nan)
        usable[t] = ok
    return out_raw, usable


def one_site(site: str):
    if not (HLS_RYAN / site).is_dir():
        return site, None
    comp, n_pix, n_dates, n_img, n_kept = composites(site)
    rows = []
    clip = {}
    tables = {k: vci_table(comp, ref) for k, ref in REFS.items()}
    for t, m in enumerate(MONTHS):
        row = {"site_id": site, "month": m, "n_dates": int(n_dates[t]), "n_pix": int(n_pix)}
        for k, (raw, us) in tables.items():
            u = us[t]
            frac = u.sum() / n_pix
            if frac >= MIN_USABLE:
                v = raw[t][u]
                row[f"vci_{k}"] = round(float(np.clip(v, 0, 100).mean()), 4)
                if k == "r7":
                    row["vci_r7wide"] = round(float(np.clip(v, -50, 150).mean()), 4)
                row[f"usable_{k}"] = round(float(frac), 4)
                lo, hi = int((v < -CLIP_TOL).sum()), int((v > 100 + CLIP_TOL).sum())
                yk = (k, m[:4])
                a = clip.setdefault(f"{k}:{m[:4]}", [0, 0, 0])
                a[0] += int(u.sum())
                a[1] += lo
                a[2] += hi
            else:
                row[f"vci_{k}"] = ""
                row[f"usable_{k}"] = round(float(frac), 4)
                if k == "r7":
                    row["vci_r7wide"] = ""
        rows.append(row)
    return site, {"rows": rows, "clip": clip, "n_img": n_img, "n_img_kept": n_kept, "n_pix": int(n_pix)}


def main() -> None:
    sites = [s["site_id"] for s in read_sites()]
    workers = max(1, (os.cpu_count() or 2) - 2)
    cols = ["site_id", "month", "n_dates", "n_pix", "vci_r7", "vci_r7wide", "usable_r7", "vci_r10", "usable_r10"]
    clip_tot, meta, done = {}, {}, 0
    with open(RES / "chip_vci.csv", "w", newline="") as fh, Pool(workers) as pool:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for site, res in pool.imap_unordered(one_site, sites, chunksize=2):
            done += 1
            if res is None:
                meta[site] = "no chip folder"
                continue
            w.writerows(res["rows"])
            for k, v in res["clip"].items():
                a = clip_tot.setdefault(k, [0, 0, 0])
                a[0] += v[0]
                a[1] += v[1]
                a[2] += v[2]
            meta[site] = {"images": res["n_img"], "images_kept": res["n_img_kept"], "pixels": res["n_pix"]}
            if done % 25 == 0:
                print(f"{done}/{len(sites)} stations", flush=True)
    clipping = {k: {"usable_pixel_months": v[0], "below_0": round(v[1] / v[0], 4) if v[0] else None,
                    "above_100": round(v[2] / v[0], 4) if v[0] else None} for k, v in sorted(clip_tot.items())}
    (RES / "chip_clipping.json").write_text(json.dumps({"clipping": clipping, "stations": meta}, indent=1))
    print("done", done)


if __name__ == "__main__":
    main()
