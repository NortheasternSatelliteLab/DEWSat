"""Per-view (single-scene) values per cell, the per-cell single-view VCI table, and monthly/weekly series built
from single views (DESIGN.md 10 weekly experiment; 11 realistic sampling)."""

from __future__ import annotations

import datetime as dt

import numpy as np

from common import CACHE, MIN_VIEW_CLEAR, MONTHS, N_MONTHS, REF_MIN_RANGE, REF_MIN_YEARS

EPOCH = dt.date(2015, 1, 1)


def day_of(yyyymmdd: np.ndarray) -> np.ndarray:
    y, m, d = yyyymmdd // 10000, (yyyymmdd // 100) % 100, yyyymmdd % 100
    return np.array([(dt.date(int(a), int(b), int(c)) - EPOCH).days for a, b, c in zip(y, m, d)], np.int64)


MONTH_END_DAY = np.array([(dt.date(int(m[:4]), int(m[5:]), 1) + dt.timedelta(days=32)).replace(day=1).toordinal()
                          - 1 - EPOCH.toordinal() for m in MONTHS], np.int64)
MONTH_START_DAY = np.array([dt.date(int(m[:4]), int(m[5:]), 1).toordinal() - EPOCH.toordinal() for m in MONTHS],
                           np.int64)


def region_views(name: str):
    """(ids, keep, views) from the region cache; views: day (V,), month index (V,), year (V,), cal (V,),
    clear (V, C), ndvi (V, C), ndvi_std, ndvi_p10 (V, C). None if the region has no per-view record."""
    with np.load(CACHE / "regions" / f"{name}.npz") as z:
        if "view_date" not in z.files:
            return None
        date = z["view_date"].astype(np.int64)
        out = {"day": day_of(date), "clear": z["view_clear_frac"], "ndvi": z["view_ndvi_mean"],
               "ndvi_std": z["view_ndvi_std"], "ndvi_p10": z["view_ndvi_p10"], "ids": z["ids"], "keep": z["keep"]}
    y, m = date // 10000, (date // 100) % 100
    out["year"], out["cal"] = y, m
    out["month"] = np.array([MONTHS.index(f"{a:04d}-{b:02d}") if f"{a:04d}-{b:02d}" in MONTHS else -1
                             for a, b in zip(y, m)], np.int64)
    return out


def cell_views(rv: dict, c: int, allowed_days: np.ndarray | None = None):
    """One cell's clear views (>= 30% clear, finite NDVI), one per day (highest clear fraction), sorted by day."""
    ok = (rv["clear"][:, c] >= MIN_VIEW_CLEAR) & np.isfinite(rv["ndvi"][:, c]) & (rv["month"] >= 0)
    if allowed_days is not None:
        ok &= np.isin(rv["day"], allowed_days)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return None
    order = np.lexsort((-rv["clear"][idx, c], rv["day"][idx]))
    idx = idx[order]
    first = np.r_[True, rv["day"][idx][1:] != rv["day"][idx][:-1]]
    idx = idx[first]
    return {"i": idx, "day": rv["day"][idx], "ndvi": rv["ndvi"][idx, c].astype(np.float64),
            "p10": rv["ndvi_p10"][idx, c].astype(np.float64), "std": rv["ndvi_std"][idx, c].astype(np.float64),
            "clear": rv["clear"][idx, c].astype(np.float64), "month": rv["month"][idx], "year": rv["year"][idx],
            "cal": rv["cal"][idx]}


def single_view_vci(cv: dict, ref_mask: np.ndarray, value: str = "ndvi") -> np.ndarray:
    """Per-view VCI against the cell's single-view table: per calendar month, min/max of view NDVI over views in
    reference months; a view in a reference month leaves its own year out; >= 3 years and range >= 0.05."""
    v = cv[value]
    out = np.full(len(v), np.nan)
    ref = ref_mask[cv["month"]]
    for cal in range(1, 13):
        sel = np.where(cv["cal"] == cal)[0]
        if len(sel) == 0:
            continue
        rs = sel[ref[sel]]
        if len(rs) == 0:
            continue
        years, inv = np.unique(cv["year"][rs], return_inverse=True)
        ny = len(years)
        ymin = np.full(ny, np.inf)
        ymax = np.full(ny, -np.inf)
        np.minimum.at(ymin, inv, cv["ndvi"][rs])
        np.maximum.at(ymax, inv, cv["ndvi"][rs])
        o_min, o_max = np.argsort(ymin), np.argsort(-ymax)
        min1, max1 = ymin[o_min[0]], ymax[o_max[0]]
        min2 = ymin[o_min[1]] if ny > 1 else np.nan
        max2 = ymax[o_max[1]] if ny > 1 else np.nan
        yi = np.searchsorted(years, cv["year"][sel])
        in_ref = ref[sel] & (yi < ny) & (years[np.clip(yi, 0, ny - 1)] == cv["year"][sel])
        yc = np.clip(yi, 0, ny - 1)
        lo = np.where(in_ref & (yc == o_min[0]), min2, min1)
        hi = np.where(in_ref & (yc == o_max[0]), max2, max1)
        n_k = ny - in_ref.astype(int)
        ok = (n_k >= REF_MIN_YEARS) & ((hi - lo) >= REF_MIN_RANGE)
        with np.errstate(invalid="ignore", divide="ignore"):
            val = np.clip(100.0 * (v[sel] - lo) / (hi - lo), 0.0, 100.0)
        out[sel] = np.where(ok & np.isfinite(v[sel]), val, np.nan)
    return out


def weekly_series(cv: dict, vci: np.ndarray, T: np.ndarray, weeks: int = 52):
    """(len(T), weeks, 7) weekly inputs ending on the last day of each month T, and the latest view's VCI/age at the
    end. Channels: VCI/100, NDVI, days since the view/365 (cap 1), new view this week, any view so far, sin, cos."""
    good = np.isfinite(vci)
    day, nd, vv = cv["day"][good], cv["ndvi"][good], vci[good]
    n = len(T)
    X = np.zeros((n, weeks, 7), np.float32)
    end = MONTH_END_DAY[T]
    E = end[:, None] - 7 * (weeks - 1 - np.arange(weeks))[None, :]          # week end days
    start = end - 7 * weeks + 1
    doy = (E + EPOCH.timetuple().tm_yday - 1) % 365.25
    X[:, :, 5] = np.sin(2 * np.pi * doy / 365.25)
    X[:, :, 6] = np.cos(2 * np.pi * doy / 365.25)
    latest_vci = np.full(n, np.nan)
    latest_age = np.full(n, np.nan)
    if len(day) == 0:
        return X, latest_vci, latest_age
    j = np.searchsorted(day, E, side="right") - 1
    jc = np.clip(j, 0, None)
    valid = (j >= 0) & (day[jc] >= start[:, None])
    age = np.where(valid, E - day[jc], 0)
    jn = np.searchsorted(day, E - 6, side="left")
    new = valid & (jn <= j)
    X[:, :, 0] = np.where(valid, vv[jc] / 100.0, 0.0)
    X[:, :, 1] = np.where(valid, nd[jc], 0.0)
    X[:, :, 2] = np.where(valid, np.minimum(age / 365.0, 1.0), 1.0)
    X[:, :, 3] = new
    X[:, :, 4] = valid
    last = valid[:, -1]
    latest_vci[last] = vv[jc[last, -1]]
    latest_age[last] = age[last, -1]
    return X, latest_vci, latest_age


def _unused():
    return N_MONTHS
