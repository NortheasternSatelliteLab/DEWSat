"""Shared paths and helpers for the station check (PLAN.md). Read-only on every input folder."""

from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path

import numpy as np

HOME = Path.home()
OUT = HOME / "Desktop" / "DEWSat_station_check"
RES = OUT / "results"
STATIONS = HOME / "Desktop" / "stations"
HLS_RYAN = HOME / "Desktop" / "hls Ryan"
NLDAS_RYAN = HOME / "Desktop" / "nldas Ryan"
V2 = HOME / "Desktop" / "DEWSat_BiLSTM_v2"
PAPER_REGIONS = HOME / "Desktop" / "DEWSat_Fatemeh_package" / "regions.csv"

SEED = 20261111
N_BOOT = 2000
REF_YEARS = (2001, 2022)          # label reference window (the paper's)
MIN_REF_YEARS = 8                 # matched reference years needed per station and calendar month
MIN_DAYS = 20                     # valid days needed for a monthly value
SM_RANGE = (0.0, 0.7)             # plausible volumetric water content, exclusive
DROUGHT_Q = 0.2
SCORE_MONTHS = [f"{y}-{m:02d}" for y in range(2016, 2025) for m in range(1, 13)]
GROWING = set(range(4, 11))       # April-October


def month_list(a: str, b: str) -> list[str]:
    y, m = int(a[:4]), int(a[5:])
    out = []
    while f"{y}-{m:02d}" <= b:
        out.append(f"{y}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def read_sites() -> list[dict]:
    with open(STATIONS / "sites.csv") as fh:
        return list(csv.DictReader(fh))


def monthly_from_daily(path: Path, col: str, check_range: bool) -> dict[str, float]:
    """Mean of valid daily values per calendar month; months with fewer than MIN_DAYS valid days are dropped."""
    acc: dict[str, list[float]] = {}
    with gzip.open(path, "rt") as fh:
        for r in csv.DictReader(fh):
            v = r.get(col, "")
            if v == "":
                continue
            x = float(v)
            if not np.isfinite(x):
                continue
            if check_range and not (SM_RANGE[0] < x < SM_RANGE[1]):
                continue
            acc.setdefault(r["date"][:7], []).append(x)
    return {k: float(np.mean(v)) for k, v in acc.items() if len(v) >= MIN_DAYS}


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1, default=float))


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC with ties; higher score = more likely label 1."""
    from scipy.stats import rankdata
    label = label.astype(bool)
    n1 = int(label.sum())
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    rk = rankdata(score)
    return float((rk[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    po = float(np.mean(a == b))
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def boot_stations(groups: np.ndarray, fn, n: int = N_BOOT, seed: int = SEED):
    """Resample whole stations; fn(index_array) -> float or tuple of floats. Returns the array of results."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    pos = {g: np.flatnonzero(groups == g) for g in ids}
    out = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([pos[g] for g in pick])
        out.append(fn(idx))
    return np.array(out, dtype=float)


def ci(x: np.ndarray) -> list[float]:
    x = x[np.isfinite(x)] if x.ndim == 1 else x
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
