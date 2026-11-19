"""Weekly or monthly samples with NLDAS-2 as the answer key (FLDAS has no weekly or daily US product).

Same model, inputs and splits as V2 (`satellite_anomaly` in dew_stat_data.py); what changes:
    answer key   NLDAS-2 Noah soil moisture 10-40 cm, daily 12:00 UTC (data-pipeline/practice/nldas_daily.py),
                 averaged over each week (Monday-Sunday) or calendar month
    normal map   each cell's NLDAS average for that week of the year (52 slots, smoothed over ±1 week) or calendar
                 month, from 2001 to train_end
    inputs       step="month": the same monthly NDVI / VCI tables as V2 (vci.csv), so only the answer key differs.
                 step="week": NDVI = median of the week's clear passes (hls_observations.csv, >= 30% of the cell
                 clear); NDVI anomaly and VCI use that cell's usual NDVI / lowest / highest for the calendar month,
                 from weekly values up to train_end (3 years of photos is too few for 52 separate week slots).
    samples      targets from 2020 on (HLS on Planetary Computer is effectively 2020+), at least one clear photo in
                 the window. Only those windows are built, to keep memory low at long weekly windows.
Splits: train = train regions, targets <= 2022; val = val regions, 2020+ (val_years="all") or 2023 only ("gap");
test = test regions, 2024-2025 (NLDAS pulled through 2025); test_same = train regions, 2024-2025.

inputs="satellite" (no normal map): same inputs, but the model predicts the soil-moisture level itself, as V1 does.

Used by train.py --answer nldas_week / nldas_month. `when` holds the step's first day (YYYY-MM-DD).
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from dew_stat_data import ALL_FEATURES, PRACTICE, REGIONS_CSV, REGIONS_DIR, REGIONS_NEW_CSV, Region, DewStatData

STEPS = ("week", "month")
FIRST_TARGET = "2020-01-01"
# Satellite features in the order of ALL_FEATURES' satellite part (see dew_stat_data.SATELLITE_FEATURES).
SAT_COLS = ["season_sin", "season_cos", "ndvi", "ndvi_anom", "vci", "has_photo", "latitude", "longitude"]


def step_start(days: pd.DatetimeIndex, step: str) -> pd.DatetimeIndex:
    """First day of the week (Monday) or month each day falls in."""
    if step == "week":
        return (days - pd.to_timedelta(days.weekday, unit="D")).normalize()
    return days.to_period("M").start_time


def season_slot(starts: pd.DatetimeIndex, step: str) -> np.ndarray:
    """Week of the year (0..51, by the week's middle day) or calendar month (0..11): the normal map's key."""
    if step == "week":
        return np.minimum((starts + pd.Timedelta(days=3)).dayofyear.values // 7, 51)
    return starts.month.values - 1


def slot_mean(x: np.ndarray, slot: np.ndarray, fit: np.ndarray, n_slots: int) -> np.ndarray:
    """Per cell and slot: the mean over fit steps -> cells x n_slots (NaN where never seen)."""
    out = np.full((x.shape[0], n_slots), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(n_slots):
            sel = fit & (slot == k)
            if sel.any():
                out[:, k] = np.nanmean(x[:, sel], axis=1)
    return out


def read_rows(regions_csv: list[Path]) -> list[dict]:
    rows = []
    for path in regions_csv:
        with Path(path).open(encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    return rows


class DewStatNLDAS(DewStatData):
    def __init__(self, step: str = "week", regions_csv: Path | list[Path] = (REGIONS_CSV, REGIONS_NEW_CSV),
                 regions_dir: Path = REGIONS_DIR, window: int = 13, batch_size: int = 512, location: bool = True,
                 train_end: str = "2022-12-31", val_end: str = "2023-12-31", inputs: str = "satellite_anomaly",
                 val_years: str = "all"):
        if step not in STEPS:
            raise ValueError(f"step must be one of {STEPS}, not {step!r}")
        if inputs not in ("satellite_anomaly", "satellite"):
            raise ValueError(f"inputs must be satellite_anomaly or satellite, not {inputs!r}")
        super().__init__(regions_csv=list(regions_csv), regions_dir=regions_dir, window=window,
                         batch_size=batch_size, train_end=train_end, val_end=val_end, inputs=inputs,
                         location=location, val_years=val_years)
        self.step = step

    def _sat_month(self, d: Path, keys: list[tuple[float, float]], starts: pd.DatetimeIndex) -> np.ndarray:
        """NDVI and VCI (0-100) from the same monthly tables V2 uses -> cells x steps x 2."""
        vc = pd.read_csv(d / "vci.csv")
        sat = np.full((len(keys), len(starts), 2), np.nan)
        ci = {k: i for i, k in enumerate(keys)}
        ti = {s.strftime("%Y-%m"): i for i, s in enumerate(starts)}
        for lon, lat, mo, ndvi, vci in zip(vc["cell_longitude"].round(2), vc["cell_latitude"].round(2), vc["month"],
                                           vc["ndvi"], vc["vci"]):
            c, t = ci.get((lon, lat)), ti.get(mo)
            if c is not None and t is not None:
                sat[c, t] = ndvi, vci
        return sat

    def _sat_week(self, obs_path: Path, keys: list[tuple[float, float]], starts: pd.DatetimeIndex) -> np.ndarray:
        """Weekly NDVI (median of clear passes) and VCI from each cell's calendar-month min/max -> cells x steps x 2."""
        o = pd.read_csv(obs_path, usecols=["cell_longitude", "cell_latitude", "date", "ndvi"])
        o["key"] = list(zip(o["cell_longitude"].round(2), o["cell_latitude"].round(2)))
        o["start"] = step_start(pd.DatetimeIndex(pd.to_datetime(o["date"])), "week")
        med = o.groupby(["key", "start"])["ndvi"].median()
        ndvi = np.full((len(keys), len(starts)), np.nan)
        ci = {k: i for i, k in enumerate(keys)}
        ti = {s: i for i, s in enumerate(starts)}
        for (k, s), v in med.items():
            c, t = ci.get(k), ti.get(s)
            if c is not None and t is not None:
                ndvi[c, t] = v
        # VCI = where this week's NDVI sits between the cell's lowest and highest weekly NDVI in that calendar month
        # (weeks up to train_end), like make_vci.py --no-clip.
        mo = (starts + pd.Timedelta(days=3)).month.values - 1
        fit = starts <= pd.Timestamp(self.train_end)
        lo, hi = np.full((len(keys), 12), np.nan), np.full((len(keys), 12), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for k in range(12):
                seen = ndvi[:, fit & (mo == k)]
                n = np.isfinite(seen).sum(axis=1)
                lo[:, k] = np.where(n >= 2, np.nanmin(seen, axis=1), np.nan)
                hi[:, k] = np.where(n >= 2, np.nanmax(seen, axis=1), np.nan)
        lo, hi = lo[:, mo], hi[:, mo]
        with np.errstate(all="ignore"):
            vci = np.where(hi > lo, 100 * (ndvi - lo) / (hi - lo), np.nan)
        return np.stack([ndvi, vci], axis=-1)

    def setup(self, stage: str | None = None) -> None:
        if self._ready:
            return
        z = np.load(PRACTICE / "nldas_daily.npz")
        days = pd.DatetimeIndex(pd.to_datetime(z["days"]))
        cells_all = pd.read_csv(PRACTICE / "nldas_cells.csv")
        daily = pd.DataFrame(z["sm"], index=days)
        groups = daily.groupby(step_start(days, self.step))
        sm_all = groups.mean()
        sm_all = sm_all[groups.size().values >= (7 if self.step == "week" else 28)]  # drop partial first/last step
        starts = pd.DatetimeIndex(sm_all.index)
        when_all = np.array([s.strftime("%Y-%m-%d") for s in starts])
        slot = season_slot(starts, self.step)
        n_slots = 52 if self.step == "week" else 12
        fit = when_all <= self.train_end
        frac = (((starts + pd.Timedelta(days=3)).dayofyear.values - 1) / 365.25 if self.step == "week"
                else slot / 12)
        season = np.stack([np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)], axis=-1)

        rows = read_rows(self.regions_csv)
        self.missing, regions = [], []
        for r in rows:
            idx = cells_all.index[cells_all["region"] == r["name"]].values
            d = self.regions_dir / r["name"]
            obs = d / "hls_observations.csv"
            if not len(idx) or not obs.exists() or not (d / "vci.csv").exists():
                self.missing.append(r["name"])
                continue
            sm = sm_all[idx].to_numpy().T  # cells x steps
            keep = np.isfinite(sm).mean(axis=1) >= 0.9
            idx, sm = idx[keep], sm[keep]
            keys = [(round(float(lo), 2), round(float(la), 2)) for lo, la in cells_all.loc[idx, ["lon", "lat"]].values]

            clim = slot_mean(sm, slot, fit, n_slots)
            if self.step == "week":  # 22 years per week slot is still noisy: average with the neighbouring weeks
                clim = (np.roll(clim, 1, axis=1) + clim + np.roll(clim, -1, axis=1)) / 3
            normal = clim[:, slot]

            sat = self._sat_week(obs, keys, starts) if self.step == "week" else self._sat_month(d, keys, starts)
            ndvi, vci = sat[..., 0], sat[..., 1]
            has = np.isfinite(ndvi)
            # NDVI anomaly: this step's NDVI minus the cell's usual NDVI for the calendar month (steps <= train_end).
            mo = (starts + pd.Timedelta(days=3)).month.values - 1 if self.step == "week" else slot
            usual = slot_mean(ndvi, mo, fit, 12)[:, mo]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fallback = float(np.nanmean(ndvi[:, fit])) if np.isfinite(ndvi[:, fit]).any() else 0.0
            usual = np.nan_to_num(usual, nan=fallback)
            loc = np.array([(la, lo) for lo, la in keys])
            T = len(starts)
            named = {
                "season_sin": np.broadcast_to(season[:, 0], (len(keys), T)),
                "season_cos": np.broadcast_to(season[:, 1], (len(keys), T)),
                "ndvi": np.where(has, ndvi, 0.0),
                "ndvi_anom": np.where(has, ndvi - usual, 0.0),
                "vci": np.where(has & np.isfinite(vci), vci / 100.0, 0.0),
                "has_photo": has.astype(float),
                "latitude": np.broadcast_to(loc[:, :1], (len(keys), T)),
                "longitude": np.broadcast_to(loc[:, 1:], (len(keys), T)),
            }
            feats = np.stack([named[f] for f in self.features], axis=-1)
            regions.append(Region(r["name"], r["split"], keys, list(when_all), feats, sm, normal))
        if not any(g.split == "train" for g in regions):
            raise RuntimeError("no train regions with NLDAS and photo data")
        self.regions = regions
        assert all(f in ALL_FEATURES for f in self.features)

        # Feature scaling from train regions, 2020-01 .. train_end (the years with photos).
        span = (when_all >= FIRST_TARGET) & fit
        fit_rows = np.concatenate([g.feats[:, span].reshape(-1, len(self.features)) for g in regions
                                   if g.split == "train"])
        mean, std = fit_rows.mean(axis=0), fit_rows.std(axis=0)
        std = np.where(std > 1e-6, std, 1.0)
        h = self.features.index("has_photo")
        mean[h], std[h] = 0.0, 1.0
        self.feat_mean, self.feat_std = mean, std

        # Windows: only targets from FIRST_TARGET on with a clear photo in the window, built directly.
        W = self.window
        back = np.arange(-W + 1, 1)
        parts = {k: [] for k in ("X", "prev", "y", "normal", "when", "region", "cell")}
        t0 = max(W - 1, int(np.searchsorted(when_all, FIRST_TARGET)))
        for gi, g in enumerate(regions):
            photo = g.feats[..., h] > 0
            run = np.cumsum(np.pad(photo, ((0, 0), (1, 0))), axis=1)  # photos in window = run[t+1] - run[t+1-W]
            t = np.arange(t0, len(when_all))
            any_photo = (run[:, t + 1] - run[:, t + 1 - W]) > 0
            ok = any_photo & np.isfinite(g.sm[:, t]) & np.isfinite(g.normal[:, t])
            c_idx, t_idx = np.nonzero(ok)
            t_idx = t[t_idx]
            scaled = ((g.feats - mean) / std).astype(np.float32)
            parts["X"].append(scaled[c_idx[:, None], t_idx[:, None] + back])
            parts["prev"].append(g.sm[c_idx, t_idx - 1])
            parts["y"].append(g.sm[c_idx, t_idx])
            parts["normal"].append(g.normal[c_idx, t_idx])
            parts["when"].append(when_all[t_idx])
            parts["region"].append(np.full(len(c_idx), gi))
            parts["cell"].append(c_idx)
        X = np.concatenate(parts["X"])
        del parts["X"]
        self.prev, self.y = np.concatenate(parts["prev"]), np.concatenate(parts["y"])
        self.normal = np.concatenate(parts["normal"])
        self.when, self.region = np.concatenate(parts["when"]), np.concatenate(parts["region"])
        self.cell = np.concatenate(parts["cell"])

        split_of = np.array([g.split for g in regions])[self.region]
        early, late = self.when <= self.train_end, self.when > self.val_end
        self.split = {
            "train": (split_of == "train") & early,
            "val": (split_of == "val") & ((~early & ~late) if self.val_years == "gap" else True),
            "test": (split_of == "test") & late,
            "test_same": (split_of == "train") & late,
        }
        raw = self.y - self.normal if self.inputs == "satellite_anomaly" else self.y  # no map: the level itself
        tr = self.split["train"]
        self.t_center, self.t_scale = float(raw[tr].mean()), float(raw[tr].std())
        self.d_scale = self.t_scale
        target = ((raw - self.t_center) / self.t_scale).astype(np.float32)[:, None]
        self.sets = {k: TensorDataset(torch.from_numpy(np.ascontiguousarray(X[m])), torch.from_numpy(target[m]))
                     for k, m in self.split.items()}
        # Only the ridge baseline reads X afterwards (last 3 steps, train.RIDGE_MONTHS): keep just those to halve
        # memory at long weekly windows.
        self.X, self.target = np.ascontiguousarray(X[:, -3:]), target
        self.input_size = X.shape[-1]
        self._ready = True
