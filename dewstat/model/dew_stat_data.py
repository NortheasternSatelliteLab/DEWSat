"""Turns the per-region CSVs into tensors the Bi-LSTM can train on.

Data comes from data-pipeline/practice/regions/<name>/ (made by pull_regions.py), one folder per
region listed in practice/regions.csv and practice/regions_new.csv. Each folder has two CSVs, joined in memory:

    fldas.csv   soil moisture, rain, evaporation, temperature ... per cell per month (2001 ..)
    vci.csv     NDVI and VCI per cell per month (HLS from Planetary Computer: effectively 2020 ..)
    hls="full" reads hls_full.csv + vci_full.csv instead (merge_hls_years.py: NASA 2016-2019 + the above)

Steps:
  1. Per region, load both into a cube: cells x months x variables.
  2. Per cell and month T, build the numbers the model sees (16 features):
       10 FLDAS variables as anomalies (wetter/drier than usual for that cell and calendar month),
          from month T-1 because NASA publishes FLDAS about a month late
        2 season (sin/cos of the calendar month)
        4 satellite: NDVI, NDVI anomaly, VCI/100, has-a-clear-photo flag (month T)
  3. Slide a window over each cell -> one sample:
       x = (window, 16) features for months T-window+1 .. T
       y = soil moisture 10-40 cm CHANGE from T-1 to T, scaled to ~unit size
  4. Split by region AND time (column `split` in regions.csv):
       train       train regions, targets 2002 .. 2022
       val         val regions, every year      (never trained on; picks the best epoch)
                   (val_years="gap": 2023 only, no year shared with train or test)
       test        test regions, 2024 ..        (new place + new time: the honest score)
       test_same   train regions, 2024 ..       (same place, new time; for comparison)
     Averages ("usual for this month") use each cell's own months <= 2022.
     Feature scaling and target scaling come from the train set only.

Cells in one region share most of their weather each month (~89% in Kansas), so what really
counts is the number of regions x months, not cells. That is why regions are held out whole.

inputs="satellite" (what a satellite could compute on its own, no FLDAS inputs at all):
    features = season sin/cos, NDVI, NDVI anomaly, VCI, has_photo, + cell latitude/longitude
               (location=False drops latitude/longitude)
    y        = soil moisture 10-40 cm LEVEL at T (scaled), not the change: a satellite does not
               know last month's soil moisture. FLDAS is used only as the answer label.
    samples  = only windows with at least one clear photo (no photo at all = nothing to go on)

inputs="satellite_anomaly" (same inputs as "satellite"; only the answer labels change):
    normal   = that cell's average soil moisture for that calendar month, months <= train_end
               (the "normal map", built from FLDAS on the ground; never an input)
    y        = soil moisture - normal: wetter (+) or drier (-) than usual for this spot and month.
               to_soil_moisture() adds the normal back, so scores are still in m3/m3.

inputs="satellite_swir" (version 3: same labels as satellite_anomaly, VCI swapped for SWIR indices):
    features = season, NDVI, NDVI anomaly, NDMI, NDMI anomaly, NDWI, NDWI anomaly, has_photo, lat/lon
               NDMI = (NIR - SWIR1) / (NIR + SWIR1), NDWI = (NIR - SWIR2) / (NIR + SWIR2), from
               regions/<name>/swir/hls_monthly.csv (pull_regions.py --steps swir), or for regions pulled
               with the combined "bands" step, the NDMI/NDWI columns of regions/<name>/hls_monthly.csv
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PRACTICE = Path(__file__).resolve().parent.parent / "data-pipeline" / "practice"
REGIONS_DIR = PRACTICE / "regions"
REGIONS_CSV = PRACTICE / "regions.csv"
REGIONS_NEW_CSV = PRACTICE / "regions_new.csv"  # 80 more regions (pass both CSVs to train on all 102)
HLS_SOURCES = ("pc", "full")  # pc: Planetary Computer tables (effectively 2020+); full: + NASA 2016-2019

TARGET = "SoilMoi10_40cm_tavg"
FLDAS_VARS = [
    "SoilMoi00_10cm_tavg",
    "SoilMoi10_40cm_tavg",
    "SoilMoi40_100cm_tavg",
    "SoilMoi100_200cm_tavg",
    "Evap_tavg",
    "Qs_tavg",
    "Qsb_tavg",
    "RadT_tavg",
    "Rainf_f_tavg",
    "Tair_f_tavg",
]
FLUX_COLS = [FLDAS_VARS.index(v) for v in ("Evap_tavg", "Qs_tavg", "Qsb_tavg", "Rainf_f_tavg")]
FEATURES = [f"{v}_anom_lag1" for v in FLDAS_VARS] + [
    "season_sin",
    "season_cos",
    "ndvi",
    "ndvi_anom",
    "vci",
    "has_photo",
]
SWIR_FEATURES = ["ndmi", "ndmi_anom", "ndwi", "ndwi_anom"]
LOCATION = ["latitude", "longitude"]
ALL_FEATURES = FEATURES + SWIR_FEATURES + LOCATION
SATELLITE_FEATURES = ["season_sin", "season_cos", "ndvi", "ndvi_anom", "vci", "has_photo"] + LOCATION
SATELLITE_SWIR_FEATURES = ["season_sin", "season_cos", "ndvi", "ndvi_anom", *SWIR_FEATURES, "has_photo"] + LOCATION
ANOMALY_MODES = ("satellite_anomaly", "satellite_swir")  # label = wetter/drier than the cell's normal
MODES = ("full", "satellite", *ANOMALY_MODES)
SPLITS = ("train", "val", "test", "test_same")


def month_index(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= end:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def load_cubes(fldas_path: Path, vci_path: Path, need_all: bool = True, swir_path: Path | None = None):
    """One region's CSVs -> (cells, months, fldas[cells, months, 10], sat[cells, months, 4]).

    sat channels: NDVI, VCI, NDMI, NDWI (the last two only when swir_path is given, else NaN).
    need_all=False (satellite inputs): only the target column has to be there; others become NaN.
    """
    fl = pd.read_csv(fldas_path)
    absent = [v for v in FLDAS_VARS if v not in fl]
    if absent and (need_all or TARGET in absent):
        raise KeyError(f"{fldas_path} is missing {', '.join(absent)}; re-pull it with pull_regions.py")
    for v in absent:
        fl[v] = np.nan
    lon_key = "cell_longitude" if "cell_longitude" in fl else "longitude"
    lat_key = "cell_latitude" if "cell_latitude" in fl else "latitude"
    fl["lon"], fl["lat"] = fl[lon_key].round(2), fl[lat_key].round(2)
    fl["month"] = fl["time"].str[:7]
    vc = pd.read_csv(vci_path)
    vc["lon"], vc["lat"] = vc["cell_longitude"].round(2), vc["cell_latitude"].round(2)

    cells = sorted(set(zip(fl["lon"], fl["lat"])))
    months = month_index(fl["month"].min(), fl["month"].max())
    ci = {c: i for i, c in enumerate(cells)}
    mi = {m: i for i, m in enumerate(months)}

    c_idx = np.array([ci[c] for c in zip(fl["lon"], fl["lat"])])
    m_idx = fl["month"].map(mi).to_numpy()
    fldas = np.full((len(cells), len(months), len(FLDAS_VARS)), np.nan)
    vals = fl[FLDAS_VARS].to_numpy(dtype=float)
    vals = np.where(vals <= -9000, np.nan, vals)  # -9999 = fill value
    # Water fluxes come in kg/m2/s (~1e-8 for runoff). Convert to mm/day so their spread is not
    # mistaken for "constant" during scaling (that silently zeroed out Qs before).
    vals[:, FLUX_COLS] *= 86400
    fldas[c_idx, m_idx] = vals

    sat = np.full((len(cells), len(months), 4), np.nan)  # NDVI, VCI, NDMI, NDWI
    vc = vc[[(c in ci) and (m in mi) for c, m in zip(zip(vc["lon"], vc["lat"]), vc["month"])]]
    vc_c = np.array([ci[c] for c in zip(vc["lon"], vc["lat"])], dtype=int)
    vc_m = vc["month"].map(mi).to_numpy(dtype=int)
    sat[vc_c, vc_m, 0] = vc["ndvi"].to_numpy(dtype=float)
    sat[vc_c, vc_m, 1] = vc["vci"].to_numpy(dtype=float)
    if swir_path is not None:
        sw = pd.read_csv(swir_path)
        sw["lon"], sw["lat"] = sw["cell_longitude"].round(2), sw["cell_latitude"].round(2)
        sw = sw[[(c in ci) and (m in mi) for c, m in zip(zip(sw["lon"], sw["lat"]), sw["month"])]]
        sw_c = np.array([ci[c] for c in zip(sw["lon"], sw["lat"])], dtype=int)
        sw_m = sw["month"].map(mi).to_numpy(dtype=int)
        sat[sw_c, sw_m, 2] = pd.to_numeric(sw["ndmi_median"], errors="coerce").to_numpy(dtype=float)
        sat[sw_c, sw_m, 3] = pd.to_numeric(sw["ndwi_median"], errors="coerce").to_numpy(dtype=float)

    # Drop water / fill cells (target mostly missing). A month or two missing from the pull is fine:
    # those samples are skipped in setup().
    keep = np.isfinite(fldas[:, :, FLDAS_VARS.index(TARGET)]).mean(axis=1) >= 0.9
    return [c for c, k in zip(cells, keep) if k], months, fldas[keep], sat[keep]


def climatology(x: np.ndarray, months: list[str], fit: np.ndarray) -> np.ndarray:
    """Average per cell and calendar month, from `fit` months only, expanded back to every month."""
    cal = np.array([int(m[5:]) - 1 for m in months])
    clim = np.full(x.shape[:1] + (12,) + x.shape[2:], np.nan)
    for k in range(12):
        sel = (cal == k) & fit
        if sel.any():
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns stay NaN
                clim[:, k] = np.nanmean(x[:, sel], axis=1)
    return clim[:, cal]


def raw_features(fldas: np.ndarray, sat: np.ndarray, months: list[str], fit_end: str,
                 sat_fit_start: str | None = None) -> np.ndarray:
    """Per cell and month T: the 16 features in FEATURES, NOT yet scaled. Averages from months <= fit_end.

    sat_fit_start: the satellite indices' "usual value" (their anomalies) uses only months from here to fit_end.
    """
    n_cells, n_months, _ = fldas.shape
    fit = np.array([m <= fit_end for m in months])
    sat_fit = fit & np.array([sat_fit_start is None or m >= sat_fit_start for m in months])

    anom = fldas - climatology(fldas, months, fit)
    lagged = np.full_like(anom, np.nan)
    lagged[:, 1:] = anom[:, :-1]  # at month T we only know FLDAS for T-1

    cal = np.array([int(m[5:]) - 1 for m in months])
    season = np.stack([np.sin(2 * np.pi * cal / 12), np.cos(2 * np.pi * cal / 12)], axis=-1)

    ndvi, vci = sat[..., 0], sat[..., 1]
    has = ~np.isnan(ndvi)  # months before HLS starts get has_photo = 0 and zeros

    def value_and_anomaly(x: np.ndarray) -> list[np.ndarray]:
        """The index itself and how far it is from that cell's usual value for the month (fit months only)."""
        ok = ~np.isnan(x)
        clim = climatology(x[..., None], months, sat_fit)[..., 0]
        seen = x[:, sat_fit]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fallback = float(np.nanmean(seen)) if np.isfinite(seen).any() else 0.0
        return [np.where(ok, x, 0.0), np.where(ok, x - np.nan_to_num(clim, nan=fallback), 0.0)]

    sat_feats = np.stack(
        [
            *value_and_anomaly(ndvi),
            np.where(has & ~np.isnan(vci), vci / 100.0, 0.0),
            has.astype(float),
            *value_and_anomaly(sat[..., 2]),  # NDMI (NaN everywhere unless the SWIR table was loaded)
            *value_and_anomaly(sat[..., 3]),  # NDWI
        ],
        axis=-1,
    )
    return np.concatenate([lagged, np.broadcast_to(season, (n_cells, n_months, 2)), sat_feats], axis=-1)


@dataclass
class Region:
    name: str
    split: str
    cells: list[tuple[float, float]]
    months: list[str]
    feats: np.ndarray  # cells x months x features, unscaled
    sm: np.ndarray  # cells x months, target soil moisture
    normal: np.ndarray  # cells x months, that cell's usual soil moisture for the calendar month


def read_regions(regions_csv: Path | list[Path], regions_dir: Path, need_fldas_inputs: bool = True,
                 need_swir: bool = False, hls: str = "pc", vci_name: str = "vci_full.csv"
                 ) -> tuple[list[dict], list[str]]:
    """Regions whose pull finished (.done markers), plus the names still missing.

    need_fldas_inputs=False (satellite inputs): fldas.csv only supplies the label, so any existing
    fldas.csv will do, even one from before a FLDAS re-pull. need_swir: also needs the SWIR pull.
    hls="full": also needs the NASA 2016-2019 backfill joined in (hls_full.csv + vci_full.csv).
    """
    rows = []
    for path in [regions_csv] if isinstance(regions_csv, (str, Path)) else regions_csv:
        with Path(path).open(encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    ready, missing = [], []
    steps = ("hls", "vci", "swir") if need_swir else ("hls", "vci")
    for r in rows:
        d = regions_dir / r["name"]
        fldas_ok = (d / ".fldas.done").exists() or (not need_fldas_inputs and (d / "fldas.csv").exists())
        if hls == "full" and not ((d / "hls_full.csv").exists() and (d / vci_name).exists()):
            missing.append(r["name"])
            continue
        # .bands.done = the combined pull (new regions): one hls_monthly.csv with NDVI, NDMI and NDWI
        if fldas_ok and all((d / f".{s}.done").exists() or (s != "vci" and (d / ".bands.done").exists())
                            for s in steps):
            ready.append(r)
        else:
            missing.append(r["name"])
    return ready, missing


class DewStatData(L.LightningDataModule):
    def __init__(
        self,
        regions_csv: Path | list[Path] = (REGIONS_CSV, REGIONS_NEW_CSV),
        regions_dir: Path = REGIONS_DIR,
        window: int = 6,
        batch_size: int = 512,
        train_end: str = "2022-12",
        val_end: str = "2023-12",
        inputs: str = "full",
        location: bool = True,
        hls: str = "pc",
        sat_baseline_start: str | None = None,
        val_years: str = "all",
    ):
        super().__init__()
        if inputs not in MODES:
            raise ValueError(f"inputs must be one of {MODES}, not {inputs!r}")
        if hls not in HLS_SOURCES:
            raise ValueError(f"hls must be one of {HLS_SOURCES}, not {hls!r}")
        self.regions_csv = [Path(p) for p in ([regions_csv] if isinstance(regions_csv, (str, Path)) else regions_csv)]
        self.regions_dir, self.hls = Path(regions_dir), hls
        # hls="full" only: index normals and VCI range from this month on (e.g. "2020-01"), while training still
        # uses every year. Separates "more training years" from "baselines moved" (it was not the baselines).
        self.sat_baseline_start = sat_baseline_start
        self.vci_name = "vci_full.csv" if sat_baseline_start is None else f"vci_full_b{sat_baseline_start[:4]}.csv"
        self.window, self.batch_size = window, batch_size
        self.train_end, self.val_end = train_end, val_end
        # val_years="all": val regions, every year (overlaps train and test years). "gap": val regions, only the
        # months after train_end up to val_end (2023), so validation shares no years with training or test.
        if val_years not in ("all", "gap"):
            raise ValueError(f"val_years must be 'all' or 'gap', not {val_years!r}")
        self.val_years = val_years
        self.inputs = inputs
        if inputs == "full":
            self.features = list(FEATURES)
        else:
            base = SATELLITE_SWIR_FEATURES if inputs == "satellite_swir" else SATELLITE_FEATURES
            self.features = [f for f in base if location or f not in LOCATION]
        self._ready = False

    def setup(self, stage: str | None = None) -> None:
        if self._ready:
            return
        full = self.inputs == "full"
        swir = self.inputs == "satellite_swir"
        rows, self.missing = read_regions(self.regions_csv, self.regions_dir, need_fldas_inputs=full, need_swir=swir,
                                          hls=self.hls, vci_name=self.vci_name)
        regions = []
        cols = [ALL_FEATURES.index(f) for f in self.features]
        for r in rows:
            d = self.regions_dir / r["name"]
            # SWIR columns: swir/hls_monthly.csv (original regions) or hls_monthly.csv (combined pull);
            # hls="full": the joined NASA 2016-2019 + Planetary Computer 2020+ table and its VCI
            swir_path = d / "swir" / "hls_monthly.csv"
            if not swir_path.exists():
                swir_path = d / "hls_monthly.csv"
            vci_path = d / "vci.csv"
            if self.hls == "full":
                swir_path, vci_path = d / "hls_full.csv", d / self.vci_name
            cells, months, fldas, sat = load_cubes(d / "fldas.csv", vci_path, need_all=full,
                                                   swir_path=swir_path if swir else None)
            if not cells:
                continue
            loc = np.array([(lat, lon) for lon, lat in cells])[:, None, :]
            feats = np.concatenate([
                raw_features(fldas, sat, months, self.train_end, self.sat_baseline_start),
                np.broadcast_to(loc, (len(cells), len(months), 2)),
            ], axis=-1)[..., cols]
            sm = fldas[..., FLDAS_VARS.index(TARGET)]
            fit = np.array([m <= self.train_end for m in months])
            normal = climatology(sm[..., None], months, fit)[..., 0]  # the "normal map", training years only
            regions.append(Region(r["name"], r["split"], cells, months, feats, sm, normal))
        if not any(g.split == "train" for g in regions):
            raise RuntimeError(f"no finished train regions in {self.regions_dir}; run pull_regions.py first")
        self.regions = regions

        # Feature scaling from the train split only (train regions, months <= train_end).
        fit_rows = np.concatenate([
            g.feats[:, [m <= self.train_end for m in g.months]].reshape(-1, len(self.features))
            for g in regions if g.split == "train"
        ])
        with np.errstate(all="ignore"):
            mean = np.nan_to_num(np.nanmean(fit_rows, axis=0))
            std = np.nanstd(fit_rows, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
        h = self.features.index("has_photo")
        mean[h], std[h] = 0.0, 1.0  # keep has_photo a clean 0/1
        self.feat_mean, self.feat_std = mean, std

        # Sliding windows per region, then stacked together.
        W = self.window
        parts = {k: [] for k in ("X", "prev", "y", "normal", "when", "region", "cell", "photo")}
        for gi, g in enumerate(regions):
            feats = np.nan_to_num((g.feats - mean) / std).astype(np.float32)
            win = np.lib.stride_tricks.sliding_window_view(feats, W, axis=1)[:, 1:].transpose(0, 1, 3, 2)
            t = np.arange(W, len(g.months))
            parts["X"].append(win.reshape(-1, W, feats.shape[-1]))
            photo = np.lib.stride_tricks.sliding_window_view(g.feats[..., h], W, axis=1)[:, 1:]
            parts["photo"].append((photo > 0).any(axis=-1).reshape(-1))
            parts["prev"].append(g.sm[:, t - 1].reshape(-1))
            parts["y"].append(g.sm[:, t].reshape(-1))
            parts["normal"].append(g.normal[:, t].reshape(-1))
            parts["when"].append(np.tile(np.array(g.months)[t], len(g.cells)))
            parts["region"].append(np.full(len(g.cells) * len(t), gi))
            parts["cell"].append(np.repeat(np.arange(len(g.cells)), len(t)))
        X = np.concatenate(parts["X"])
        self.prev, self.y = np.concatenate(parts["prev"]), np.concatenate(parts["y"])
        self.when, self.region = np.concatenate(parts["when"]), np.concatenate(parts["region"])
        self.cell = np.concatenate(parts["cell"])
        self.normal = np.concatenate(parts["normal"])

        split_of = np.array([g.split for g in regions])[self.region]
        early = self.when <= self.train_end
        late = self.when > self.val_end
        self.split = {
            "train": (split_of == "train") & early,
            "val": (split_of == "val") & ((~early & ~late) if self.val_years == "gap" else True),
            "test": (split_of == "test") & late,
            "test_same": (split_of == "train") & late,
        }
        ok = np.isfinite(self.y) & (np.isfinite(self.prev) if self.inputs == "full" else True)
        if self.inputs != "full":
            ok &= np.concatenate(parts["photo"])
        if self.inputs in ANOMALY_MODES:
            ok &= np.isfinite(self.normal)
        self.split = {k: m & ok for k, m in self.split.items()}

        # Full: predict the change from last month. Satellite: the level itself.
        # Anomaly modes: wetter/drier than this cell's normal for the month.
        raw = self.y - self.prev if full else self.y - self.normal if self.inputs in ANOMALY_MODES else self.y
        tr = self.split["train"]
        self.t_center = 0.0 if self.inputs == "full" else float(raw[tr].mean())
        self.t_scale = float(raw[tr].std())  # target scaling from train only
        self.d_scale = self.t_scale
        target = ((raw - self.t_center) / self.t_scale).astype(np.float32)[:, None]  # (samples, 1) like the model output
        self.X, self.target = X, target
        self.sets = {
            k: TensorDataset(torch.from_numpy(np.ascontiguousarray(X[m])), torch.from_numpy(target[m]))
            for k, m in self.split.items()
        }
        self.input_size = X.shape[-1]
        self._ready = True

    def region_names(self, split: str) -> list[str]:
        return sorted({self.regions[i].name for i in np.unique(self.region[self.split[split]])})

    def to_soil_moisture(self, scaled: np.ndarray, split: str) -> np.ndarray:
        """Model output (scaled change / level / anomaly) -> soil moisture in m3/m3."""
        value = self.t_center + np.asarray(scaled).reshape(-1) * self.t_scale
        m = self.split[split]
        if self.inputs == "full":
            return self.prev[m] + value  # add last month back
        if self.inputs in ANOMALY_MODES:
            return self.normal[m] + value  # add the normal map back
        return value

    def loader(self, split: str, shuffle: bool = False) -> DataLoader:
        # Everything fits in memory; workers=0 avoids process spawning (fastest, also on Windows).
        return DataLoader(self.sets[split], batch_size=self.batch_size, shuffle=shuffle, num_workers=0)

    def train_dataloader(self):
        return self.loader("train", True)

    def val_dataloader(self):
        return self.loader("val")

    def test_dataloader(self):
        return self.loader("test")

    def predict_dataloader(self):
        return self.loader("test")
