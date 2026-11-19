"""What the final weekly model learned from the data, plus the raw weekly table it learned from (for the team lead).

Uses the weekly NLDAS run picked by validation loss in --runs-dir (default runs_valgap_week: validation = val regions
in 2023 only). Rebuilds the same data (DewStatNLDAS, same window and split) and joins each test prediction to that
week's inputs.

Run from the model folder:   .venv-train/bin/python make_trends.py [--runs-dir runs_valgap_week]
Writes report/trends/: trends.md, 1_greenness.png, 2_seasons.png, 3_test_average.png, 4_big_vs_small.png,
5_by_region.png, and raw/ (not in git): weekly_all_cells.csv.gz, predictions_test.csv.gz, regions.csv, README.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dew_stat_data import PRACTICE  # noqa: E402
from dew_stat_nldas import DewStatNLDAS  # noqa: E402
from compare_setups import fair_pick  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "report" / "trends"
INK, MUTED, GRAY, BLUE, ORANGE = "#1f2328", "#59636e", "#8c959f", "#2a6fdb", "#d9822b"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
KEY = ["region", "longitude", "latitude", "week_start"]
RAW_README = """# DewStat raw data (weekly, 102 regions, 2020-2025)

One cell = 0.125° NLDAS grid cell (about 11 × 14 km). Weeks run Monday to Sunday; `week_start` is the Monday.

## weekly_all_cells.csv.gz: everything the weekly model learns from
| Column | Meaning | Source |
|---|---|---|
| region, split | study region and its role: train (learned from, weeks up to 2022), val (picks settings, 2023 only), test (scored, 2024-2025) | regions.csv |
| longitude, latitude | cell centre | NLDAS grid |
| has_photo | 1 if at least one clear photo that week (>= 30% of the cell clear after the Fmask cloud/shadow/snow/water mask) | HLS v2 (Landsat 8/9 + Sentinel-2), Planetary Computer |
| ndvi | (NIR − Red) / (NIR + Red), median of the week's clear photos; empty without a photo | HLS |
| ndvi_anom | ndvi minus the cell's usual NDVI for the calendar month (2020-2022) | HLS |
| vci | Vegetation Condition Index 0-100: where ndvi sits between the cell's lowest and highest for the month (2020-2022) | HLS |
| nldas_sm_m3m3 | the answer key: soil moisture 10-40 cm, m³/m³, weekly mean of daily 12:00 UTC values | NASA NLDAS-2 Noah |
| normal_sm_m3m3 | the normal map: the cell's usual soil moisture for that week of the year (2001-2022, ±1 week) | NLDAS-2 |

The model also gets the season (sine/cosine of the date) and latitude/longitude, computed from the columns above.
Soil moisture is never an input: it is only the answer key (and, through the normal map, the starting point).

## predictions_test.csv.gz: the final model on the 22 test regions, 2024-2025 (run `{run}`)
actual = NLDAS; normal = normal map alone; model = DewStat (normal map + satellite correction, 3-seed average);
persistence = last week's actual (needs ground data the satellite does not have); actual_anom / model_anom =
minus the normal; plus that week's satellite inputs.

## regions.csv: region, split, number of cells
"""


def tidy(ax, title: str) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#eaeef2")


def corr(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else np.nan


def weekly_table(dm: DewStatNLDAS) -> pd.DataFrame:
    """Every cell-week of every region: the inputs (unscaled), the answer key and the normal map."""
    parts = []
    for g in dm.regions:
        f = {name: g.feats[..., k] for k, name in enumerate(dm.features)}
        n_c, n_t = g.sm.shape
        lon, lat = np.array(g.cells).T
        has = f["has_photo"] > 0
        parts.append(pd.DataFrame({
            "region": g.name, "split": g.split,
            "longitude": np.repeat(lon, n_t), "latitude": np.repeat(lat, n_t),
            "week_start": np.tile(np.asarray(g.months).astype(str), n_c),
            "has_photo": has.ravel().astype(int),
            "ndvi": np.where(has, f["ndvi"], np.nan).ravel(),
            "ndvi_anom": np.where(has, f["ndvi_anom"], np.nan).ravel(),
            "vci": np.where(has, f["vci"] * 100, np.nan).ravel(),
            "nldas_sm_m3m3": g.sm.ravel(), "normal_sm_m3m3": g.normal.ravel(),
        }))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["week_start"] >= "2020-01-01"]
    return df.round({"ndvi": 4, "ndvi_anom": 4, "vci": 1, "nldas_sm_m3m3": 5, "normal_sm_m3m3": 5})


def last_photo_anom(table: pd.DataFrame, window: int) -> pd.Series:
    """NDVI anomaly from the latest clear photo in the model's window (what the model mostly reacts to)."""
    t = table.sort_values(KEY).copy()
    t["last_anom"] = t.groupby(["region", "longitude", "latitude"])["ndvi_anom"].transform(
        lambda s: s.ffill(limit=window - 1))
    return t.set_index(KEY)["last_anom"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs_valgap_week")
    args = ap.parse_args()
    runs = HERE / args.runs_dir
    name = fair_pick(runs)
    cfg = json.loads((runs / name / "config.json").read_text())
    (OUT / "raw").mkdir(parents=True, exist_ok=True)

    dm = DewStatNLDAS(step="week", window=cfg["window"], inputs=cfg["inputs"],
                      val_years=cfg.get("val_years", "all"),
                      regions_csv=[PRACTICE / "regions.csv", PRACTICE / "regions_new.csv"])
    dm.setup()
    table = weekly_table(dm)

    p = pd.read_csv(runs / name / "predictions_test.csv").rename(
        columns={"cell_longitude": "longitude", "cell_latitude": "latitude", "month": "week_start", "bilstm": "model"})
    p = p.drop(columns=["ridge"]).round({"longitude": 4, "latitude": 4})
    table[["longitude", "latitude"]] = table[["longitude", "latitude"]].round(4)
    p = p.merge(table[KEY + ["ndvi", "ndvi_anom", "vci", "has_photo"]], on=KEY, how="left")
    p = p.join(last_photo_anom(table, cfg["window"]), on=KEY)
    # Persistence (last week's actual), from the same table.
    prev = table.sort_values(KEY).groupby(["region", "longitude", "latitude"])["nldas_sm_m3m3"].shift(1)
    p = p.join(table.sort_values(KEY).assign(persistence=prev).set_index(KEY)["persistence"], on=KEY)
    p["actual_anom"] = p["actual"] - p["normal"]
    p["model_anom"] = p["model"] - p["normal"]
    p["date"] = pd.to_datetime(p["week_start"])
    p["mo"] = p["date"].dt.month
    lines = [f"# What the weekly model learned (run `{args.runs_dir}/{name}`, picked by validation loss)", "",
             "22 test regions, 2024-2025: places and years the model never trained on. Validation: "
             + ("val regions in 2023 only." if cfg.get("val_years") == "gap" else "val regions, all years."), "",
             f"- Test cell-weeks: {len(p):,}. R² model {1 - ((p.model - p.actual) ** 2).sum() / ((p.actual - p.actual.mean()) ** 2).sum():.3f}, "
             f"normal map {1 - ((p.normal - p.actual) ** 2).sum() / ((p.actual - p.actual.mean()) ** 2).sum():.3f}.",
             f"- Wetter/drier than normal: model vs actual correlation {corr(p.model_anom, p.actual_anom):.2f}."]

    # 1. Greenness: NDVI anomaly (latest photo) vs soil-moisture anomaly, actual and as the model predicts it.
    q = p.dropna(subset=["last_anom"])
    q = q.assign(bin=pd.qcut(q["last_anom"], 10, duplicates="drop"))
    b = q.groupby("bin", observed=True).agg(x=("last_anom", "median"), act=("actual_anom", "mean"),
                                            pred=("model_anom", "mean"), n=("model", "size"))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.axhline(0, color=GRAY, lw=1)
    ax.plot(b.x, b.act, color=INK, lw=2, marker="o", ms=5, label="Actual (NLDAS)")
    ax.plot(b.x, b.pred, color=BLUE, lw=2, marker="o", ms=5, label="Model's prediction")
    ax.set_xlabel("NDVI anomaly from the latest clear photo (greener than usual →)", color=MUTED)
    ax.set_ylabel("Soil moisture − normal (m³/m³)", color=MUTED)
    ax.legend(frameon=False, fontsize=9)
    tidy(ax, "Trend 1: greener than usual → wetter than usual (test regions, 10 equal-size groups)")
    fig.tight_layout()
    fig.savefig(OUT / "1_greenness.png", dpi=150)
    plt.close(fig)
    lines += ["", "## 1. Greener than usual → wetter than usual", "",
              f"Correlation of the latest NDVI anomaly with the actual soil-moisture anomaly: "
              f"{corr(q.last_anom, q.actual_anom):.2f}; with the model's predicted anomaly: {corr(q.last_anom, q.model_anom):.2f}.",
              "", "| NDVI anomaly (group median) | Actual SM − normal | Model SM − normal | Cell-weeks |", "|---|---|---|---|"]
    lines += [f"| {r.x:+.3f} | {r.act:+.4f} | {r.pred:+.4f} | {r.n:,} |" for r in b.itertuples()]

    # 2. Seasons: how well the model tracks wetter/drier than normal, per calendar month.
    s = p.groupby("mo").apply(lambda g: pd.Series({
        "r": corr(g.model_anom, g.actual_anom),
        "mse_model": np.mean((g.model - g.actual) ** 2), "mse_normal": np.mean((g.normal - g.actual) ** 2),
        "photo": g.has_photo.mean()}), include_groups=False)
    s["better"] = 1 - s.mse_model / s.mse_normal
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.axhline(0, color=GRAY, lw=1)
    ax.bar(s.index, s.better * 100, 0.6, color=[BLUE if v > 0 else ORANGE for v in s.better])
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_ylabel("Error cut vs the normal map alone (%)", color=MUTED)
    tidy(ax, "Trend 2: when in the year the satellite helps (test regions; blue = better than the map)")
    fig.tight_layout()
    fig.savefig(OUT / "2_seasons.png", dpi=150)
    plt.close(fig)
    lines += ["", "## 2. When in the year the satellite helps", "",
              "| Month | Error cut vs normal map | Anomaly correlation | Weeks with a clear photo |", "|---|---|---|---|"]
    lines += [f"| {MONTHS[m - 1]} | {r.better:+.0%} | {r.r:.2f} | {r.photo:.0%} |" for m, r in s.iterrows()]

    # 3. Average over all test regions, week by week.
    w = p.groupby("date")[["actual", "normal", "model", "persistence"]].mean()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(w.index, w.actual, color=INK, lw=2, label="Actual (NLDAS)")
    ax.plot(w.index, w.normal, color=GRAY, lw=2, ls="--", label="Normal map")
    ax.plot(w.index, w.model, color=BLUE, lw=2, label="Model")
    ax.set_ylabel("m³/m³ (average of 22 test regions)", color=MUTED)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    tidy(ax, "Trend 3: average soil moisture across the 22 test regions, week by week, 2024-2025")
    fig.tight_layout()
    fig.savefig(OUT / "3_test_average.png", dpi=150)
    plt.close(fig)
    lines += ["", "## 3. Across all test regions, week by week", "",
              f"Correlation of the weekly average anomaly (model vs actual): "
              f"{corr(w.model - w.normal, w.actual - w.normal):.2f}. See 3_test_average.png."]

    # 4. Big vs small departures from normal: does the model get the direction right?
    p["size"] = pd.cut(p.actual_anom.abs(), [0, 0.01, 0.02, 0.04, 0.08, 1],
                       labels=["< 0.01", "0.01-0.02", "0.02-0.04", "0.04-0.08", "> 0.08"])
    d = p.groupby("size", observed=True).apply(lambda g: pd.Series({
        "right_dir": (np.sign(g.model_anom) == np.sign(g.actual_anom)).mean(),
        "share_caught": (g.model_anom / g.actual_anom).clip(-1, 2).median(), "n": len(g)}), include_groups=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(range(len(d)), d.right_dir * 100, 0.6, color=BLUE)
    ax.axhline(50, color=GRAY, lw=1, ls="--")
    ax.text(len(d) - 0.6, 50, "coin flip", color=MUTED, fontsize=9, ha="left", va="center")
    ax.set_xlim(-0.5, len(d) + 0.2)
    for i, v in enumerate(d.right_dir):
        ax.text(i, v * 100 + 1.5, f"{v:.0%}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(d)), d.index)
    ax.set_xlabel("How far the actual soil moisture is from normal (m³/m³)", color=MUTED)
    ax.set_ylabel("Weeks with the right direction (%)", color=MUTED)
    ax.set_ylim(0, 105)
    tidy(ax, "Trend 4: the model gets wetter vs drier right more often when the departure is big")
    fig.tight_layout()
    fig.savefig(OUT / "4_big_vs_small.png", dpi=150)
    plt.close(fig)
    lines += ["", "## 4. Big vs small departures from normal", "",
              "| Actual departure from normal | Right direction (wetter/drier) | Median share of the departure predicted | Cell-weeks |",
              "|---|---|---|---|"]
    lines += [f"| {k} m³/m³ | {r.right_dir:.0%} | {r.share_caught:.0%} | {int(r.n):,} |" for k, r in d.iterrows()]

    # 5. By region: where the satellite signal works.
    rg = p.groupby("region").apply(lambda g: pd.Series({
        "r": corr(g.model_anom, g.actual_anom), "green_r": corr(g.last_anom.fillna(0), g.actual_anom),
        "better": 1 - np.mean((g.model - g.actual) ** 2) / np.mean((g.normal - g.actual) ** 2),
        "sm": g.normal.mean(), "lon": g.longitude.mean()}), include_groups=False).sort_values("better")
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    y = np.arange(len(rg))
    ax.axvline(0, color=GRAY, lw=1)
    ax.barh(y, rg.better * 100, 0.6, color=[BLUE if v > 0 else ORANGE for v in rg.better])
    ax.set_yticks(y, [n.replace("_", " ").title().replace(" Lp", " Lower Peninsula") for n in rg.index], fontsize=8.5)
    ax.set_xlabel("Error cut vs the normal map alone (%)", color=MUTED)
    tidy(ax, "Trend 5: test regions where the satellite helps (blue) or hurts (orange)")
    ax.grid(axis="x", color="#eaeef2")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "5_by_region.png", dpi=150)
    plt.close(fig)
    lines += ["", "## 5. By region", "",
              f"Regions where the model beats the map: {(rg.better > 0).sum()} of {len(rg)}. Across regions, the "
              f"error cut correlates {corr(rg.lon, rg.better):.2f} with longitude (east +) and "
              f"{corr(rg.sm, rg.better):.2f} with the region's usual soil moisture.", "",
              "| Region | Error cut vs map | Model anomaly correlation | NDVI anomaly correlation | Usual SM |",
              "|---|---|---|---|---|"]
    lines += [f"| {n} | {r.better:+.0%} | {r.r:.2f} | {r.green_r:.2f} | {r.sm:.3f} |"
              for n, r in rg.sort_values("better", ascending=False).iterrows()]
    (OUT / "trends.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # Raw data for the team (not in git).
    table.to_csv(OUT / "raw" / "weekly_all_cells.csv.gz", index=False)
    p.drop(columns=["date", "mo", "size", "last_anom"]).to_csv(OUT / "raw" / "predictions_test.csv.gz", index=False)
    pd.DataFrame([(g.name, g.split, len(g.cells)) for g in dm.regions],
                 columns=["region", "split", "cells"]).to_csv(OUT / "raw" / "regions.csv", index=False)
    (OUT / "raw" / "README.md").write_text(RAW_README.format(run=f"{args.runs_dir}/{name}"), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
