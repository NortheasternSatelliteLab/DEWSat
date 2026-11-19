"""Daily vs weekly vs monthly: how hard is persistence to beat, and how often is there a clear photo?

Answer key: daily NLDAS-2 Noah soil moisture 10-40 cm (FLDAS has no daily US product), from
data-pipeline/practice/nldas_daily.py. Photos: data-pipeline/practice/regions/*/hls_observations.csv (clear passes,
>= 30% of the cell clear, all HLS satellites), 2021-2025.

For each time step (day, week, month), on 2024-2025 (fit years 2019-2023):
    normal map   = each cell's average for that day-of-year (±15 days) / week-of-year / calendar month, fit years only
    persistence  = last step's value
Scores: R², RMSE, persistence ÷ normal map (MSE), and "anomaly kept" = R² of last step's anomaly as a guess of this
step's anomaly (how much of wetter/drier-than-normal is already known one step earlier).

Run from the model folder:   .venv-train/bin/python cadence_check.py
Writes report/weekly/ (results.md, PNGs)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PRACTICE = HERE.parent / "data-pipeline" / "practice"
OUT = HERE / "report" / "weekly"
FIT_END, EVAL_START = "2023-12-31", "2024-01-01"
INK, MUTED, BLUE, ORANGE, GRAY = "#1f2328", "#59636e", "#2a6fdb", "#d9822b", "#8c959f"


def r2(pred: np.ndarray, actual: np.ndarray) -> float:
    ok = np.isfinite(pred) & np.isfinite(actual)
    p, a = pred[ok], actual[ok]
    return float(1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2))


def mse(pred: np.ndarray, actual: np.ndarray) -> float:
    ok = np.isfinite(pred) & np.isfinite(actual)
    return float(np.mean((pred[ok] - actual[ok]) ** 2))


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    z = np.load(PRACTICE / "nldas_daily.npz")
    cells = pd.read_csv(PRACTICE / "nldas_cells.csv")
    sm = pd.DataFrame(z["sm"], index=pd.to_datetime(z["days"]))
    return sm, cells


def series(sm: pd.DataFrame, step: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Soil moisture at this time step (mean over the step) and each row's 'season' key for the normal map."""
    if step == "day":
        s = sm.asfreq("D")
        return s, s.index.dayofyear.values
    if step == "week":
        s = sm.resample("W-SUN").mean()
        return s, np.minimum(s.index.isocalendar().week.values.astype(int), 52)
    s = sm.resample("MS").mean()
    return s, s.index.month.values


def normal_map(s: pd.DataFrame, key: np.ndarray, step: str) -> pd.DataFrame:
    fit = (s.index <= FIT_END)
    clim = s[fit].groupby(key[fit]).mean()
    if step == "day":  # smooth the day-of-year normal over ±15 days (5 years is too few for raw daily means)
        clim = clim.reindex(range(1, 367)).interpolate(limit_direction="both")
        wrap = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]])
        clim = wrap.rolling(31, center=True).mean().iloc[15:-15]
        clim.index = range(1, 367)
    return pd.DataFrame(clim.loc[key].values, index=s.index, columns=s.columns)


def scores(sm: pd.DataFrame, cols: np.ndarray) -> dict[str, dict[str, float]]:
    res = {}
    for step in ("day", "week", "month"):
        s, key = series(sm[cols], step)
        norm = normal_map(s, key, step)
        prev = s.shift(1)
        ev = s.index >= EVAL_START
        a, p, n = s[ev].values.ravel(), prev[ev].values.ravel(), norm[ev].values.ravel()
        anom, anom_prev = (s - norm)[ev].values.ravel(), (prev - norm.shift(1))[ev].values.ravel()
        res[step] = {
            "r2_persist": r2(p, a), "r2_normal": r2(n, a),
            "rmse_persist": mse(p, a) ** 0.5, "rmse_normal": mse(n, a) ** 0.5,
            "persist_over_normal": mse(p, a) / mse(n, a),
            "anom_kept": r2(anom_prev, anom),
            "typical_change": float(np.nanstd((s - prev)[ev].values)),
        }
    return res


def lag_curve(sm: pd.DataFrame, cols: np.ndarray, lags: range) -> tuple[list[float], list[float]]:
    s = sm[cols].asfreq("D")
    key = s.index.dayofyear.values
    norm = normal_map(s, key, "day")
    ev = s.index >= EVAL_START
    a = s[ev].values.ravel()
    anom = (s - norm)[ev].values.ravel()
    lvl, an = [], []
    for k in lags:
        lvl.append(r2(s.shift(k)[ev].values.ravel(), a))
        an.append(r2((s - norm).shift(k)[ev].values.ravel(), anom))
    return lvl, an


def photo_stats() -> pd.DataFrame:
    regs = pd.concat([pd.read_csv(PRACTICE / "regions.csv"), pd.read_csv(PRACTICE / "regions_new.csv")])
    d0, d1 = pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31")
    n_days = (d1 - d0).days + 1
    n_weeks = len(pd.date_range(d0, d1, freq="D").to_period("W").unique())
    n_months = 60
    rows = []
    for _, r in regs.iterrows():
        f = PRACTICE / "regions" / r["name"] / "hls_observations.csv"
        o = pd.read_csv(f, usecols=["cell_longitude", "cell_latitude", "date", "sensor"])
        o["date"] = pd.to_datetime(o["date"])
        o = o[(o["date"] >= d0) & (o["date"] <= d1)]
        for sats, sub in (("all HLS (4-5 satellites)", o), ("Landsat only (2 satellites)", o[o["sensor"] == "L30"])):
            for _, g in sub.groupby(["cell_longitude", "cell_latitude"]):
                d = pd.DatetimeIndex(g["date"].unique())
                rows.append({"region": r["name"], "satellites": sats, "day": len(d) / n_days,
                             "week": len(d.to_period("W").unique()) / n_weeks,
                             "month": len(d.to_period("M").unique()) / n_months})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    sm, cells = load()
    groups = {"22 test regions": cells.index[cells["split"] == "test"].values, "all 102 regions": cells.index.values}
    res = {g: scores(sm, c) for g, c in groups.items()}
    photos = photo_stats()
    ph = photos.groupby("satellites")[["day", "week", "month"]].mean()

    labels = {"day": "Daily", "week": "Weekly", "month": "Monthly"}
    md = ["# Daily vs weekly vs monthly", "",
          "Answer key: NLDAS-2 Noah soil moisture 10-40 cm, daily (12:00 UTC), at the same 2,625 cells as the model. "
          "Scored on 2024-2025; normal maps from 2019-2023. Persistence = last step's value (needs ground data).", ""]
    for g, rr in res.items():
        md += [f"## {g}", "",
               "| Step | Persistence R² | Normal map R² | Persistence RMSE | Normal map RMSE | Persistence ÷ normal (MSE) "
               "| Anomaly already known one step earlier (R²) | Typical change per step |",
               "|---|---|---|---|---|---|---|---|"]
        for step, v in rr.items():
            md.append(f"| {labels[step]} | {v['r2_persist']:.3f} | {v['r2_normal']:.3f} | {v['rmse_persist']:.4f} | "
                      f"{v['rmse_normal']:.4f} | {v['persist_over_normal']:.2f}× | {v['anom_kept']:.3f} | "
                      f"{v['typical_change']:.4f} |")
        md.append("")
    md += ["## Clear photos (≥ 30% of the cell cloud-free), 2021-2025, average over 2,625 cells", "",
           "| Satellites | Days with a photo | Weeks with a photo | Months with a photo |", "|---|---|---|---|"]
    for sats, v in ph.iterrows():
        md.append(f"| {sats} | {v['day']:.0%} | {v['week']:.0%} | {v['month']:.0%} |")
    md.append("")
    md.append("For reference, on FLDAS (monthly, 22 test regions, 2024+): model V2 R² 0.79, persistence 0.78, "
              "normal map 0.74.")
    (OUT / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # Chart 1: how fast persistence decays with lag (days), test regions.
    lags = range(1, 91)
    lvl, an = lag_curve(sm, groups["22 test regions"], lags)
    n_r2 = res["22 test regions"]["day"]["r2_normal"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(list(lags), lvl, color=BLUE, lw=2, label="Persistence: soil moisture k days ago")
    ax.plot(list(lags), an, color=ORANGE, lw=2, label="Anomaly k days ago (wetter/drier than normal)")
    ax.axhline(n_r2, color=GRAY, lw=1.5, ls="--", label=f"Normal map alone (R² {n_r2:.2f})")
    for k, txt in ((1, "1 day"), (7, "1 week"), (30, "1 month")):
        ax.axvline(k, color="#d0d7de", lw=1, zorder=0)
        ax.annotate(f"{txt}\nR² {lvl[k - 1]:.2f}", (k, lvl[k - 1]), xytext=(6, 8), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_xlabel("How old the ground value is (days)", color=MUTED)
    ax.set_ylabel("R² against today's soil moisture", color=MUTED)
    ax.set_title("Persistence is near-perfect at a day and a week, and fades by a month", loc="left", color=INK)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 90)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#eaeef2")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "1_persistence_by_lag.png", dpi=150)
    plt.close(fig)

    # Chart 2: share of days / weeks / months with a clear photo.
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    for i, (sats, col) in enumerate((("all HLS (4-5 satellites)", BLUE), ("Landsat only (2 satellites)", ORANGE))):
        v = ph.loc[sats, ["day", "week", "month"]].values
        bars = ax.bar(x + (i - 0.5) * 0.36, v, 0.34, color=col, label=sats)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.015, f"{val:.0%}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, ["Days", "Weeks", "Months"])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Share with ≥ 1 clear photo", color=MUTED)
    ax.set_title("How often a cell gets a clear photo (2021-2025, 2,625 cells)", loc="left", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#eaeef2")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "2_photos.png", dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
