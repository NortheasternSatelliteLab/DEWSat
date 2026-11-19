"""Export the normal map (V2's "usual soil moisture per cell and calendar month") with pictures for the team.

The normal map = each cell's FLDAS soil moisture 10-40 cm averaged per calendar month over 2001-2022 (training years
only), exactly the values dew_stat_data.py adds back to V2's prediction.

Run from the model folder:   .venv-train/bin/python make_normal_map.py
Writes normal_map/: normal_map.csv (region, cell, month, value), normal_map_float16.bin (cells x 12, the on-board
format), 1_map.png, 2_through_the_year.png, 3_how_v2_uses_it.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dew_stat_data import PRACTICE, DewStatData  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "normal_map"
V2_RUN = HERE / "models" / "v2_satellite_anomaly"  # the chosen V2
INK, MUTED, GRAY = "#1f2328", "#59636e", "#8c959f"
LINES = ["#2a6fdb", "#d9822b", "#1a9e8f", "#8250df"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def tidy(ax) -> None:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def export() -> pd.DataFrame:
    dm = DewStatData(window=12, inputs="satellite_anomaly", hls="pc",
                     regions_csv=[PRACTICE / "regions.csv", PRACTICE / "regions_new.csv"])
    dm.setup()
    rows = []
    for g in dm.regions:
        cal = np.array([int(m[5:]) for m in g.months])
        first = [int(np.argmax(cal == k)) for k in range(1, 13)]  # the normal repeats every year
        for c, (lon, lat) in enumerate(g.cells):
            for k in range(12):
                rows.append((g.name, g.split, lon, lat, k + 1, round(float(g.normal[c, first[k]]), 4)))
    df = pd.DataFrame(rows, columns=["region", "split", "longitude", "latitude", "month", "normal_sm_m3m3"])
    df.to_csv(OUT / "normal_map.csv", index=False)
    wide = df.pivot_table(index=["region", "longitude", "latitude"], columns="month", values="normal_sm_m3m3")
    wide.to_numpy().astype(np.float16).tofile(OUT / "normal_map_float16.bin")
    return df


def map_figure(df: pd.DataFrame) -> None:
    states = json.loads((HERE / "assets" / "us_states.json").read_text())["features"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    vmin, vmax = df["normal_sm_m3m3"].quantile([0.02, 0.98])
    for ax, month in zip(axes, (4, 8)):
        for f in states:
            geom = f["geometry"]
            polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
            for poly in polys:
                ring = np.array(poly[0])
                ax.plot(ring[:, 0], ring[:, 1], color="#d0d7de", lw=0.6, zorder=1)
        d = df[df["month"] == month]
        sc = ax.scatter(d["longitude"], d["latitude"], c=d["normal_sm_m3m3"], cmap="Blues", vmin=vmin, vmax=vmax,
                        s=4, marker="s", linewidths=0, zorder=2)
        ax.set_title(f"Normal soil moisture in {MONTHS[month - 1]}", loc="left", color=INK, fontsize=11)
        ax.set_xlim(-125, -66.5)
        ax.set_ylim(24.5, 49.5)
        ax.set_aspect(1.25)
        ax.axis("off")
    cb = fig.colorbar(sc, ax=axes, shrink=0.75, pad=0.02)
    cb.set_label("m³/m³ (10-40 cm)", color=MUTED)
    fig.suptitle("The normal map: each 9 × 11 km cell's usual soil moisture for the month (FLDAS 2001-2022), "
                 "102 regions, 2,624 cells", x=0.02, ha="left", color=INK, fontsize=11)
    fig.savefig(OUT / "1_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def year_figure(df: pd.DataFrame) -> None:
    picks = [("washington_columbia_basin", "Eastern Washington (wet winter, dry summer)"),
             ("louisiana_rice", "Louisiana rice"), ("kansas_ref", "Kansas"),
             ("arizona_desert", "Arizona desert")]
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    for (name, label), col in zip(picks, LINES):
        m = df[df["region"] == name].groupby("month")["normal_sm_m3m3"].mean()
        if m.empty:
            continue
        ax.plot(m.index, m.values, color=col, lw=2, marker="o", ms=4)
        ax.annotate(label, (12, m.values[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_xlim(0.7, 15.8)
    ax.set_ylabel("Normal soil moisture (m³/m³)", color=MUTED)
    ax.set_title("12 numbers per cell: the usual soil moisture for each month (region averages)", loc="left",
                 color=INK, fontsize=11)
    ax.grid(axis="y", color="#eaeef2")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "2_through_the_year.png", dpi=150)
    plt.close(fig)


def use_figure() -> str:
    p = pd.read_csv(V2_RUN / "predictions_test.csv")
    region = "texas_blackland" if "texas_blackland" in set(p["region"]) else p["region"].iloc[0]
    g = p[p["region"] == region].groupby("month")[["actual", "normal", "bilstm"]].mean()
    x = pd.to_datetime(g.index)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    ax.plot(x, g["actual"], color=INK, lw=2, label="Actual (FLDAS, the answer key)")
    ax.plot(x, g["normal"], color=GRAY, lw=2, ls="--", label="Normal map (stored, the same every year)")
    ax.plot(x, g["bilstm"], color=LINES[0], lw=2, label="V2 = normal map + model's correction")
    ax.set_ylabel("m³/m³", color=MUTED)
    ax.set_title(f"How V2 uses the map: a test region it never trained on ({region.replace('_', ' ')}, 2024+)",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(axis="y", color="#eaeef2")
    tidy(ax)
    ax = axes[1]
    ax.axhline(0, color=GRAY, lw=1)
    ax.plot(x, g["actual"] - g["normal"], color=INK, lw=2, label="Actual − normal (wetter + / drier −)")
    ax.plot(x, g["bilstm"] - g["normal"], color=LINES[0], lw=2, label="Model's correction (what the satellite predicts)")
    ax.set_ylabel("m³/m³", color=MUTED)
    ax.set_title("The part the satellite model predicts: wetter or drier than normal", loc="left", color=INK,
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(axis="y", color="#eaeef2")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "3_how_v2_uses_it.png", dpi=150)
    plt.close(fig)
    return region


def main() -> int:
    OUT.mkdir(exist_ok=True)
    df = export()
    map_figure(df)
    year_figure(df)
    region = use_figure()
    print(f"{df['region'].nunique()} regions, {len(df) // 12} cells; example region {region}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
