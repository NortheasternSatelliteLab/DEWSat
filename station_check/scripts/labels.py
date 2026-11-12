"""Monthly station, NLDAS-2 and FLDAS values at every station; matched-reference drought flags and anomalies (PLAN.md 2).

Writes results/labels_monthly.csv and results/sites_check.csv. No satellite data is read here.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from common import (DROUGHT_Q, MIN_REF_YEARS, NLDAS_RYAN, PAPER_REGIONS, REF_YEARS, RES, STATIONS, V2,
                    monthly_from_daily, read_sites, save_json)


def v2_regions() -> list[tuple]:
    new = set(json.load(open(V2 / "data" / "FROZEN_REGIONS.json"))["new_regions"])
    regs = []
    for r in csv.DictReader(open(V2 / "data" / "plan" / "regions_plan.csv")):
        if r["name"] in new:
            regs.append((r["name"], "dev" if r["role"] == "dev" else "test", r["side"],
                         *map(float, (r["west"], r["south"], r["east"], r["north"]))))
    for r in csv.DictReader(open(PAPER_REGIONS)):
        regs.append((r["region_id"], "dev", r["side_of_98W"], *map(float, (r["west"], r["south"], r["east"], r["north"]))))
    return regs


def main() -> None:
    RES.mkdir(exist_ok=True)
    sites = read_sites()
    z = np.load(V2 / "data" / "locked" / "fldas_conus.npz")
    fv, fmonths, flons, flats = z["values"], [str(m) for m in z["months"]], z["lons"], z["lats"]
    fmi = {m: i for i, m in enumerate(fmonths)}
    lon0, lat0 = float(flons[0]) - 0.05, float(flats[0]) - 0.05
    regs = v2_regions()
    site_rows, rows = [], []
    stats = {"sites": len(sites), "no_fldas_cell": 0, "fldas_nan_cell": 0}
    for s in sites:
        sid, lat, lon = s["site_id"], float(s["lat"]), float(s["lon"])
        ix, iy = int(np.floor((lon - lon0) / 0.1 + 1e-9)), int(np.floor((lat - lat0) / 0.1 + 1e-9))
        inside = [g for g in regs if g[3] <= lon < g[5] and g[4] <= lat < g[6]]
        reg = inside[0] if inside else None
        cell_id = ""
        if reg:
            clon = np.floor(lon * 10 + 1e-9) / 10 + 0.05
            clat = np.floor(lat * 10 + 1e-9) / 10 + 0.05
            cell_id = f"{reg[0]}:{clon:.2f}:{clat:.2f}"
        site_rows.append({"site_id": sid, "network": s["network"], "name": s["name"], "lat": lat, "lon": lon,
                          "fldas_ix": ix, "fldas_iy": iy, "v2_region": reg[0] if reg else "",
                          "v2_role": reg[1] if reg else "", "side": reg[2] if reg else ("west" if lon < -98 else "east"),
                          "v2_cell_id": cell_id})
        if not (0 <= ix < len(flons) and 0 <= iy < len(flats)):
            stats["no_fldas_cell"] += 1
            continue
        fl = {m: float(fv[fmi[m], iy, ix]) for m in fmonths if np.isfinite(fv[fmi[m], iy, ix])}
        if not fl:
            stats["fldas_nan_cell"] += 1
            continue
        st = monthly_from_daily(STATIONS / f"{sid}.csv.gz", "sm_10_40", check_range=True)
        nl = {}
        nd = NLDAS_RYAN / sid
        if nd.is_dir():
            for f in sorted(nd.glob("*.csv.gz")):
                nl.update(monthly_from_daily(f, "sm_10_40", check_range=False))
        # FLDAS as trained on: its own 2001-2022 reference (needs 20 values), the paper's rule
        trained = {}
        for cal in range(1, 13):
            ref = [fl[m] for m in fl if REF_YEARS[0] <= int(m[:4]) <= REF_YEARS[1] and int(m[5:]) == cal]
            if len(ref) >= 20:
                trained[cal] = float(np.quantile(ref, DROUGHT_Q))
        for cal in range(1, 13):
            yrs = [y for y in range(REF_YEARS[0], REF_YEARS[1] + 1)
                   if f"{y}-{cal:02d}" in st and f"{y}-{cal:02d}" in nl and f"{y}-{cal:02d}" in fl]
            ref = {k: [src[f"{y}-{cal:02d}"] for y in yrs] for k, src in (("stn", st), ("nldas", nl), ("fldas", fl))}
            ok = len(yrs) >= MIN_REF_YEARS
            thr = {k: float(np.quantile(v, DROUGHT_Q)) for k, v in ref.items()} if ok else {}
            mu = {k: float(np.mean(v)) for k, v in ref.items()} if ok else {}
            for y in range(2000, 2025):
                m = f"{y}-{cal:02d}"
                if m not in st and m not in nl and m not in fl:
                    continue
                row = {"site_id": sid, "month": m, "cal": cal, "n_ref": len(yrs),
                       "stn": st.get(m, ""), "nldas": nl.get(m, ""), "fldas": fl.get(m, ""),
                       "fldas_dr_trained": int(fl[m] < trained[cal]) if (m in fl and cal in trained) else ""}
                for k, src in (("stn", st), ("nldas", nl), ("fldas", fl)):
                    if ok and m in src:
                        row[f"{k}_dr"] = int(src[m] < thr[k])
                        row[f"{k}_anom"] = round(src[m] - mu[k], 6)
                    else:
                        row[f"{k}_dr"] = ""
                        row[f"{k}_anom"] = ""
                rows.append(row)
    cols = ["site_id", "month", "cal", "n_ref", "stn", "nldas", "fldas", "stn_dr", "nldas_dr", "fldas_dr",
            "stn_anom", "nldas_anom", "fldas_anom", "fldas_dr_trained"]
    with open(RES / "labels_monthly.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(RES / "sites_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(site_rows[0]))
        w.writeheader()
        w.writerows(site_rows)
    full = [r for r in rows if r["stn_dr"] != "" and r["nldas_dr"] != "" and r["fldas_dr"] != ""]
    stats.update({"rows": len(rows), "rows_all_three_labels": len(full),
                  "stations_with_all_three": len({r["site_id"] for r in full}),
                  "fldas_range": [float(np.nanmin(fv)), float(np.nanmax(fv))],
                  "in_v2_regions": sum(1 for r in site_rows if r["v2_region"]),
                  "in_v2_dev": sum(1 for r in site_rows if r["v2_role"] == "dev")})
    save_json(stats, RES / "labels_stats.json")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
