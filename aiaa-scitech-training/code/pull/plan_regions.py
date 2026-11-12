"""Plan the new regions: candidate 5 x 5-cell boxes on the 0.1-degree FLDAS grid across the contiguous U.S., their
roles (development or fresh test), and the seeded pull batches of 50.

Rules (DECISIONS.md; amendment 2 of 2026-11-07):
- A candidate box is 5 x 5 FLDAS cells (0.5 x 0.5 degrees) with corners on the 0.1-degree grid. All 25 cell centres
  lie inside the contiguous U.S. (Census 2023 cartographic state boundaries, 1:20M, without AK, HI and the
  territories), and at least VEG_MIN of its area is vegetated in ESA CCI Land Cover 2020 (300 m).
- Distance between two boxes = Euclidean distance between their nearest points, in degrees.
- Development regions: >= 0.25 degrees from every other development region and from the existing 102.
- Fresh-test regions: >= 0.5 degrees from every development region (existing or new) and every other test region.
- Test sites are placed first on a coarse lattice (snapped to the nearest feasible box), then development regions
  fill the rest greedily (row by row, south to north, west to east). The lattice pitch is chosen so about a quarter
  of the new regions are fresh-test regions.
- Batches of 50: 25 West and 25 East of 98 W (box centroid), each side 6 test + 19 development, every role drawn
  one per latitude band (equal-count bands of what is left), seeded. A batch is formed only while both sides can
  fill it, so stopping after any batch leaves a balanced sample.

Writes data/plan/: candidates.npz, regions_plan.csv, batches.json, plan.json, plan_map.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parents[2]  # the repository root
LANDCOVER = V2 / "data" / "landcover" / "esa_cci_lc_2020_conus.tif"
STATES = V2 / "data" / "boundaries" / "cb_2023_us_state_20m" / "cb_2023_us_state_20m.shp"
EXISTING = [V2 / "inputs" / "paper_code" / "paper_hls" / "regions.csv",
            V2 / "inputs" / "paper_code" / "paper_hls" / "regions_new.csv"]
OUT = V2 / "data" / "plan"

W0, S0, E0, N0 = -125.0, 24.0, -66.5, 49.5      # planning domain (degrees)
NX, NY = round((E0 - W0) * 10), round((N0 - S0) * 10)
BOX = 5                                         # cells per side
LC_PER_CELL = 36                                # 1/360-degree land-cover pixels per 0.1-degree cell
NON_CONUS_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
WEST_OF = -98.0
D_DEV = 2.5                                     # 0.25 degrees, in 0.1-degree units
D_TEST = 5.0                                    # 0.5 degrees
PLAN_SEED = 20261107
BATCH_TEST_PER_SIDE = 6
BATCH_DEV_PER_SIDE = 19

# ESA CCI LCCS classes
VEG = {10, 11, 12, 20, 30, 40, 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110, 120, 121, 122, 130, 140,
       160, 170, 180}
GROUPS = {
    "cropland": {10, 11, 12, 20, 30},
    "crop_natural_mosaic": {40},
    "tree": {50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100},
    "shrubland": {110, 120, 121, 122},
    "grassland": {130, 140},
    "flooded_vegetation": {160, 170, 180},
    "sparse_vegetation": {150, 151, 152, 153},
    "urban": {190},
    "bare": {200, 201, 202},
    "water": {210},
    "snow_ice": {220},
    "no_data": {0},
}


def load_existing() -> list[dict]:
    out = []
    for p in EXISTING:
        with p.open() as fh:
            for r in csv.DictReader(fh):
                w, s, e, n = (float(r[k]) for k in ("west", "south", "east", "north"))
                out.append({"name": r["name"], "x": round((w - W0) * 10), "y": round((s - S0) * 10),
                            "w": round((e - w) * 10), "h": round((n - s) * 10), "west": w, "south": s, "east": e,
                            "north": n})
    return out


def cell_fractions() -> dict[str, np.ndarray]:
    """Per 0.1-degree cell (NY, NX, rows from the south): vegetated fraction and the land-cover group fractions."""
    import rasterio
    with rasterio.open(LANDCOVER) as ds:
        lc = ds.read(1)
        b = ds.bounds
    assert abs(b.left - W0) < 1e-6 and abs(b.top - N0) < 1e-6, b
    lc = lc[: NY * LC_PER_CELL, : NX * LC_PER_CELL]
    out = {}

    def frac(mask):
        f = mask.reshape(NY, LC_PER_CELL, NX, LC_PER_CELL).mean(axis=(1, 3))
        return f[::-1].astype(np.float32)          # row 0 = southernmost

    out["veg"] = frac(np.isin(lc, list(VEG)))
    for g, cls in GROUPS.items():
        out[g] = frac(np.isin(lc, list(cls)))
    return out


def conus_cells() -> np.ndarray:
    """(NY, NX) True where the 0.1-degree cell centre lies inside the contiguous U.S."""
    import shapefile
    import shapely
    from shapely.geometry import shape
    from shapely.ops import unary_union
    with shapefile.Reader(str(STATES)) as sf:
        fields = [f[0] for f in sf.fields[1:]]
        geoms = []
        for rec, shp in zip(sf.records(), sf.shapes()):
            d = dict(zip(fields, rec))
            if d["STATEFP"] in NON_CONUS_FIPS:
                continue
            geoms.append(shape(shp.__geo_interface__))
    conus = unary_union(geoms)
    shapely.prepare(conus)
    xs = W0 + (np.arange(NX) + 0.5) / 10
    ys = S0 + (np.arange(NY) + 0.5) / 10
    X, Y = np.meshgrid(xs, ys)
    return shapely.contains_xy(conus, X.ravel(), Y.ravel()).reshape(NY, NX)


def box_mean(a: np.ndarray) -> np.ndarray:
    """Mean over every 5 x 5 window; index [y, x] = box with lower-left cell (x, y)."""
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(a, (BOX, BOX)).mean(axis=(2, 3))


def box_min(a: np.ndarray) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(a, (BOX, BOX)).min(axis=(2, 3))


def stencil(bw: int, bh: int, d: float):
    """Offsets (dx, dy) of a new 5x5 box's corner, relative to a placed box's corner, that violate distance d."""
    D = math.ceil(d)
    offs = []
    for dx in range(-BOX - D, bw + D + 1):
        gx = max(0, -dx - BOX, dx - bw)
        for dy in range(-BOX - D, bh + D + 1):
            gy = max(0, -dy - BOX, dy - bh)
            if gx * gx + gy * gy < d * d - 1e-9:
                offs.append((dx, dy))
    return np.array(offs, np.int64)


class Layout:
    def __init__(self, cand: np.ndarray):
        self.cand = cand
        self.forbid_dev = np.zeros_like(cand)
        self.forbid_test = np.zeros_like(cand)
        self.placed: list[dict] = []
        self._st = {}

    def _stamp(self, grid, x, y, w, h, d):
        key = (w, h, d)
        if key not in self._st:
            self._st[key] = stencil(w, h, d)
        o = self._st[key]
        xx, yy = x + o[:, 0], y + o[:, 1]
        ok = (xx >= 0) & (yy >= 0) & (xx < grid.shape[1]) & (yy < grid.shape[0])
        grid[yy[ok], xx[ok]] = True

    def add(self, x, y, w, h, role, name=None):
        if role == "test":
            self._stamp(self.forbid_dev, x, y, w, h, D_TEST)
            self._stamp(self.forbid_test, x, y, w, h, D_TEST)
        else:   # dev or existing
            self._stamp(self.forbid_dev, x, y, w, h, D_DEV)
            self._stamp(self.forbid_test, x, y, w, h, D_TEST)
        self.placed.append({"x": x, "y": y, "w": w, "h": h, "role": role, "name": name})


def make_layout(cand, existing, pitch: int, phase: tuple[int, int], snap: int = 3) -> Layout:
    L = Layout(cand)
    for e in existing:
        L.add(e["x"], e["y"], e["w"], e["h"], "existing", e["name"])
    ny, nx = cand.shape
    # test sites: lattice points, each snapped to the nearest feasible corner within `snap` cells
    for gy in range(phase[1], ny, pitch):
        for gx in range(phase[0], nx, pitch):
            best = None
            for dy in range(-snap, snap + 1):
                for dx in range(-snap, snap + 1):
                    x, y = gx + dx, gy + dy
                    if 0 <= x < nx and 0 <= y < ny and cand[y, x] and not L.forbid_test[y, x]:
                        dd = dx * dx + dy * dy
                        if best is None or dd < best[0]:
                            best = (dd, x, y)
            if best:
                L.add(best[1], best[2], BOX, BOX, "test")
    # development fill: row by row from the south, west to east
    for y in range(ny):
        row_ok = cand[y] & ~L.forbid_dev[y]
        x = 0
        while x < nx:
            if cand[y, x] and not L.forbid_dev[y, x]:
                L.add(x, y, BOX, BOX, "dev")
            x += 1
    return L


def check_layout(L: Layout) -> dict:
    """Independent pairwise check of every distance rule."""
    P = L.placed
    bad = {"dev_dev": 0, "dev_existing": 0, "test_any": 0}
    xs = np.array([p["x"] for p in P]); ys = np.array([p["y"] for p in P])
    ws = np.array([p["w"] for p in P]); hs = np.array([p["h"] for p in P])
    roles = np.array([p["role"] for p in P])
    for i, p in enumerate(P):
        if p["role"] == "existing":
            continue
        gx = np.maximum(0, np.maximum(xs - (p["x"] + p["w"]), p["x"] - (xs + ws)))
        gy = np.maximum(0, np.maximum(ys - (p["y"] + p["h"]), p["y"] - (ys + hs)))
        dist = np.sqrt(gx ** 2 + gy ** 2) / 10
        dist[i] = np.inf
        if p["role"] == "dev":
            bad["dev_dev"] += int(((roles == "dev") & (dist < 0.25 - 1e-9)).sum())
            bad["dev_existing"] += int(((roles == "existing") & (dist < 0.25 - 1e-9)).sum())
            bad["test_any"] += int(((roles == "test") & (dist < 0.5 - 1e-9)).sum())
        else:
            bad["test_any"] += int((dist < 0.5 - 1e-9).sum())
    return bad


def side_of(x: int, w: int) -> str:
    return "west" if W0 + (x + w / 2) / 10 < WEST_OF else "east"


def make_batches(regions: list[dict]) -> list[list[str]]:
    """Seeded batches of 50: per side 6 test + 19 dev, one per equal-count latitude band of what is left."""
    pools = {(s, r): [g for g in regions if g["side"] == s and g["role"] == r]
             for s in ("west", "east") for r in ("test", "dev")}
    need = {"test": BATCH_TEST_PER_SIDE, "dev": BATCH_DEV_PER_SIDE}
    batches = []
    b = 0
    while all(len(pools[(s, r)]) >= need[r] for s in ("west", "east") for r in ("test", "dev")):
        b += 1
        rng = np.random.default_rng([PLAN_SEED, b])
        batch = []
        for s in ("west", "east"):
            for r in ("test", "dev"):
                pool = sorted(pools[(s, r)], key=lambda g: (g["lat"], g["lon"]))
                k = need[r]
                bands = np.array_split(np.arange(len(pool)), k)
                pick = [int(band[rng.integers(len(band))]) for band in bands]
                chosen = [pool[i] for i in pick]
                for g in chosen:
                    g["batch"] = b
                    batch.append(g["name"])
                keep = set(pick)
                pools[(s, r)] = [g for i, g in enumerate(pool) if i not in keep]
        batches.append(batch)
    return batches


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--veg-min", type=float, default=0.75)
    ap.add_argument("--pitches", default="20,21,22,23,24,25,26,27,28,30")
    ap.add_argument("--target-test", type=float, default=6 / 25)
    ap.add_argument("--explore", action="store_true", help="print the pitch/phase search and stop")
    ap.add_argument("--pitch", type=int, default=None, help="fix the test-lattice pitch (0.1-degree units)")
    ap.add_argument("--phase", default=None, help="fix the test-lattice phase, e.g. 15,12")
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fr = cell_fractions()
    inside = conus_cells()
    veg_box = box_mean(fr["veg"])
    conus_box = box_min(inside.astype(np.int8)) > 0
    cand = conus_box & (veg_box >= a.veg_min)
    existing = load_existing()
    # existing regions' vegetated fraction (for the log)
    ex_veg = {}
    for e in existing:
        sub = fr["veg"][e["y"]:e["y"] + e["h"], e["x"]:e["x"] + e["w"]]
        ex_veg[e["name"]] = float(sub.mean())
    print(f"candidates: {int(cand.sum())} corners pass (CONUS {int(conus_box.sum())}); "
          f"existing veg fraction: min {min(ex_veg.values()):.2f}, median {np.median(list(ex_veg.values())):.2f}, "
          f"below {a.veg_min}: {sum(v < a.veg_min for v in ex_veg.values())}", flush=True)
    # pitch / phase search
    results = []
    for p in (int(v) for v in a.pitches.split(",")):
        phases = [(px, py) for px in range(0, p, 3) for py in range(0, p, 3)]
        for ph in phases:
            L = make_layout(cand, existing, p, ph)
            n_test = sum(q["role"] == "test" for q in L.placed)
            n_dev = sum(q["role"] == "dev" for q in L.placed)
            sides = {}
            for q in L.placed:
                if q["role"] in ("test", "dev"):
                    key = (side_of(q["x"], q["w"]), q["role"])
                    sides[key] = sides.get(key, 0) + 1
            # balanced batches possible = min over side/role of floor(pool / need)
            nb = min(sides.get(("west", "test"), 0) // BATCH_TEST_PER_SIDE,
                     sides.get(("east", "test"), 0) // BATCH_TEST_PER_SIDE,
                     sides.get(("west", "dev"), 0) // BATCH_DEV_PER_SIDE,
                     sides.get(("east", "dev"), 0) // BATCH_DEV_PER_SIDE)
            results.append({"pitch": p, "phase": ph, "test": n_test, "dev": n_dev,
                            "test_frac": n_test / max(1, n_test + n_dev), "sides": {f"{k[0]}_{k[1]}": v for k, v in sides.items()},
                            "balanced_batches": nb})

    best = max(results, key=lambda r: (r["balanced_batches"], r["test"] + r["dev"]))
    if a.pitch is not None:
        ph = tuple(int(v) for v in a.phase.split(","))
        best = next(r for r in results if r["pitch"] == a.pitch and tuple(r["phase"]) == ph)
    print("chosen:", best, flush=True)
    if a.explore:
        return 0
    L = make_layout(cand, existing, best["pitch"], tuple(best["phase"]))
    bad = check_layout(L)
    if any(bad.values()):
        raise SystemExit(f"distance rule violated: {bad}")
    regions = []
    n = 0
    for q in L.placed:
        if q["role"] == "existing":
            continue
        n += 1
        x, y = q["x"], q["y"]
        west, south = round(W0 + x / 10, 1), round(S0 + y / 10, 1)
        g = {"name": f"n{n:04d}", "role": q["role"], "west": west, "south": south, "east": round(west + 0.5, 1),
             "north": round(south + 0.5, 1), "lon": round(west + 0.25, 2), "lat": round(south + 0.25, 2),
             "side": side_of(x, q["w"]), "veg_frac": round(float(veg_box[y, x]), 4), "batch": 0}
        for grp in GROUPS:
            g[f"lc_{grp}"] = round(float(fr[grp][y:y + BOX, x:x + BOX].mean()), 4)
        regions.append(g)
    batches = make_batches(regions)
    cols = ["name", "batch", "role", "side", "west", "south", "east", "north", "lon", "lat", "veg_frac"] + \
           [f"lc_{g}" for g in GROUPS]
    regions.sort(key=lambda g: (g["batch"] if g["batch"] else 10 ** 6, g["name"]))
    with (OUT / "regions_plan.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for g in regions:
            w.writerow({k: g[k] for k in cols})
    (OUT / "batches.json").write_text(json.dumps({"seed": PLAN_SEED, "batches": batches}, indent=1))
    np.savez_compressed(OUT / "candidates.npz", cand=cand, veg_box=veg_box.astype(np.float32),
                        conus_box=conus_box, cell_veg=fr["veg"], cell_conus=inside)
    summary = {
        "made": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1),
        "rules": {"veg_min": a.veg_min, "veg_classes": sorted(VEG), "dev_min_distance_deg": 0.25,
                  "test_min_distance_deg": 0.5, "box_cells": BOX, "west_of_lon": WEST_OF,
                  "landcover": "ESA CCI Land Cover 2020 v2.1.1 (300 m), Planetary Computer esa-cci-lc",
                  "boundary": "U.S. Census Bureau cartographic boundary file cb_2023_us_state_20m (CONUS)"},
        "candidate_corners": int(cand.sum()), "existing": len(existing),
        "existing_veg_frac": ex_veg, "search": results, "chosen": best, "distance_check_violations": bad,
        "n_test": sum(g["role"] == "test" for g in regions), "n_dev": sum(g["role"] == "dev" for g in regions),
        "n_batches": len(batches), "in_batches": sum(len(b) for b in batches),
        "not_in_any_batch": sum(1 for g in regions if not g["batch"]),
    }
    (OUT / "plan.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("existing_veg_frac", "search")}, indent=1))
    try:
        plot(regions, existing, fr["veg"], inside)
    except Exception as e:  # noqa: BLE001
        print("plot failed:", e)
    return 0


def plot(regions, existing, veg, inside):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(15, 8))
    show = np.where(inside, veg, np.nan)
    ax.imshow(show, origin="lower", extent=(W0, E0, S0, N0), cmap="Greens", vmin=0, vmax=1, alpha=0.6)
    for e in existing:
        ax.add_patch(Rectangle((e["west"], e["south"]), e["east"] - e["west"], e["north"] - e["south"],
                               fc="none", ec="black", lw=0.8))
    for g in regions:
        c = "tab:red" if g["role"] == "test" else "tab:blue"
        ax.add_patch(Rectangle((g["west"], g["south"]), 0.5, 0.5, fc=c if g["batch"] else "none", ec=c, lw=0.6,
                               alpha=0.7))
    ax.axvline(WEST_OF, color="k", ls="--", lw=0.8)
    ax.set_xlim(W0, E0); ax.set_ylim(S0, N0); ax.set_aspect(1 / math.cos(math.radians(37)))
    ax.set_title("Planned regions: existing 102 (black outline), new development (blue), fresh test (red); "
                 "hollow = not in a balanced batch")
    fig.tight_layout()
    fig.savefig(OUT / "plan_map.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
