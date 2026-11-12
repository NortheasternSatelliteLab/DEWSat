"""Development and fresh-test regions, their grids, sources, and the outer-fold region groups (DESIGN.md 1-2)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from common import (DATA, GROUP_SEED, N_FOLDS, NEW_REGIONS, PAPER_REGIONS, REEXTRACT, V2, WEST_OF_LON, read_json)

EXISTING_FILES = (V2 / "inputs" / "paper_code" / "paper_hls" / "regions.csv",
                  V2 / "inputs" / "paper_code" / "paper_hls" / "regions_new.csv")
PLAN_FILE = DATA / "plan" / "regions_plan.csv"
FROZEN_REGIONS = DATA / "FROZEN_REGIONS.json"


def _box(r):
    return {k: float(r[k]) for k in ("west", "south", "east", "north")}


def all_regions() -> list[dict]:
    out = []
    for p in EXISTING_FILES:
        with p.open() as fh:
            for r in csv.DictReader(fh):
                out.append({"name": r["name"], "origin": "existing", "role": "dev", "batch": 0, **_box(r)})
    with PLAN_FILE.open() as fh:
        for r in csv.DictReader(fh):
            out.append({"name": r["name"], "origin": "new", "role": r["role"], "batch": int(r["batch"]), **_box(r)})
    for g in out:
        lon = (g["west"] + g["east"]) / 2
        g["side"] = "west" if lon < WEST_OF_LON else "east"
        g["lon"], g["lat"] = lon, (g["south"] + g["north"]) / 2
    return out


def selected_regions(frozen: bool = True, only_complete_new: bool = False) -> list[dict]:
    """The frozen dataset's regions (default), or for sandboxes the 102 plus every new region whose pull is complete.

    Sandboxes may set DEWSAT_PSEUDO_TEST to development region names that then play the fresh-test role, so a
    Phase 5 code check never reads a real fresh-test label."""
    import os
    regs = all_regions()
    pseudo = set(filter(None, os.environ.get("DEWSAT_PSEUDO_TEST", "").split(",")))
    if pseudo:
        for g in regs:
            if g["name"] in pseudo:
                g["role"] = "test"
            elif g["role"] == "test":
                g["role"] = "excluded"
    if frozen:
        fz = read_json(FROZEN_REGIONS)
        if fz is None:
            raise SystemExit(f"{FROZEN_REGIONS} missing: the dataset is not frozen yet")
        keep = set(fz["new_regions"])
        return [g for g in regs if g["origin"] == "existing" or g["name"] in keep]
    if only_complete_new:
        st = read_json(DATA / "pull_state.json", {}) or {}
        ok = {n for b in st.get("batches", {}).values() for n in b.get("ok", [])}
        return [g for g in regs if g["origin"] == "existing" or g["name"] in ok]
    return [g for g in regs if (g["origin"] == "existing" or g["batch"] > 0) and g["role"] != "excluded"]


def composite_dir(g: dict) -> Path:
    """Monthly per-pixel composites: the paper's for the 102, this pull's for new regions."""
    return (PAPER_REGIONS if g["origin"] == "existing" else NEW_REGIONS) / g["name"] / "hls"


def scene_dir(g: dict) -> Path | None:
    """Per-view values (and NDMI composites): new regions, and the 102 once re-extracted."""
    d = (REEXTRACT if g["origin"] == "existing" else NEW_REGIONS) / g["name"] / "hls"
    return d if d.is_dir() else None


def assign_groups(dev: list[dict]) -> dict[str, int]:
    """Seeded shuffle within each side, dealt round-robin; East continues where West stopped (DESIGN.md 2)."""
    rng = np.random.default_rng(GROUP_SEED)
    groups, start = {}, 0
    for side in ("west", "east"):
        names = sorted(g["name"] for g in dev if g["side"] == side)
        order = rng.permutation(len(names))
        for j, idx in enumerate(order):
            groups[names[idx]] = int((start + j) % N_FOLDS)
        start = (start + len(names)) % N_FOLDS
    return groups


def grid_of(g: dict) -> dict:
    """The region's pixel grid (same rule as code/pull/grid.py)."""
    p = composite_dir(g) / "grid.json"
    return json.loads(p.read_text())


def cell_centers(grid: dict) -> list[tuple[float, float]]:
    out = []
    for i in range(grid["rows"] * grid["cols"]):
        row, col = divmod(i, grid["cols"])
        out.append((round(grid["west"] + (col + 0.5) * 0.1, 2), round(grid["north"] - (row + 0.5) * 0.1, 2)))
    return out


def _unused():
    return math
