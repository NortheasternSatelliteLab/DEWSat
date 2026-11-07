"""Build every region's features in parallel (cached), then assemble development and fresh-test arrays."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import CACHE, MONTHS, REFS, git_state, log, read_json, save_npz, write_json
from regions import assign_groups, selected_regions


def _work(args):
    import features
    g, with_labels = args
    return features.process_region(g, with_labels)


def build(frozen: bool = True, workers: int = 8, only_complete_new: bool = False, names: list[str] | None = None,
          include_test: bool = True) -> dict:
    regs = selected_regions(frozen=frozen, only_complete_new=only_complete_new)
    if names:
        regs = [g for g in regs if g["name"] in set(names)]
    if not include_test:
        regs = [g for g in regs if g["role"] == "dev"]
    todo = [g for g in regs if not (CACHE / "regions" / f"{g['name']}.npz").exists()]
    log(f"features: {len(regs)} regions, {len(todo)} to build, {workers} workers")
    t0 = time.time()
    if todo:
        todo.sort(key=lambda g: g["name"] != "kansas_ref")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, res in enumerate(ex.map(_work, [(g, g["role"] == "dev") for g in todo]), 1):
                if i % 25 == 0 or i == len(todo):
                    log(f"  features {i}/{len(todo)} {res['name']} {res.get('seconds', 0)}s")
    return assemble(regs, time.time() - t0)


def assemble(regs: list[dict], seconds: float = 0.0) -> dict:
    dev = [g for g in regs if g["role"] == "dev"]
    test = [g for g in regs if g["role"] == "test"]
    groups = assign_groups(dev)
    meta = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state(), "refs": list(REFS),
            "months": [MONTHS[0], MONTHS[-1]], "seconds": round(seconds, 1)}
    for part, rl in (("dev", dev), ("test", test)):
        if not rl:
            continue
        keys = ("ids", "drought", "feats", "obs", "cellf", "fixed", "obs_c", "ndmi")
        parts = {k: [] for k in keys}
        region_idx, has_ndmi, has_views, dropped, n_dates = [], [], [], 0, []
        for ri, g in enumerate(rl):
            with np.load(CACHE / "regions" / f"{g['name']}.npz") as z:
                keep = z["keep"] if part == "dev" else np.ones(len(z["ids"]), bool)
                dropped += int((~keep).sum())
                for k in keys[:-1]:
                    a = z[k]
                    parts[k].append(a[:, keep] if a.ndim >= 3 and k in ("feats", "obs", "cellf") else a[keep])
                nk = int(keep.sum())
                if "ndmi" in z.files:
                    parts["ndmi"].append(z["ndmi"][:, keep])
                    has_ndmi.append(np.ones(nk, bool))
                else:
                    parts["ndmi"].append(np.full((len(REFS), nk, len(MONTHS), 2), np.nan, np.float32))
                    has_ndmi.append(np.zeros(nk, bool))
                has_views.append(np.full(nk, "view_date" in z.files))
                region_idx.append(np.full(nk, ri, np.int32))
                n_dates.append(z["n_dates"])
        arrays = {k: np.concatenate(v, axis=1 if k in ("feats", "obs", "cellf", "ndmi") else 0)
                  for k, v in parts.items()}
        if part == "test":
            arrays.pop("drought")                      # fresh-test labels stay locked
        arrays["cell_id"] = arrays.pop("ids")
        arrays["region"] = np.concatenate(region_idx)
        arrays["has_ndmi"] = np.concatenate(has_ndmi)
        arrays["has_views"] = np.concatenate(has_views)
        arrays["region_n_dates"] = np.stack(n_dates)
        save_npz(CACHE / f"{part}.npz", **arrays)
        meta[part] = {"regions": [g["name"] for g in rl], "sides": [g["side"] for g in rl],
                      "origins": [g["origin"] for g in rl], "batches": [g["batch"] for g in rl],
                      "lon": [g["lon"] for g in rl], "lat": [g["lat"] for g in rl],
                      "n_cells": int(len(arrays["cell_id"])), "cells_dropped_no_label": dropped,
                      "cells_with_ndmi": int(arrays["has_ndmi"].sum()), "cells_with_views": int(arrays["has_views"].sum())}
        if part == "dev":
            meta[part]["groups"] = [groups[g["name"]] for g in rl]
    write_json(CACHE / "dataset.json", meta)
    log(f"assembled: dev {meta.get('dev', {}).get('n_cells')} cells in {len(dev)} regions; "
        f"test {meta.get('test', {}).get('n_cells')} cells in {len(test)} regions")
    return meta


def load(part: str = "dev"):
    meta = read_json(CACHE / "dataset.json")
    with np.load(CACHE / f"{part}.npz") as z:
        arrays = {k: z[k] for k in z.files}
    return meta, arrays
