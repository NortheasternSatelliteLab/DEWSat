"""Data counts for STATUS.md, report.md and data_card.md (scores/data_counts.json), from the files on disk."""

from __future__ import annotations

import json
import subprocess
from collections import Counter

import numpy as np

from common import CACHE, DATA, N_FOLDS, NEW_REGIONS, PAPER_REGIONS, REEXTRACT, SCORES, V2, log, read_json, write_json
from regions import composite_dir, scene_dir, selected_regions


def du_bytes(path) -> int:
    try:
        out = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True, timeout=600).stdout.split()
        return int(out[0]) * 1024
    except Exception:  # noqa: BLE001
        return -1


def granules(g) -> dict:
    """Searched granules by archive and sensor (unique ids), from the search caches of the pull that made the region's
    composites: this pull for new regions, the paper run's for the original 102."""
    out = Counter()
    d = composite_dir(g)
    for name, arch in (("search_pc.jsonl", "pc"), ("search_nasa.jsonl", "nasa")):
        p = d / name
        if not p.exists():
            continue
        seen = set()
        for line in p.read_text().splitlines():
            e = json.loads(line)
            for sc in e["scenes"]:
                if sc["id"] not in seen:
                    seen.add(sc["id"])
                    out[f"{arch}_{e['sensor']}"] += 1
                    out["v20"] += sc["id"].endswith(".v2.0")
    return dict(out)


def run(frozen: bool = True) -> dict:
    regs = selected_regions(frozen=frozen)
    meta = read_json(CACHE / "dataset.json")
    res = {"regions": {}, "granules": {}, "views": {}, "disk": {}}
    role_side = Counter((g["role"], g["origin"], g["side"]) for g in regs)
    res["regions"] = {f"{r}_{o}_{s}": n for (r, o, s), n in role_side.items()}
    tot = Counter()
    views = Counter()
    for g in regs:
        for k, v in granules(g).items():
            tot[f"{g['role']}_{g['origin']}_{k}"] += v
            tot[k] += v
        sd = scene_dir(g)
        if sd is None or not (CACHE / "regions" / f"{g['name']}.npz").exists():
            continue
        z = None
        try:
            z = np.load(CACHE / "regions" / f"{g['name']}.npz")
            if "view_date" in z.files:
                cf = z["view_clear_frac"]
                views["views"] += cf.shape[0]
                views["view_cell_rows"] += cf.size
                views["view_cell_rows_clear30"] += int((cf >= 0.3).sum())
                views[f"views_{g['role']}"] += cf.shape[0]
                sens = z["view_sensor"]
                views["views_L30"] += int((sens == 0).sum())
                views["views_S30"] += int((sens == 1).sum())
        finally:
            if z is not None:
                z.close()
    res["granules"] = dict(tot)
    res["views"] = dict(views)
    if meta:
        res["cells"] = {"dev": meta["dev"]["n_cells"], "dev_dropped_no_label": meta["dev"]["cells_dropped_no_label"],
                        "test": meta.get("test", {}).get("n_cells", 0),
                        "dev_with_ndmi": meta["dev"]["cells_with_ndmi"], "dev_with_views": meta["dev"]["cells_with_views"]}
        # windows per job and fold, base rates (development only)
        try:
            import dataset
            from windows import Base, Sets
            m, arr = dataset.load("dev")
            base = Base(m, arr, "dev")
            win = {}
            for job in ("job1", "job2", "job3"):
                for k in range(N_FOLDS):
                    s = Sets(base, job, k, arr["obs"][k])
                    win[f"{job}_fold{k + 1}"] = {"outer_train": int(len(s.outer_train)), "outer_val": int(len(s.outer_val)),
                                                 "inner_train": int(len(s.inner_train)), "inner_val": int(len(s.inner_val))}
            res["windows"] = win
            lab = arr["drought"][:, :96]
            west = base.region_west[base.region]
            ok = lab >= 0
            res["drought_rate_dev"] = {"all": float(lab[ok].mean()), "west": float(lab[west][ok[west]].mean()),
                                       "east": float(lab[~west][ok[~west]].mean())}
            obs4 = arr["obs"][4][:, :96]
            res["observed_share_dev_all_table"] = float(obs4[ok].mean())
            nd = arr["region_n_dates"]
            res["clear_dates_per_region_month"] = {"median": float(np.median(nd)), "min": int(nd.min()), "max": int(nd.max())}
        except Exception as exc:  # noqa: BLE001
            res["windows_error"] = str(exc)[:300]
    res["disk"] = {"new_regions_bytes": du_bytes(NEW_REGIONS), "reextract_bytes": du_bytes(REEXTRACT) if REEXTRACT.exists() else 0,
                   "paper_regions_copy_bytes": du_bytes(PAPER_REGIONS), "cache_bytes": du_bytes(CACHE)}
    res["summary"] = {"regions": len(regs), "granules_searched": sum(v for k, v in tot.items() if k[:3] in ("pc_", "nas")),
                      "views": views.get("views", 0), "view_cell_rows": views.get("view_cell_rows", 0)}
    write_json(SCORES / "data_counts.json", res)
    log(f"data counts: {res['summary']}")
    return res


def _unused():
    return DATA, V2
