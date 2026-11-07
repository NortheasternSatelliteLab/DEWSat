"""Check every job's fold sets against DESIGN.md 2-4: disjoint cells, buffers, inner year, months, eligibility."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dewsat2"))
import numpy as np
import dataset
from common import BLOCK_IDX, MONTHS, MI, DEV_END, N_FOLDS, LOGS, write_json
from windows import Base, Sets, obs_stats
meta, arr = dataset.load("dev")
base = Base(meta, arr, "dev")
res, bad = {}, []
for job in ("job1", "job2", "job3"):
    for k in range(N_FOLDS):
        s = Sets(base, job, k, arr["obs"][k])
        buf = 15 if job == "job3" else 12
        bs, be = BLOCK_IDX[k]
        r = {}
        T = base.w_T
        for nm in ("outer_train", "outer_val", "inner_train", "inner_val"):
            idx = getattr(s, nm)
            r[nm] = {"n": int(len(idx)), "T_min": MONTHS[T[idx].min()] if len(idx) else None,
                     "T_max": MONTHS[T[idx].max()] if len(idx) else None,
                     "groups": sorted(set(base.w_group[idx].tolist())), "months": int(len(np.unique(T[idx])))}
        cells = {nm: set(base.w_cell[getattr(s, nm)].tolist()) for nm in r}
        for a, b in (("outer_train", "outer_val"), ("inner_train", "inner_val"), ("inner_train", "outer_val"),
                     ("inner_val", "outer_val")):
            if cells[a] & cells[b]:
                bad.append(f"{job} f{k+1}: cells shared by {a} and {b}")
        To = T[s.outer_train]
        gap = min(np.min(np.where(To < bs, bs - To, 999)), np.min(np.where(To > be, To - be, 999)))
        if gap <= buf:
            bad.append(f"{job} f{k+1}: outer training T only {gap} months from the block")
        Ti, Tv = T[s.inner_train], T[s.inner_val]
        y = s.inner_year
        ys, ye = MI[f"{y}-01"], MI[f"{y}-12"]
        if not ((Tv >= ys) & (Tv <= ye)).all():
            bad.append(f"{job} f{k+1}: inner validation outside its year")
        gi = min(np.min(np.where(Ti < ys, ys - Ti, 999)), np.min(np.where(Ti > ye, Ti - ye, 999)))
        if gi <= buf:
            bad.append(f"{job} f{k+1}: inner training T only {gi} months from the inner year")
        if job == "job3" and (T[s.outer_train] + 3 > DEV_END).any():
            bad.append(f"{job} f{k+1}: forecast training target beyond 2023-12")
        if (T[s.outer_val] > DEV_END).any() or (T[s.outer_train] > DEV_END).any():
            bad.append(f"{job} f{k+1}: development set reaches past 2023-12")
        t_obs, n_prev = obs_stats(arr["obs"][k], base)
        v = s.outer_val
        if job in ("job1",) and not t_obs[v].all():
            bad.append(f"{job} f{k+1}: job-1 window with T unobserved")
        if job == "job2" and (t_obs[v].any() or (n_prev[v] < 8).any()):
            bad.append(f"{job} f{k+1}: job-2 window breaks its rule")
        r["gap_outer"], r["gap_inner"], r["inner_year"] = int(gap), int(gi), y
        res[f"{job}_fold{k+1}"] = r
out = {"ok": not bad, "problems": bad, "sets": res}
write_json(LOGS / "verify_folds.json", out)
print("ok" if not bad else bad)
for kk, v in res.items():
    print(kk, {nm: (v[nm]["n"], v[nm]["T_min"], v[nm]["T_max"], v[nm]["groups"]) for nm in ("outer_train", "outer_val", "inner_train", "inner_val")}, "gaps", v["gap_outer"], v["gap_inner"])
