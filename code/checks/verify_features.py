"""Check the ported feature code: with the paper's reference (2016-2022, leave-one-year-out) it must reproduce
tables_loyo/hls_fldas_drought.csv (features and target) for the listed regions. Writes logs/verify_features.json."""
import csv, json, os, sys, tempfile
from pathlib import Path
import numpy as np
V2 = Path(__file__).resolve().parents[2]  # the repository root
os.environ["DEWSAT_V2_OUT"] = tempfile.mkdtemp(dir=V2 / "tmp")
sys.path.insert(0, str(V2 / "code" / "dewsat2"))
import common, features, regions
names = sys.argv[1].split(",")
idx = np.arange(common.N_MONTHS)
paper_ref = idx <= common.MI["2022-12"]
features.reference_mask = lambda ref: paper_ref           # every table = the paper's reference
regs = {g["name"]: g for g in regions.all_regions()}
tol = {"NDVI_mean": 6e-6, "NDVI_std": 6e-6, "VCI_mean": 6e-4, "VCI_p10": 6e-4, "valid_frac": 6e-5}
res = {"regions": names, "rows": 0, "missing": 0, "not_observed": 0, "over_tol": {k: 0 for k in tol},
       "max_diff": {k: 0.0 for k in tol}, "target_mismatch": 0, "extra_observed": 0}
rows = {}
with open(V2 / "inputs/paper_data/tables_loyo/hls_fldas_drought.csv") as fh:
    for r in csv.DictReader(fh):
        if r["region_id"] in names:
            rows[(r["cell_id"], r["month"])] = r
for n in names:
    features.process_region(regs[n], True)
    z = np.load(Path(os.environ["DEWSAT_V2_OUT"]) / "cache" / "regions" / f"{n}.npz")
    ids = list(z["ids"]); f = z["feats"][0]; o = z["obs"][0]; dr = z["drought"]; keep = z["keep"]
    seen = np.zeros(o.shape, bool)
    for (cid, m), r in rows.items():
        if not cid.startswith(n + ":"):
            continue
        res["rows"] += 1
        i, k = ids.index(cid), common.MI[m]
        seen[i, k] = True
        if not o[i, k]:
            res["not_observed"] += 1
            continue
        for j, nm in enumerate(tol):
            d = abs(float(f[i, k, j]) - float(r[nm]))
            res["max_diff"][nm] = max(res["max_diff"][nm], d)
            res["over_tol"][nm] += d > tol[nm]
        res["target_mismatch"] += int(r["target"]) != 1 - int(dr[i, k])
    res["extra_observed"] += int((o[:, :common.MI["2026-09"] + 1] & (dr >= 0) & keep[:, None] & ~seen).sum())
res["ok"] = res["missing"] == 0 and res["not_observed"] == 0 and not any(res["over_tol"].values()) and res["target_mismatch"] == 0 and res["extra_observed"] == 0
(V2 / "logs" / "verify_features.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
