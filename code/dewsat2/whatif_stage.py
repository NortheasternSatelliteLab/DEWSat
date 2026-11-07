"""Secondary what-ifs, job 1 (DESIGN.md 10): NDMI added (not DEWSat's camera) and VCI per cell vs per pixel."""

from __future__ import annotations

import numpy as np

import scoring
import simple
import units
from common import N_FOLDS, SCORES, git_state, log, read_json, write_json
from windows import raw_windows

SEEDS = (42, 43, 44)


def best_job1_family() -> str:
    sc = read_json(SCORES / "scores.json")
    return sc["selection"]["job1"]["best_network"]


def specs() -> list[dict]:
    fam = best_job1_family()
    out = []
    for v in units.WHATIF_VARIANTS:
        for k in range(N_FOLDS):
            cfg = units.chosen_config("job1", fam, k)
            for s in SEEDS:
                out.append({"type": "whatif", "variant": v, "k": k, "cfg": cfg, "seed": s, "job": "job1"})
    return out


def _sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def collect(fam: str) -> dict:
    c = units.ctx()
    base, arrays = c["base"], c["arrays"]
    out = {}
    for pair in (("ndmi_off", "ndmi_on"), ("pixel", "cell")):
        ys, regs, preds = [], [], {}
        skipped = []
        for k in range(N_FOLDS):
            d0 = units.prep_variant(pair[0], k)
            if len(d0["ev"]) == 0 or len(d0["fit"]) == 0:
                skipped.append(k + 1)
                continue
            ys.append(d0["Y_ev"][:, 0].astype(np.int64))
            regs.append(base.w_region[d0["ev"]])
            for v in pair:
                ps = []
                for s in SEEDS:
                    with np.load(units.whatif_dir(v, fam, k, s) / "pred.npz") as z:
                        assert np.array_equal(z["wid"], d0["ev"])
                        ps.append(_sig(z["logit"][:, 0]))
                preds.setdefault(f"{fam}_{v}", []).append(np.mean(ps, axis=0))
            # simple models on the same windows
            dv = units.prep_variant(pair[0], k)
            feats, obs = arrays["feats"][k], arrays["obs"][k]
            if pair[0] == "ndmi_off":
                nd = arrays["ndmi"][k]
                def X(idx, with_nd):
                    vt = feats[base.w_cell[idx], base.w_T[idx], 2].astype(np.float64)
                    cols = [vt]
                    if with_nd:
                        cols += [nd[base.w_cell[idx], base.w_T[idx], 0], nd[base.w_cell[idx], base.w_T[idx], 1]]
                    return np.stack(cols, axis=1).astype(np.float64)
                for nm, wn in (("lr_vci_t", False), ("lr_vci_t_ndmi", True)):
                    m = simple.LR(["x"] * (3 if wn else 1)).fit(X(dv["fit"], wn), dv["Y_fit"][:, 0])
                    preds.setdefault(nm, []).append(m.predict(X(dv["ev"], wn)))
            else:
                cellf = arrays["cellf"][k]
                for nm, src in (("lr_vci_t_pixel", "pixel"), ("lr_vci_t_cell", "cell")):
                    def X(idx):
                        if src == "pixel":
                            return feats[base.w_cell[idx], base.w_T[idx], 2][:, None].astype(np.float64)
                        return cellf[base.w_cell[idx], base.w_T[idx], 0][:, None].astype(np.float64)
                    m = simple.LR(["v"]).fit(X(dv["fit"]), dv["Y_fit"][:, 0])
                    preds.setdefault(nm, []).append(m.predict(X(dv["ev"])))
        y = np.concatenate(ys)
        reg = np.concatenate(regs)
        west = base.region_west[reg]
        pr = {m: np.concatenate(v) for m, v in preds.items()}
        if pair[0] == "ndmi_off":
            pairs = [(f"{fam}_ndmi_on", f"{fam}_ndmi_off"), ("lr_vci_t_ndmi", "lr_vci_t"), (f"{fam}_ndmi_off", "lr_vci_t"),
                     (f"{fam}_ndmi_on", "lr_vci_t_ndmi")]
            name = "ndmi"
        else:
            pairs = [(f"{fam}_cell", f"{fam}_pixel"), ("lr_vci_t_cell", "lr_vci_t_pixel"), (f"{fam}_pixel", "lr_vci_t_pixel"),
                     (f"{fam}_cell", "lr_vci_t_cell")]
            name = "vci_cell_vs_pixel"
        out[name] = scoring.score_subset(name, y, reg, west, pr, base.region_group, base.n_regions, N_FOLDS, pairs)
        out[name]["regions"] = int(len(np.unique(reg)))
        out[name]["folds_skipped_empty"] = skipped
        log(f"  what-if {name}: " + ", ".join(f"{m} {v['all']['auc_within'][0]:.4f}" for m, v in out[name]["models"].items()))
    return out


def run(status=None) -> dict:
    import run as runner
    fam = best_job1_family()
    sp = specs()
    runner.run_specs("whatif", sp, n_cpu=3, n_mps=2, threads=3, status=status)
    res = {"git": git_state(), "family": fam, **collect(fam)}
    write_json(SCORES / "whatif.json", res)
    return res
