"""Weekly-input experiment (amendment 3; DESIGN.md 10). Secondary; it does not change the gate.

On development cells with per-scene values: 52 weekly steps ending on the label month's last day, built from single
clear views (>= 30% of the cell clear) with a per-cell single-view VCI table from each fold's reference months
(leave-one-year-out). LR on the latest VCI + its age (days) vs the best job-1 network family with its fold-chosen
setting (masking off, 3 seeds). Same folds, buffers and metric as job 1. Training windows are capped at 300,000 per
fold (seeded subsample, the same for both models) to keep the 52-step inputs in memory.
"""

from __future__ import annotations

import time

import numpy as np

import nets
import scoring
import simple
import units
import views
from common import CACHE, N_FOLDS, RUNS, SCORES, git_state, log, read_json, save_npz, write_json
from features import reference_mask
from windows import Sets

SEEDS = (42, 43, 44)
MAX_TRAIN = 300_000
WEEKS = 52


def unit_dir(spec):
    return RUNS / "weekly" / spec["cfg"]["family"] / f"fold{spec['k'] + 1}" / f"seed{spec['seed']}"


def fold_path(k):
    return CACHE / "weekly" / f"fold{k + 1}.npz"


def build_fold(k: int) -> dict:
    """Weekly inputs for fold k's training (subsampled) and validation windows (cached)."""
    p = fold_path(k)
    if p.exists():
        with np.load(p) as z:
            return {kk: z[kk] for kk in z.files}
    t0 = time.time()
    c = units.ctx()
    base, arrays, meta = c["base"], c["arrays"], c["meta"]
    obs = arrays["obs"][k]
    sets = Sets(base, "job1", k, obs)
    # every labelled window of cells with views, in fold k's training months / validation block (T need not be
    # observed in the composite: the weekly series has its own views)
    from common import BLOCK_IDX, BUFFER_NOWCAST
    bs, be = BLOCK_IDX[k]
    T, grp = base.w_T, base.w_group
    has_views = arrays["has_views"][base.w_cell] & (base.label0 >= 0)
    tr = np.where(has_views & (grp != k) & ((T < bs - BUFFER_NOWCAST) | (T > be + BUFFER_NOWCAST)))[0]
    va = np.where(has_views & (grp == k) & (T >= bs) & (T <= be))[0]
    rng = np.random.default_rng([20261108, k, 52])
    if len(tr) > MAX_TRAIN:
        tr = np.sort(rng.choice(tr, MAX_TRAIN, replace=False))
    rm = reference_mask(f"f{k + 1}")
    names = meta["dev"]["regions"]
    out = {"tr": tr, "va": va}
    for part, idx in (("tr", tr), ("va", va)):
        X = np.zeros((len(idx), WEEKS, 7), np.float16)
        lv = np.full(len(idx), np.nan, np.float32)
        la = np.full(len(idx), np.nan, np.float32)
        cells = base.w_cell[idx]
        order = np.argsort(cells, kind="stable")
        cur_region, rv, id_pos = None, None, None
        uc, first = np.unique(cells[order], return_index=True)
        bounds = list(first) + [len(order)]
        for ci, cell in enumerate(uc):
            rows = order[bounds[ci]:bounds[ci + 1]]
            r = int(base.region[cell])
            if r != cur_region:
                cur_region = r
                rv = views.region_views(names[r])
                id_pos = {s: i for i, s in enumerate(rv["ids"])} if rv is not None else None
            if rv is None:
                continue
            cv = views.cell_views(rv, id_pos[str(arrays["cell_id"][cell])])
            if cv is None:
                continue
            vci = views.single_view_vci(cv, rm)
            Xc, a, b = views.weekly_series(cv, vci, base.w_T[idx[rows]], WEEKS)
            X[rows], lv[rows], la[rows] = Xc.astype(np.float16), a, b
        out[f"X_{part}"], out[f"lv_{part}"], out[f"la_{part}"] = X, lv, la
        out[f"y_{part}"] = base.label0[idx].astype(np.int8)
    out["seconds"] = np.array(time.time() - t0)
    save_npz(p, **out)
    log(f"  weekly fold {k + 1}: {len(tr)} training and {len(va)} validation windows ({time.time() - t0:.0f}s)")
    return out


def run_unit(spec: dict, device: str, threads: int) -> dict:
    d_out = unit_dir(spec)
    if (d_out / "DONE").exists():
        return read_json(d_out / "meta.json")
    k, cfg = spec["k"], dict(spec["cfg"])
    cfg["mask_p"] = 0.0
    f = build_fold(k)
    ok_tr = np.isfinite(f["lv_tr"])          # at least one clear view in the window
    ok_va = np.isfinite(f["lv_va"])
    Xtr = f["X_tr"][ok_tr].astype(np.float32)
    Xva = f["X_va"][ok_va].astype(np.float32)
    ytr = f["y_tr"][ok_tr].astype(np.float32)[:, None]
    data = {"X_fit": Xtr, "Y_fit": ytr, "M_fit": np.ones_like(ytr, bool), "hide_ok": np.zeros(WEEKS, bool),
            "obs_ch": 3, "X_eval": [("ev", Xva, None)]}
    if cfg["family"] == "resbilstm":
        lr = simple.LR(["v", "a"]).fit(np.stack([f["lv_tr"][ok_tr], f["la_tr"][ok_tr]], 1).astype(np.float64), ytr[:, 0])
        data["base_fit"] = lr.logit(np.stack([f["lv_tr"][ok_tr], f["la_tr"][ok_tr]], 1).astype(np.float64))[:, None].astype(np.float32)
        data["X_eval"] = [("ev", Xva, lr.logit(np.stack([f["lv_va"][ok_va], f["la_va"][ok_va]], 1).astype(np.float64))[:, None].astype(np.float32))]
    res = nets.train(cfg, data, spec["seed"], device=device, threads=threads)
    z = res["logits"]["ev"]
    d_out.mkdir(parents=True, exist_ok=True)
    save_npz(d_out / "pred.npz", wid=f["va"][ok_va].astype(np.int64), logit=z.astype(np.float32))
    meta = {"k": k + 1, "cfg": cfg, "seed": spec["seed"], "device": device, "git": git_state(),
            "seconds": res["seconds"], "n_fit": int(ok_tr.sum()), "n_ev": int(ok_va.sum()),
            "metric": {"metric": simple.auc(z[:, 0], f["y_va"][ok_va])}}
    write_json(d_out / "meta.json", meta)
    (d_out / "DONE").write_text(meta["git"]["commit"] + "\n")
    return meta


def run(status=None) -> dict:
    import run as runner
    import whatif_stage
    fam = whatif_stage.best_job1_family()
    usable = []
    for k in range(N_FOLDS):
        f = build_fold(k)
        if np.isfinite(f["lv_tr"]).any() and np.isfinite(f["lv_va"]).any():
            usable.append(k)
    sp = [{"type": "weekly", "job": "weekly", "k": k, "cfg": units.chosen_config("job1", fam, k), "seed": s}
          for k in usable for s in SEEDS]
    runner.run_specs("weekly", sp, n_cpu=2, n_mps=2, threads=4, status=status)
    c = units.ctx()
    base = c["base"]
    ys, regs, p_lr, p_net, cover = [], [], [], [], {}
    for k in usable:
        f = build_fold(k)
        ok_tr, ok_va = np.isfinite(f["lv_tr"]), np.isfinite(f["lv_va"])
        lr = simple.LR(["latest_vci", "age_days"]).fit(
            np.stack([f["lv_tr"][ok_tr], f["la_tr"][ok_tr]], 1).astype(np.float64), f["y_tr"][ok_tr].astype(float))
        p_lr.append(lr.predict(np.stack([f["lv_va"][ok_va], f["la_va"][ok_va]], 1).astype(np.float64)))
        ps = []
        for s in SEEDS:
            with np.load(unit_dir({"cfg": {"family": fam}, "k": k, "seed": s}) / "pred.npz") as z:
                ps.append(1 / (1 + np.exp(-z["logit"][:, 0])))
        p_net.append(np.mean(ps, axis=0))
        ys.append(f["y_va"][ok_va].astype(np.int64))
        regs.append(base.w_region[f["va"][ok_va]])
        cover[k + 1] = {"validation_windows": int(len(f["va"])), "with_a_clear_view": int(ok_va.sum()),
                        "weeks_with_new_view_share": float(f["X_va"][ok_va][:, :, 3].astype(np.float32).mean())}
    y = np.concatenate(ys)
    reg = np.concatenate(regs)
    preds = {"lr_latest_age_weekly": np.concatenate(p_lr), f"{fam}_weekly": np.concatenate(p_net)}
    sc = scoring.score_subset("weekly", y, reg, base.region_west[reg], preds, base.region_group, base.n_regions,
                              N_FOLDS, [(f"{fam}_weekly", "lr_latest_age_weekly")])
    res = {"git": git_state(), "family": fam, "scores": sc, "coverage": cover, "max_train_windows": MAX_TRAIN,
           "folds_used": [k + 1 for k in usable]}
    write_json(SCORES / "weekly.json", res)
    log(f"  weekly: LR {sc['models']['lr_latest_age_weekly']['all']['auc_within'][0]:.4f}, "
        f"{fam} {sc['models'][f'{fam}_weekly']['all']['auc_within'][0]:.4f}")
    return res
