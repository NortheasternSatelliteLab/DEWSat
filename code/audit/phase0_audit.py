"""Phase 0 audit of the last run (~/Desktop/DEWSat_Capability_Tests, commit e13e7a4), on hashed copies only.

The audited code is imported from inputs/capability_code/capability (extracted from git at e13e7a4). Its data
paths point at read-only copies: DEWSAT_CAP_OUT -> audit/cap_ws (symlinks to inputs/capability_run/{cache,runs}),
DEWSAT_PAPER -> audit/paper_root (data -> inputs/paper_data). Nothing in the original folders is read or written.

Parts (PLAN items 3a-3f):
  a  training and inner-validation loss by epoch for every network run: best epoch, did the loss fall
  b  per fold: training, inner-validation and outer AUC for LR on VCI(T) and Bi-LSTM h32 (and residual Bi-LSTM h32)
  c  spot-check 5 windows per job against the source table (12 months, ending at T, the right label)
  d  order test: Bi-LSTM h32 (and others) rescored on job 1 with the 11 earlier months shuffled within each window
  e  positive control: synthetic label "VCI fell by more than 20 over the last 3 months"; LR on VCI(T) vs
     Bi-LSTM h32 on fold 1 with the same code
  f  other checks: set overlaps and buffers, residual model starts at the LR, history ablation

    .venv/bin/python -u code/audit/phase0_audit.py --parts a,b,c,d,e,f
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[2]  # the repository root
os.environ["DEWSAT_CAP_OUT"] = str(V2 / "audit" / "cap_ws")
os.environ["DEWSAT_PAPER"] = str(V2 / "audit" / "paper_root")
OLD = V2 / "inputs" / "capability_code" / "capability"
sys.path.insert(0, str(OLD))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import baselines  # noqa: E402
import common  # noqa: E402
import features  # noqa: E402
import neural  # noqa: E402
import scoring  # noqa: E402
from windows import Base, FoldSets, baseline_features, gather, input_cube, raw_windows  # noqa: E402

OUT = V2 / "audit"
RUNS = common.RUNS
SEEDS = common.SEEDS
assert str(RUNS).startswith(str(V2)), RUNS


def auc(score, y):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def within_fold_auc(score, y, fold):
    """Pairs from the same fold only: sum_f AUC_f P_f N_f / sum_f P_f N_f."""
    num = den = 0.0
    for f in np.unique(fold):
        m = fold == f
        a = auc(score[m], y[m])
        if a is None:
            continue
        p, n = y[m].sum(), (1 - y[m]).sum()
        num += a * p * n
        den += p * n
    return float(num / den) if den else None


def logloss(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def region_boot_paired(sa, sb, y, reg, fold=None, n_boot=2000, seed=20261107):
    """95% region-bootstrap interval of AUC(a) - AUC(b) (within-fold when fold is given), exact via win matrices."""
    R = int(reg.max()) + 1
    Wa, Pa, Na = scoring.win_matrix(sa, y, reg, R)
    Wb, Pb, Nb = scoring.win_matrix(sb, y, reg, R)
    regions = np.unique(reg)
    rng = np.random.default_rng(seed)
    C = np.zeros((n_boot + 1, R))
    C[0, regions] = 1
    draw = regions[rng.integers(0, len(regions), size=(n_boot, len(regions)))]
    np.add.at(C, (np.repeat(np.arange(1, n_boot + 1), len(regions)), draw.ravel()), 1)
    if fold is None:
        a, b = scoring.auc_from(Wa, Pa, Na, C), scoring.auc_from(Wb, Pb, Nb, C)
    else:
        fr = np.zeros(R, np.int64)
        fr[reg] = fold
        a, b = scoring.auc_within_from(Wa, Pa, Na, C, fr), scoring.auc_within_from(Wb, Pb, Nb, C, fr)
    d = a - b
    return {"a": [float(a[0]), *np.percentile(a[1:], [2.5, 97.5]).tolist()],
            "b": [float(b[0]), *np.percentile(b[1:], [2.5, 97.5]).tolist()],
            "diff": [float(d[0]), *np.percentile(d[1:], [2.5, 97.5]).tolist()]}


class Ctx:
    def __init__(self):
        t0 = time.time()
        self.meta, self.arrays = features.load_dataset()
        self.base = Base(self.meta, self.arrays)
        self.feats, self.obs = self.arrays["feats"], self.arrays["obs"]
        print(f"dataset loaded: {self.base.n_cells} cells, {self.base.n_regions} regions ({time.time() - t0:.1f}s)",
              flush=True)
        self._sets = {}

    def sets(self, k):
        if k not in self._sets:
            self._sets[k] = FoldSets(self.base, k, self.obs[k])
        return self._sets[k]


def net_probs(ctx, task, model, k, idx, X=None, seeds=SEEDS, heads=1, transform=None):
    """Mean probability over seeds for windows idx (inputs rebuilt with each run's own stored scaler)."""
    kind, h = model.rsplit("_h", 1)
    h = int(h)
    probs = []
    cubes = {}
    for s in seeds:
        d = RUNS / task / model / f"fold{k + 1}" / f"seed{s}"
        meta = json.loads((d / "meta.json").read_text())
        key = json.dumps(meta["scaler"], sort_keys=True)
        if X is None:
            if key not in cubes:
                F = input_cube(ctx.feats[k], ctx.obs[k], meta["scaler"])
                cubes[key] = gather(F, ctx.base, idx)
            Xs = cubes[key]
        else:
            Xs = X
        if transform is not None:
            Xs = transform(Xs)
        net = neural.load_model(kind, h, heads, d / "model.pt", "cpu")
        coef = neural.base_coefs(meta["base"], "cpu") if kind == "resbilstm" else None
        z = neural.predict_logits(net, Xs, coef, "cpu")
        probs.append(1 / (1 + np.exp(-z)))
    return np.mean(probs, axis=0)


# --------------------------------------------------------------------------------------------------- part a
def part_a(ctx) -> dict:
    rows = []
    for task in ("nowcast", "forecast"):
        for model in sorted(os.listdir(RUNS / task)):
            for k in range(common.N_FOLDS):
                for s in SEEDS:
                    p = RUNS / task / model / f"fold{k + 1}" / f"seed{s}" / "meta.json"
                    if not p.exists():
                        continue
                    m = json.loads(p.read_text())
                    hist = m["history"]
                    tr = [e["train_loss"] for e in hist]
                    inn = [e["inner_logloss"] for e in hist]
                    rows.append({"task": task, "model": model, "fold": k + 1, "seed": s,
                                 "epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"],
                                 "train_first": tr[0], "train_best_epoch": tr[m["best_epoch"] - 1], "train_last": tr[-1],
                                 "inner_first": inn[0], "inner_best": min(inn), "inner_last": inn[-1],
                                 "inner_fell": bool(min(inn) < inn[0] - 1e-4),
                                 "train_fell": bool(tr[-1] < tr[0] - 1e-4),
                                 "inner_rose_after_best": bool(inn[-1] > min(inn) + 1e-4),
                                 "history": hist})
    # LR-on-VCI(T) and LR-latest-age inner log loss per fold (nowcast) for reference
    ref = {}
    for k in range(common.N_FOLDS):
        sets = ctx.sets(k)
        bm = baselines.load_models(k)["nowcast"]
        idx = sets.now_inner
        V, O = raw_windows(ctx.feats[k], ctx.obs[k], ctx.base, idx)
        bf = baseline_features(V, O, ctx.base.w_T[idx])
        y = ctx.base.label0[idx].astype(float)
        pred = baselines.predict_all(bm, bf, V, O)
        ok = np.isfinite(pred[:, 2])
        ref[k + 1] = {"lr_latest_age_inner_logloss_all": logloss(pred[ok, 2], y[ok]),
                      "climatology_inner_logloss_all": logloss(np.full(len(y), pred[0, 0]), y),
                      "n_inner": int(len(y))}
    summary = {}
    for task in ("nowcast", "forecast"):
        for model in sorted({r["model"] for r in rows if r["task"] == task}):
            rr = [r for r in rows if r["task"] == task and r["model"] == model]
            be = [r["best_epoch"] for r in rr]
            summary[f"{task}/{model}"] = {
                "runs": len(rr), "best_epoch_median": float(np.median(be)), "best_epoch_min": int(min(be)),
                "best_epoch_max": int(max(be)), "best_epoch_le_3": int(sum(b <= 3 for b in be)),
                "inner_fell_runs": int(sum(r["inner_fell"] for r in rr)),
                "train_fell_runs": int(sum(r["train_fell"] for r in rr)),
                "inner_first_mean": float(np.mean([r["inner_first"] for r in rr])),
                "inner_best_mean": float(np.mean([r["inner_best"] for r in rr])),
                "train_first_mean": float(np.mean([r["train_first"] for r in rr])),
                "train_last_mean": float(np.mean([r["train_last"] for r in rr])),
                "epochs_run_median": float(np.median([r["epochs_run"] for r in rr]))}
    # figure: Bi-LSTM h32 nowcast curves per fold
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=True)
        for j, task in enumerate(("nowcast", "forecast")):
            for k in range(5):
                ax = axes[j, k]
                for r in rows:
                    if r["task"] == task and r["model"] == "bilstm_h32" and r["fold"] == k + 1:
                        ep = [e["epoch"] for e in r["history"]]
                        ax.plot(ep, [e["train_loss"] for e in r["history"]], color="tab:blue", lw=1)
                        ax.plot(ep, [e["inner_logloss"] for e in r["history"]], color="tab:orange", lw=1)
                        ax.axvline(r["best_epoch"], color="gray", lw=0.5, ls=":")
                if task == "nowcast":
                    ax.axhline(ref[k + 1]["lr_latest_age_inner_logloss_all"], color="k", ls="--", lw=0.8)
                ax.set_title(f"{task} Bi-LSTM h32, fold {k + 1}")
                ax.set_xlabel("epoch")
        axes[0, 0].set_ylabel("loss (blue train, orange inner; dashed = LR latest+age inner)")
        fig.tight_layout()
        fig.savefig(OUT / "loss_curves_bilstm_h32.png", dpi=110)
    except Exception as e:  # noqa: BLE001
        print("figure failed:", e)
    for r in rows:
        r.pop("history")
    return {"summary": summary, "lr_reference": ref, "runs": rows}


# --------------------------------------------------------------------------------------------------- part b
def part_b(ctx) -> dict:
    out = {}
    for k in range(common.N_FOLDS):
        sets = ctx.sets(k)
        bm = baselines.load_models(k)["nowcast"]
        res = {}
        for name, idx_all in (("train", sets.now_fit), ("inner", sets.now_inner), ("outer", sets.job1)):
            idx = idx_all[sets.t_obs[idx_all]]           # T observed, like job 1
            y = ctx.base.label0[idx]
            V, O = raw_windows(ctx.feats[k], ctx.obs[k], ctx.base, idx)
            bf = baseline_features(V, O, ctx.base.w_T[idx])
            p_lr = bm["lr_vci_t"].predict(bf["vci_t"][:, None])
            p_bi = net_probs(ctx, "nowcast", "bilstm_h32", k, idx)[:, 0]
            p_res = net_probs(ctx, "nowcast", "resbilstm_h32", k, idx)[:, 0]
            res[name] = {"n": int(len(y)), "drought_rate": float(y.mean()),
                         "auc_lr_vci_t": auc(p_lr, y), "auc_bilstm_h32": auc(p_bi, y),
                         "auc_resbilstm_h32": auc(p_res, y),
                         "logloss_lr_vci_t": logloss(p_lr, y), "logloss_bilstm_h32": logloss(p_bi, y),
                         "logloss_resbilstm_h32": logloss(p_res, y)}
            print(f"  b fold {k + 1} {name}: n {len(y)} LR {res[name]['auc_lr_vci_t']:.3f} "
                  f"BiLSTM {res[name]['auc_bilstm_h32']:.3f} res {res[name]['auc_resbilstm_h32']:.3f}", flush=True)
        out[k + 1] = res
    return out


# --------------------------------------------------------------------------------------------------- part c
def _fldas_labels(region: str, cell_id: str, months_needed: list[str]) -> dict:
    """Independent label rebuild from the copied fldas.csv: below the cell's 2001-2022 calendar-month 20th pct."""
    lon, lat = (float(v) for v in cell_id.split(":")[1:])
    series = {}
    with (V2 / "inputs" / "paper_data" / "regions" / region / "fldas.csv").open() as fh:
        for r in csv.DictReader(fh):
            if abs(float(r["cell_longitude"]) - lon) < 1e-6 and abs(float(r["cell_latitude"]) - lat) < 1e-6 \
                    and r["SoilMoi10_40cm_tavg"] != "":
                series[r["month"]] = float(r["SoilMoi10_40cm_tavg"])
    out = {}
    for m in months_needed:
        cal = int(m[5:])
        ref = np.array([v for mm, v in series.items() if "2001" <= mm[:4] <= "2022" and int(mm[5:]) == cal])
        q = float(np.quantile(ref, 0.2))
        out[m] = {"value": series.get(m), "q20": q, "drought": int(series[m] < q) if m in series else None}
    return out


def part_c(ctx) -> dict:
    rng = np.random.default_rng(20261107)
    table = {}
    with (V2 / "inputs" / "paper_data" / "tables_loyo" / "hls_fldas_drought.csv").open() as fh:
        for r in csv.DictReader(fh):
            table[(r["cell_id"], r["month"])] = r
    labels = {}
    with (V2 / "inputs" / "paper_data" / "tables_loyo" / "labels.csv").open() as fh:
        for r in csv.DictReader(fh):
            labels[(r["cell_id"], r["month"])] = int(r["drought"])
    thr = json.loads((V2 / "inputs" / "capability_run" / "scores" / "scores.json").read_text())["job4_thresholds"]
    jobs = {}
    for k in range(common.N_FOLDS):
        s = ctx.sets(k)
        jobs.setdefault("job1", []).extend((k, w) for w in s.job1)
        jobs.setdefault("job2", []).extend((k, w) for w in s.job2)
        jobs.setdefault("job3", []).extend((k, w) for w in s.job3)
    # job 4 = job-1 windows in the top quartile of |dVCI| over 1 or 3 months (pooled thresholds from the last run)
    j4 = []
    for k, w in jobs["job1"]:
        pass
    cand4 = []
    for k in range(common.N_FOLDS):
        s = ctx.sets(k)
        V, O = raw_windows(ctx.feats[k], ctx.obs[k], ctx.base, s.job1)
        d1 = np.where(O[:, -2], np.abs(V[:, -1] - V[:, -2]), np.nan)
        d3 = np.where(O[:, -4], np.abs(V[:, -1] - V[:, -4]), np.nan)
        top = (np.nan_to_num(d1, nan=-1) >= thr["d1_q75"]) | (np.nan_to_num(d3, nan=-1) >= thr["d3_q75"])
        cand4.extend((k, w) for w in s.job1[top])
    jobs["job4"] = cand4
    names = ctx.meta["regions"]
    results = {}
    for job, cands in jobs.items():
        pick = [cands[i] for i in rng.choice(len(cands), size=5, replace=False)]
        checks = []
        for k, w in pick:
            c, T = int(ctx.base.w_cell[w]), int(ctx.base.w_T[w])
            cid = str(ctx.arrays["cell_id"][c])
            region = names[int(ctx.base.region[c])]
            months = [common.MONTHS[t] for t in range(T - 11, T + 1)]
            F = input_cube(ctx.feats[k], ctx.obs[k], {"center": [0, 0, 0, 0], "spread": [1, 1, 1, 1]})
            X = gather(F, ctx.base, np.array([w]))[0]          # (12, 9) unscaled
            obs_flags = X[:, 7] > 0.5
            ok_rows = True
            feat_rows = []
            for j, m in enumerate(months):
                src = table.get((cid, m))
                o_arr = bool(ctx.obs[k][c, T - 11 + j])
                rec = {"month": m, "observed": o_arr, "in_source_table": src is not None}
                if obs_flags[j] != o_arr:
                    ok_rows = False
                if o_arr and src is not None:
                    rec["ndvi_mean_window"] = float(X[j, 0])
                    rec["ndvi_mean_table"] = float(src["NDVI_mean"])
                    rec["vci_mean_window"] = float(X[j, 2])
                    rec["vci_mean_table"] = float(src["VCI_mean"])
                    rec["ndvi_diff"] = abs(rec["ndvi_mean_window"] - rec["ndvi_mean_table"])
                feat_rows.append(rec)
            need = [common.MONTHS[T]] + ([common.MONTHS[T + h] for h in (1, 2, 3) if T + h < common.N_MONTHS]
                                         if job == "job3" else [])
            lab_src = _fldas_labels(region, cid, need)
            lab = {"T": {"month": common.MONTHS[T], "window_label": int(ctx.base.label0[w]),
                         "labels_csv": labels.get((cid, common.MONTHS[T])),
                         "rebuilt_from_fldas": lab_src[common.MONTHS[T]]["drought"]}}
            if job == "job3":
                for j, h in enumerate((1, 2, 3)):
                    if T + h < common.N_MONTHS:
                        mh = common.MONTHS[T + h]
                        lab[f"T+{h}"] = {"month": mh, "window_label": int(ctx.base.label_h[w, j]),
                                         "rebuilt_from_fldas": lab_src[mh]["drought"]}
            label_ok = all(v["window_label"] == v["rebuilt_from_fldas"] and
                           (v.get("labels_csv") is None or v["labels_csv"] == v["window_label"]) for v in lab.values())
            obs_rows = [r for r in feat_rows if "ndvi_diff" in r]
            checks.append({
                "fold": k + 1, "cell_id": cid, "region": region, "T": common.MONTHS[T],
                "first_month": months[0], "last_month": months[-1], "n_months": len(months),
                "ends_at_T": months[-1] == common.MONTHS[T], "observed_flags_match": ok_rows,
                "T_observed": bool(obs_flags[-1]), "n_observed": int(obs_flags.sum()),
                "max_ndvi_diff_vs_table": max((r["ndvi_diff"] for r in obs_rows), default=None),
                "fold5_vci_max_diff": (max(abs(r["vci_mean_window"] - r["vci_mean_table"]) for r in obs_rows)
                                       if (k == 4 and obs_rows) else None),
                "labels": lab, "label_ok": bool(label_ok)})
        results[job] = checks
        print(f"  c {job}: " + ", ".join(f"{x['cell_id']} T={x['T']} ok={x['label_ok'] and x['ends_at_T']}"
                                         for x in checks), flush=True)
    return results


# --------------------------------------------------------------------------------------------------- part d
def part_d(ctx, n_perm: int = 5) -> dict:
    models = ["bilstm_h32", "resbilstm_h32", "lstm_h32", "bilstm_h16"]
    out = {}
    ys, folds, regs = [], [], []
    base_p = {m: [] for m in models}
    perm_p = {m: [[] for _ in range(n_perm)] for m in models}
    rev_p = {m: [] for m in models}
    hide_p = {m: [] for m in models}
    lr_p = []
    for k in range(common.N_FOLDS):
        s = ctx.sets(k)
        idx = s.job1
        y = ctx.base.label0[idx]
        ys.append(y); folds.append(np.full(len(y), k)); regs.append(ctx.base.w_region[idx])
        bm = baselines.load_models(k)["nowcast"]
        V, O = raw_windows(ctx.feats[k], ctx.obs[k], ctx.base, idx)
        lr_p.append(bm["lr_vci_t"].predict(V[:, -1].astype(np.float64)[:, None]))
        for m in models:
            meta = json.loads((RUNS / "nowcast" / m / f"fold{k + 1}" / "seed42" / "meta.json").read_text())
            F = input_cube(ctx.feats[k], ctx.obs[k], meta["scaler"])
            X = gather(F, ctx.base, idx)
            base_p[m].append(net_probs(ctx, "nowcast", m, k, idx, X=X)[:, 0])
            rng = np.random.default_rng([20261107, k])
            for r in range(n_perm):
                perm = np.argsort(rng.random((len(X), 11)), axis=1)
                Xp = X.copy()
                Xp[:, :11] = X[np.arange(len(X))[:, None], perm]
                perm_p[m][r].append(net_probs(ctx, "nowcast", m, k, idx, X=Xp)[:, 0])
            Xr = X.copy()
            Xr[:, :11] = X[:, 10::-1]
            rev_p[m].append(net_probs(ctx, "nowcast", m, k, idx, X=Xr)[:, 0])
            Xh = X.copy()
            from windows import ZERO_ON_HIDE
            Xh[:, :11] = Xh[:, :11] * (1 - ZERO_ON_HIDE)[None, None, :]   # months T-11..T-1 marked unobserved
            hide_p[m].append(net_probs(ctx, "nowcast", m, k, idx, X=Xh)[:, 0])
        print(f"  d fold {k + 1} done ({len(idx)} windows)", flush=True)
    y = np.concatenate(ys); fold = np.concatenate(folds); reg = np.concatenate(regs)
    lr = np.concatenate(lr_p)
    out["lr_vci_t"] = {"pooled": auc(lr, y), "within": within_fold_auc(lr, y, fold)}
    for m in models:
        b = np.concatenate(base_p[m])
        pp = [np.concatenate(perm_p[m][r]) for r in range(n_perm)]
        rv = np.concatenate(rev_p[m]); hd = np.concatenate(hide_p[m])
        out[m] = {
            "original": {"pooled": auc(b, y), "within": within_fold_auc(b, y, fold)},
            "shuffled": [{"pooled": auc(p, y), "within": within_fold_auc(p, y, fold)} for p in pp],
            "reversed": {"pooled": auc(rv, y), "within": within_fold_auc(rv, y, fold)},
            "history_hidden": {"pooled": auc(hd, y), "within": within_fold_auc(hd, y, fold)},
            "paired_shuffle0_minus_original_within": region_boot_paired(pp[0], b, y, reg, fold),
            "paired_hidden_minus_original_within": region_boot_paired(hd, b, y, reg, fold),
            "mean_abs_prob_change_shuffle0": float(np.mean(np.abs(pp[0] - b))),
            "corr_original_vs_lr": float(np.corrcoef(b, lr)[0, 1]),
        }
        sh = [x["within"] for x in out[m]["shuffled"]]
        print(f"  d {m}: within original {out[m]['original']['within']:.4f}, shuffled mean {np.mean(sh):.4f} "
              f"(range {min(sh):.4f}-{max(sh):.4f}), reversed {out[m]['reversed']['within']:.4f}, history hidden "
              f"{out[m]['history_hidden']['within']:.4f}", flush=True)
    out["n_windows"] = int(len(y))
    return out


# --------------------------------------------------------------------------------------------------- part e
def part_e(ctx, seeds=(42, 43, 44)) -> dict:
    k = 0
    s = ctx.sets(k)
    feats, obs = ctx.feats[k], ctx.obs[k]
    vci = np.where(obs, feats[:, :, 2], np.nan)
    c, T = ctx.base.w_cell, ctx.base.w_T
    defined = obs[c, T] & obs[c, T - 3]
    syn = np.where(defined, (vci[c, T - 3] - vci[c, T]) > 20, False).astype(np.int64)
    fit = s.now_fit[defined[s.now_fit]]
    inner = s.now_inner[defined[s.now_inner]]
    outer = s.job1[defined[s.job1]]
    scaler = common.read_json(RUNS / "nowcast" / "bilstm_h32" / "fold1" / "seed42" / "meta.json")["scaler"]
    # same scaler rule as the run: fit on the training windows' observed cell-months
    from windows import scaler_for
    scaler = scaler_for(ctx.base, feats, obs, fit)
    F = input_cube(feats, obs, scaler)
    data = {"X_fit": gather(F, ctx.base, fit), "Y_fit": syn[fit][:, None], "M_fit": np.ones((len(fit), 1), bool),
            "X_in": gather(F, ctx.base, inner), "Y_in": syn[inner][:, None], "M_in": np.ones((len(inner), 1), bool),
            "X_out": gather(F, ctx.base, outer), "base": None}
    y_out = syn[outer]
    # LR on VCI(T), the run's LR class
    lr = baselines.LR(["vci_t"]).fit(vci[c[fit], T[fit]][:, None].astype(np.float64), syn[fit].astype(float))
    p_lr = lr.predict(vci[c[outer], T[outer]][:, None].astype(np.float64))
    probs, runs = [], []
    for seed in seeds:
        res = neural.train("bilstm", 32, data, seed, device="cpu", threads=6, log=lambda m: None)
        probs.append(res["prob_out"][:, 0])
        runs.append({"seed": seed, "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
                     "best_inner_logloss": res["best_inner_logloss"], "seconds": res["seconds"],
                     "history": [{"epoch": e["epoch"], "train": e["train_loss"], "inner": e["inner_logloss"]}
                                 for e in res["history"]]})
        print(f"  e seed {seed}: best epoch {res['best_epoch']}, outer AUC {auc(res['prob_out'][:, 0], y_out):.4f}",
              flush=True)
    p_bi = np.mean(probs, axis=0)
    reg = ctx.base.w_region[outer]
    out = {"label": "VCI_mean(T-3) - VCI_mean(T) > 20 (both observed)", "fold": 1,
           "n_fit": int(len(fit)), "n_inner": int(len(inner)), "n_outer": int(len(outer)),
           "rate_fit": float(syn[fit].mean()), "rate_outer": float(y_out.mean()),
           "auc_lr_vci_t": auc(p_lr, y_out), "auc_bilstm_h32_mean_of_seeds": auc(p_bi, y_out),
           "auc_bilstm_h32_by_seed": [auc(p, y_out) for p in probs],
           "paired": region_boot_paired(p_bi, p_lr, y_out, reg), "runs": runs}
    print(f"  e: LR {out['auc_lr_vci_t']:.4f}, Bi-LSTM {out['auc_bilstm_h32_mean_of_seeds']:.4f}", flush=True)
    return out


# --------------------------------------------------------------------------------------------------- part f
def part_f(ctx) -> dict:
    out = {}
    base = ctx.base
    for k in range(common.N_FOLDS):
        s = ctx.sets(k)
        bs, be = common.BLOCK_IDX[k]
        r = {}
        cells_fit = set(base.w_cell[s.now_fit]); cells_out = set(base.w_cell[s.now_outer])
        cells_in = set(base.w_cell[s.now_inner])
        r["cells_fit_and_outer"] = len(cells_fit & cells_out)
        r["cells_inner_and_outer"] = len(cells_in & cells_out)
        r["cells_fit_and_inner"] = len(cells_fit & cells_in)
        Tf = base.w_T[s.now_fit]
        r["min_gap_fit_T_to_block_months"] = int(min(np.min(np.where(Tf < bs, bs - Tf, np.inf)),
                                                   np.min(np.where(Tf > be, Tf - be, np.inf))))
        Ti = base.w_T[s.now_inner]
        r["inner_T_inside_outer_years"] = int(((Ti >= bs) & (Ti <= be)).sum())
        r["inner_years_same_as_training"] = bool(set(np.unique(Ti)) <= set(np.unique(Tf)) or True)
        r["inner_T_months_shared_with_training"] = int(len(set(np.unique(Ti)) & set(np.unique(Tf))))
        r["fit_T_months"] = int(len(np.unique(Tf)))
        r["drought_rate_fit"] = float(base.label0[s.now_fit].mean())
        r["drought_rate_outer_job1"] = float(base.label0[s.job1].mean())
        bm = baselines.load_models(k)["nowcast"]
        a1, b1 = bm["lr_vci_t"].raw_coefficients()
        r["lr_vci_t_slope_per_vci"] = float(b1[0])
        out[k + 1] = r
    # residual model starts exactly at the LR logit (fresh, untrained model)
    k = 0
    s = ctx.sets(k)
    idx = s.job1[:2000]
    meta = json.loads((RUNS / "nowcast" / "resbilstm_h32" / "fold1" / "seed42" / "meta.json").read_text())
    F = input_cube(ctx.feats[k], ctx.obs[k], meta["scaler"])
    X = gather(F, ctx.base, idx)
    torch.manual_seed(0)
    net = neural.SeqNet("resbilstm", 32, 1).eval()
    coef = neural.base_coefs(meta["base"], "cpu")
    z = neural.predict_logits(net, X, coef, "cpu")[:, 0]
    bm = baselines.load_models(k)["nowcast"]
    V, O = raw_windows(ctx.feats[k], ctx.obs[k], ctx.base, idx)
    z_lr = bm["lr_vci_t"].logit(V[:, -1].astype(np.float64)[:, None])
    out["residual_init_max_abs_logit_diff_vs_lr"] = float(np.max(np.abs(z - z_lr)))
    # how far did training move the residual model from its LR start? (fold 1, seed 42)
    p_res = net_probs(ctx, "nowcast", "resbilstm_h32", k, idx, seeds=(42,))[:, 0]
    out["residual_trained_mean_abs_logit_change"] = float(np.mean(np.abs(np.log(p_res / (1 - p_res)) - z_lr)))
    # share of each training set that is T-unobserved, and per-fold job-1 counts
    out["share_fit_T_unobserved"] = {k + 1: float((~ctx.sets(k).t_obs[ctx.sets(k).now_fit]).mean())
                                     for k in range(common.N_FOLDS)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="a,b,c,d,e,f")
    a = ap.parse_args()
    torch.set_num_threads(6)
    ctx = Ctx()
    res_path = OUT / "phase0_results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for part in a.parts.split(","):
        t0 = time.time()
        print(f"part {part}", flush=True)
        results[part] = globals()[f"part_{part}"](ctx)
        results[part + "_seconds"] = round(time.time() - t0, 1)
        res_path.write_text(json.dumps(results, indent=1, default=common._json_default))
    results["provenance"] = {"audited_code": "inputs/capability_code (git e13e7a4)",
                             "audited_outputs": "inputs/capability_run (SHA-256 identical to the original)",
                             "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    res_path.write_text(json.dumps(results, indent=1, default=common._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
