"""ReLU Bi-LSTM for job 1 (DESIGN.md amendment 7; secondary, no pass/fail, development folds only).

`bilstm_relu` is the `bilstm` with ReLU in place of tanh inside the LSTM cell; the gates stay sigmoid
(nets.HandLSTM). It trains with each fold's chosen `bilstm` setting and the 10 final seeds, on the same windows, with
the same data order and the same initial weights as the original runs. The fresh test stays locked and is not scored.

Stages (posthoc_run.py, namespace "relu": logs/relu/, runs/relu/failures/):
    relu_pilot        fold 1 on the CPU: `bilstm_hand` (the hand-written cell with tanh) for PILOT_SEEDS, and
                      `bilstm_relu` seed 42 (kept: it is also that seed's run in the main stage)
    relu_pilot_score  scores/relu_pilot.json: each `bilstm_hand` seed's outer-validation AUC within PILOT_TOL of the
                      original `bilstm` run of that seed (trained on the CPU); the ReLU run's loss fell and stayed finite
    relu              the 40 `bilstm_relu` runs; refuses to start unless the pilot passed
    relu_score        scores/relu.json: jobs 1 and 4 (through job 1), the score stage's subsets and region draws

Runs: runs/relu/<family>/fold<k>/seed<s>/ (pred.npz, model.pt, meta.json, DONE).

    cd ~/Desktop/DEWSat_BiLSTM_v2 && source env.sh && mkdir -p logs/relu && rm -f logs/relu/STOP
    caffeinate -dimsu nohup .venv/bin/python -u code/dewsat2/posthoc_run.py --stages relu_pilot,relu_pilot_score \
        --cpu-workers 4 --mps-workers 0 --threads 2 >> logs/relu/run.log 2>&1 &
    caffeinate -dimsu nohup .venv/bin/python -u code/dewsat2/posthoc_run.py --stages relu,relu_score \
        --cpu-workers 3 --mps-workers 2 --threads 2 >> logs/relu/run.log 2>&1 &
"""

from __future__ import annotations

import os
import time

import numpy as np

from common import FINAL_SEEDS, N_FOLDS, RUNS, SCORES, git_state, read_json, save_npz, write_json

MODEL = "bilstm_relu"
CHECK = "bilstm_hand"
SEEDS = FINAL_SEEDS
PILOT_FOLD = 0
PILOT_SEEDS = tuple(s for s in (42, 43, 46) if s in FINAL_SEEDS)
PILOT_TOL = 0.001
LABELS = {MODEL: "ReLU Bi-LSTM", CHECK: "Bi-LSTM, hand-written cell (tanh)"}


def cfg_for(k: int, family: str = MODEL) -> dict:
    """Fold k's chosen `bilstm` setting with the family swapped (no re-tuning)."""
    import units
    cfg = units.chosen_config("job1", "bilstm", k)
    if cfg is None:
        raise RuntimeError(f"bilstm fold {k + 1}: no chosen setting")
    return {**cfg, "family": family}


def stage_specs(stage: str) -> list[dict]:
    if stage == "relu_pilot":
        specs = [{"type": "relu", "job": "job1", "k": PILOT_FOLD, "cfg": cfg_for(PILOT_FOLD, CHECK), "seed": s,
                  "cpu_only": True} for s in PILOT_SEEDS]
        return specs + [{"type": "relu", "job": "job1", "k": PILOT_FOLD, "cfg": cfg_for(PILOT_FOLD), "seed": SEEDS[0],
                         "cpu_only": True}]
    if stage == "relu":
        pilot = read_json(SCORES / "relu_pilot.json")
        if not pilot or not pilot.get("passed"):
            raise SystemExit("relu: the pilot has not passed (scores/relu_pilot.json); nothing started")
        return [{"type": "relu", "job": "job1", "k": k, "cfg": cfg_for(k), "seed": s}
                for k in range(N_FOLDS) for s in SEEDS]
    raise ValueError(stage)


def run_dir(family: str, k: int, seed: int):
    return RUNS / "relu" / family / f"fold{k + 1}" / f"seed{seed}"


def spec_dir(spec: dict):
    return run_dir(spec["cfg"]["family"], spec["k"], spec["seed"])


def spec_name(spec: dict) -> str:
    return f"relu/{spec['cfg']['family']}/fold{spec['k'] + 1}/seed{spec['seed']}"


def device_for(cfg: dict) -> str:
    """Where the pipeline places a bilstm of this size (GPU from 128 recurrent units in total)."""
    import search
    return search.device_for({**cfg, "family": "bilstm"})


def run_unit(spec: dict, device: str, threads: int) -> dict:
    """One run, trained exactly as the original `bilstm` final run of the same fold and seed (units.run_net)."""
    import nets
    import posthoc
    import torch
    import units
    out_dir = spec_dir(spec)
    if (out_dir / "DONE").exists():
        return read_json(out_dir / "meta.json")
    if spec.get("cpu_only") and device != "cpu":
        raise RuntimeError(f"{spec_name(spec)} is a CPU-only pilot run, got {device}")
    cfg, k, seed = spec["cfg"], spec["k"], spec["seed"]
    orig = read_json(units.final_dir("job1", "bilstm", k, seed) / "meta.json") or {}
    if {**orig.get("cfg", {}), "family": cfg["family"]} != cfg:
        raise RuntimeError(f"fold {k + 1} seed {seed}: setting differs from the original bilstm run's")
    d = units.prep("job1", k, "outer")
    data = {"X_fit": d["X_fit"], "Y_fit": d["Y_fit"], "M_fit": d["M_fit"], "hide_ok": d["hide_ok"],
            "base_fit": None, "V_fit": None, "base_coef": None, "X_eval": [("ev", d["X_ev"], None)]}
    t0 = time.time()
    res = nets.train(cfg, data, seed, device=device, threads=threads)
    z = res["logits"]["ev"]
    met = units.eval_metric("job1", z, d)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npz(out_dir / "pred.npz", wid=d["ev"].astype(np.int64), logit=z.astype(np.float32))
    torch.save(res["state"], out_dir / "model.pt")
    meta = {"job": "job1", "family": cfg["family"], "fold": k + 1, "split": "outer", "cfg": cfg, "seed": seed,
            "device": device, "threads": threads, "git": git_state(), "n_params": res["n_params"],
            "multiply_adds": posthoc.lstm_macs(cfg["hidden"], cfg.get("layers", 1), d["X_fit"].shape[2],
                                               d["Y_fit"].shape[1]),
            "seconds": res["seconds"], "unit_seconds": round(time.time() - t0, 1), "prep_seconds": d["seconds"],
            "history": res["history"], "metric": met, "scaler": d["scaler"], "n_fit": int(len(d["fit"])),
            "n_ev": int(len(d["ev"])), "original": {"device": orig.get("device"), "seconds": orig.get("seconds"),
                                                    "metric": (orig.get("metric") or {}).get("metric")},
            "pid": os.getpid()}
    write_json(out_dir / "meta.json", meta)
    (out_dir / "DONE").write_text(meta["git"]["commit"] + "\n")
    return meta


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _loss_ok(meta: dict) -> tuple[bool, bool]:
    """(training loss finite throughout, last epoch's loss below the first's)."""
    losses = [h["train_loss"] for h in meta.get("history", [])]
    finite = bool(losses) and bool(np.all(np.isfinite(losses)))
    return finite, bool(finite and losses[-1] < losses[0])


# --------------------------------------------------------------------------------------------- pilot
def pilot_check() -> dict:
    """scores/relu_pilot.json: the hand-written tanh cell reproduces the original CPU runs; the ReLU run trains."""
    import units
    rows, bad = [], []
    for s in PILOT_SEEDS:
        hd, od = run_dir(CHECK, PILOT_FOLD, s), units.final_dir("job1", "bilstm", PILOT_FOLD, s)
        mh, mo = read_json(hd / "meta.json"), read_json(od / "meta.json")
        if mh is None or mo is None or not (hd / "DONE").exists():
            bad.append(f"seed {s}: run missing")
            continue
        with np.load(hd / "pred.npz") as a, np.load(od / "pred.npz") as b:
            same = bool(np.array_equal(a["wid"], b["wid"]))
            dz = float(np.abs(a["logit"] - b["logit"]).max()) if same else None
            dp = float(np.abs(_sig(a["logit"]) - _sig(b["logit"])).max()) if same else None
            r = float(np.corrcoef(a["logit"][:, 0], b["logit"][:, 0])[0, 1]) if same else None
        ah, ao = mh["metric"]["metric"], mo["metric"]["metric"]
        row = {"seed": s, "original_device": mo["device"], "auc_hand": ah, "auc_original": ao,
               "auc_difference": ah - ao, "max_logit_difference": dz, "max_probability_difference": dp,
               "logit_correlation": r, "loss_hand": [h["train_loss"] for h in mh["history"]],
               "loss_original": [h["train_loss"] for h in mo["history"]], "seconds_hand": mh["seconds"],
               "seconds_original": mo["seconds"]}
        rows.append(row)
        if mo["device"] != "cpu":
            bad.append(f"seed {s}: the original run was trained on {mo['device']}, not the CPU")
        if not same:
            bad.append(f"seed {s}: window ids differ")
        elif abs(ah - ao) > PILOT_TOL:
            bad.append(f"seed {s}: AUC differs by {ah - ao:+.4f} (tolerance {PILOT_TOL})")
    mr = read_json(run_dir(MODEL, PILOT_FOLD, SEEDS[0]) / "meta.json")
    relu = None
    if mr is None:
        bad.append("the ReLU pilot run is missing")
    else:
        finite, fell = _loss_ok(mr)
        relu = {"seed": SEEDS[0], "loss": [h["train_loss"] for h in mr["history"]], "finite": finite, "fell": fell,
                "seconds": mr["seconds"], "auc": mr["metric"]["metric"]}
        if not (finite and fell):
            bad.append(f"ReLU run: loss finite {finite}, fell {fell}")
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state(), "note": "DESIGN.md amendment 7 pilot",
           "fold": PILOT_FOLD + 1, "tolerance_auc": PILOT_TOL, "tanh_reproduction": rows, "relu_run": relu,
           "failures": bad, "passed": not bad}
    write_json(SCORES / "relu_pilot.json", out)
    return out


# --------------------------------------------------------------------------------------------- scoring
def final_probs(k: int, wid: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Mean probability (n, 1) of the finished seeds; None unless all SEEDS are done."""
    ps = []
    for s in SEEDS:
        d = run_dir(MODEL, k, s)
        if not (d / "DONE").exists():
            continue
        with np.load(d / "pred.npz") as z:
            if not np.array_equal(z["wid"], wid):
                raise RuntimeError(f"{MODEL}/fold{k + 1}/seed{s}: window ids differ from the simple models'")
            ps.append(_sig(z["logit"].astype(float)))
    return (np.mean(ps, axis=0) if len(ps) == len(SEEDS) else None), len(ps)


def reading(d_all: list) -> str:
    """The amendment's reading of the primary pair (all regions)."""
    lo, hi = d_all[1], d_all[2]
    if lo is None:
        return "not scored"
    if lo > 0:
        return "ReLU helps in development"
    if hi < 0:
        return "ReLU is worse in development"
    return "no detectable difference"


def reproduce(k: int) -> float:
    """Largest |logit| difference between fold k's saved seed-42 model, run again, and its scored predictions."""
    import nets
    import torch
    import units
    d = units.prep("job1", k, "outer")
    rd = run_dir(MODEL, k, SEEDS[0])
    meta = read_json(rd / "meta.json")
    cfg = meta["cfg"]
    model = nets.SeqNet(cfg["family"], d["X_ev"].shape[2], 1, cfg["hidden"], cfg.get("layers", 1),
                        cfg.get("dropout", 0.2), cfg.get("bidir"), nets.OBS_CH)
    model.load_state_dict(torch.load(rd / "model.pt", map_location="cpu"))
    z = nets.predict_logits(model, d["X_ev"], None, "cpu")
    with np.load(rd / "pred.npz") as p:
        return float(np.abs(z - p["logit"]).max())


def score(write: bool = True) -> dict:
    """scores/relu.json: `bilstm_relu` on jobs 1 and 4 beside `bilstm`, LR on VCI(T) and the gate's best simple
    model, pooled and resampled exactly as the score stage (its subsets and region draws). write=False: the check
    code/checks/verify_relu_score.py, which points the ReLU runs at the original ones."""
    import dataset
    import score_stage
    import scoring
    import units
    from windows import Base
    meta_ds, arrays = dataset.load("dev")
    base = Base(meta_ds, arrays, "dev")
    R, fold_r = base.n_regions, base.region_group
    used = {}
    orig = score_stage.load_preds

    def load_plus(job, k, nh):
        wid, preds, u = orig(job, k, nh)
        if wid is not None and job == "job1":
            p, n = final_probs(k, wid)
            u[MODEL] = n
            if p is not None:
                preds[MODEL] = p
        used[f"{job}/fold{k + 1}"] = u
        return wid, preds, u

    score_stage.load_preds = load_plus
    try:
        col = score_stage.collect(base, arrays)
    finally:
        score_stage.load_preds = orig
    subs = col["subsets"]
    gate = score_stage.GATE
    wording = gate["wording"]
    official = read_json(SCORES / "scores.json") or {}
    sel = official.get("selection") or {}
    jobs = {}
    for g in gate["jobs"]:
        job = g["job"]
        if job not in ("job1", "job4"):
            continue
        name = g["subset"]
        s = subs[name]
        if MODEL not in s["preds"]:
            raise RuntimeError(f"{name}: {MODEL} is not complete ({used})")
        best_simple = (sel.get(job) or {}).get("best_simple")
        models = [m for m in dict.fromkeys([MODEL, "bilstm", "lr_vci_t", best_simple]) if m and m in s["preds"]]
        pairs = [(a, b) for a, b in ((MODEL, "bilstm"), (MODEL, "lr_vci_t"), (MODEL, best_simple),
                                     ("bilstm", "lr_vci_t")) if a in models and b in models and a != b]
        res = scoring.score_subset(name, s["y"], s["region"], s["west"], {m: s["preds"][m] for m in models}, fold_r,
                                   R, N_FOLDS, pairs)
        paired = {kk: {sd: v[sd]["dauc_within"] for sd in scoring.SIDES} for kk, v in res["paired"].items()}
        verdicts = {kk: scoring.verdict(v["all"], v["west"], v["east"], wording) for kk, v in paired.items()}
        verdicts[f"{MODEL} - bilstm"] = reading(paired[f"{MODEL} - bilstm"]["all"])
        off = ((official.get("subsets") or {}).get(name) or {}).get("models") or {}
        check = {m: {"here": res["models"][m]["all"]["auc_within"][0],
                     "score_stage": ((off.get(m) or {}).get("all") or {}).get("auc_within", [None])[0]}
                 for m in models if m != MODEL}
        jobs[job] = {"label": g["label"], "subset": name, "n": res["n"], "n_pos": res["n_pos"],
                     "best_simple": best_simple,
                     "models": {m: {sd: v[sd]["auc_within"] for sd in scoring.SIDES} | {
                         "per_fold_auc": v["per_fold_auc"], "brier": v["all"]["brier"]}
                         for m, v in res["models"].items()},
                     "paired": paired, "verdicts": verdicts, "matches_score_stage": check}
    runs = []
    for k in range(N_FOLDS):
        for s in SEEDS:
            m = read_json(run_dir(MODEL, k, s) / "meta.json")
            o = read_json(units.final_dir("job1", "bilstm", k, s) / "meta.json") or {}
            finite, fell = _loss_ok(m)
            runs.append({"fold": k + 1, "seed": s, "device": m["device"], "seconds": m["seconds"],
                         "seconds_original": o.get("seconds"), "device_original": o.get("device"),
                         "auc": m["metric"]["metric"], "auc_original": (o.get("metric") or {}).get("metric"),
                         "loss_finite": finite, "loss_fell": fell, "loss_last": m["history"][-1]["train_loss"],
                         "loss_last_original": (o.get("history") or [{}])[-1].get("train_loss")})
    sizes = []
    for k in range(N_FOLDS):
        m = read_json(run_dir(MODEL, k, SEEDS[0]) / "meta.json")
        sizes.append({"fold": k + 1, "cfg": m["cfg"], "n_params": m["n_params"],
                      "multiply_adds": m.get("multiply_adds")})     # the original bilstm runs (the check) lack it
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state(),
           "note": "DESIGN.md amendment 7: secondary, no pass/fail, development folds only; the fresh test is not scored",
           "labels": LABELS, "seeds": list(SEEDS), "seeds_used": used, "jobs": jobs, "runs": runs,
           "all_losses_fell": all(r["loss_fell"] for r in runs), "all_losses_finite": all(r["loss_finite"] for r in runs),
           "reproduce_max_logit_difference": {f"fold{k + 1}": reproduce(k) for k in range(N_FOLDS)},
           "sizes": sizes, "pilot": read_json(SCORES / "relu_pilot.json")}
    if write:
        write_json(SCORES / "relu.json", out)
    return out
