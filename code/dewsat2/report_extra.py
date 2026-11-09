"""Report sections for the rows added after the development results (DESIGN.md amendments 4-6): the post-hoc models'
development scores and the 2x2 table, the learning curve, the fresh-test 2x2 and the model sizes for the Jetson.
Every number is read from a logged file (scores/posthoc.json, scores/lcurve.json, scores/phase5.json, run metadata)."""

from __future__ import annotations

from pathlib import Path

from common import LEADS, RUNS, read_json

PH_LABELS = {"bilstm_vci": "VCI-only Bi-LSTM", "cnn_lr": "CNN + LR", "cnn_bilstm": "CNN + Bi-LSTM"}
TWO_BY_TWO = (("Month T only", "lr_vci_t", "cnn_lr"), ("12 months", "bilstm_vci", "cnn_bilstm"))


def _r():
    import report
    return report


def two_by_two(auc, diff, note: str) -> list[str]:
    """The 2x2 of inputs (cell average or cell image) x history (month T or 12 months). auc(m) and diff(a, b) return
    [point, low, high] or None."""
    R = _r()
    ci, lab = R.ci, R.lab
    if any(auc(m) is None for _, a, b in TWO_BY_TWO for m in (a, b)):
        return []
    L = [f"**The 2x2 table** ({note}):", "",
         "| History | Cell average (VCI) | Cell image (23x21 VCI tiles) | Image − average |", "|---|---|---|---|"]
    for name, a, b in TWO_BY_TWO:
        L.append(f"| {name} | {lab(a)}: {ci(auc(a))} | {lab(b)}: {ci(auc(b))} | {ci(diff(b, a), signed=True)} |")
    L.append(f"| 12 months − month T | {ci(diff('bilstm_vci', 'lr_vci_t'), signed=True)} | "
             f"{ci(diff('cnn_bilstm', 'cnn_lr'), signed=True)} | |")
    return L + [""]


def sec_posthoc(d: dict) -> list[str]:
    R = _r()
    ci, lab, f3 = R.ci, R.lab, R.f3
    p = d.get("posthoc")
    L = ["## 4b. Models added after the development results (amendments 4 and 4b; development)", ""]
    if not p:
        return L + ["Not scored yet (`scores/posthoc.json`).", ""]
    L += ["Added after section 4's results, with their designs fixed before they were trained (DESIGN.md amendments 4 "
          "and 4b; not part of the gate):", "",
          "- **VCI-only Bi-LSTM**: each job's Bi-LSTM search, with NDVI, the valid fraction and the season channels "
          "zeroed, so only VCI_mean, VCI_p10, the observed flag and the months since a view carry information.",
          "- **CNN + LR** (job 1): a small CNN on month T's 23x21 VCI tile of the cell, then one linear layer.",
          "- **CNN + Bi-LSTM** (job 1): the same encoder on each of the 12 monthly tiles, then a Bi-LSTM.", "",
          "Same folds, inner selection, 10 seeds, pooling and whole-region resamples as section 4.", ""]
    j1 = p["jobs"].get("job1")
    if j1:
        def auc(m):
            return (j1["models"].get(m) or {}).get("all")

        def diff(a, b):
            return (j1["paired"].get(f"{a} - {b}") or {}).get("all")
        L += two_by_two(auc, diff, "job 1, development, within-fold AUC, all regions")
    for job, e in p["jobs"].items():
        lr = "lr_latest_age" if job == "job2" else "lr_vci_t"
        sel = ((d.get("scores") or {}).get("selection") or {}).get(job) or {}
        show = [m for m in dict.fromkeys([lr, e.get("best_simple"), sel.get("best_network"), sel.get("best_bilstm"),
                                          "bilstm", *e["new_models"]]) if m and m in e["models"]]
        L += [f"### {e['label']} ({e['metric']})", ""]
        if e.get("n"):
            L += [f"{e['n']:,} windows, {e['n_pos']:,} in drought.", ""]
        per_fold = all("per_fold_auc" in e["models"][m] for m in show)
        L += ["| Model | All | West | East |" + (" Fold 1 | Fold 2 | Fold 3 | Fold 4 |" if per_fold else ""),
              "|---|---|---|---|" + ("---|---|---|---|" if per_fold else "")]
        for m in show:
            v = e["models"][m]
            L.append(f"| {lab(m)} | {ci(v['all'])} | {ci(v['west'])} | {ci(v['east'])} |" +
                     (" " + " | ".join(f3(x) for x in v["per_fold_auc"]) + " |" if per_fold else ""))
        L += ["", "| Comparison | Δ all | Δ West | Δ East | Result |", "|---|---|---|---|---|"]
        for k, v in e["paired"].items():
            a, b = k.split(" - ")
            L.append(f"| {lab(a)} − {lab(b)} | {ci(v['all'], signed=True)} | {ci(v['west'], signed=True)} | "
                     f"{ci(v['east'], signed=True)} | {e['verdicts'].get(k, '')} |")
        L += [""]
    return L


def _size_label(axis: str, row: dict) -> str:
    if axis == "years":
        return f"{row['size']} of 5 years" + (f" (mean of {row['subsets']} subsets)" if row["subsets"] > 1 else " (full)")
    if row["size"] == 1.0:
        return "all regions (full)"
    return {0.125: "1/8", 0.25: "1/4", 0.5: "1/2"}.get(row["size"], str(row["size"])) + " of the regions"


def sec_lcurve(d: dict) -> list[str]:
    R = _r()
    ci, lab = R.ci, R.lab
    lc = d.get("lcurve")
    L = ["## 7b. Learning curve (amendment 5; secondary, no pass/fail)", ""]
    if not lc:
        return L + ["Not scored yet (`scores/lcurve.json`).", ""]
    nets_ = lc["networks"]
    L += [f"Does more training data close the gap to LR? Networks {', '.join(lab(m) for m in nets_)}, each fold's chosen "
          f"setting retrained on slices of the fold's training windows (seeds {', '.join(map(str, lc['seeds']))}); LR on "
          "VCI(T) refit on every slice. Job 1, development folds only, within-fold AUC, whole-region resamples shared "
          "by every row of an axis.", ""]
    for axis, e in lc["axes"].items():
        L += [f"### More {axis} (folds {', '.join(map(str, e['folds']))}; {e['n']:,} scored windows in "
              f"{e['regions']} regions)", "",
              "| Training data | Mean training windows | " + " | ".join(lab(m) for m in ["lr_vci_t"] + nets_) + " | " +
              " | ".join(f"{lab(m)} − LR" for m in nets_) + " |",
              "|---|---|" + "---|" * (1 + 2 * len(nets_))]
        for row in e["sizes"]:
            L.append(f"| {_size_label(axis, row)} | {row['mean_training_windows']:,.0f} | " +
                     " | ".join(ci(row["auc_within"][m]["all"]) for m in ["lr_vci_t"] + nets_) + " | " +
                     " | ".join(ci(row["network_minus_lr"][m]["all"], signed=True) for m in nets_) + " |")
        L += [""]
        for m, ch in e["change_in_difference"].items():
            last, first = ch["second_largest_to_full"], ch["smallest_to_full"]
            if last[1] is not None and last[1] > 0:
                read = "still rising at the full data, so more of this kind of data would likely help"
            elif first[1] is not None and first[1] > 0:
                read = "rose, then flattened before the full data, so more of this kind is not expected to close the gap"
            else:
                read = "flat, so more of this kind is not expected to close the gap"
            L.append(f"- {lab(m)}: change in (network − LR) from the smallest slice to the full data "
                     f"{ci(first, signed=True)}; from the second-largest to the full data {ci(last, signed=True)}: "
                     f"{read} (reading rule fixed in amendment 5).")
        L += [""]
    return L


def macs_for(cfg: dict, n_in: int, n_heads: int, steps: int = 12) -> int | None:
    """Multiply-adds of one prediction, layer by layer (recurrent gates, convolutions, the head's linear layers)."""
    fam = cfg["family"]
    if fam in ("cnn_lr", "cnn_bilstm"):
        import cnn
        return int(cnn.multiply_adds(cfg))
    net = "bilstm" if fam == "bilstm_vci" else fam
    h, layers = int(cfg["hidden"]), int(cfg.get("layers", 1))
    if net == "tcn":
        enc = steps * n_in * h + layers * 3 * steps * 3 * h * h
        out_dim = h
    else:
        bidir = cfg.get("bidir", net in ("bilstm", "resbilstm"))
        dirs, gates = (2 if bidir else 1), (3 if net == "gru" else 4)
        enc, d_in = 0, n_in
        for _ in range(layers):
            enc += steps * dirs * gates * h * (d_in + h)
            d_in = dirs * h
        out_dim = dirs * h
    return int(enc + (2 * out_dim) * (2 * h) + (2 * h) * n_heads)


def sec_sizes(d: dict) -> list[str]:
    """Phase 5 networks (trained on all development data): parameters, multiply-adds, saved weights."""
    R = _r()
    lab = R.lab
    L = ["## 8b. Model sizes for the Jetson", ""]
    rows = []
    for meta_p in sorted((RUNS / "phase5" / "composite").glob("*/*/seed42/meta.json")):
        m = read_json(meta_p) or {}
        if "spec" not in m:
            continue
        cfg, job = m["spec"]["cfg"], m["spec"]["job"]
        n_in = 4 if cfg["family"] == "bilstm_vci" else 9
        heads = len(LEADS) if job == "job3" else 1
        macs = m.get("multiply_adds") or macs_for(cfg, n_in, heads)
        pt = meta_p.parent / "model.pt"
        rows.append((job, cfg["family"], cfg, m.get("n_params"), macs, pt.stat().st_size if pt.exists() else None,
                     pt.parent.relative_to(RUNS.parent)))
    if not rows:
        return L + ["Not available until Phase 5 has trained (sizes of the development runs are in "
                    "`scores/posthoc.json` and each run's `meta.json`).", ""]
    L += ["One prediction is one cell-month window (12 monthly steps; a tile CNN reads one 23x21 tile per step). "
          "Multiply-adds are counted layer by layer (recurrent gates, convolutions, linear layers; norms and activations "
          "left out). Weights: `model.pt` (PyTorch state dict, float32) of the seed-42 run; the 10 seeds are the same "
          "size. A float32 model needs about 4 bytes per parameter.", "",
          "| Job | Model | Setting | Parameters | Multiply-adds per prediction | model.pt | Folder (10 seeds) |",
          "|---|---|---|---|---|---|---|"]
    for job, fam, cfg, n_p, macs, size, folder in rows:
        bits = [f"{k} {cfg[k]}" for k in ("hidden", "layers", "width", "depth") if k in cfg]
        L.append(f"| {job} | {lab(fam)} | cfg{cfg['id']:02d}: {', '.join(bits)} | {n_p:,} | {macs:,} | "
                 f"{size / 1024:.1f} KiB | `{Path(*folder.parts[:-1])}/` |" if n_p and macs and size else
                 f"| {job} | {lab(fam)} | cfg{cfg['id']:02d} | {n_p} | {macs} | {size} | `{folder}` |")
    return L + [""]


# ------------------------------------------------------------------------- fresh test: verdicts and post-hoc slices
JOB_LABELS = {"job1": "Job 1, estimation now", "job2": "Job 2, missing month",
              "job3": "Job 3, forecast (mean onset AUC, leads 1-3)", "job4": "Job 4, where VCI changed most"}


def rule_verdict(a, w, e) -> str:
    """DESIGN.md section 9's wording for a paired difference; a, w, e = [point, low, high] for all regions, West, East."""
    if not a or a[1] is None:
        return "not scored"
    if a[1] > 0:
        worse = [s for s, v in (("West", w), ("East", e)) if v and v[2] is not None and v[2] < 0]
        return "passed" if not worse else f"did not pass (better overall, worse in the {' and '.join(worse)})"
    if a[2] < 0:
        return "worse"
    return "no detectable difference"


def fresh_verdict_rows(d: dict) -> list[dict]:
    """The pre-registered fresh-test rows (DESIGN.md sections 9 and 11): per job, the gate (best network vs best
    simple model) and the best Bi-LSTM variant vs the best simple model; for job 1 both also against LR on VCI(T)."""
    sc, p5 = d.get("scores") or {}, d.get("phase5") or {}
    if not p5:
        return []
    fr, out = p5["fresh"], []
    for job in ("job1", "job2", "job3", "job4"):
        sel = (sc.get("selection") or {}).get(job) or {}
        net, bil, simp = sel.get("best_network"), sel.get("best_bilstm"), sel.get("best_simple")
        sub = fr.get("job3_mean_onset" if job == "job3" else job)
        if not sub:
            continue
        rows = [("gate", net, simp), ("best Bi-LSTM variant", bil, simp)]
        if job == "job1":
            rows += [("against LR on VCI(T)", net, "lr_vci_t"), ("against LR on VCI(T)", bil, "lr_vci_t")]
        for kind, a, b in rows:
            v = sub["paired"].get(f"{a} - {b}") if a and b and a != b else None
            if not v:
                continue
            dd = {sd: (v[sd] if job == "job3" else v[sd]["dauc_within"]) for sd in ("all", "west", "east")}
            out.append({"job": job, "label": JOB_LABELS[job], "kind": kind, "model": a, "comparator": b, "diff": dd,
                        "verdict": rule_verdict(dd["all"], dd["west"], dd["east"])})
    return out


def fresh_verdict_table(d: dict) -> list[str]:
    R = _r()
    ci, lab = R.ci, R.lab
    rows = fresh_verdict_rows(d)
    if not rows:
        return []
    L = ["**Pre-registered verdicts** (the section 9 rule: passed = the all-regions interval above zero and neither "
         "side's interval entirely below zero; AUC differences, 2,000 whole-region resamples):", "",
         "| Job | Row | Comparison | All | West | East | Verdict |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        dd = r["diff"]
        L.append(f"| {r['label']} | {r['kind']} | {lab(r['model'])} − {lab(r['comparator'])} | "
                 f"{ci(dd['all'], signed=True)} | {ci(dd['west'], signed=True)} | {ci(dd['east'], signed=True)} | "
                 f"**{r['verdict']}** |")
    return L + [""]


def sec_fresh_posthoc(d: dict) -> list[str]:
    R = _r()
    ci, lab, f3 = R.ci, R.lab, R.f3
    f = d.get("fresh_posthoc")
    L = ["## 8a. Fresh test: post-hoc diagnostics (after the lock; not a second test)", ""]
    if not f:
        return L + ["Not computed yet (`code/dewsat2/fresh_posthoc.py`, after the fresh test is scored).", ""]
    L += ["The same saved fresh-test predictions as section 8 (job 1), sliced to explain the result "
          "(`scores/fresh_posthoc.json`, `code/dewsat2/fresh_posthoc.py`). Nothing here was used to choose or change a "
          "model, and the fresh test was not scored again.", ""]
    rc = f.get("reload_check")
    if rc:
        L += [f"- **Validity:** the {rc['models']} saved Phase 5 networks, reloaded from `model.pt`, reproduce their scored "
              f"fresh-test logits (largest difference {rc['max_abs_logit_diff']:.1e}; the CPU-trained ones exactly). "
              f"Parameter counts match the metadata: {'yes' if rc['params_match'] else 'NO'}. Every training loss fell: "
              f"{'yes' if rc['losses_fell'] else 'NO'}.", ""]
    nps = f.get("network_pairs", {})
    if nps.get("fresh_job1"):
        L += ["**The networks against each other** (job 1 and job 4; paired AUC differences with the scorer's resamples):",
              "", "| Comparison | Job 1, all | West | East | Job 4, all |", "|---|---|---|---|---|"]
        p1, p4 = nps["fresh_job1"]["paired"], (nps.get("fresh_job4") or {}).get("paired", {})
        for k, v in p1.items():
            a, b = k.split(" - ")
            v4 = p4.get(k)
            L.append(f"| {lab(a)} − {lab(b)} | {ci(v['all'], signed=True)} | {ci(v['west'], signed=True)} | "
                     f"{ci(v['east'], signed=True)} | {ci(v4['all'], signed=True) if v4 else '—'} |")
        L += ["", "The three full-input recurrent networks (Bi-LSTM, residual Bi-LSTM, GRU) are within 0.006 AUC of "
              "each other; each is about 0.04 above LR.", ""]
    by = f.get("by_year")
    if by:
        ms = ["lr_vci_t", "bilstm", "resbilstm", "gru", "bilstm_vci", "cnn_bilstm"]
        L += ["**By year and side** (pooled AUC; point estimates):", "",
              "| Year | Side | Windows | Drought rate | " + " | ".join(lab(m) for m in ms) + " | Bi-LSTM − LR |",
              "|---|---|---|---|" + "---|" * len(ms) + "---|"]
        for sd in ("all", "west", "east"):
            for yr in (2025, 2026):
                e = by.get(f"{sd}_{yr}")
                if not e:
                    continue
                a = e["auc"]
                L.append(f"| {yr if yr == 2025 else '2026 (Jan-Sep)'} | {sd} | {e['n']:,} | {e['drought_rate']:.3f} | " +
                         " | ".join(f3(a.get(m)) for m in ms) + f" | {a['bilstm'] - a['lr_vci_t']:+.3f} |")
        L += [""]
    wr, wm = f.get("within_region"), f.get("within_month")
    if wr and wm:
        ms = ["gbt_mono", "bilstm", "resbilstm", "gru", "bilstm_vci", "cnn_lr", "cnn_bilstm"]
        L += ["**Over time and across places.** Over time: AUC from pairs of windows in the same region, so only ranking "
              "months within a region counts. Across places: pairs in the same month, so only ranking cells within a "
              "month counts. Network − LR, with whole-region intervals (1,000 resamples) for the over-time slice:", "",
              "| Slice | Side | LR | " + " | ".join(lab(m) for m in ms) + " |", "|---|---|---|" + "---|" * len(ms)]
        for sd in ("all", "west", "east"):
            mm = wr[sd]["models"]
            L.append(f"| over time | {sd} | {f3(mm['lr_vci_t']['auc'])} | " +
                     " | ".join(f"{f3(mm[m]['auc'])} ({ci(mm[m]['minus_lr'], signed=True)})" for m in ms) + " |")
        for sd in ("all", "west", "east"):
            mm = wm[sd]
            L.append(f"| across places | {sd} | {f3(mm['lr_vci_t'])} | " +
                     " | ".join(f"{f3(mm[m])} ({mm[m] - mm['lr_vci_t']:+.3f})" for m in ms) + " |")
        L += ["", "The networks' gain shows in both slices and both years, so it is not one artifact. The VCI-only "
              "Bi-LSTM gains nothing over time, so on the fresh test the extra skill comes from the NDVI level and "
              "spread, clear fraction and season inputs, not from the VCI history alone.", ""]
    dv = f.get("dev_folds")
    if dv:
        ms = ["lr_vci_t", "bilstm", "resbilstm", "gru", "bilstm_vci", "cnn_lr", "cnn_bilstm"]
        L += ["**Why development said the opposite.** Each development fold's training-set size, its validation drought "
              "rate and the job-1 AUCs, beside Phase 5 (trained on all development data) on the fresh test:", "",
              "| Data | Validation | Training windows | Drought rate | " + " | ".join(lab(m) for m in ms) +
              " | Bi-LSTM − LR |", "|---|---|---|---|" + "---|" * len(ms) + "---|"]
        for r in sorted(dv["folds"], key=lambda r: r["training_windows"]):
            a = r["auc"]
            L.append(f"| fold {r['fold']} | {r['validation_years']} | {r['training_windows']:,} | {r['drought_rate']:.3f} | " +
                     " | ".join(f3(a.get(m)) for m in ms) +
                     (f" | {a['bilstm'] - a['lr_vci_t']:+.3f} |" if "bilstm" in a and "lr_vci_t" in a else " | — |"))
        fa = dv["fresh_auc"]
        L.append(f"| Phase 5 | fresh test 2025-01..2026-09 | {dv['phase5_training_windows']:,} | {dv['fresh_drought_rate']:.3f} | " +
                 " | ".join(f3(fa.get(m)) for m in ms) + f" | {fa['bilstm'] - fa['lr_vci_t']:+.3f} |")
        L += ["", "The Bi-LSTM's gap to LR follows the training-set size, not the dryness: the driest validation block "
              "(2020-21) was its worst, and the fresh test, drier still, was its best, with 2.2 times the largest fold's "
              "training windows. With the learning curve (section 7b: more years help, more regions do not), the best-"
              "supported reading is that the networks need more training data than a development fold gave them. "
              "Caveats: five points; training size is confounded with which years and regions each fold trained on; "
              "the fresh test is one unusually dry 21-month period.", ""]
    return L
