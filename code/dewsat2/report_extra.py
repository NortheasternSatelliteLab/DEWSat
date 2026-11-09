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
