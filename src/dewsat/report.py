"""Plots and a written report for each run.

A run that only leaves metrics.json behind is hard to defend later, so every
run also gets a REPORT.md stating what was measured, what it was measured
against, and which claims it does not support.
"""
import os
from pathlib import Path

import numpy as np


def _format(value, digits=5):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def plot_results(run_dir, histories, target, prediction, result, calendar_months, data_kind):
    run_dir = Path(run_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(run_dir.resolve() / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    accent, muted = "#177b88", "#b4553a"
    member_colors = ["#177b88", "#b4553a", "#7a6a9b", "#3f7f3f", "#8a6d1f"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    for position, member in enumerate(histories):
        axes[0, 0].plot([row["epoch"] for row in member], [row["val_rmse"] for row in member],
                        color=member_colors[position % len(member_colors)], alpha=0.85,
                        marker="o", markersize=2.5, linewidth=1.2,
                        label=f"seed {member[0]['seed']}")
    axes[0, 0].set(xlabel="Epoch", ylabel="Validation RMSE", title="Checkpoint selection")
    if len(histories) > 1:
        axes[0, 0].legend(fontsize=8)

    axes[0, 1].scatter(target, prediction, alpha=0.35, s=10, color=accent)
    axes[0, 1].plot([0, 1], [0, 1], "--", color="#333333", linewidth=1)
    axes[0, 1].set(xlabel="Target", ylabel="Prediction", xlim=(0, 1), ylim=(0, 1),
                   title=f"Held-out test (RMSE {result['test']['rmse']:.4f})")

    monthly = result["test"].get("by_calendar_month", {})
    if monthly:
        labels = sorted(monthly)
        axes[1, 0].bar(labels, [monthly[k]["rmse"] for k in labels], color=accent)
        axes[1, 0].set(xlabel="Calendar month of target", ylabel="RMSE",
                       title="Where the error concentrates")

    names = ["model"] + sorted(result["baselines"])
    values = [result["test"]["rmse"]] + [result["baselines"][n]["rmse"] for n in sorted(result["baselines"])]
    colors = [accent] + [muted] * (len(names) - 1)
    # barh draws bottom-up, so reverse everything together or the model gets a
    # baseline's colour.
    axes[1, 1].barh(names[::-1], values[::-1], color=colors[::-1])
    axes[1, 1].set(xlabel="Test RMSE", title="Model against reference predictors")
    for position, value in enumerate(values[::-1]):
        axes[1, 1].text(value, position, f" {value:.4f}", va="center", fontsize=8)

    suffix = " | software demonstration only" if data_kind == "synthetic" else ""
    figure.suptitle(f"DEWSat v2 | {data_kind.upper()} data | lead {result['lead_months']} month(s){suffix}")
    figure.savefig(run_dir / "results.png", dpi=150)
    plt.close(figure)


def write_report(run_dir, config, audit, result, histories, bundle, splits, prediction):
    run_dir = Path(run_dir)
    windows = bundle["windows"]
    target = windows["y"][splits["test"]]
    calendar = [windows["metadata"][i]["target_calendar_month"] for i in splits["test"]]
    plot_results(run_dir, histories, target, prediction, result, calendar,
                 bundle["manifest"]["data_kind"])

    lines = [f"# Run report: {run_dir.name}", "",
             f"- Data: **{result['data_kind']}** (`{result['target_name']}`)",
             f"- Task: lead **{result['lead_months']}** month(s), "
             f"{config['sequence_length']}-month input window",
             f"- Split mode: **{audit['splits']['mode']}**, counts {audit['splits']['counts']}",
             f"- Model: {audit['parameter_count']:,} parameters "
             f"({config['num_layers']}x{config['hidden_size']} bi-LSTM, pool `{config['pool']}`)",
             f"- Trained in {audit['training_seconds']:.1f}s on `{audit['device']}` "
             f"({config['n_ensemble']} member(s))", ""]

    lines += ["## What this run does and does not show", "", "| Question | Answer from this run |",
              "|---|---|",
              f"| Does the pipeline run end to end? | Yes: {audit['windows']['total']} windows, "
              f"reload difference {result['reload_max_absolute_difference']:.2e} |",
              f"| Better than carrying the last value forward? | RMSE skill "
              f"{_format(result['skill_vs']['persistence'], 3)} vs persistence |",
              f"| Better than seasonal climatology? | RMSE skill "
              f"{_format(result['skill_vs']['seasonal_climatology'], 3)} |",
              f"| Better than a linear model on the same window? | RMSE skill "
              f"{_format(result['skill_vs']['ridge'], 3)} vs ridge |",
              f"| Drought accuracy on real observations? | "
              f"{'No: synthetic data' if result['data_kind'] == 'synthetic' else 'Only for this target, unit and period'} |",
              "| Jetson latency, energy or INT8 accuracy? | No: run `dewsat bench` on the device |", "",
              f"_{result['interpretation']}_", ""]

    header = ("| Predictor | RMSE | MAE | R2 | Spearman | Dry RMSE | POD | FAR | CSI | HSS |\n"
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines += ["## Held-out test metrics", "", header]

    def row(name, metrics):
        event = metrics["event"]
        return (f"| {name} | {_format(metrics['rmse'])} | {_format(metrics['mae'])} | "
                f"{_format(metrics['r2'], 4)} | {_format(metrics['spearman_r'], 4)} | "
                f"{_format(metrics['dry_rmse'])} | {_format(event['pod'], 3)} | "
                f"{_format(event['far'], 3)} | {_format(event['csi'], 3)} | {_format(event['hss'], 3)} |")

    lines.append(row("**Bi-LSTM**", result["test"]))
    for name in sorted(result["baselines"]):
        lines.append(row(name, result["baselines"][name]))
    lines += ["", f"Dry event threshold: target < {config['dry_threshold']}, "
                  f"base rate {_format(result['test']['event']['base_rate'], 3)}. "
                  f"POD is hit rate, FAR is false alarm ratio, CSI is critical success index, "
                  f"HSS is Heidke skill score.", ""]

    if "ensemble" in result:
        lines += ["## Ensemble", "",
                  f"- Members: {result['ensemble']['members']}",
                  f"- Mean member RMSE: {_format(result['ensemble']['mean_member_rmse'])} "
                  f"(ensemble {_format(result['test']['rmse'])})",
                  f"- Mean prediction spread across members: "
                  f"{_format(result['ensemble']['mean_member_spread'])}", ""]

    lines += ["## Data admission", "",
              f"- Rows: {audit['table']['rows']} over {audit['table']['cells']} cells, "
              f"{audit['table']['first_month']}..{audit['table']['last_month']}",
              f"- Observed fraction of the within-cell month span: "
              f"{audit['table']['observed_fraction_of_span']}",
              f"- Interior missing months: {audit['table']['interior_missing_months']} "
              f"(imputed with the observed flag set to 0)",
              f"- Rows dropped by min_valid_frac: {audit['table']['dropped_low_quality_rows']}",
              f"- Windows rejected: {audit['windows']['rejected'] or 'none'}",
              f"- Mean observed months per window: {audit['windows']['mean_observed_months']} "
              f"of {config['sequence_length']}",
              f"- Scaler ({audit['scaler']['kind']}) fitted on {audit['scaler']['fit_monthly_rows']} "
              f"monthly rows ending {audit['scaler']['fit_last_month']}", ""]
    if audit["splits"].get("blocks"):
        blocks = audit["splits"]["blocks"]
        lines += [f"- Held-out blocks: val {blocks['val']}, test {blocks['test']}",
                  f"- Train/test block overlap: {audit['splits']['block_overlap'] or 'none'}", ""]
    if audit["warnings"]:
        lines += ["### Warnings", "", *[f"- {w}" for w in audit["warnings"]], ""]

    lines += ["## Selection", "", "| Member | Seed | Epochs run | Best epoch | Validation RMSE |",
              "|---:|---:|---:|---:|---:|"]
    for entry in result["selection"]:
        marker = " (kept)" if entry["member"] == result["best_member"] else ""
        lines.append(f"| {entry['member']}{marker} | {entry['seed']} | {entry['epochs_run']} | "
                     f"{entry['best_epoch']} | {_format(entry['val_rmse'])} |")

    environment = audit["environment"]
    lines += ["", "## Provenance", "",
              f"- Command: `{environment['command']}`",
              f"- Input: `{environment['csv']}`", f"- SHA256: `{environment['csv_sha256']}`",
              f"- dewsat {environment['dewsat_version']}, Python {environment['python']}, "
              f"torch {environment['torch']}, numpy {environment['numpy']}",
              f"- Platform: {environment['platform']} ({environment['machine']})",
              f"- Checkpoint: `best.pt`, {result['checkpoint_bytes']:,} bytes", ""]

    (run_dir / "REPORT.md").write_text("\n".join(lines))
    return run_dir / "REPORT.md"
