"""Configuration, the training loop, distillation, and run artifacts.

Differences from v1 that matter in practice:

* the epoch loop gathers batches from row indices instead of holding a
  materialised window cube, so memory tracks the table size;
* `torch.use_deterministic_algorithms` is no longer forced on, because the
  cuDNN LSTM backward has no deterministic implementation and a CUDA run would
  abort. Determinism is a config flag that warns instead of failing;
* checkpoint selection can average several seeds, which is the cheapest honest
  way to report uncertainty;
* every run writes a report, the baseline comparison, and the environment it
  ran in, so a result can be defended weeks later.
"""
import copy
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import baselines as baseline_module
from . import metrics as metric_module
from .contract import (CSV_FEATURES, MODEL_FEATURES, contract_fingerprint, month_number,
                       month_string, read_manifest, sha256_file, write_json)
from .data import build_inputs, fit_scaler, make_windows, read_monthly_csv, split_windows
from .model import DroughtBiLSTM, count_parameters, distillation_loss, predict_batches, weighted_huber

VERSION = "2.0.0"
CHECKPOINT_FORMAT = "dewsat-v2-checkpoint"

DEFAULT_CONFIG = {
    # reproducibility and hardware
    "seed": 42, "device": "cpu", "cpu_threads": 2, "deterministic": True,
    # what is being predicted
    "sequence_length": 12, "lead_months": 0, "dry_threshold": 0.2,
    # data admission
    "min_valid_frac": 0.0, "min_observed_frac": 0.6, "require_anchor": True,
    "scaler_kind": "zscore",
    # architecture
    "hidden_size": 48, "num_layers": 1, "dropout": 0.0, "mlp_hidden": 64,
    "pool": "final_masked_mean",
    # optimisation
    "batch_size": 64, "epochs": 40, "patience": 8, "learning_rate": 1e-3,
    "weight_decay": 1e-4, "lr_schedule": "cosine", "min_lr_factor": 0.05,
    "huber_delta": 0.1, "dry_weight": 2.0, "gradient_clip": 1.0,
    # evaluation
    "n_ensemble": 1, "ridge_penalty": 1.0,
    # splits
    "split_mode": "temporal", "embargo_months": 0, "split_seed": 42,
    "val_block_frac": 0.2, "test_block_frac": 0.2,
    "train_start": "2016-01", "train_end": "2021-12",
    "val_start": "2022-01", "val_end": "2022-12",
    "test_start": "2023-01", "test_end": "2024-12",
    # distillation (used by the distill command)
    "distill_alpha": 0.5,
}

POSITIVE_INTEGERS = ("sequence_length", "hidden_size", "num_layers", "mlp_hidden",
                     "batch_size", "epochs", "patience", "cpu_threads", "n_ensemble")
POSITIVE_FLOATS = ("learning_rate", "huber_delta", "gradient_clip")
NONNEGATIVE = ("dry_weight", "weight_decay", "lead_months", "embargo_months", "ridge_penalty")


def load_config(path=None, overrides=None):
    from .contract import read_json
    config = dict(DEFAULT_CONFIG)
    if path:
        supplied = read_json(path)
        unknown = sorted(set(supplied) - set(DEFAULT_CONFIG))
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}. Check spelling against DEFAULT_CONFIG.")
        config.update(supplied)
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return validate_config(config)


def validate_config(config):
    for name in POSITIVE_INTEGERS:
        if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name in POSITIVE_FLOATS:
        if not isinstance(config[name], (int, float)) or config[name] <= 0:
            raise ValueError(f"{name} must be positive.")
    for name in NONNEGATIVE:
        if not isinstance(config[name], (int, float)) or config[name] < 0:
            raise ValueError(f"{name} must be nonnegative.")
    if not 0 <= config["dropout"] < 1:
        raise ValueError("dropout must be in [0,1).")
    if not 0 < config["min_observed_frac"] <= 1:
        raise ValueError("min_observed_frac must be in (0,1].")
    if not 0 <= config["min_valid_frac"] <= 1:
        raise ValueError("min_valid_frac must be in [0,1].")
    if not 0 < config["dry_threshold"] < 1:
        raise ValueError("dry_threshold must be in (0,1).")
    if config["lr_schedule"] not in {"none", "cosine"}:
        raise ValueError("lr_schedule must be 'none' or 'cosine'.")
    if not 0 < config["min_lr_factor"] <= 1:
        raise ValueError("min_lr_factor must be in (0,1].")
    if not 0 <= config["distill_alpha"] <= 1:
        raise ValueError("distill_alpha must be in [0,1].")
    for name in ("train", "val", "test"):
        month_number(config[f"{name}_start"]), month_number(config[f"{name}_end"])
    return config


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda, mps, or auto.")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("device 'cuda' requested but no CUDA device is visible.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("device 'mps' requested but Metal is not available.")
    return name


def seed_everything(seed, cpu_threads=2, deterministic=True):
    """Seed every generator this project uses and ask for reproducible kernels.

    `warn_only=True` is deliberate: cuDNN's LSTM backward has no deterministic
    implementation, so strict mode turns any GPU training run into a crash.
    Run-to-run CPU results are reproducible; GPU results are close, not equal.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(max(1, cpu_threads))
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def log(message):
    print(message, flush=True)


def prepare(csv_path, config, manifest_path=None, require_target=True):
    """Load everything the model needs and report how the data was admitted."""
    manifest = read_manifest(csv_path, manifest_path)
    table = read_monthly_csv(csv_path, require_target=require_target,
                             min_valid_frac=config["min_valid_frac"])
    windows = make_windows(table, sequence_length=config["sequence_length"],
                           lead_months=config["lead_months"],
                           min_observed_frac=config["min_observed_frac"],
                           require_anchor=config["require_anchor"],
                           require_target=require_target)
    return {"manifest": manifest, "table": table, "windows": windows}


def _learning_rate(config, epoch):
    if config["lr_schedule"] == "none":
        return config["learning_rate"]
    floor = config["learning_rate"] * config["min_lr_factor"]
    progress = (epoch - 1) / max(1, config["epochs"] - 1)
    return floor + 0.5 * (config["learning_rate"] - floor) * (1 + math.cos(math.pi * progress))


def train_member(bundle, splits, scaler, config, device, seed, teacher_predictions=None):
    """Train one model, early-stopping on validation RMSE, and return its best state."""
    table, windows = bundle["table"], bundle["windows"]
    seed_everything(seed, config["cpu_threads"], config["deterministic"])
    model_config = {name: config[name] for name in ("hidden_size", "num_layers", "dropout",
                                                    "mlp_hidden", "pool")}
    model_config["sequence_length"] = config["sequence_length"]
    model = DroughtBiLSTM(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    train_index = splits["train"]
    y = windows["y"]
    rng = np.random.default_rng(seed)
    history, best = [], {"rmse": math.inf, "epoch": 0, "state": None}
    stale = 0
    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        rate = _learning_rate(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        order = rng.permutation(train_index)
        total, seen, soft_total = 0.0, 0, 0.0
        for start in range(0, len(order), config["batch_size"]):
            batch = order[start:start + config["batch_size"]]
            x = torch.from_numpy(build_inputs(table, windows, batch, scaler)).to(device)
            target = torch.from_numpy(y[batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            if teacher_predictions is None:
                loss = weighted_huber(prediction, target, config["huber_delta"], config["dry_weight"])
                soft = torch.zeros((), device=device)
            else:
                teacher = torch.from_numpy(teacher_predictions[batch]).to(device)
                loss, _, soft = distillation_loss(prediction, teacher, target, config["distill_alpha"],
                                                  config["huber_delta"], config["dry_weight"])
            if not torch.isfinite(loss):
                raise RuntimeError("Loss became nonfinite; inspect scaling and learning rate.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.item()) * len(batch)
            soft_total += float(soft) * len(batch)
            seen += len(batch)
        validation = predict_batches(model, table, windows, splits["val"], scaler, device)
        val = metric_module.regression_metrics(y[splits["val"]], validation, config["dry_threshold"])
        history.append({"epoch": epoch, "learning_rate": rate, "train_loss": total / seen,
                        "train_teacher_term": soft_total / seen, "val_rmse": val["rmse"],
                        "val_mae": val["mae"], "val_dry_rmse": val["dry_rmse"],
                        "seconds": round(time.perf_counter() - started, 3)})
        if val["rmse"] < best["rmse"] - 1e-9:
            best = {"rmse": val["rmse"], "epoch": epoch,
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
            stale = 0
        else:
            stale += 1
        log(f"  seed {seed} epoch {epoch:03d}  loss={total/seen:.5f}  val_rmse={val['rmse']:.5f}"
            f"  lr={rate:.2e}  {history[-1]['seconds']:.1f}s")
        if stale >= config["patience"]:
            log(f"  seed {seed}: no validation gain for {stale} epochs; stopping at {epoch}.")
            break
    model.load_state_dict(best["state"])
    model.to(device).eval()
    return model, model_config, history, best


def save_checkpoint(path, model, model_config, config, scaler, manifest, bundle, extra=None):
    payload = {
        "format": CHECKPOINT_FORMAT, "dewsat_version": VERSION, "schema_version": 2,
        "csv_features": CSV_FEATURES, "model_features": MODEL_FEATURES,
        "sequence_length": config["sequence_length"], "lead_months": config["lead_months"],
        "model_config": model_config,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler": scaler, "data_manifest": manifest,
        "contract_fingerprint": contract_fingerprint(manifest),
        "config": config, "torch_version": str(torch.__version__),
    }
    payload.update(extra or {})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT}. v1 checkpoints are not loadable: "
                         "the model gained an observed-month input channel, so retrain.")
    if checkpoint["model_features"] != MODEL_FEATURES:
        raise ValueError("Checkpoint feature order does not match this build.")
    model = DroughtBiLSTM(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def _environment(csv_path, config):
    return {"command": " ".join(sys.argv), "dewsat_version": VERSION,
            "python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": str(np.__version__), "platform": platform.platform(),
            "machine": platform.machine(), "csv": str(csv_path), "csv_sha256": sha256_file(csv_path)}


def _split_audit(bundle, splits):
    windows = bundle["windows"]
    audit = {"mode": splits["_mode"],
             "counts": {name: int(len(splits[name])) for name in ("train", "val", "test")},
             "target_month_range": {
                 name: [month_string(int(windows["target_months"][splits[name]].min())),
                        month_string(int(windows["target_months"][splits[name]].max()))]
                 for name in ("train", "val", "test")},
             "mean_observed_months": {
                 name: round(float(windows["observed_months"][splits[name]].mean()), 3)
                 for name in ("train", "val", "test")}}
    if "_blocks" in splits:
        audit["blocks"] = splits["_blocks"]
        audit["block_overlap"] = sorted(
            set(splits["_blocks"]["train"]) & set(splits["_blocks"]["test"]))
    return audit


def evaluate_run(bundle, splits, scaler, config, models, device, ridge_penalty=None):
    """Score the model and every baseline on the held-out split."""
    table, windows = bundle["table"], bundle["windows"]
    y = windows["y"]
    test_index = splits["test"]
    calendar = np.asarray([windows["metadata"][i]["target_calendar_month"] for i in test_index])
    member_predictions = [predict_batches(m, table, windows, test_index, scaler, device) for m in models]
    prediction = np.mean(member_predictions, axis=0)
    references = baseline_module.build_all(table, windows, splits, scaler,
                                           ridge_penalty if ridge_penalty is not None
                                           else config["ridge_penalty"])
    reference_predictions = {name: fn(test_index) for name, (fn, _) in references.items()}
    result = {
        "test": metric_module.evaluate_all(y[test_index], prediction, calendar, config["dry_threshold"]),
        "baselines": {name: metric_module.evaluate_all(y[test_index], values, None,
                                                       config["dry_threshold"])
                      for name, values in reference_predictions.items()},
        "baseline_details": {name: info for name, (_, info) in references.items()},
    }
    result["skill_vs"] = {
        name: metric_module.skill_score(result["test"]["rmse"], result["baselines"][name]["rmse"])
        for name in reference_predictions}
    if len(models) > 1:
        member_rmse = [float(np.sqrt(np.mean((p - y[test_index]) ** 2))) for p in member_predictions]
        result["ensemble"] = {"members": len(models), "member_test_rmse": member_rmse,
                              "mean_member_rmse": float(np.mean(member_rmse)),
                              "mean_member_spread": float(np.mean(np.std(member_predictions, axis=0)))}
    return result, prediction, reference_predictions


def write_test_predictions(path, bundle, splits, prediction, reference_predictions):
    import csv as csv_module
    windows = bundle["windows"]
    index = splits["test"]
    names = sorted(reference_predictions)
    fields = ["cell_id", "group", "input_start_month", "input_end_month", "target_month",
              "observed_months", "target", "prediction", *names]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, i in enumerate(index):
            meta = windows["metadata"][int(i)]
            writer.writerow({
                "cell_id": meta["cell_id"], "group": meta["group"],
                "input_start_month": meta["input_start_month"],
                "input_end_month": meta["input_end_month"], "target_month": meta["target_month"],
                "observed_months": meta["observed_months"],
                "target": float(windows["y"][int(i)]),
                "prediction": float(prediction[position]),
                **{name: float(reference_predictions[name][position]) for name in names}})


def train(csv_path, run_dir, config, manifest_path=None, teacher_checkpoint=None):
    """Full run: admit data, split, fit, evaluate against baselines, write artifacts."""
    from .report import write_report
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} is not empty. Use a new run name so earlier results survive.")
    device = resolve_device(config["device"])
    bundle = prepare(csv_path, config, manifest_path)
    manifest, table, windows = bundle["manifest"], bundle["table"], bundle["windows"]
    splits = split_windows(windows, config)
    scaler = fit_scaler(table, windows, splits["train"], config["scaler_kind"])
    if month_number(scaler["fit_last_month"]) >= month_number(config["val_start"]):
        raise RuntimeError("The scaler saw a month at or after val_start; this would leak.")

    teacher_predictions, teacher_info = None, None
    if teacher_checkpoint:
        teacher_predictions, teacher_info = _teacher_targets(teacher_checkpoint, bundle, splits,
                                                             scaler, config, device)

    run_dir.mkdir(parents=True, exist_ok=True)
    for warning in table["warnings"]:
        log(f"WARNING: {warning}")
    log(f"{manifest['data_kind'].upper()} data | lead {config['lead_months']} month(s) | "
        f"{len(windows['row_indices'])} windows | splits {_split_audit(bundle, splits)['counts']} | "
        f"device {device}")

    started = time.perf_counter()
    models, histories, selections = [], [], []
    for member in range(config["n_ensemble"]):
        seed = config["seed"] + member
        model, model_config, history, best = train_member(bundle, splits, scaler, config, device,
                                                          seed, teacher_predictions)
        models.append(model)
        histories.append([{**row, "member": member, "seed": seed} for row in history])
        selections.append({"member": member, "seed": seed, "best_epoch": best["epoch"],
                           "val_rmse": best["rmse"], "epochs_run": len(history)})
    training_seconds = time.perf_counter() - started
    best_member = int(np.argmin([s["val_rmse"] for s in selections]))

    audit = {"data_kind": manifest["data_kind"], "lead_months": config["lead_months"],
             "sequence_length": config["sequence_length"],
             "table": {**table["coverage"], "dropped_low_quality_rows": table["dropped_low_quality_rows"],
                       "group_column": table["group_column"]},
             "windows": {"total": int(len(windows["row_indices"])), "rejected": windows["rejected"],
                         "mean_observed_months": round(float(windows["observed_months"].mean()), 3)},
             "splits": _split_audit(bundle, splits),
             "scaler": {"kind": scaler["kind"], "fit_monthly_rows": scaler["fit_monthly_rows"],
                        "fit_last_month": scaler["fit_last_month"]},
             "parameter_count": count_parameters(models[best_member]),
             "device": device, "training_seconds": round(training_seconds, 2),
             "warnings": table["warnings"], "environment": _environment(csv_path, config)}
    write_json(run_dir / "audit.json", audit)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "data_manifest.json", manifest)
    write_json(run_dir / "scaler.json", scaler)
    _write_history(run_dir / "history.csv", [row for member in histories for row in member])

    checkpoint_path = run_dir / "best.pt"
    save_checkpoint(checkpoint_path, models[best_member], models[best_member].config, config,
                    scaler, manifest, bundle,
                    extra={"best_epoch": selections[best_member]["best_epoch"],
                           "validation_rmse": selections[best_member]["val_rmse"],
                           "csv_sha256": audit["environment"]["csv_sha256"],
                           "teacher": teacher_info})
    if config["n_ensemble"] > 1:
        for member, model in enumerate(models):
            save_checkpoint(run_dir / "members" / f"member_{member:02d}.pt", model, model.config,
                            config, scaler, manifest, bundle, extra={**selections[member]})

    result, prediction, reference_predictions = evaluate_run(bundle, splits, scaler, config,
                                                             models, device)
    reloaded, saved = load_checkpoint(checkpoint_path, "cpu")
    reload_prediction = predict_batches(reloaded, table, windows, splits["test"], saved["scaler"], "cpu")
    single_prediction = predict_batches(models[best_member], table, windows, splits["test"], scaler, device)
    reload_difference = float(np.max(np.abs(reload_prediction - single_prediction)))
    if reload_difference > 1e-4:
        raise RuntimeError(f"Reloading the checkpoint changed predictions by {reload_difference:.2e}; "
                           "the saved model is not the model that was evaluated.")

    result.update({
        "data_kind": manifest["data_kind"], "target_name": manifest["target_name"],
        "lead_months": config["lead_months"], "selection": selections,
        "best_member": best_member, "validation_rmse": selections[best_member]["val_rmse"],
        "reload_max_absolute_difference": reload_difference,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "parameter_count": audit["parameter_count"],
        "training_seconds": audit["training_seconds"],
        "interpretation": _interpretation(manifest, config, splits),
    })
    write_json(run_dir / "metrics.json", result)
    write_test_predictions(run_dir / "test_predictions.csv", bundle, splits, prediction,
                           reference_predictions)
    write_report(run_dir, config, audit, result, histories, bundle, splits, prediction)
    log(f"\nTest RMSE {result['test']['rmse']:.5f} | persistence {result['baselines']['persistence']['rmse']:.5f}"
        f" | climatology {result['baselines']['seasonal_climatology']['rmse']:.5f}")
    log(f"Artifacts in {run_dir}")
    return result


def _interpretation(manifest, config, splits):
    parts = []
    if manifest["data_kind"] == "synthetic":
        parts.append("Synthetic data: these numbers test the software, not drought accuracy.")
    else:
        parts.append("Real data: skill applies to this target definition, spatial unit and period only.")
    parts.append("Nowcast (lead 0): current-condition estimation, not early warning."
                 if config["lead_months"] == 0 else
                 f"Forecast at {config['lead_months']}-month lead using only prior months.")
    parts.append("Held-out spatial blocks, so this also tests transfer to unseen locations."
                 if splits["_mode"] == "block_holdout" else
                 "Temporal holdout only; every test location was seen in training.")
    return " ".join(parts)


def _teacher_targets(teacher_checkpoint, bundle, splits, scaler, config, device):
    """Precompute frozen teacher predictions for the training windows."""
    teacher, checkpoint = load_checkpoint(teacher_checkpoint, device)
    if checkpoint["sequence_length"] != config["sequence_length"] or \
            checkpoint["lead_months"] != config["lead_months"]:
        raise ValueError("Teacher and student must share sequence_length and lead_months.")
    if checkpoint["contract_fingerprint"] != contract_fingerprint(bundle["manifest"]):
        raise ValueError("Teacher was trained on a different data contract.")
    predictions = np.zeros(len(bundle["windows"]["y"]), dtype=np.float32)
    index = splits["train"]
    # The teacher's own scaler is used, so the student inherits its exact view of the inputs.
    predictions[index] = predict_batches(teacher, bundle["table"], bundle["windows"], index,
                                         checkpoint["scaler"], device)
    log(f"Teacher {teacher_checkpoint}: {count_parameters(teacher):,} parameters, "
        f"{len(index)} soft targets, alpha={config['distill_alpha']}")
    return predictions, {"checkpoint": str(teacher_checkpoint),
                         "parameters": count_parameters(teacher),
                         "alpha": config["distill_alpha"],
                         "validation_rmse": checkpoint.get("validation_rmse")}


def _write_history(path, rows):
    import csv as csv_module
    with Path(path).open("w", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predict_csv(checkpoint_path, csv_path, output_path, target_month=None, manifest_path=None,
                allow_drift=False, device="cpu"):
    """Reload a checkpoint and write one prediction per window, without targets."""
    import csv as csv_module
    from .contract import compare_manifests
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"{output_path} exists; choose a new prediction output path.")
    device = resolve_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    manifest = read_manifest(csv_path, manifest_path)
    if contract_fingerprint(manifest) != checkpoint["contract_fingerprint"]:
        raise ValueError("Feature or target definitions differ from training. New months are fine; "
                         "changed definitions need a newly trained model.")
    drift = compare_manifests(checkpoint["data_manifest"], manifest)
    if drift and not allow_drift:
        raise ValueError(f"Manifest fields changed outside the model contract: {sorted(drift)}. "
                         "Pass --allow-manifest-drift to record and proceed.")
    bundle = prepare(csv_path, config, manifest_path, require_target=False)
    windows = bundle["windows"]
    index = np.arange(len(windows["row_indices"]))
    if target_month:
        index = np.asarray([i for i in index
                            if windows["metadata"][i]["target_month"] == target_month], dtype=np.int64)
        if not len(index):
            raise ValueError(f"No windows predict {target_month}. Available: "
                             f"{windows['metadata'][0]['target_month']}.."
                             f"{windows['metadata'][-1]['target_month']}")
    prediction = predict_batches(model, bundle["table"], windows, index, checkpoint["scaler"], device)
    fields = ["cell_id", "group", "input_start_month", "input_end_month", "target_month",
              "observed_months", "prediction", "lead_months", "data_kind", "target_name"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, i in enumerate(index):
            meta = windows["metadata"][int(i)]
            writer.writerow({key: meta[key] for key in ("cell_id", "group", "input_start_month",
                                                        "input_end_month", "target_month",
                                                        "observed_months")}
                            | {"prediction": float(prediction[position]),
                               "lead_months": checkpoint["lead_months"],
                               "data_kind": manifest["data_kind"],
                               "target_name": manifest["target_name"]})
    return {"predictions": int(len(prediction)), "path": str(output_path),
            "lead_months": checkpoint["lead_months"], "data_kind": manifest["data_kind"],
            "manifest_drift": sorted(drift), "mean_prediction": float(prediction.mean())}
