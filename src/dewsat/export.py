"""Turn a checkpoint into deployable artifacts and measure what that costs.

v1 stopped at a pickled checkpoint, which only loads inside a matching PyTorch
install. Nothing in it told a device how to build an input tensor. This module
writes ONNX and TorchScript, a calibration tensor drawn from the training split
only, and a deploy manifest that fully specifies preprocessing, then verifies
that every exported form still produces the training-time numbers.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .contract import MODEL_FEATURES, month_string, sha256_file, write_json
from .data import build_inputs
from .train import load_checkpoint, prepare, resolve_device, split_windows

DEFAULT_OPSET = 17
PARITY_TOLERANCE = 1e-4


def _calibration_inputs(checkpoint, data_csv, manifest_path, size, seed=42):
    """Draw representative scaled windows from the TRAINING split only.

    Calibrating a quantized engine on validation or test windows leaks held-out
    data into the deployed model, which is easy to do by accident and hard to
    see afterwards.
    """
    config = checkpoint["config"]
    bundle = prepare(data_csv, config, manifest_path)
    splits = split_windows(bundle["windows"], config)
    rng = np.random.default_rng(seed)
    train_index = splits["train"]
    chosen = np.sort(rng.choice(train_index, size=min(size, len(train_index)), replace=False))
    months = bundle["windows"]["target_months"][chosen]
    return build_inputs(bundle["table"], bundle["windows"], chosen, checkpoint["scaler"]), {
        "source_csv": str(data_csv), "split": "train", "windows": int(len(chosen)),
        "available_train_windows": int(len(train_index)),
        "target_month_range": [month_string(int(months.min())), month_string(int(months.max()))],
    }


def _deploy_manifest(checkpoint, artifacts, calibration_info, parity, opset):
    scaler = checkpoint["scaler"]
    return {
        "format": "dewsat-v2-deploy",
        "dewsat_version": checkpoint["dewsat_version"],
        "created_from": {"torch_version": checkpoint["torch_version"],
                         "validation_rmse": checkpoint.get("validation_rmse"),
                         "contract_fingerprint": checkpoint["contract_fingerprint"]},
        "task": {"target_name": checkpoint["data_manifest"]["target_name"],
                 "target_range": checkpoint["data_manifest"]["target_range"],
                 "lead_months": checkpoint["lead_months"],
                 "data_kind": checkpoint["data_manifest"]["data_kind"]},
        "io": {"input_name": "window", "output_name": "prediction",
               "input_shape": ["batch", checkpoint["sequence_length"], len(MODEL_FEATURES)],
               "input_dtype": "float32", "output_shape": ["batch"], "onnx_opset": opset,
               "dynamic_axes": ["batch"]},
        "input_recipe": {
            "channel_order": MODEL_FEATURES,
            "step_order": "oldest month first; the last step is the most recent input month",
            "standardization": {"applies_to_channels": scaler["scaled_indices"],
                                "kind": scaler["kind"], "center": scaler["center"],
                                "spread": scaler["spread"],
                                "formula": "(value - center) / spread"},
            "unstandardized_channels": {"valid_frac": "fraction 0..1 as supplied",
                                        "month_sin": "sin(2*pi*(month-1)/12)",
                                        "month_cos": "cos(2*pi*(month-1)/12)",
                                        "observed": "1.0 if the month was observed, else 0.0"},
            "missing_month_rule": "set the four standardized channels and valid_frac to 0.0, keep the "
                                  "true season values, and set observed to 0.0",
            "minimum_observed_months": int(np.ceil(checkpoint["config"]["min_observed_frac"]
                                                   * checkpoint["sequence_length"])),
        },
        "artifacts": artifacts,
        "calibration": calibration_info,
        "parity": parity,
        "notes": [
            "The output is a bounded score in [0,1]; lower means drier.",
            "Quantized engines must be calibrated with the supplied training-split tensor, "
            "never with validation or test windows.",
            "TensorRT INT8 support for LSTM layers is limited; expect FP16 to be the practical "
            "GPU precision on Orin and verify accuracy before claiming INT8.",
        ],
    }


def export_model(checkpoint_path, out_dir, data_csv=None, manifest_path=None, opset=DEFAULT_OPSET,
                 calibration_size=512, parity_batches=(1, 8, 64), run_onnx_check=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_checkpoint(checkpoint_path, "cpu")
    model.eval()
    sequence_length = checkpoint["sequence_length"]

    calibration_info = {"available": False,
                        "reason": "no --data supplied, so no calibration tensor was written"}
    if data_csv:
        calibration, info = _calibration_inputs(checkpoint, data_csv, manifest_path, calibration_size)
        np.savez_compressed(out_dir / "calibration.npz", window=calibration)
        calibration.astype(np.float32).tofile(out_dir / "calibration.f32")
        calibration_info = {"available": True, **info,
                            "npz": "calibration.npz", "raw_float32": "calibration.f32",
                            "shape": list(calibration.shape), "dtype": "float32",
                            "raw_layout": "C-contiguous [windows, months, channels]"}
        probe = torch.from_numpy(calibration.copy())
    else:
        torch.manual_seed(0)
        probe = torch.randn(max(parity_batches), sequence_length, len(MODEL_FEATURES))
        probe[:, :, -1] = 1.0
    # The ONNX LSTM exporter warns that tracing at batch > 1 can freeze the batch
    # size into the graph, so trace at batch 1 and then prove the dynamic axis
    # works by checking every size in parity_batches.
    example = probe[:1].clone()
    parity_batches = tuple(b for b in parity_batches if b <= len(probe)) or (1,)

    with torch.inference_mode():
        reference = model(example).numpy()

    artifacts, parity = {}, {}
    scripted_path = out_dir / "model_torchscript.pt"
    traced = torch.jit.trace(model, example, check_trace=False)
    traced = torch.jit.freeze(traced.eval())
    traced.save(scripted_path)
    with torch.inference_mode():
        reloaded = torch.jit.load(scripted_path)
        difference = float(np.max(np.abs(reloaded(example).numpy() - reference)))
    parity["torchscript_max_abs_diff"] = difference
    artifacts["torchscript"] = {"file": scripted_path.name, "bytes": scripted_path.stat().st_size,
                                "sha256": sha256_file(scripted_path)}

    onnx_path = out_dir / "model.onnx"
    exporter = _export_onnx(model, example, onnx_path, opset)
    artifacts["onnx"] = {"file": onnx_path.name, "bytes": onnx_path.stat().st_size,
                         "sha256": sha256_file(onnx_path), "opset": opset, "exporter": exporter}
    parity["onnx"] = _check_onnx(onnx_path, model, probe, parity_batches) if run_onnx_check else \
        {"checked": False, "reason": "skipped by request"}

    for name, value in parity.items():
        if isinstance(value, float) and value > PARITY_TOLERANCE:
            raise RuntimeError(f"{name} differs from PyTorch by {value:.2e}; do not deploy this export.")
    if parity["onnx"].get("checked") and parity["onnx"]["max_abs_diff"] > PARITY_TOLERANCE:
        raise RuntimeError(f"ONNX output differs by {parity['onnx']['max_abs_diff']:.2e} at batch "
                           f"{parity['onnx']['worst_batch']}; do not deploy this export.")

    manifest = _deploy_manifest(checkpoint, artifacts, calibration_info, parity, opset)
    write_json(out_dir / "deploy_manifest.json", manifest)
    return {"out_dir": str(out_dir), "artifacts": artifacts, "parity": parity,
            "calibration": calibration_info,
            "next": "Copy this folder to the Jetson, then run scripts/jetson_build_engine.py"}


def _export_onnx(model, example, path, opset):
    """Prefer the TorchScript exporter: it emits the single ONNX LSTM op TensorRT wants."""
    errors = {}
    try:
        torch.onnx.export(model, (example,), str(path), input_names=["window"],
                          output_names=["prediction"], opset_version=opset,
                          dynamic_axes={"window": {0: "batch"}, "prediction": {0: "batch"}},
                          dynamo=False)
        return "torchscript"
    except Exception as error:  # noqa: BLE001 - fall back rather than fail the export
        errors["torchscript"] = repr(error)
    torch.onnx.export(model, (example,), str(path), input_names=["window"],
                      output_names=["prediction"], opset_version=opset, dynamo=True,
                      dynamic_shapes={"x": {0: torch.export.Dim("batch")}})
    return f"dynamo (torchscript exporter failed: {errors['torchscript'][:200]})"


def _check_onnx(path, model, probe, batches):
    """Compare ONNX Runtime against PyTorch at every batch size we promise to support."""
    try:
        import onnxruntime
    except ImportError:
        return {"checked": False, "reason": "onnxruntime is not installed in this environment"}
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    per_batch, worst, worst_batch = {}, 0.0, None
    for size in batches:
        sample = probe[:size]
        with torch.inference_mode():
            expected = model(sample).numpy()
        actual = np.asarray(session.run(None, {name: sample.numpy()})[0]).reshape(-1)
        difference = float(np.max(np.abs(actual - expected)))
        per_batch[str(size)] = difference
        if difference > worst:
            worst, worst_batch = difference, size
    return {"checked": True, "runtime": onnxruntime.__version__,
            "providers": session.get_providers(), "traced_at_batch": 1,
            "max_abs_diff_by_batch": per_batch, "max_abs_diff": worst,
            "worst_batch": worst_batch}


def select_quantization_engine(preferred=None):
    """Pick an INT8 kernel backend explicitly.

    Some builds ship with the engine unset, so quantization fails with an opaque
    "NoQEngine" error. QNNPACK is the right choice on ARM, which covers both
    Apple Silicon and the Orin's Cortex-A78AE cores; x86 uses FBGEMM.
    """
    supported = list(torch.backends.quantized.supported_engines)
    order = [preferred] if preferred else []
    order += ["qnnpack", "fbgemm", "x86", "onednn"]
    for name in order:
        if name and name in supported:
            torch.backends.quantized.engine = name
            return name
    raise RuntimeError(f"No INT8 engine in this PyTorch build (supported: {supported}). "
                       "Dynamic quantization needs qnnpack on ARM or fbgemm on x86.")


def _save_quantized(quantized, example, path):
    """Save the INT8 module in the most portable form this build supports.

    Scripting a dynamically quantized LSTM works on some PyTorch builds and not
    others, and tracing is fine here because the forward pass has no data
    dependent branching. Pickling is the last resort and ties the file to the
    exact environment, so it is reported rather than hidden.
    """
    for name, factory in (("torchscript_script", lambda: torch.jit.script(quantized)),
                          ("torchscript_trace", lambda: torch.jit.trace(quantized, example,
                                                                        check_trace=False))):
        try:
            module = factory()
            torch.jit.save(module, path)
            with torch.inference_mode():
                difference = float((torch.jit.load(path)(example) - quantized(example)).abs().max())
            if difference < 1e-5:
                return name
        except Exception:  # noqa: BLE001 - fall through to the next strategy
            continue
    torch.save({"format": "dewsat-v2-int8-dynamic", "module": quantized}, path)
    return "pickled module (loads only in a matching torch environment)"


def quantize_dynamic(checkpoint_path, out_dir, data_csv, manifest_path=None, device="cpu",
                     engine=None):
    """Dynamic INT8 weights for the LSTM and linear layers, with the accuracy cost measured.

    This is the honest CPU compression path: it shrinks weights about fourfold
    and runs on Orin's ARM cores. It is not a TensorRT INT8 GPU engine, and the
    two must not be reported as if they were the same result.
    """
    from .metrics import evaluate_all
    from .model import predict_batches
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)
    engine_name = select_quantization_engine(engine)
    model, checkpoint = load_checkpoint(checkpoint_path, "cpu")
    config = checkpoint["config"]
    quantized = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8)
    quantized.eval()

    bundle = prepare(data_csv, config, manifest_path)
    splits = split_windows(bundle["windows"], config)
    index = splits["test"]
    y = bundle["windows"]["y"][index]
    float_prediction = predict_batches(model, bundle["table"], bundle["windows"], index,
                                       checkpoint["scaler"], "cpu")
    int8_prediction = predict_batches(quantized, bundle["table"], bundle["windows"], index,
                                      checkpoint["scaler"], "cpu")

    path = out_dir / "model_int8_dynamic.pt"
    example = torch.from_numpy(build_inputs(bundle["table"], bundle["windows"], index[:4],
                                            checkpoint["scaler"]))
    saved_as = _save_quantized(quantized, example, path)

    float_bytes = Path(checkpoint_path).stat().st_size
    # Compare weights to weights. File sizes also carry container overhead, which
    # dominates for a model this small and would flatter or spoil the comparison.
    float_weight_bytes = int(sum(t.numel() * t.element_size() for t in model.state_dict().values()))
    parameters = int(sum(t.numel() for t in model.state_dict().values()))
    result = {
        "int8_file": path.name, "int8_bytes": path.stat().st_size, "saved_as": saved_as,
        "engine": engine_name,
        "parameters": parameters,
        "float32_weight_bytes": float_weight_bytes,
        "int8_weight_bytes_estimate": int(round(float_weight_bytes / 4)),
        "float_checkpoint_bytes": float_bytes,
        "file_size_ratio": round(path.stat().st_size / float_bytes, 4),
        "file_size_note": "On-disk ratio includes serialization overhead. Below roughly 100k "
                          "parameters that overhead dominates and the file can even grow; the "
                          "weight-level figures are the ones to quote.",
        "quantized_modules": ["LSTM", "Linear"],
        "test_float32": evaluate_all(y, float_prediction, None, config["dry_threshold"]),
        "test_int8": evaluate_all(y, int8_prediction, None, config["dry_threshold"]),
        "max_abs_prediction_shift": float(np.max(np.abs(int8_prediction - float_prediction))),
        "note": "CPU dynamic quantization. GPU INT8 on Orin requires a TensorRT engine built with "
                "the exported calibration tensor and must be measured separately.",
        "reload_note": f"Set torch.backends.quantized.engine = '{engine_name}' before loading this "
                       "file, and expect small numeric differences on a different engine.",
    }
    result["rmse_delta"] = result["test_int8"]["rmse"] - result["test_float32"]["rmse"]
    write_json(out_dir / "quantization.json", result)
    return result
