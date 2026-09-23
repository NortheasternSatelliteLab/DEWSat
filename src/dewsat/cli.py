"""Command line interface.

Every subcommand prints a JSON summary so runs can be diffed, logged or piped
into a notebook. `--set key=value` overrides any config field without editing a
file, which keeps sweeps out of version control.
"""
import argparse
import json
from pathlib import Path

from .contract import read_json


def _parse_overrides(pairs):
    overrides = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            overrides[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key.strip()] = raw
    return overrides


def _config_arguments(parser):
    parser.add_argument("--config", help="JSON config file; unspecified fields use the defaults.")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", dest="overrides",
                        help="Override a config field, repeatable. Values are parsed as JSON.")


def build_parser():
    parser = argparse.ArgumentParser(prog="dewsat", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    synth = commands.add_parser("synth", help="Write a synthetic monthly CSV and its manifest.")
    synth.add_argument("--out", required=True)
    synth.add_argument("--cells", type=int, default=64)
    synth.add_argument("--regions", type=int, default=4)
    synth.add_argument("--seed", type=int, default=42)
    synth.add_argument("--start", default="2015-01")
    synth.add_argument("--end", default="2024-12")
    synth.add_argument("--cloud", type=float, default=0.35,
                       help="0 keeps every month, 1 removes most of the wet season.")
    synth.add_argument("--reference-end", default="2021-12",
                       help="Last month of the frozen percentile reference period.")
    synth.add_argument("--overwrite", action="store_true")

    validate = commands.add_parser("validate", help="Check the contract, windows and splits.")
    validate.add_argument("--data", required=True)
    validate.add_argument("--manifest")
    _config_arguments(validate)

    train = commands.add_parser("train", help="Train, evaluate against baselines, write a report.")
    train.add_argument("--data", required=True)
    train.add_argument("--manifest")
    train.add_argument("--out", required=True)
    _config_arguments(train)

    distill = commands.add_parser("distill", help="Train a small student against a trained teacher.")
    distill.add_argument("--data", required=True)
    distill.add_argument("--manifest")
    distill.add_argument("--teacher", required=True)
    distill.add_argument("--out", required=True)
    _config_arguments(distill)

    predict = commands.add_parser("predict", help="Predict from a checkpoint; targets not required.")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--data", required=True)
    predict.add_argument("--manifest")
    predict.add_argument("--target-month", help="Keep only windows predicting this YYYY-MM.")
    predict.add_argument("--out", required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--allow-manifest-drift", action="store_true",
                         help="Proceed when non-contract manifest fields changed, and record them.")

    export = commands.add_parser("export", help="Write ONNX, TorchScript, calibration and manifest.")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--data", help="Training CSV, used to draw the calibration tensor.")
    export.add_argument("--manifest")
    export.add_argument("--opset", type=int, default=17)
    export.add_argument("--calibration-size", type=int, default=512)

    quantize = commands.add_parser("quantize", help="Dynamic INT8 for CPU, with the accuracy cost.")
    quantize.add_argument("--checkpoint", required=True)
    quantize.add_argument("--data", required=True)
    quantize.add_argument("--manifest")
    quantize.add_argument("--out", required=True)

    bench = commands.add_parser("bench", help="Latency, throughput and energy for exported artifacts.")
    bench.add_argument("--checkpoint")
    bench.add_argument("--torchscript")
    bench.add_argument("--onnx")
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--half", action="store_true", help="CUDA fp16.")
    bench.add_argument("--int8", action="store_true",
                       help="Also time a dynamically quantized copy of --checkpoint on CPU.")
    bench.add_argument("--batch-sizes", default="1,8,64")
    bench.add_argument("--iterations", type=int, default=100)
    bench.add_argument("--warmup", type=int, default=20)
    bench.add_argument("--no-power", action="store_true")
    bench.add_argument("--out")

    commands.add_parser("probe", help="Describe this machine and, on a Jetson, its power state.")

    summary = commands.add_parser("summary", help="Print the headline numbers from a run directory.")
    summary.add_argument("run_dir")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "synth":
        from .synth import generate_synthetic
        result = generate_synthetic(args.out, n_cells=args.cells, seed=args.seed, start=args.start,
                                    end=args.end, n_regions=args.regions, cloud=args.cloud,
                                    reference_end=args.reference_end, overwrite=args.overwrite)
    elif args.command == "validate":
        result = _validate(args)
    elif args.command == "train":
        from .train import load_config, train
        config = load_config(args.config, _parse_overrides(args.overrides))
        result = train(args.data, args.out, config, args.manifest)
    elif args.command == "distill":
        from .train import load_config, train
        config = load_config(args.config, _parse_overrides(args.overrides))
        result = train(args.data, args.out, config, args.manifest, teacher_checkpoint=args.teacher)
    elif args.command == "predict":
        from .train import predict_csv
        result = predict_csv(args.checkpoint, args.data, args.out, args.target_month, args.manifest,
                             args.allow_manifest_drift, args.device)
    elif args.command == "export":
        from .export import export_model
        result = export_model(args.checkpoint, args.out, args.data, args.manifest, args.opset,
                              args.calibration_size)
    elif args.command == "quantize":
        from .export import quantize_dynamic
        result = quantize_dynamic(args.checkpoint, args.out, args.data, args.manifest)
    elif args.command == "bench":
        from .bench import format_benchmark, run_benchmark
        sizes = tuple(int(v) for v in args.batch_sizes.split(","))
        result = run_benchmark(args.out, args.checkpoint, args.torchscript, args.onnx, args.device,
                               sizes, args.iterations, args.warmup, args.half, not args.no_power,
                               int8=args.int8)
        print(format_benchmark(result), flush=True)
    elif args.command == "probe":
        from .bench import probe_device
        result = probe_device()
    else:
        result = _summary(Path(args.run_dir))
    print(json.dumps(result, indent=2, default=str))
    return 0


def _validate(args):
    from .data import split_windows
    from .train import load_config, prepare
    config = load_config(args.config, _parse_overrides(args.overrides))
    bundle = prepare(args.data, config, args.manifest)
    windows = bundle["windows"]
    splits = split_windows(windows, config)
    for warning in bundle["table"]["warnings"]:
        print(f"WARNING: {warning}", flush=True)
    return {
        "data_kind": bundle["manifest"]["data_kind"],
        "target_name": bundle["manifest"]["target_name"],
        "coverage": bundle["table"]["coverage"],
        "dropped_low_quality_rows": bundle["table"]["dropped_low_quality_rows"],
        "sequence_length": config["sequence_length"], "lead_months": config["lead_months"],
        "windows": int(len(windows["row_indices"])),
        "input_shape": [config["sequence_length"], 8],
        "mean_observed_months": round(float(windows["observed_months"].mean()), 3),
        "rejected_windows": windows["rejected"],
        "split_mode": splits["_mode"],
        "splits": {name: int(len(splits[name])) for name in ("train", "val", "test")},
        "blocks": splits.get("_blocks"),
        "warnings": bundle["table"]["warnings"],
    }


def _summary(run_dir):
    metrics = read_json(run_dir / "metrics.json")
    audit = read_json(run_dir / "audit.json")
    return {
        "run": str(run_dir), "data_kind": metrics["data_kind"], "lead_months": metrics["lead_months"],
        "parameters": metrics["parameter_count"], "test_rmse": metrics["test"]["rmse"],
        "test_mae": metrics["test"]["mae"], "test_r2": metrics["test"]["r2"],
        "dry_event": {k: metrics["test"]["event"][k] for k in ("pod", "far", "csi", "hss")},
        "baseline_rmse": {name: values["rmse"] for name, values in metrics["baselines"].items()},
        "skill_vs": metrics["skill_vs"], "split_mode": audit["splits"]["mode"],
        "device": audit["device"], "training_seconds": audit["training_seconds"],
        "interpretation": metrics["interpretation"],
    }
