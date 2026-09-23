"""Latency, throughput and energy measurement, on a laptop or on the Jetson.

The claim "runs on an Orin Nano" needs numbers attached: latency percentiles at
a stated batch size, the power the board actually drew while producing them,
and the accuracy of the exact artifact that was timed. This module produces all
three where the hardware allows and says plainly when it cannot.
"""
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .contract import MODEL_FEATURES, write_json

TEGRA_RELEASE = Path("/etc/nv_tegra_release")
DEVICE_TREE_MODEL = Path("/proc/device-tree/model")


def _read_text(path):
    try:
        return Path(path).read_text(errors="ignore").strip("\x00\n ")
    except OSError:
        return None


def _run(command):
    if not shutil.which(command[0]):
        return None
    try:
        finished = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return (finished.stdout or finished.stderr).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def probe_device():
    """Read-only description of the machine, with the Jetson bits if present."""
    info = {
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "python": platform.python_version(),
        "torch": str(torch.__version__), "torch_cuda_available": torch.cuda.is_available(),
        "torch_mps_available": bool(torch.backends.mps.is_available()),
        "cpu_count": os.cpu_count(), "is_jetson": False,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        info["cuda_device"] = {"name": properties.name,
                               "capability": f"{properties.major}.{properties.minor}",
                               "total_memory_mb": round(properties.total_memory / 1e6),
                               "multiprocessors": properties.multi_processor_count}
    model = _read_text(DEVICE_TREE_MODEL)
    if (model and "jetson" in model.lower()) or TEGRA_RELEASE.exists():
        info.update({
            "is_jetson": True, "board": model, "l4t": _read_text(TEGRA_RELEASE),
            "nvpmodel": _run(["nvpmodel", "-q"]),
            "jetson_clocks": _run(["jetson_clocks", "--show"]),
            "power_rails": sorted(str(p) for p in _power_rail_files()),
        })
    for name in ("tensorrt", "onnxruntime"):
        try:
            module = __import__(name)
            info[f"{name}_version"] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            info[f"{name}_version"] = None
    info["recommended_commands"] = [
        "sudo nvpmodel -q                 # show the active power mode",
        "sudo nvpmodel -m 0               # MAXN, for the fastest published numbers",
        "sudo nvpmodel -m 1               # the lower-power mode, for the field budget",
        "sudo jetson_clocks               # pin clocks so latency percentiles stop drifting",
        "sudo jetson_clocks --restore     # put the governor back afterwards",
        "tegrastats --interval 100        # watch power and thermals during a run",
    ]
    return info


def _power_rail_files():
    """INA3221 rails expose instantaneous power in microwatts on Jetson boards."""
    found = []
    for directory in Path("/sys/bus/i2c/drivers").glob("ina3221*/*/hwmon/hwmon*"):
        found.extend(sorted(directory.glob("power*_input")))
    if not found:
        for directory in Path("/sys/class/hwmon").glob("hwmon*"):
            name = _read_text(directory / "name") or ""
            if "ina" in name.lower() or "vdd" in name.lower():
                found.extend(sorted(directory.glob("power*_input")))
    return found


class PowerSampler:
    """Background sampler for board power, preferring sysfs over tegrastats."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.samples = []
        self.source = None
        self._files = _power_rail_files()
        self._process = None
        self._thread = None
        self._stop = threading.Event()
        if self._files:
            self.source = "ina3221_sysfs"
        elif shutil.which("tegrastats"):
            self.source = "tegrastats"

    def _sample_sysfs(self):
        while not self._stop.is_set():
            total = 0.0
            for path in self._files:
                value = _read_text(path)
                if value and value.isdigit():
                    total += int(value) / 1e6  # microwatts to watts
            if total:
                self.samples.append(total)
            self._stop.wait(self.interval)

    def _sample_tegrastats(self):
        pattern = re.compile(r"(VDD_IN|POM_5V_IN|VDD_GPU_SOC)\s+(\d+)mW")
        self._process = subprocess.Popen(
            ["tegrastats", "--interval", str(int(self.interval * 1000))],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        for line in self._process.stdout:
            if self._stop.is_set():
                break
            match = pattern.search(line)
            if match:
                self.samples.append(int(match.group(2)) / 1000)

    def __enter__(self):
        if self.source == "ina3221_sysfs":
            self._thread = threading.Thread(target=self._sample_sysfs, daemon=True)
        elif self.source == "tegrastats":
            self._thread = threading.Thread(target=self._sample_tegrastats, daemon=True)
        if self._thread:
            self._thread.start()
        return self

    def __exit__(self, *exception):
        self._stop.set()
        if self._process:
            self._process.terminate()
        if self._thread:
            self._thread.join(timeout=3)
        return False

    def summary(self):
        if not self.samples:
            return {"available": False,
                    "reason": "no INA3221 rails and no tegrastats; energy needs Jetson hardware"}
        values = np.asarray(self.samples, dtype=np.float64)
        return {"available": True, "source": self.source, "samples": int(len(values)),
                "mean_watts": float(values.mean()), "peak_watts": float(values.max()),
                "idle_estimate_watts": float(np.percentile(values, 5))}


def _synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def torch_runner(checkpoint_path, device="cpu", half=False):
    from .train import load_checkpoint, resolve_device
    device = resolve_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if half:
        if device != "cuda":
            raise ValueError("half precision benchmarking is only meaningful on CUDA here.")
        model = model.half()
    dtype = torch.float16 if half else torch.float32

    def run(batch):
        with torch.inference_mode():
            tensor = torch.from_numpy(batch).to(device=device, dtype=dtype)
            output = model(tensor)
            _synchronize(device)
            return output.float().cpu().numpy()

    return run, {"backend": "torch", "device": device, "precision": "fp16" if half else "fp32",
                 "sequence_length": checkpoint["sequence_length"]}


def int8_dynamic_runner(checkpoint_path):
    """Quantize on the fly so the timed model is exactly the one `quantize` scored."""
    from .export import select_quantization_engine
    from .train import load_checkpoint
    engine = select_quantization_engine()
    model, checkpoint = load_checkpoint(checkpoint_path, "cpu")
    quantized = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8).eval()

    def run(batch):
        with torch.inference_mode():
            return quantized(torch.from_numpy(batch)).float().numpy()

    return run, {"backend": "torch-int8-dynamic", "device": "cpu", "precision": "int8",
                 "engine": engine, "sequence_length": checkpoint["sequence_length"]}


def torchscript_runner(path, device="cpu"):
    from .train import resolve_device
    device = resolve_device(device)
    module = torch.jit.load(str(path), map_location=device)
    module.eval()

    def run(batch):
        with torch.inference_mode():
            output = module(torch.from_numpy(batch).to(device))
            _synchronize(device)
            return output.float().cpu().numpy()

    return run, {"backend": "torchscript", "device": device, "precision": "fp32"}


def onnx_runner(path, providers=None):
    import onnxruntime
    available = onnxruntime.get_available_providers()
    requested = providers or [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider",
                                          "CPUExecutionProvider") if p in available]
    session = onnxruntime.InferenceSession(str(path), providers=requested)
    name = session.get_inputs()[0].name

    def run(batch):
        return np.asarray(session.run(None, {name: batch})[0]).reshape(-1)

    return run, {"backend": "onnxruntime", "version": onnxruntime.__version__,
                 "providers": session.get_providers(), "precision": "fp32"}


def benchmark(run, sequence_length=12, batch_sizes=(1, 8, 64), iterations=100, warmup=20,
              measure_power=True, seed=0):
    """Time a runner at each batch size, after warmup, with power sampled if possible."""
    rng = np.random.default_rng(seed)
    results = []
    sampler = PowerSampler() if measure_power else None
    context = sampler if sampler else _NullContext()
    with context:
        for batch_size in batch_sizes:
            batch = rng.standard_normal((batch_size, sequence_length, len(MODEL_FEATURES))
                                        ).astype(np.float32)
            batch[:, :, -1] = 1.0
            cold = time.perf_counter()
            run(batch)
            cold_ms = (time.perf_counter() - cold) * 1000
            for _ in range(warmup):
                run(batch)
            timings = np.empty(iterations)
            for i in range(iterations):
                started = time.perf_counter()
                run(batch)
                timings[i] = (time.perf_counter() - started) * 1000
            results.append({
                "batch_size": batch_size, "iterations": iterations,
                "cold_start_ms": round(cold_ms, 4),
                "mean_ms": float(timings.mean()), "p50_ms": float(np.percentile(timings, 50)),
                "p90_ms": float(np.percentile(timings, 90)), "p99_ms": float(np.percentile(timings, 99)),
                "min_ms": float(timings.min()), "max_ms": float(timings.max()),
                "windows_per_second": float(batch_size / (timings.mean() / 1000)),
                "ms_per_window": float(timings.mean() / batch_size),
            })
    power = sampler.summary() if sampler else {"available": False, "reason": "power sampling disabled"}
    if power.get("available"):
        for entry in results:
            entry["energy_per_window_mj"] = round(
                power["mean_watts"] * entry["ms_per_window"], 4)  # W * ms == mJ
    return {"batches": results, "power": power}


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False


def run_benchmark(out_path, checkpoint=None, torchscript=None, onnx=None, device="cpu",
                  batch_sizes=(1, 8, 64), iterations=100, warmup=20, half=False,
                  measure_power=True, sequence_length=None, int8=False):
    """Benchmark every supplied artifact and write one comparable report."""
    device_info = probe_device()
    entries = []
    targets = []
    if checkpoint:
        targets.append(("checkpoint", lambda: torch_runner(checkpoint, device, half)))
    if torchscript:
        targets.append(("torchscript", lambda: torchscript_runner(torchscript, device)))
    if onnx:
        targets.append(("onnx", lambda: onnx_runner(onnx)))
    if int8:
        if not checkpoint:
            raise ValueError("--int8 quantizes a checkpoint, so --checkpoint is required.")
        targets.append(("int8_dynamic", lambda: int8_dynamic_runner(checkpoint)))
    if not targets:
        raise ValueError("Give at least one of --checkpoint, --torchscript, --onnx or --int8.")
    for name, factory in targets:
        run, info = factory()
        length = sequence_length or info.get("sequence_length") or 12
        measurement = benchmark(run, length, batch_sizes, iterations, warmup, measure_power)
        entries.append({"artifact": name, "info": info, "sequence_length": length, **measurement})
    report = {"device": device_info, "results": entries,
              "caveats": [
                  "Latency includes host-to-device transfer, which is what an operational caller pays.",
                  "Pin clocks with `sudo jetson_clocks` before quoting percentiles.",
                  "Energy is board power during the loop, so it includes idle draw; subtract the "
                  "idle estimate for a marginal figure.",
                  "These are inference numbers only; nothing here measures training on device.",
              ]}
    if out_path:
        write_json(out_path, report)
    return report


def format_benchmark(report):
    lines = ["| Artifact | Device | Precision | Batch | p50 ms | p90 ms | Windows/s | mJ/window |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for entry in report["results"]:
        info = entry["info"]
        device = info.get("device") or ",".join(info.get("providers", []))
        for batch in entry["batches"]:
            energy = batch.get("energy_per_window_mj")
            lines.append(f"| {entry['artifact']} | {device} | {info['precision']} | "
                         f"{batch['batch_size']} | {batch['p50_ms']:.3f} | {batch['p90_ms']:.3f} | "
                         f"{batch['windows_per_second']:.1f} | "
                         f"{'n/a' if energy is None else f'{energy:.3f}'} |")
    power = report["results"][0]["power"] if report["results"] else {}
    if not power.get("available"):
        lines.append("")
        lines.append(f"_Power: {power.get('reason', 'unavailable')}._")
    return "\n".join(lines)
