"""Paths, constants, provenance and small I/O helpers (DESIGN.md)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

V2 = Path(os.environ.get("DEWSAT_V2", str(Path.home() / "Desktop" / "DEWSat_BiLSTM_v2")))
CODE = Path(__file__).resolve().parent.parent     # the repository this code runs from (worktrees too)
OUT = Path(os.environ.get("DEWSAT_V2_OUT", str(V2)))       # sandboxes point this elsewhere
DATA = V2 / "data"
PAPER_REGIONS = V2 / "inputs" / "paper_data" / "regions"     # the paper's composites for the original 102
NEW_REGIONS = DATA / "regions"                                # this pull
REEXTRACT = DATA / "regions_reextract"                        # the 102 re-pulled per scene
FLDAS_DEV = DATA / "fldas" / "dev"
LOCKED = Path(os.environ.get("DEWSAT_LOCKED", str(DATA / "locked")))   # sandboxes point at their own copy
CACHE = OUT / "cache"
RUNS = OUT / "runs"
SCORES = OUT / "scores"
FIGURES = OUT / "figures"
LOGS = OUT / "logs"


def month_list(start: str, end: str) -> list[str]:
    y, m = int(start[:4]), int(start[5:])
    ye, me = int(end[:4]), int(end[5:])
    out = []
    while (y, m) <= (ye, me):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


MONTHS = month_list("2016-01", "2026-09")      # 129 months of composites
N_MONTHS = len(MONTHS)
MI = {m: i for i, m in enumerate(MONTHS)}
DEV_END = MI["2023-12"]                         # last development month (index 95)
FRESH_START, FRESH_END = MI["2025-01"], MI["2026-09"]
SEQ = 12
LEADS = (1, 2, 3)
BLOCKS = [("2016-01", "2017-12"), ("2018-01", "2019-12"), ("2020-01", "2021-12"), ("2022-01", "2023-12")]
BLOCK_IDX = [(MI[a], MI[b]) for a, b in BLOCKS]
N_FOLDS = 4
REFS = ("f1", "f2", "f3", "f4", "all")          # VCI tables: one per outer fold, and all development months
BUFFER_NOWCAST = 12
BUFFER_FORECAST = 15
WEST_OF_LON = -98.0
GROUP_SEED = 20261108
BOOT_SEED = 20261108
N_BOOT = 2000
SEARCH_SEED = 20261108
FINAL_SEEDS = tuple(range(42, 52))              # 10 seeds
N_CONFIGS = 30
SMOKE = os.environ.get("DEWSAT_SMOKE") == "1"   # sandbox code checks only: tiny search and 2 seeds
if SMOKE:
    FINAL_SEEDS = (42, 43)
    N_CONFIGS = 2

# feature rules, identical to the paper's build_dataset.py
NODATA = -32768
NDVI_SCALE = 10_000
REF_MIN_YEARS = 3
REF_MIN_RANGE = 0.05
MIN_USABLE_FRAC = 0.05
LABEL_REF = ("2001", "2022")
LABEL_MIN_REF = 20
DROUGHT_QUANTILE = 0.2
MIN_VIEW_CLEAR = 0.3

FAMILIES = ("bilstm", "resbilstm", "gru", "lstm", "tcn")
SIMPLE = ("climatology", "lr_vci_t", "lr_latest_age", "lr_trend", "gbt_mono", "state_ewma")
LABELS = {
    "climatology": "Climatology", "lr_vci_t": "LR on VCI(T)", "lr_latest_age": "LR on latest VCI + age",
    "lr_trend": "LR + trend features", "gbt_mono": "Boosted trees (monotone in VCI)",
    "state_ewma": "Smoothed-VCI state model", "persistence": "Persistence (uses ground data)",
    "bilstm": "Bi-LSTM", "resbilstm": "Residual Bi-LSTM", "gru": "GRU", "lstm": "One-direction LSTM",
    "tcn": "Temporal CNN",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(CODE), *args], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    head = run("rev-parse", "HEAD")
    dirty = bool(run("status", "--porcelain", "--", "dewsat2", "design"))
    return {"commit": head, "short": head[:7], "dirty": dirty}


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default) + "\n")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return default


def save_npz(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path, set)):
        return str(o) if isinstance(o, Path) else sorted(o)
    return str(o)


def record_failure(name: str, exc: BaseException) -> None:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    write_text(LOGS / "failures" / f"{name.replace('/', '_')}.txt", tb)
    path = RUNS / "failures.json"
    items = read_json(path, []) or []
    items.append({"run": name, "error": f"{type(exc).__name__}: {exc}"[:500], "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    write_json(path, items)
    log(f"FAILED {name}: {type(exc).__name__}: {exc}")


def env_summary() -> dict:
    import platform

    import torch
    return {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
            "machine": platform.machine(), "platform": platform.platform()}
