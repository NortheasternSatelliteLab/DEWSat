"""Random-search configurations (DESIGN.md 6), drawn once per (job, family) with a fixed seed."""

from __future__ import annotations

import math
import zlib

import numpy as np

from common import N_CONFIGS, SEARCH_SEED, SMOKE

HIDDEN = (16, 24, 32, 48, 64, 96, 128)
EPOCHS = (6, 10, 15, 20, 30)


def seed_for(*parts) -> int:
    return (SEARCH_SEED + zlib.crc32("|".join(map(str, parts)).encode())) % (2 ** 32)


def net_configs(job: str, family: str) -> list[dict]:
    rng = np.random.default_rng(seed_for("net", job, family))
    out = []
    for i in range(N_CONFIGS):
        mask_on = bool(rng.random() < 0.5)
        cfg = {"id": i, "family": family, "hidden": int(rng.choice(HIDDEN)), "layers": int(rng.choice([1, 2])),
               "lr": float(math.exp(rng.uniform(math.log(2e-4), math.log(3e-3)))),
               "weight_decay": float(math.exp(rng.uniform(math.log(1e-6), math.log(1e-1)))),
               "dropout": float(rng.uniform(0.0, 0.5)),
               "mask_p": float(rng.uniform(0.1, 0.4)) if mask_on else 0.0,
               "epochs": int(rng.choice(EPOCHS)), "batch": 512}
        if SMOKE:
            cfg["epochs"] = 1
        bid = bool(rng.random() < 0.5)          # drawn for every family so the streams stay aligned
        if family == "gru":
            cfg["bidir"] = bid
        elif family in ("bilstm", "resbilstm"):
            cfg["bidir"] = True
        elif family == "lstm":
            cfg["bidir"] = False
        out.append(cfg)
    return out


def gbt_configs(job: str) -> list[dict]:
    rng = np.random.default_rng(seed_for("gbt", job))
    out = []
    for i in range(N_CONFIGS):
        out.append({"id": i, "learning_rate": float(math.exp(rng.uniform(math.log(0.02), math.log(0.3)))),
                    "max_iter": int(rng.choice([50, 100, 200, 400, 800])),
                    "max_leaf_nodes": int(rng.choice([7, 15, 31, 63])),
                    "min_samples_leaf": int(rng.choice([20, 100, 400, 1600])),
                    "l2_regularization": float(math.exp(rng.uniform(math.log(1e-3), math.log(10.0))))})
    return out


def device_for(cfg: dict) -> str:
    """MPS for temporal CNNs, the tile CNNs (post-hoc cnn_lr, cnn_bilstm) and large recurrent networks (benchmark:
    logs/bench_nets.json), CPU otherwise."""
    if cfg["family"] in ("tcn", "cnn_lr", "cnn_bilstm"):
        return "mps"
    size = cfg["hidden"] * cfg.get("layers", 1) * (2 if cfg.get("bidir") else 1)
    return "mps" if size >= 128 else "cpu"


def cost(cfg: dict, n_windows: int) -> float:
    """Rough seconds for one run on its preferred device (for scheduling and estimates only)."""
    if cfg["family"] in ("cnn_lr", "cnn_bilstm"):           # GPU rates of the post-hoc runner (posthoc_run._est)
        per = 1.5 + 0.15 * cfg["width"] if cfg["family"] == "cnn_lr" else 20.0 + 0.6 * cfg["width"]
        return per * cfg["epochs"] * n_windows / 1e5 + 15
    per100k = {"cpu": 1.95 * (cfg["hidden"] / 32) ** 0.95 * cfg.get("layers", 1) * (1.0 if cfg.get("bidir") else 0.55),
               "mps": 2.0 + 0.012 * cfg["hidden"] * cfg.get("layers", 1)}
    dev = device_for(cfg)
    return per100k[dev] * cfg["epochs"] * n_windows / 1e5 + 15
