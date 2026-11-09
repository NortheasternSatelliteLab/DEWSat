"""DESIGN.md amendment 7, first check: the hand-written LSTM cell (nets.HandLSTM) against nn.LSTM.

For each Bi-LSTM setting chosen in development (hidden 24 x 1 layer, 128 x 1, 128 x 2) and two seeds:
  - the same seed gives `bilstm`, `bilstm_hand` (tanh) and `bilstm_relu` exactly the same initial weights;
  - `bilstm_hand` reproduces `bilstm`'s logits (largest difference below 1e-5), on CPU and, if present, MPS;
  - `bilstm_relu` gives finite logits that differ from `bilstm`'s (the activation is really changed).
Writes logs/relu/verify_cell.json; exits 1 if any check fails.

    cd ~/Desktop/DEWSat_BiLSTM_v2 && source env.sh && .venv/bin/python code/checks/verify_relu_cell.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dewsat2"))
import numpy as np
import torch

import nets
from common import LOGS, write_json

TOL = 1e-5
SETTINGS = ((24, 1), (128, 1), (128, 2))


def build(family, hidden, layers, seed):
    torch.manual_seed(seed)
    return nets.SeqNet(family, 9, 1, hidden, layers, 0.23, True, nets.OBS_CH)


def inputs(n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 12, 9, generator=g)
    x[:, :, nets.OBS_CH] = (torch.rand(n, 12, generator=g) < 0.7).float()     # observed flag, 0/1
    return x


def main() -> int:
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    rows, bad = [], []
    x = inputs()
    for hidden, layers in SETTINGS:
        for seed in (42, 43):
            m = {f: build(f, hidden, layers, seed).eval() for f in ("bilstm", "bilstm_hand", "bilstm_relu")}
            sd = {f: mm.state_dict() for f, mm in m.items()}
            same_keys = all(sd[f].keys() == sd["bilstm"].keys() for f in sd)
            same_init = same_keys and all(torch.equal(sd[f][k], sd["bilstm"][k]) for f in sd for k in sd["bilstm"])
            row = {"hidden": hidden, "layers": layers, "seed": seed, "same_keys": same_keys, "same_init": same_init,
                   "n_params": {f: nets.count_parameters(mm) for f, mm in m.items()}}
            for dev in devices:
                with torch.no_grad():
                    z = {f: mm.to(dev)(x.to(dev)).float().cpu() for f, mm in m.items()}
                row[dev] = {"hand_minus_lstm": float((z["bilstm_hand"] - z["bilstm"]).abs().max()),
                            "relu_minus_lstm": float((z["bilstm_relu"] - z["bilstm"]).abs().max()),
                            "relu_finite": bool(torch.isfinite(z["bilstm_relu"]).all())}
                if row[dev]["hand_minus_lstm"] >= TOL:
                    bad.append(f"{hidden}x{layers} seed {seed} {dev}: tanh cell differs by {row[dev]['hand_minus_lstm']:.2e}")
                if not row[dev]["relu_finite"] or row[dev]["relu_minus_lstm"] < 1e-3:
                    bad.append(f"{hidden}x{layers} seed {seed} {dev}: ReLU logits not finite or unchanged")
            if not same_init:
                bad.append(f"{hidden}x{layers} seed {seed}: initial weights differ")
            if len(set(row["n_params"].values())) != 1:
                bad.append(f"{hidden}x{layers} seed {seed}: parameter counts differ {row['n_params']}")
            rows.append(row)
            print(row, flush=True)
    out = {"tolerance": TOL, "devices": devices, "rows": rows, "failures": bad, "passed": not bad,
           "torch": torch.__version__}
    write_json(LOGS / "relu" / "verify_cell.json", out)
    print("PASSED" if not bad else "FAILED:\n  " + "\n  ".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
