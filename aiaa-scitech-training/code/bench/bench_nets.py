"""Epoch time of each network family on synthetic data, CPU (3 threads) vs MPS."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dewsat2"))
import nets

n, steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000, 12
rng = np.random.default_rng(0)
X = rng.standard_normal((n, steps, 9)).astype(np.float32)
X[:, :, 7] = (rng.random((n, steps)) < 0.8)
Y = (rng.random((n, 1)) < 0.3).astype(np.float32)
data = {"X_fit": X, "Y_fit": Y, "M_fit": np.ones_like(Y, bool), "hide_ok": np.r_[np.ones(11, bool), False]}
res = []
for dev in ("cpu", "mps"):
    for fam, h, L in (("bilstm", 32, 1), ("bilstm", 128, 2), ("gru", 64, 1), ("lstm", 128, 2), ("tcn", 64, 2), ("tcn", 128, 1)):
        cfg = {"family": fam, "hidden": h, "layers": L, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.2,
               "mask_p": 0.2, "epochs": 2, "batch": 512}
        t0 = time.time()
        out = nets.train(cfg, data, 0, device=dev, threads=3)
        ep = [e["seconds"] for e in out["history"]]
        res.append({"device": dev, "family": fam, "hidden": h, "layers": L, "epoch_s": ep, "n": n})
        print(dev, fam, h, L, "epoch s", ep, flush=True)
Path(sys.argv[2] if len(sys.argv) > 2 else "bench.json").write_text(json.dumps(res, indent=1))
