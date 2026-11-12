"""Time a few configurations on real windows (sandbox with the original 102 regions): seconds per epoch per
100k windows on CPU (3 threads) and MPS, against search.cost's estimate. Writes logs/bench_real.json."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dewsat2"))
import numpy as np
import nets, search, units
from common import V2
d = units.prep("job1", 0, "inner")
n = len(d["X_fit"])
print("inner-train windows", n, "prep s", d["seconds"], flush=True)
data = {"X_fit": d["X_fit"], "Y_fit": d["Y_fit"], "M_fit": d["M_fit"], "hide_ok": d["hide_ok"]}
cfgs = [("bilstm", 32, 1, True), ("bilstm", 128, 2, True), ("gru", 48, 1, False), ("gru", 96, 2, True),
        ("lstm", 64, 1, False), ("lstm", 128, 2, False), ("tcn", 32, 1, None), ("tcn", 128, 2, None)]
out = []
for fam, h, L, bd in cfgs:
    cfg = {"family": fam, "hidden": h, "layers": L, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.2, "mask_p": 0.2,
           "epochs": 2, "batch": 512, "bidir": bd}
    for dev in ("cpu", "mps"):
        if fam == "tcn" and dev == "cpu" and h > 32:
            continue
        r = nets.train(cfg, data, 0, device=dev, threads=3)
        ep = r["history"][-1]["seconds"]
        out.append({"family": fam, "hidden": h, "layers": L, "bidir": bd, "device": dev, "sec_epoch": ep,
                    "per100k": ep / n * 1e5, "est_per100k": search.cost(cfg, 1e5) / 2 - 15 / 2})
        print(out[-1], flush=True)
(V2 / "logs" / "bench_real.json").write_text(json.dumps({"n": n, "runs": out}, indent=1))
