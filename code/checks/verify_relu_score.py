"""DESIGN.md amendment 7, check of the scoring path: relu.score() with the ReLU runs pointed at the original `bilstm`
runs must give `bilstm_relu - bilstm` exactly 0, the score stage's own within-fold AUCs for `bilstm`, LR on VCI(T)
and the best simple model, and saved models that reproduce their predictions. Writes nothing to results/scores/; the result
goes to logs/relu/verify_score.json. Exits 1 if any check fails.

    source env.sh && .venv/bin/python code/checks/verify_relu_score.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dewsat2"))
import relu
import units
from common import LOGS, write_json

orig_dir = relu.run_dir
relu.run_dir = lambda fam, k, s: units.final_dir("job1", "bilstm", k, s) if fam == relu.MODEL else orig_dir(fam, k, s)
res = relu.score(write=False)
bad = []
for job, e in res["jobs"].items():
    d = e["paired"][f"{relu.MODEL} - bilstm"]
    if any(abs(v) > 1e-12 for sd in d.values() for v in sd if v is not None):
        bad.append(f"{job}: bilstm_relu - bilstm is not 0 with the same runs: {d}")
    for m, c in e["matches_score_stage"].items():
        if c["score_stage"] is None or abs(c["here"] - c["score_stage"]) > 1e-12:
            bad.append(f"{job} {m}: within-fold AUC {c['here']} here, {c['score_stage']} in the score stage")
for f, v in res["reproduce_max_logit_difference"].items():
    if v > 1e-4:
        bad.append(f"{f}: saved model differs from its predictions by {v}")
out = {"jobs": {j: {"models": {m: v["all"] for m, v in e["models"].items()}, "paired": e["paired"],
                    "matches_score_stage": e["matches_score_stage"]} for j, e in res["jobs"].items()},
       "reproduce": res["reproduce_max_logit_difference"], "failures": bad, "passed": not bad}
write_json(LOGS / "relu" / "verify_score.json", out)
for j, e in out["jobs"].items():
    print(j, {m: round(v[0], 4) for m, v in e["models"].items()}, e["matches_score_stage"])
print("reproduce", out["reproduce"])
print("PASSED" if not bad else "FAILED:\n  " + "\n  ".join(bad))
raise SystemExit(0 if not bad else 1)
