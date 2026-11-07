"""Freeze the new-region set once no further batch will be pulled (DECISIONS.md #24, #37).

The dataset's regions are frozen when the driver has stopped adding new-region batches: every planned batch is
complete, or a stop rule fired. Only whole batches count; within them, regions whose pull failed after every retry
are left out and listed. The re-extraction of the original 102 continues afterwards: it adds per-scene values and
SWIR for regions already in the dataset and changes no region or composite.

    .venv/bin/python -I code/pull/freeze_regions.py            # refuses while a new-region batch is still running
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

V2 = Path.home() / "Desktop" / "DEWSat_BiLSTM_v2"


def main() -> int:
    st = json.loads((V2 / "data" / "pull_state.json").read_text())
    plan = json.loads((V2 / "data" / "plan" / "batches.json").read_text())
    batches = st.get("batches", {})
    running = [b for b, v in batches.items() if v.get("state") == "running"]
    whole = sorted(int(b) for b, v in batches.items() if v.get("state") == "complete")
    all_planned_done = len(whole) == len(plan["batches"])
    import subprocess
    driver_alive = subprocess.run(["pgrep", "-f", "code/pull/pull_driver.py"], capture_output=True).returncode == 0
    moved_on = bool(st.get("originals")) or not driver_alive      # the final retry of failed regions is over
    stopped = any(e.get("event", "").startswith("frozen") for e in st.get("events", []))
    if running or not ((all_planned_done and moved_on) or stopped):
        print(f"not frozen: batches running {running}, complete {len(whole)}/{len(plan['batches'])}")
        return 1
    new = sorted(n for b in whole for n in batches[str(b)]["ok"])
    failed = sorted(n for b in whole for n in batches[str(b)].get("failed", []))
    import csv
    roles = {r["name"]: r for r in csv.DictReader((V2 / "data" / "plan" / "regions_plan.csv").open())}
    out = {"frozen": time.strftime("%Y-%m-%d %H:%M:%S"), "batches": whole, "new_regions": new, "failed_regions": failed,
           "counts": {f"{role}_{side}": sum(1 for x in new if roles[x]["role"] == role and roles[x]["side"] == side)
                      for role in ("dev", "test") for side in ("west", "east")},
           "pull_elapsed_h": round((time.time() - st["started_epoch"]) / 3600, 2),
           "note": "new regions of whole batches; the original 102 are always development regions"}
    (V2 / "data" / "FROZEN_REGIONS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "new_regions"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
