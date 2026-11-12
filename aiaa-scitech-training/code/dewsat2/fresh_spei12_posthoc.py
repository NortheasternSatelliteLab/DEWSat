"""Post-hoc, after the lock: the fresh-test predictions against SPEI-12 drought, the benchmark the manuscript names.
scores/fresh_spei12_posthoc.json.

gridMET's 1-year SPEI is the 12-month SPEI. Phase 6 compared it with the FLDAS flag on development cells only (AUC
0.776, fourth of eight timescales; DECISIONS.md #77) and fetched it for no fresh-test region. This script fetches it for
the 144 fresh-test regions (phase6_spei.fetch: same server, centre point, month rule and cache,
data/phase6/spei/series/1y__*.npy) and runs fresh_labels_posthoc.py unchanged with 1y added to its timescales, so
SPEI-12 drought (SPEI <= -0.84) is scored on the same windows, with the same models, pairs and 2,000 whole-region
resamples as the other labels. The output goes to its own file; scores/fresh_labels_posthoc.json is not touched. When
the windows common to every label are those of fresh_labels_posthoc.json, its rows must come out identical here; the
check is written into the output. Not a second test of the FLDAS label.

    cd ~/Desktop/DEWSat_BiLSTM_v2 && source env.sh && .venv/bin/python -u code/dewsat2/fresh_spei12_posthoc.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import fresh_labels_posthoc as F
from common import SCORES, log, read_json, write_json

PUBLISHED = SCORES / "fresh_labels_posthoc.json"
TARGET = SCORES / "fresh_spei12_posthoc.json"


def main() -> dict:
    F.EXTRA_SPEI = (*F.EXTRA_SPEI, "1y")
    F.write_json = lambda path, obj: write_json(TARGET if Path(path) == PUBLISHED else path, obj)
    F.log = lambda msg: log(msg.replace(PUBLISHED.name, TARGET.name))
    before = PUBLISHED.read_bytes()
    F.run()
    if PUBLISHED.read_bytes() != before:
        raise RuntimeError(f"{PUBLISHED} changed")
    out, pub = read_json(TARGET), read_json(PUBLISHED)
    check = {}
    for job, e in out["jobs"].items():
        p = pub["jobs"][job]
        check[job] = {"common_windows": e["common_windows"], "published_common_windows": p["common_windows"],
                      "published_rows_identical": {k: e["labels"].get(k) == v for k, v in p["labels"].items()}}
    out["note"] = ("post-hoc, after the lock: SPEI-12 (gridMET 1y) added to the labels of fresh_labels_posthoc.py; "
                   "scores/fresh_labels_posthoc.json is unchanged")
    out["added_label"] = "spei_1y"
    out["check_against_fresh_labels_posthoc"] = check
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(TARGET, out)
    log(f"fresh_spei12_posthoc: wrote {TARGET}; check {check}")
    return out


if __name__ == "__main__":
    main()
