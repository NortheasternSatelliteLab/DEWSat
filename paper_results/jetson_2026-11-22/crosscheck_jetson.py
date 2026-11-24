#!/usr/bin/env python3
"""Independent checks of the Jetson numbers behind the paper tables (2026-11-22).

1. trtexec: re-reads every raw trtexec log the boards saved (results/logs/trtexec_*.log) and compares its own
   "Throughput" and "Latency ... median / percentile(99%)" lines, and the per-iteration times file's median, with the
   bench JSON the tables are built from. Also checks the command line in the log names the same engine and batch.
2. Soaks: recomputes the hottest sensor, loaded power and probe stability from the 10-second interval records
   instead of the runner's summary.
3. Energy: mJ per item = 1000 x W / items per second, recomputed for every row.

Reads no labels. Prints PASS/FAIL lines and writes CROSSCHECK.txt next to this file.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASES = {"v2": Path.home() / "Desktop/DEWSat_BiLSTM_v2/jetson_kit/results_from_jetson",
         "Prithvi": Path.home() / "Desktop/DEWSat_Prithvi_pilot/jetson_ft/results_from_jetson"}
lines, fails = [], 0


def say(s):
    print(s)
    lines.append(s)


def results_dirs(base):
    """Each fetched results/ folder (newest fetch last); skip partial copies."""
    return sorted(p for p in base.glob("*/results") if p.is_dir())


def close(a, b, rel=1e-3, abs_=1e-6):
    return a is not None and b is not None and abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


SEEN = set()    # (kind, tag, when): a later fetch repeats the earlier fetches' files; check each file once


def check_trtexec(kit, rdir):
    global fails
    n_ok = n_bad = n_missing = n_seen = 0
    for bf in sorted(rdir.glob("bench_*.json")):
        b = json.loads(bf.read_text())
        tag = b.get("tag")
        key = (kit, "bench", tag, b.get("when"))
        if key in SEEN:
            n_seen += 1
            continue
        SEEN.add(key)
        for e in b.get("trtexec") or []:
            if not e.get("throughput_qps"):
                continue
            v = e.get("variant", "default")
            stem = f"{e['engine']}_b{e['batch']}{'' if v == 'default' else '_' + v}_{tag}"
            log = rdir / "logs" / f"trtexec_{stem}.log"
            if not log.exists():
                n_missing += 1
                continue
            text = log.read_text(errors="ignore")
            cmd = text.splitlines()[0] if text else ""
            m_q = re.findall(r"Throughput: ([\d.]+) qps", text)
            m_l = re.findall(r"\] Latency: min = [\d.]+ ms, max = [\d.]+ ms, mean = [\d.]+ ms, median = ([\d.]+) ms.*?"
                             r"percentile\(99%\) = ([\d.]+) ms", text)
            problems = []
            if f"{e['engine']}.engine" not in cmd:
                problems.append("engine not in the command line")
            shp = re.search(r"--shapes=\S*?:(\d+)x", cmd)
            if not shp or int(shp.group(1)) != e["batch"]:
                problems.append(f"batch in the command line {shp.group(1) if shp else '?'} != {e['batch']}")
            if not m_q or not close(float(m_q[-1]), e["throughput_qps"]):
                problems.append(f"throughput log {m_q[-1] if m_q else '?'} vs json {e['throughput_qps']}")
            if not m_l or not close(float(m_l[-1][0]), e["latency_ms"].get("p50")):
                problems.append(f"median log {m_l[-1][0] if m_l else '?'} vs json p50 {e['latency_ms'].get('p50')}")
            if m_l and not close(float(m_l[-1][1]), e["latency_ms"].get("p99")):
                problems.append(f"p99 log {m_l[-1][1]} vs json {e['latency_ms'].get('p99')}")
            ft = ((e.get("from_times_file") or {}).get("latency_ms") or {}).get("p50")
            if ft is not None and not close(ft, e["latency_ms"].get("p50"), rel=0.03):
                problems.append(f"per-iteration times median {ft:.4f} vs trtexec median {e['latency_ms']['p50']:.4f}")
            items = e.get("windows_per_s") or e.get("items_per_s")
            if items and not close(items, e["throughput_qps"] * e["batch"]):
                problems.append(f"items/s {items} != qps x batch")
            if e.get("mean_w") and items:
                mj = 1000 * e["mean_w"] / items
                got = e.get("mj_per_window", e.get("mj_per_item"))
                if got is not None and not close(mj, got, rel=1e-9):
                    problems.append(f"energy {got} != 1000 W / items {mj}")
            if problems:
                n_bad += 1
                fails += 1
                say(f"  [FAIL] {kit} {bf.name} {e['label']}: " + "; ".join(problems))
            else:
                n_ok += 1
    say(f"  [{'PASS' if n_bad == 0 else 'FAIL'}] {kit} {rdir.parent.name}: {n_ok} trtexec rows match their raw logs"
        f"{f', {n_bad} do not' if n_bad else ''}{f', {n_missing} rows have no log here' if n_missing else ''}"
        f"{f' ({n_seen} benchmark files already checked in an earlier fetch)' if n_seen else ''}")


def check_soaks(rdir):
    global fails
    for sf in sorted(rdir.glob("soak_*.json")):
        s = json.loads(sf.read_text())
        key = ("v2", "soak", s.get("tag"), s.get("when"))
        if key in SEEN:
            continue
        SEEN.add(key)
        sm = s.get("summary") or {}
        iv = s.get("intervals") or []
        hot = max((max(i["temps_c"].values()) for i in iv if i.get("temps_c")), default=None)
        on = [i["power_w"] for i in iv if i.get("phase") == "on" and i.get("power_w") is not None]
        loaded = sum(on) / len(on) if on else None
        # off-phase intervals of a duty-cycle soak run no inference, so they carry no probe result (None)
        probed = [i["probe_identical_to_first"] for i in iv if i.get("probe_identical_to_first") is not None]
        probe_ok = all(probed)
        p = []
        if hot is not None and sm.get("hottest_c") is not None and hot > sm["hottest_c"] + 0.01:
            p.append(f"intervals reach {hot:.2f} C, summary says {sm['hottest_c']:.2f} C")
        if loaded is not None and sm.get("loaded_power_w_mean") is not None and not close(loaded, sm["loaded_power_w_mean"], rel=0.02):
            p.append(f"on-phase mean {loaded:.3f} W vs summary {sm['loaded_power_w_mean']:.3f} W")
        if not probe_ok and not s.get("probe_changes"):
            p.append("an interval's probe output changed but the summary says 0 changes")
        mins = (iv[-1]["t_s"] / 60) if iv else 0
        if s.get("minutes_requested") and mins < 0.95 * s["minutes_requested"]:
            p.append(f"intervals cover {mins:.1f} of {s['minutes_requested']} min")
        if s.get("errors"):
            p.append(f"{s['errors']} errors")
        if p:
            fails += 1
        say(f"  [{'FAIL' if p else 'PASS'}] soak {s.get('tag')}: hottest {hot:.1f} C from {len(iv)} intervals "
            f"(summary {sm.get('hottest_c')}), on-phase power {loaded:.2f} W (summary {sm.get('loaded_power_w_mean'):.2f} W), "
            f"probe identical to the first in all {len(probed)} intervals that ran it: {probe_ok}"
            + (f"; {'; '.join(p)}" if p else ""))


def main():
    for kit, base in BASES.items():
        dirs = results_dirs(base)
        if not dirs:
            say(f"{kit}: no fetched results yet")
            continue
        say(f"{kit}:")
        for d in dirs:
            check_trtexec(kit, d)
            if kit == "v2":
                check_soaks(d)
    say(f"{'ALL PASS' if fails == 0 else f'{fails} FAIL'}")
    (OUT / "CROSSCHECK.txt").write_text("\n".join(lines) + "\n")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
