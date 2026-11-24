#!/usr/bin/env python3
"""Paper tables from the Jetson Orin Nano runs of 2026-11-21 and 2026-11-22.

Two kits ran on the same board (Jetson Orin Nano Engineering Reference Developer Kit Super, 8 GB, JetPack 6.2.1):
  v2       the paper's model: v2's plain Bi-LSTM, ten-seed ensemble (DEWSat_BiLSTM_v2/jetson_kit)
  Prithvi  the Prithvi pilot's fresh-test model: O3's GRU on Prithvi anomaly features, 5 seeds, with the frozen
           Prithvi-EO-2.0-tiny encoder (DEWSat_Prithvi_pilot/jetson_ft)

Reads only what the boards wrote (bench_*.json, soak_*.json, ort/trt/build .json, board_intervals.json computed on the
Mac from the board's saved predictions, boot_times.txt). Reads no labels and computes no accuracy: the accuracy columns
are the board runner's own parity checks, copied. Every number is rounded once, here.

    python3 make_jetson_tables.py        # writes the CSVs and jetson_tables.html next to this file
"""
from __future__ import annotations

import csv
import html
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent
V2_BASE = Path.home() / "Desktop/DEWSat_BiLSTM_v2/jetson_kit/results_from_jetson"
FT_BASE = Path.home() / "Desktop/DEWSat_Prithvi_pilot/jetson_ft/results_from_jetson"
TRIP_C = 70.0          # passive trip of the cpu/gpu/soc thermal zones on this board (soak trip_points); also the paper's limit
MODE_ORDER = {"7W": 0, "15W": 1, "25W": 2, "MAXN_SUPER": 3}
CLOCK_NAMES = {"dvfs": "dynamic (DVFS)", "pinned": "pinned (jetson_clocks)"}
NOTES = []             # problems found while building; printed at the end and written to BUILD_NOTES.txt


# ------------------------------------------------------------------------------------------------ small helpers

def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception as e:  # noqa: BLE001
        NOTES.append(f"could not read {p}: {type(e).__name__}: {e}")
        return None


def files(base, pattern):
    """Result files under base (any fetched date), skipping partial copies and the engine layer reports."""
    if not base.exists():
        return []
    return sorted(p for p in base.rglob(pattern)
                  if ".partial" not in str(p) and ".truncated" not in str(p) and "engines" not in p.parts
                  and "logs" not in p.parts)


def newest_by(items, key):
    """One item per key: the one with the latest 'when' (the same file can come back in two fetches)."""
    best = {}
    for path, d in items:
        k = key(d)
        if k not in best or str(d.get("when", "")) > str(best[k][1].get("when", "")):
            best[k] = (path, d)
    return [best[k] for k in best]


def r(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    return f"{x:.{nd}f}"


def sig(x, n=3):
    """n significant figures, never scientific notation."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    if x == 0:
        return "0"
    nd = max(n - 1 - int(math.floor(math.log10(abs(x)))), 0)
    return f"{x:.{nd}f}"


def sci(x):
    return "" if x is None else f"{x:.1e}"


def write_csv(name, header, rows, title):
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    TABLES.append((name, title, header, rows))
    print(f"  wrote {name}: {len(rows)} rows")


TABLES = []
SOAKS = []
BENCH_OC = {}          # (kit, tag) -> over-current protection events counted during that benchmark
BOOT_AGE = {}          # (kit, tag) -> minutes from the board's boot to the start of that benchmark (its idle reading)
RESTED = []            # benchmarks whose session logged a rest of 4+ min right before them (their idle counts as rested)
BUDGET_IDLE = {}       # (mode, clocks) -> (watts, which reading): rested where measured, else the benchmark's start
FT_OK = {}             # Prithvi parity on the board: ("trt", engine) or ("ort", graph, precision) -> (status, reasons)
_SESSIONS = {}
_RESTS = {}


def _naive(ts):
    try:
        return datetime.strptime(str(ts)[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def session_starts(base):
    """(session start, board boot) pairs from the board's session logs ('== <t> EST session|step X start; ...;
    up since <boot>'). All times are the board's local time."""
    if base not in _SESSIONS:
        pat = re.compile(r"^== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \S+ (?:session|step) \S+ start; .*up since "
                         r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", re.M)
        seen = set()
        for p in files(base, "*.log"):
            for m in pat.finditer(p.read_text(errors="ignore")):
                seen.add((_naive(m.group(1)), _naive(m.group(2))))
        _SESSIONS[base] = sorted(x for x in seen if x[0] and x[1])
    return _SESSIONS[base]


def session_rests(base):
    """tag -> seconds of rest that a session log records right before that benchmark's command line."""
    if base not in _RESTS:
        out = {}
        for p in files(base, "*.log"):
            eq = [ln for ln in p.read_text(errors="ignore").splitlines() if ln.startswith("== ")]
            for i, ln in enumerate(eq):
                m = re.search(r"\bbench\b.*--tag (\S+)", ln)
                mr = re.search(r"rest (\d+) s first", eq[i - 1]) if (m and i > 0) else None
                if mr:
                    out[m.group(1)] = int(mr.group(1))
        _RESTS[base] = out
    return _RESTS[base]


def bench_context(kit, b):
    """Note a benchmark's over-current events, how long after boot its idle reading was taken, and any logged rest."""
    tag = b.get("tag")
    rest_s = session_rests(V2_BASE if kit == "v2" else FT_BASE).get(tag)
    idle = ((b.get("idle") or {}).get("total_w") or {}).get("mean")
    if rest_s and rest_s >= 240 and idle is not None:
        mode, clocks = (mode_clock_v2 if kit == "v2" else mode_clock_ft)(b)
        RESTED.append({"kit": kit, "tag": tag, "mode": mode, "clocks": clocks, "idle_w": idle, "rest_s": rest_s})
    oc = b.get("over_current_events") or {}
    BENCH_OC[(kit, tag)] = sum(v for v in oc.values() if isinstance(v, (int, float)))
    when = _naive(b.get("when"))
    prior = [x for x in session_starts(V2_BASE if kit == "v2" else FT_BASE) if when and x[0] <= when]
    if prior:
        BOOT_AGE[(kit, tag)] = (when - prior[-1][1]).total_seconds() / 60


def rest_sources(mode, clocks, kits=("v2",)):
    """Idle readings taken with no load for the 4-10 min before: each soak's baseline (after a 10-min cool-down), F11's
    240-s rest phases, and benchmarks that began after a logged rest. [(watts, where)], same mode and clocks."""
    out = []
    for s in SOAKS:
        if s["mode"] == mode and s["clocks"] == clocks:
            out += [(v, src) for v, src in ((s["idle_w"], f"{s['name']} baseline"),
                                            (s["off_w"], f"{s['name']} 240-s rest phases")) if v is not None]
    for x in RESTED:
        if x["kit"] in kits and x["mode"] == mode and x["clocks"] == clocks:
            out.append((x["idle_w"], f"{'v2' if x['kit'] == 'v2' else 'Prithvi'} benchmark {x['tag']} after a "
                                     f"{x['rest_s'] // 60}-min rest"))
    return out


def rest_idle(mode, clocks, kits=("v2",)):
    return [v for v, _ in rest_sources(mode, clocks, kits)]


def idle_for(x):
    """The budget idle of a benchmark row's mode and clocks (falls back to that benchmark's own idle reading)."""
    return BUDGET_IDLE.get((x["mode"], x["clocks"]), (x["idle_w"], ""))[0]


def ft_match(x):
    """Did this Prithvi row's runtime and file give the reference outputs on the board (the board's own parity check)?"""
    st = FT_OK.get(("ort", x["family"], x["precision"]) if x["runtime"] == "ONNX Runtime CPU" else ("trt", x["engine"]))
    if not st or not st[0]:
        return ""
    if st[0] == "PASS":
        return "yes"
    if st[0] == "WARN":
        return "yes (warned: FP16 rounding above the warning limit, below the fail limit)"
    return "NO" + (f": {st[1]}" if st[1] else "")


def both_match(enc, head):
    a, b = ft_match(enc), ft_match(head)
    if a.startswith("yes") and b.startswith("yes"):
        return "yes"
    bad = [n for n, m in (("encoder", a), ("head", b)) if not m.startswith("yes")]
    return "NO: the " + " and the ".join(bad) + " engines gave wrong outputs on the board (P2)"


def rest_txt(vals):
    if not vals:
        return ""
    return r(vals[0], 2) if len(vals) == 1 else f"{r(min(vals), 2)}-{r(max(vals), 2)}"


def soak_why(s, sm, hot, oc_n):
    """The board runner's WARN/FAIL rules for a soak, in words (empty when it passed)."""
    if s.get("status") == "PASS":
        return ""
    why = []
    if s.get("errors"):
        why.append(f"{s['errors']} errors")
    if s.get("probe_changes"):
        why.append(f"probe output changed {s['probe_changes']} times")
    if s.get("stopped"):
        why.append(f"stopped at {(s['stopped'] or {}).get('temp_c')} C")
    if (sm.get("p50_drift_pct") or 0) > 25:
        why.append("p50 drift over 25%")
    if (sm.get("rss_growth_mb") or 0) > 64:
        why.append("memory grew over 64 MB")
    if hot is not None and s.get("stop_at_c") is not None and hot >= s["stop_at_c"]:
        why.append(f"reached the {s['stop_at_c']:g} C stop line")
    if oc_n:
        why.append(f"the module's over-current protection fired {oc_n} times")
    if s.get("board_problems"):
        why.append("board state: " + "; ".join(map(str, s["board_problems"])))
    if s.get("interrupted"):
        why.append("interrupted")
    return "; ".join(why) or "see the board's SUMMARY.md"


def mode_clock_v2(b):
    pm = (b.get("power_mode") or {}).get("name")
    pinned = bool((b.get("clocks_before") or {}).get("pinned"))
    tag = b.get("tag", "")
    if pm is None:
        pm = tag.rsplit("_", 1)[0]
    if tag.endswith("_pinned") != pinned:
        NOTES.append(f"v2 bench {tag}: tag says {'pinned' if tag.endswith('_pinned') else 'dvfs'}, clocks say pinned={pinned}")
    return pm, ("pinned" if pinned else "dvfs")


def mode_clock_ft(b):
    bs = b.get("board_state") or {}
    pm = bs.get("power_mode") or b.get("tag", "").rsplit("_", 1)[0]
    pinned = bool((bs.get("clocks") or {}).get("pinned"))
    return pm, ("pinned" if pinned else "dvfs")


def mode_key(mode, clocks):
    return (MODE_ORDER.get(mode, 9), 0 if clocks == "dvfs" else 1)


def hottest(temps):
    if not temps:
        return None, None
    k = max(temps, key=lambda z: temps[z])
    return temps[k], k


# ------------------------------------------------------------------------------------------------ benchmark rows

def bench_rows(path, b, kit):
    """Every timed row of one bench file, normalised: one dict per (runtime, engine/model, batch, variant)."""
    mode, clocks = (mode_clock_v2 if kit == "v2" else mode_clock_ft)(b)
    idle = ((b.get("idle") or {}).get("total_w") or {}).get("mean")
    out = []
    for bucket, runtime in (("trtexec", "TensorRT (trtexec)"), ("trt_python", "TensorRT (Python)"),
                            ("ort", "ONNX Runtime CPU")):
        for e in b.get(bucket) or []:
            lat, gpu = e.get("latency_ms") or {}, e.get("gpu_compute_ms") or {}
            pw = e.get("power") or {}
            tw = pw.get("total_w") or {}
            if kit == "v2":
                item = "window"
                ips = e.get("windows_per_s")
                if bucket == "ort":
                    family, precision = e["model"].rsplit("_", 1)
                    profile, engine = "-", e["model"]
                else:
                    family, precision, profile, engine = e["family"], e["precision"], e["profile"], e["engine"]
            else:
                item = e.get("item")
                ips = e.get("items_per_s")
                family, precision = e.get("graph"), e.get("precision")
                profile = e.get("profile", "-") if bucket != "ort" else "-"
                engine = e.get("engine") or f"{family}_{precision}"
            mean_w = e.get("mean_w")
            mj = 1000.0 * mean_w / ips if (mean_w and ips) else None
            marg = 1000.0 * max(mean_w - idle, 0.0) / ips if (mean_w and ips and idle is not None) else None
            stored = e.get("mj_per_window") if kit == "v2" else e.get("mj_per_item")
            if mj is not None and stored is not None and abs(mj - stored) > 1e-9 * max(1.0, abs(stored)):
                NOTES.append(f"{path.name} {e.get('label')}: energy recomputed {mj} != stored {stored}")
            t_hot, z_hot = hottest(pw.get("max_temp_c") or {})
            mem = e.get("memory") or {}
            out.append({
                "kit": kit, "file": str(path), "when": b.get("when"), "tag": b.get("tag"), "mode": mode, "clocks": clocks,
                "runtime": runtime, "variant": e.get("variant", "default") if bucket == "trtexec" else "-",
                "family": family, "precision": precision, "profile": profile, "engine": engine,
                "threads": e.get("threads", "") if bucket == "ort" else "", "batch": e.get("batch"), "item": item,
                "p50": lat.get("p50"), "p99": lat.get("p99"), "mean": lat.get("mean"), "gpu_p50": gpu.get("p50"),
                "ips": ips, "mean_w": mean_w, "max_w": tw.get("max"), "idle_w": idle, "mj": mj, "marg_mj": marg,
                "hot_c": t_hot, "hot_zone": z_hot, "label": e.get("label"),
                "engine_mb": (mem.get("engine_file_bytes") or 0) / 1e6 if mem.get("engine_file_bytes") else None,
                "device_mem_mb": (mem.get("device_memory_bytes") or 0) / 1e6 if mem.get("device_memory_bytes") else None,
                "rss_mb": e.get("rss_mb"),
            })
    return out


def find(rows, **kw):
    hits = [x for x in rows if all(x.get(k) == v for k, v in kw.items())]
    return hits[0] if len(hits) == 1 else (hits[0] if hits else None)


def layer_precision_v2(builds):
    """engine -> 'FP16 (30 of 33 layers)' etc., from build.json's layer reports (what TensorRT actually chose)."""
    out = {}
    for name, e in builds.items():
        lay = (e or {}).get("layers") or {}
        bp = lay.get("by_output_precision") or {}
        if not lay.get("available"):
            continue
        n = lay.get("layers")
        parts = [f"{k.upper()} {bp[k]}" for k in ("int8", "fp16", "fp32") if bp.get(k)]
        main = max(("int8", "fp16", "fp32"), key=lambda k: bp.get(k, 0))
        out[name] = {"text": f"{', '.join(parts)} of {n} layers", "fp16": bp.get("fp16", 0), "int8": bp.get("int8", 0),
                     "main": main.upper(), "bytes": e.get("engine_bytes"), "seconds": e.get("seconds")}
    return out


# ------------------------------------------------------------------------------------------------ v2

def v2_tables():
    print("v2 (plain Bi-LSTM, the paper's model)")
    benches = newest_by([(p, d) for p in files(V2_BASE, "bench_*.json") if (d := load(p))], key=lambda d: d.get("tag"))
    rows = []
    for p, b in benches:
        rows += bench_rows(p, b, "v2")
        bench_context("v2", b)
    builds = {}
    for p in files(V2_BASE, "build.json"):
        d = load(p) or {}
        builds.update(d.get("engines") or {})
    lp = layer_precision_v2(builds)
    runs = sorted({(x["mode"], x["clocks"], x["tag"]) for x in rows}, key=lambda t: mode_key(t[0], t[1]))
    print("  benchmark runs:", ", ".join(t[2] for t in runs))

    # ---- J1: the paper model vs one copy, per window / region / tile, every power mode (trtexec, default timing)
    units = [("b1", 1, "one window (batch-1 engine)"), ("dyn", 25, "one region: 25 windows (5 x 5 cells)"),
             ("dyn", 100, "one tile: 100 windows (about one HLS tile)")]
    models = [("fused10", "Bi-LSTM, ten-seed ensemble (paper model)"), ("seed42", "Bi-LSTM, one seed (one copy)")]
    j1 = []
    for mode, clocks, tag in runs:
        for fam, mname in models:
            for prof, batch, unit in units:
                eng = f"{fam}_fp16_{prof}"
                x = find(rows, tag=tag, runtime="TensorRT (trtexec)", variant="default", engine=eng, batch=batch)
                py = find(rows, tag=tag, runtime="TensorRT (Python)", engine=eng, batch=batch)
                if not x:
                    continue
                L = lp.get(eng, {})
                j1.append([mode, CLOCK_NAMES[clocks], mname, unit, batch, eng, L.get("text", ""),
                           r(x["p50"]), r(x["p99"]), r(x["gpu_p50"]), r(py["p50"]) if py else "",
                           f"{x['ips']:.0f}", r(x["mean_w"], 2), r(x["max_w"], 2), r(x["idle_w"], 2),
                           sig(x["mj"] * batch), sig(x["marg_mj"] * batch), r(x["hot_c"], 1)])
    write_csv("J1_v2_speed_power_energy.csv",
              ["Power mode", "Clocks", "Model", "Unit", "Windows", "TensorRT engine", "Layers TensorRT ran in",
               "Latency p50 (ms)", "Latency p99 (ms)", "GPU compute p50 (ms)", "Latency from Python p50 (ms)",
               "Throughput (windows/s)", "Module power while running (W)", "Peak module power sample (W)",
               "Idle module power (W)", "Energy per unit (mJ)", "Energy per unit above idle (mJ)",
               "Hottest sensor during the run (C)"], j1,
              "Speed, power and energy of the paper model (ten seeds) and one copy, every power mode")

    # ---- J0: one line per power mode for the power budget
    j0 = []
    for mode, clocks, tag in runs:
        tile = find(rows, tag=tag, runtime="TensorRT (trtexec)", variant="default", engine="fused10_fp16_dyn", batch=100)
        reg = find(rows, tag=tag, runtime="TensorRT (trtexec)", variant="default", engine="fused10_fp16_dyn", batch=25)
        win = find(rows, tag=tag, runtime="TensorRT (trtexec)", variant="default", engine="fused10_fp16_b1", batch=1)
        one = find(rows, tag=tag, runtime="TensorRT (trtexec)", variant="default", engine="seed42_fp16_dyn", batch=100)
        mine = [x for x in rows if x["tag"] == tag and x["mean_w"]]
        top = max(mine, key=lambda x: x["mean_w"]) if mine else None
        peak = max((x["max_w"] for x in mine if x["max_w"]), default=None)
        hot = max((x["hot_c"] for x in rows if x["tag"] == tag and x["hot_c"]), default=None)
        age = BOOT_AGE.get(("v2", tag))
        rv = rest_idle(mode, clocks)
        if rv:
            BUDGET_IDLE[(mode, clocks)] = (sum(rv) / len(rv), f"at rest, mean of {len(rv)} reading{'s' if len(rv) != 1 else ''}")
        elif rows_idle(rows, tag) is not None:
            BUDGET_IDLE[(mode, clocks)] = (rows_idle(rows, tag), "start of the benchmark, " + (
                f"{age:.0f} min after boot" if age is not None else "time since boot not logged"))
        bi = BUDGET_IDLE.get((mode, clocks), (None, ""))
        above_rest = (1000.0 * (tile["mean_w"] - sum(rv) / len(rv)) / tile["ips"] * 100
                      if (tile and rv and tile["ips"] and tile["mean_w"]) else None)
        j0.append([mode, CLOCK_NAMES[clocks], r(rows_idle(rows, tag), 2), r(age, 0) if age is not None else "",
                   rest_txt(rv), r(bi[0], 2), bi[1], sig(above_rest) if above_rest is not None else "",
                   r(win["p50"]) if win else "", r(reg["p50"]) if reg else "", r(tile["p50"]) if tile else "",
                   r(tile["mean_w"], 2) if tile else "", sig(tile["mj"] * 100) if tile else "",
                   sig(tile["marg_mj"] * 100) if tile else "",
                   r(one["p50"]) if one else "", sig(one["mj"] * 100) if one else "",
                   r(top["mean_w"], 2) if top else "", (top["label"] or "") if top else "", r(peak, 2), r(hot, 1)])
    write_csv("J0_v2_power_budget_by_mode.csv",
              ["Power mode", "Clocks", "Idle module power at the start of the benchmark (W)",
               "Minutes after boot of that idle reading", "Idle at rest, no load for the 4-10 min before (W)",
               "Idle for the budget (W)", "Which idle reading that is",
               "Paper model, energy per tile above the rested idle (mJ)",
               "Paper model, one window p50 (ms)",
               "Paper model, one region (25 windows) p50 (ms)", "Paper model, one tile (100 windows) p50 (ms)",
               "Paper model, module power during tiles (W)", "Paper model, energy per tile (mJ)",
               "Paper model, energy per tile above idle (mJ)", "One copy, one tile p50 (ms)",
               "One copy, energy per tile (mJ)", "Highest mean module power in the benchmark (W)",
               "Which benchmark row drew it", "Highest module power sample (W)",
               "Hottest sensor in the benchmark (C)"], j0,
              "Power budget per power mode (paper model on TensorRT; every row in J1/J5)")

    # ---- J5: every benchmark row
    j5 = []
    order = {"TensorRT (trtexec)": 0, "TensorRT (Python)": 1, "ONNX Runtime CPU": 2}
    for x in sorted(rows, key=lambda x: (mode_key(x["mode"], x["clocks"]), order[x["runtime"]], x["engine"],
                                         str(x["threads"]), x["batch"] or 0, x["variant"])):
        L = lp.get(x["engine"], {})
        j5.append([x["mode"], CLOCK_NAMES[x["clocks"]], x["runtime"], x["variant"], x["engine"], x["family"],
                   x["precision"], L.get("text", "") if x["runtime"] != "ONNX Runtime CPU" else "FP32 (CPU)",
                   x["threads"], x["batch"], r(x["p50"]), r(x["p99"]), r(x["mean"]), r(x["gpu_p50"]),
                   f"{x['ips']:.1f}" if x["ips"] else "", r(x["mean_w"], 2), r(x["max_w"], 2), r(x["idle_w"], 2),
                   sig(x["mj"]), sig(x["marg_mj"]), sig(x["mj"] * x["batch"]) if x["mj"] else "", r(x["hot_c"], 1),
                   x["tag"], (x["when"] or "")[:19]])
    write_csv("J5_v2_all_benchmark_rows.csv",
              ["Power mode", "Clocks", "Runtime", "Timing variant", "Engine or model file", "Model family",
               "Precision asked for", "Precision TensorRT ran", "CPU threads", "Batch (windows)", "Latency p50 (ms)",
               "Latency p99 (ms)", "Latency mean (ms)", "GPU compute p50 (ms)", "Throughput (windows/s)",
               "Module power while running (W)", "Peak module power sample (W)", "Idle module power (W)",
               "Energy per window (mJ)", "Energy per window above idle (mJ)", "Energy per batch (mJ)",
               "Hottest sensor during the row (C)", "Run tag", "Measured at"], j5,
              "Every v2 benchmark row (all runtimes, engines, batches and power modes)")

    # ---- J6: engines on the board
    j6 = []
    for name in sorted(lp):
        L = lp[name]
        j6.append([name, L["text"], sig((L["bytes"] or 0) / 1e6), r(L["seconds"], 1)])
    write_csv("J6_v2_tensorrt_engines.csv",
              ["Engine", "Layers by precision TensorRT chose", "Engine file (MB)", "Build time on the board (s)"], j6,
              "TensorRT engines built on the board (size, build time, the precision TensorRT actually used)")
    return rows


def rows_idle(rows, tag):
    xs = [x["idle_w"] for x in rows if x["tag"] == tag and x["idle_w"] is not None]
    return xs[0] if xs else None


def soak_tables():
    print("v2 soaks (heat)")
    soaks = newest_by([(p, d) for p in files(V2_BASE, "soak_*.json") if (d := load(p))], key=lambda d: d.get("tag"))
    step = {"flight15w": "F10", "duty15w": "F11", "worstcase": "F12"}
    out = []
    SOAKS.clear()
    for p, s in sorted(soaks, key=lambda t: str(t[1].get("when"))):
        sm = s.get("summary") or {}
        ivs = s.get("intervals") or []
        on = [i["power_w"] for i in ivs if i.get("phase") == "on" and i.get("power_w") is not None]
        off = [i["power_w"] for i in ivs if i.get("phase") == "off" and i.get("power_w") is not None]
        allp = [i["power_w"] for i in ivs if i.get("power_w") is not None]
        mode = (s.get("power_mode") or {}).get("name")
        clocks = s.get("clocks_expected") or ("pinned" if (s.get("clocks_start") or {}).get("pinned") else "dvfs")
        hot_c, hot_z = sm.get("hottest_c"), None
        if sm.get("max_temp_c"):
            hot_c, hot_z = hottest(sm["max_temp_c"])
        duty = s.get("duty")
        load_txt = f"TensorRT {s.get('what', '').replace('TensorRT ', '')}, batch {s.get('batch')}, " + \
                   (f"duty {duty} s on/off" if duty else "continuous")
        idle = ((s.get("idle") or {}).get("total_w") or {}).get("mean")
        pre = s.get("preset")
        name = f"{step.get(pre, 'first run')} {pre}" if pre else f"first run ({s.get('tag')})"
        fan = sm.get("fan_pwm_min_max") or ["", ""]
        rpm = sm.get("fan_rpm_min_max") or ["", ""]
        gpu = sm.get("gpu_mhz_min_max") or ["", ""]
        oc = s.get("over_current_events") or {}
        oc_n = sum(v for v in oc.values() if isinstance(v, (int, float)))
        oc_txt = f"{oc_n} ({', '.join(f'{k} {v}' for k, v in oc.items())})" if oc_n else "0"
        why = soak_why(s, sm, hot_c, oc_n)
        SOAKS.append({"name": name, "preset": pre, "mode": mode, "clocks": clocks, "hot_c": hot_c, "hot_zone": hot_z,
                      "loaded_w": sm.get("loaded_power_w_mean"), "idle_w": idle,
                      "mean_w": sum(allp) / len(allp) if allp else None,
                      "on_w": sum(on) / len(on) if (on and duty) else None,
                      "off_w": sum(off) / len(off) if (off and duty) else None, "minutes": s.get("minutes_run"),
                      "windows": s.get("windows"), "errors": s.get("errors"), "probe_changes": s.get("probe_changes"),
                      "status": s.get("status"), "batch": s.get("batch"), "duty": duty,
                      "p50_drift_pct": sm.get("p50_drift_pct"), "oc": oc_n, "oc_txt": oc_txt, "why": why})
        out.append([name, mode, CLOCK_NAMES.get(clocks, clocks), load_txt, r(s.get("minutes_run"), 1),
                    f"{s.get('windows'):,}" if s.get("windows") is not None else "", s.get("errors"),
                    s.get("probe_changes"), r(sm.get("p50_first_ms")), r(sm.get("p50_last_ms")),
                    r(sm.get("p50_drift_pct"), 2), r(sm.get("p99_worst_ms")), r(idle, 2),
                    r(sm.get("loaded_power_w_mean"), 2), r(sum(on) / len(on), 2) if (on and duty) else "",
                    r(sum(off) / len(off), 2) if (off and duty) else "", r(sum(allp) / len(allp), 2) if allp else "",
                    r(hot_c, 1), hot_z or "", r(TRIP_C - hot_c, 1) if hot_c is not None else "",
                    f"{fan[0]}-{fan[1]}", f"{rpm[0]}-{rpm[1]}", f"{r(gpu[0], 0)}-{r(gpu[1], 0)}",
                    r(sm.get("rss_growth_mb"), 1), oc_txt, s.get("status"), why, (s.get("when") or "")[:19]])
    # minute-by-minute record of every soak (10 s intervals), for plotting heat and power over time
    ts = []
    zones = ["cpu-thermal", "gpu-thermal", "soc0-thermal", "soc1-thermal", "soc2-thermal", "tj-thermal"]
    for p, s in sorted(soaks, key=lambda t: str(t[1].get("when"))):
        pre = s.get("preset")
        name = f"{step.get(pre, 'first run')} {pre}" if pre else f"first run ({s.get('tag')})"
        for i in s.get("intervals") or []:
            tc = i.get("temps_c") or {}
            lat = i.get("latency_ms") or {}
            fan = ((i.get("fan") or {}).get("pwm") or {})
            ts.append([name, (s.get("power_mode") or {}).get("name"), s.get("clocks_expected") or "",
                       r(i.get("t_s") / 60, 3), i.get("phase"), r(i.get("power_w"), 3),
                       r(max(tc.values()), 3) if tc else ""] + [r(tc.get(z), 3) for z in zones] +
                      [next(iter(fan.values()), ""), r(i.get("gpu_mhz"), 0), r(lat.get("p50")), r(lat.get("p99")),
                       i.get("calls", ""), r(i.get("rss_mb"), 1), i.get("mem_available_mb", ""),
                       {True: "yes", False: "no"}.get(i.get("probe_identical_to_first"), "")])
    with (OUT / "J8_v2_soak_timeseries.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Test", "Power mode", "Clocks", "Minutes", "Phase", "Module power (W)", "Hottest sensor (C)"]
                   + [f"{z} (C)" for z in zones] + ["Fan PWM (0-255)", "GPU clock (MHz)", "Latency p50 (ms)",
                                                    "Latency p99 (ms)", "Batches run in the interval",
                                                    "Process memory (MB)", "Free memory (MB)",
                                                    "Probe output identical to the first"])
        w.writerows(ts)
    print(f"  wrote J8_v2_soak_timeseries.csv: {len(ts)} rows (not in the HTML page)")
    write_csv("J2_v2_heat_soaks.csv",
              ["Test", "Power mode", "Clocks", "Load", "Minutes", "Windows processed", "Errors",
               "Times the probe output changed", "Latency p50 at start (ms)", "Latency p50 at end (ms)",
               "p50 drift (%)", "Worst p99 (ms)", "Idle module power before (W)",
               "Module power while loaded (W)", "Mean power in on-phases (W)", "Mean power in off-phases (W)",
               "Mean power over the whole test (W)", "Hottest sensor (C)", "Which sensor",
               "Margin to the 70 C passive trip (C)", "Fan PWM range", "Fan RPM range", "GPU clock range (MHz)",
               "Memory growth (MB)", "Over-current events (the module's protection)", "Status",
               "Why not PASS", "Started at"], out,
              "Heat: sustained-load soaks (room air, stock fan)")
    return soaks


def parity_v2():
    print("v2 parity (deployment checks; copied from the board's own results)")
    out = []
    ort = newest_by([(p, d) for p in files(V2_BASE, "ort.json") if (d := load(p))], key=lambda d: "ort")
    trts = newest_by([(p, d) for p in files(V2_BASE, "trt.json") + files(V2_BASE, "trt_*.json") if (d := load(p))],
                     key=lambda d: d.get("when"))
    bi = {}
    for p in files(V2_BASE, "board_intervals.json"):
        bi = (load(p) or {}).get("paths") or bi

    def ci(v):
        return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" if v else ""

    def cis(v):
        return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" if v else ""

    if ort:
        o = ort[0][1]
        st = o.get("v2_stored") or {}
        stored_auc = (st.get("auc") or [None])[0]
        out.append(["v2 stored result (Mac, PyTorch)", "ten-seed mean", "FP32", "-", ci(st.get("auc")), "",
                    cis((st.get("minus_lr") or {}).get("all")), r((st.get("brier") or [None])[0]), r(st.get("ece")),
                    "", "", "", "reference"])
        for name, m in (o.get("models") or {}).items():
            out.append(parity_row_v2("ONNX Runtime CPU (board)", name, m, bi.get(name), stored_auc))
    else:
        stored_auc = None
    for p, t in sorted(trts, key=lambda pt: str(pt[1].get("when"))):
        tag = "TensorRT (board)" if p.name == "trt.json" else "TensorRT after a reboot (board, F13)"
        for name, e in (t.get("engines") or {}).items():
            out.append(parity_row_v2(tag, name, e, bi.get(name), stored_auc))
    write_csv("J3_v2_board_parity.csv",
              ["Runtime", "Model or engine", "Precision", "Compared with",
               "AUC on the 65,170 fresh-test windows [95% CI]", "AUC minus the stored FP32 AUC",
               "Bi-LSTM minus LR on VCI(T) [95% CI]", "Brier score", "ECE",
               "AUC difference from that reference [paired 95% CI]",
               "Largest probability difference from that reference", "Same output on a rerun", "Status"], out,
              "Does the board give the paper's answers? (deployment parity; the fresh test was scored once on the Mac)")


PREC_NAMES = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8 (Q/DQ, FP16 elsewhere)",
              "int8s": "INT8 strict (Q/DQ, FP32 elsewhere)"}


def parity_row_v2(runtime, name, e, interval, stored_auc):
    fam = name.split("_")[0]
    prec_key = name.split("_")[1] if "_" in name else ""
    prec = PREC_NAMES.get(prec_key, prec_key.upper())
    if runtime.startswith("ONNX") and prec_key == "int8":
        prec = "INT8 (Q/DQ file, FP32 elsewhere on the CPU)"
    if e.get("vs_mac_fp32") is not None:
        vs, against = e["vs_mac_fp32"], "Mac FP32 reference (ONNX)"
    else:
        vs, against = e.get("vs_mac_ort") or {}, "Mac: the same ONNX file on ONNX Runtime"
    t = e.get("test") if e.get("metrics") and fam != "seed42" else None   # never an AUC for one seed
    d = e.get("delta_vs_mac_fp32") or e.get("delta_vs_mac") or {}
    pc = (e.get("paired_vs_mac_fp32") or {}).get("auc_ci95")
    vs_fp32 = ""
    if t and stored_auc is not None:
        dd = t["auc"] - stored_auc
        vs_fp32 = "0.0000" if abs(dd) < 5e-5 else f"{dd:+.4f}"
    auc = ""
    lr = ""
    if t:
        auc = f"{t['auc']:.3f}"
        lr = f"{(e.get('minus_lr') or {}).get('auc', float('nan')):+.3f}"
        if interval:
            auc = f"{interval['auc']['all'][0]:.3f} [{interval['auc']['all'][1]:.3f}, {interval['auc']['all'][2]:.3f}]"
            m = interval["minus_lr"]["all"]
            lr = f"{m[0]:+.3f} [{m[1]:+.3f}, {m[2]:+.3f}]"
    dauc = ""
    if t and d.get("auc") is not None:
        dauc = f"{d['auc']:+.1e}" + (f" [{pc[0]:+.1e}, {pc[1]:+.1e}]" if pc else "")
    note = "" if t else ("one seed: output parity only, no accuracy" if fam == "seed42" else "batch-1 engine: parity only")
    return [runtime, name, prec, against, auc or note, vs_fp32, lr, r(t["brier"]) if t else "",
            r(t["ece"]) if t else "", dauc, sci(vs.get("max_abs")),
            {True: "yes", False: "no"}.get(e.get("deterministic_rerun"), ""), e.get("status")]


def boot_table():
    print("boot")
    out = []
    for p in files(V2_BASE, "boot_times.txt"):
        text = p.read_text(errors="ignore")
        blocks = re.split(r"^== ", text, flags=re.M)
        for blk in blocks:
            m = re.match(r"(\S+ \S+ \S+): reboot command to SSH (\d+) s", blk)
            if not m:
                continue
            s = re.search(r"Startup finished in (.+?) = ([\d.]+)s", blk)
            g = re.search(r"graphical\.target reached after ([\d.]+)s", blk) or \
                re.search(r"multi-user\.target reached after ([\d.]+)s", blk)
            out.append([m.group(1), m.group(2), s.group(2) if s else "", s.group(1) if s else "",
                        g.group(1) if g else ""])
    seen, rows = set(), []
    for x in out:
        if x[0] not in seen:
            seen.add(x[0])
            rows.append(x)
    write_csv("J4_boot_time.csv",
              ["Reboot at", "Reboot command to SSH login (s, measured on the Mac)", "systemd boot total (s)",
               "systemd breakdown", "Target reached after (s)"], rows,
              "Boot time (each reboot of the overnight run)")
    return rows


# ------------------------------------------------------------------------------------------------ Prithvi

FT_GRAPH = {"encoder": "Prithvi-EO-2.0-tiny encoder (frozen)", "gru_o3": "GRU head, 5 seeds (fresh-test model)",
            "bilstm_o3_exploratory": "Bi-LSTM head (exploratory, no verdict)"}


def ft_tables():
    print("Prithvi fresh-test model")
    benches = newest_by([(p, d) for p in files(FT_BASE, "bench_*.json") if (d := load(p))], key=lambda d: d.get("tag"))
    rows = []
    for p, b in benches:
        rows += bench_rows(p, b, "ft")
        bench_context("ft", b)
        FT_META[b.get("tag")] = (b.get("memory") or {})
    if not rows:
        print("  no Prithvi benchmark files yet")
        return []
    runs = sorted({(x["mode"], x["clocks"], x["tag"]) for x in rows}, key=lambda t: mode_key(t[0], t[1]))
    print("  benchmark runs:", ", ".join(t[2] for t in runs))
    builds = {}
    for p in files(FT_BASE, "build.json"):
        builds.update((load(p) or {}).get("engines") or {})

    # ---- P1: per power mode, the units that matter
    units = [("encoder", "b1", 1, "one chip (224 x 224 px, batch-1 engine)"),
             ("encoder", "dyn", 25, "25 chips"),
             ("encoder", "dyn", 100, "100 chips: one region pass (25 cells x 4 chips)"),
             ("gru_o3", "b1", 1, "one window (batch-1 engine)"),
             ("gru_o3", "dyn", 25, "one region-month: 25 windows"),
             ("gru_o3", "dyn", 100, "100 windows")]
    p1 = []
    for mode, clocks, tag in runs:
        for prec in ("fp16", "fp32"):
            for g, prof, batch, unit in units:
                eng = f"{g}_{prec}_{prof}"
                x = find(rows, tag=tag, runtime="TensorRT (trtexec)", variant="default", engine=eng, batch=batch)
                py = find(rows, tag=tag, runtime="TensorRT (Python)", engine=eng, batch=batch)
                if not x:
                    continue
                L = layer_text_ft(builds.get(eng))
                p1.append([mode, CLOCK_NAMES[clocks], "TensorRT", FT_GRAPH[g], prec.upper(), L, ft_match(x), unit,
                           batch, r(x["p50"]), r(x["p99"]), r(x["gpu_p50"]), r(py["p50"]) if py else "",
                           f"{x['ips']:.1f}", r(x["mean_w"], 2), r(x["max_w"], 2), r(x["idle_w"], 2),
                           sig(x["mj"] * batch), sig(x["marg_mj"] * batch), r(x["hot_c"], 1)])
        for g, batch, unit in (("encoder", 1, "one chip"), ("encoder", 25, "25 chips"), ("gru_o3", 1, "one window"),
                               ("gru_o3", 25, "one region-month: 25 windows")):
            x = find(rows, tag=tag, runtime="ONNX Runtime CPU", family=g, batch=batch)
            if not x:
                continue
            p1.append([mode, CLOCK_NAMES[clocks], "ONNX Runtime (CPU)", FT_GRAPH[g], (x["precision"] or "").upper(),
                       "(CPU, no TensorRT)", ft_match(x), unit, batch, r(x["p50"]), r(x["p99"]), "", "",
                       f"{x['ips']:.1f}", r(x["mean_w"], 2), r(x["max_w"], 2), r(x["idle_w"], 2), sig(x["mj"] * batch),
                       sig(x["marg_mj"] * batch), r(x["hot_c"], 1)])
    write_csv("P1_prithvi_speed_power_energy.csv",
              ["Power mode", "Clocks", "Runtime", "Graph", "Precision asked for", "Layers TensorRT ran in",
               "Output matches the reference on the board", "Unit", "Items",
               "Latency p50 (ms)", "Latency p99 (ms)", "GPU compute p50 (ms)", "Latency from Python p50 (ms)",
               "Throughput (items/s)", "Module power while running (W)", "Peak module power sample (W)",
               "Idle module power (W)", "Energy per unit (mJ)", "Energy per unit above idle (mJ)",
               "Hottest sensor during the run (C)"], p1,
              "Prithvi fresh-test model: speed, power and energy per power mode. Quote only rows whose output "
              "matches: the TensorRT encoder engines and the head's variable-batch engines gave wrong outputs")

    # ---- P4: per cell and per region, encoder + head, one pass a month
    def p4_row(mode, clocks, path, match, enc_ms, head_ms, enc_mj, head_mj, enc_marg, head_marg, enc_w, head_w):
        cell_mj = 4 * enc_mj + head_mj              # enc_mj per chip, head_mj per window
        cell_marg = 4 * enc_marg + head_marg
        return [mode, CLOCK_NAMES[clocks], path, match, r(enc_ms), r(head_ms), r(enc_ms + head_ms),
                sig(enc_mj * 100), sig(head_mj * 25), sig(25 * cell_mj), sig(25 * cell_marg), sig(cell_mj),
                sig(4 * enc_mj), sig(4 * enc_mj * 25), r(enc_w, 2), r(head_w, 2)]

    p4 = []
    for mode, clocks, tag in runs:
        ex = find(rows, tag=tag, runtime="ONNX Runtime CPU", family="encoder", batch=25)
        hx = find(rows, tag=tag, runtime="ONNX Runtime CPU", family="gru_o3", batch=25)
        if ex and hx:
            p4.append(p4_row(mode, clocks, "CPU: ONNX Runtime FP32 (encoder as 4 batches of 25 chips)",
                             both_match(ex, hx), 4 * ex["p50"], hx["p50"], ex["mj"], hx["mj"], ex["marg_mj"],
                             hx["marg_mj"], ex["mean_w"], hx["mean_w"]))
        else:
            p4.append([mode, CLOCK_NAMES[clocks], "CPU: ONNX Runtime",
                       "not measured" + (": ONNX Runtime aborts at start-up in 7W (CPU cores 4-5 offline)"
                                         if mode == "7W" else "")] + [""] * 12)
        for prec in ("fp16", "fp32"):
            enc = find(rows, tag=tag, runtime="TensorRT (trtexec)", variant="default", engine=f"encoder_{prec}_dyn", batch=100)
            head = find(rows, tag=tag, runtime="TensorRT (trtexec)", variant="default", engine=f"gru_o3_{prec}_dyn", batch=25)
            if enc and head:
                p4.append(p4_row(mode, clocks, f"TensorRT {prec.upper()} (variable-batch engines; timing only)",
                                 both_match(enc, head), enc["p50"], head["p50"], enc["mj"], head["mj"], enc["marg_mj"],
                                 head["marg_mj"], enc["mean_w"], head["mean_w"]))
    write_csv("P4_prithvi_per_cell_and_region.csv",
              ["Power mode", "Clocks", "Path", "Outputs match the reference on the board",
               "Encoder, one region pass (100 chips) p50 (ms)",
               "GRU head, one region-month (25 windows) p50 (ms)", "One region-month with one pass, total (ms)",
               "Encoder energy per region pass (mJ)", "Head energy per region-month (mJ)",
               "Region-month energy with one pass (mJ)", "Region-month energy above idle (mJ)",
               "Per cell-month with one pass (mJ)", "Each extra pass per cell (mJ)", "Each extra pass per region (mJ)",
               "Module power, encoder (W)", "Module power, head (W)"], p4,
              "Prithvi model per cell and per region (4 chips per cell-pass; the head runs once per cell-month). "
              "Only the CPU path gave the reference outputs on the board")

    # ---- P3: every row
    p3 = []
    order = {"TensorRT (trtexec)": 0, "TensorRT (Python)": 1, "ONNX Runtime CPU": 2}
    for x in sorted(rows, key=lambda x: (mode_key(x["mode"], x["clocks"]), order[x["runtime"]], x["engine"],
                                         x["batch"] or 0, x["variant"])):
        p3.append([x["mode"], CLOCK_NAMES[x["clocks"]], x["runtime"], x["variant"], x["engine"],
                   FT_GRAPH.get(x["family"], x["family"]), (x["precision"] or "").upper(),
                   layer_text_ft(builds.get(x["engine"])) if x["runtime"] != "ONNX Runtime CPU" else "FP32 (CPU)",
                   ft_match(x), x["threads"], x["batch"], x["item"], r(x["p50"]), r(x["p99"]), r(x["mean"]), r(x["gpu_p50"]),
                   f"{x['ips']:.1f}" if x["ips"] else "", r(x["mean_w"], 2), r(x["max_w"], 2), r(x["idle_w"], 2),
                   sig(x["mj"]), sig(x["marg_mj"]), r(x["hot_c"], 1), sig(x["engine_mb"]), sig(x["device_mem_mb"]),
                   r(x["rss_mb"], 0), x["tag"], (x["when"] or "")[:19]])
    write_csv("P3_prithvi_all_benchmark_rows.csv",
              ["Power mode", "Clocks", "Runtime", "Timing variant", "Engine or model file", "Graph",
               "Precision asked for", "Precision TensorRT ran", "Output matches the reference on the board",
               "CPU threads", "Batch", "Item", "Latency p50 (ms)",
               "Latency p99 (ms)", "Latency mean (ms)", "GPU compute p50 (ms)", "Throughput (items/s)",
               "Module power while running (W)", "Peak module power sample (W)", "Idle module power (W)",
               "Energy per item (mJ)", "Energy per item above idle (mJ)", "Hottest sensor during the row (C)",
               "Engine file (MB)", "Engine device memory (MB)", "Process memory (MB)", "Run tag", "Measured at"], p3,
              "Every Prithvi benchmark row")
    return rows


def layer_text_ft(e):
    lay = ((e or {}).get("layers") or {})
    bp = lay.get("by_output_precision") or {}
    if not lay.get("available"):
        return ""
    parts = [f"{k.upper()} {bp[k]}" for k in ("int8", "fp16", "fp32") if bp.get(k)]
    return f"{', '.join(parts)} of {lay.get('layers')} layers"


def parity_ft():
    print("Prithvi parity")
    out = []
    ort = newest_by([(p, d) for p in files(FT_BASE, "ort.json") if (d := load(p))], key=lambda d: "ort")
    trt = newest_by([(p, d) for p in files(FT_BASE, "trt.json") if (d := load(p))], key=lambda d: "trt")
    if ort:
        o = ort[0][1]
        for g, ent in (o.get("graphs") or {}).items():
            for prec in ("fp32", "fp16"):
                pe = ent.get(prec) or {}
                ref = pe.get("vs_stored") or pe.get("vs_pytorch") or {}
                out.append(parity_row_ft("ONNX Runtime CPU (board)", g, prec, "-", ent.get("items"), ref, None,
                                         pe.get("status"), pe.get("error")))
                FT_OK[("ort", g, prec)] = (pe.get("status"), pe.get("error") or "")
    if trt:
        t = trt[0][1]
        for name, e in (t.get("engines") or {}).items():
            n = e.get("items") or e.get("b1_items")
            out.append(parity_row_ft("TensorRT (board)", e.get("graph"), e.get("precision"), e.get("profile"), n,
                                     e.get("vs_reference") or {}, e.get("deterministic_rerun"), e.get("status"),
                                     "; ".join(e.get("reasons") or [])))
            FT_OK[("trt", name)] = (e.get("status"), "; ".join(e.get("reasons") or []))
    if not out:
        print("  no Prithvi parity files yet")
        return
    write_csv("P2_prithvi_board_parity.csv",
              ["Runtime", "Graph", "Precision", "Engine profile", "Items compared", "Compared with",
               "Largest difference", "Mean difference", "ESI output: largest difference",
               "ESI ranking agreement (Spearman)", "Encoder: lowest cosine similarity", "Same output on a rerun",
               "Status", "Notes"], out,
              "Prithvi model: does the board give the stored fresh-test predictions? (no labels on the board)")


def parity_row_ft(runtime, g, prec, prof, n, ref, rerun, status, note):
    enc = g == "encoder"
    return [runtime, FT_GRAPH.get(g, g), (prec or "").upper(), prof, n if n is not None else "",
            "Mac PyTorch features of the same chips" if enc else "stored 5-seed mean the fresh test scored",
            sci(ref.get("max_abs")), sci(ref.get("mean_abs")), "" if enc else sci(ref.get("esi_max_abs")),
            "" if enc or ref.get("esi_spearman") is None else f"{ref['esi_spearman']:.6f}",
            f"{ref['min_cosine']:.6f}" if enc and ref.get("min_cosine") is not None else "",
            {True: "yes", False: "no"}.get(rerun, ""), status or "", note or ""]


# ------------------------------------------------------------------------------------------------ claims

def claims_table(v2_rows, boots, ft_rows):
    """The manuscript's power, heat and speed statements (Space to Soil draft, written for 12U) against the board."""
    print("claims")
    runs = sorted({(x["mode"], x["clocks"], x["tag"]) for x in v2_rows}, key=lambda t: mode_key(t[0], t[1]))

    def tile(tag, eng="fused10_fp16_dyn", batch=100):
        return find(v2_rows, tag=tag, runtime="TensorRT (trtexec)", variant="default", engine=eng, batch=batch)

    tiles = [(m, c, tile(t)) for m, c, t in runs if tile(t)]
    idles = [(m, c, rows_idle(v2_rows, t)) for m, c, t in runs if rows_idle(v2_rows, t) is not None]
    out = []

    # heat
    hot_b = max((x for x in v2_rows + ft_rows if x["hot_c"] is not None), key=lambda x: x["hot_c"], default=None)
    hot_s = max((s for s in SOAKS if s["hot_c"] is not None), key=lambda s: s["hot_c"], default=None)
    soak_txt = "; ".join(f"{s['name']} ({s['mode']}, {CLOCK_NAMES.get(s['clocks'], s['clocks'])}, "
                         f"{r(s['minutes'], 0)} min): {r(s['hot_c'], 1)} C" for s in SOAKS)
    if hot_s or hot_b:
        worst = max([v for v in (hot_s and hot_s["hot_c"], hot_b and hot_b["hot_c"]) if v is not None])
        out.append(["ML processors stay below 70 C",
                    "Hottest on-chip sensor (CPU, GPU, SoC, junction) in every soak and benchmark",
                    soak_txt + (f"; hottest in any benchmark row {r(hot_b['hot_c'], 1)} C "
                                f"({hot_b['mode']}, {hot_b['label']})" if hot_b else ""),
                    f"Met: highest {r(worst, 1)} C, {r(TRIP_C - worst, 1)} C below the board's 70 C passive trip"
                    if worst < TRIP_C else f"NOT met: {r(worst, 1)} C",
                    "Room air with the stock fan, on the developer kit's carrier board. Vacuum (no convection) was "
                    "not tested; in orbit the heat must leave by conduction, so this does not show the flight unit "
                    "stays below 70 C."])
    # power envelope
    if tiles and idles:
        bud = [w for w, _ in BUDGET_IDLE.values() if w is not None]
        lo_i, hi_i = (min(bud), max(bud)) if bud else (min(i for _, _, i in idles), max(i for _, _, i in idles))
        oc_parts = [f"{s['oc']} times in {s['name']} ({s['mode']}, {CLOCK_NAMES.get(s['clocks'], s['clocks'])}, "
                    f"{r(s['minutes'], 0)} min; {s['errors']} errors)" for s in SOAKS if s["oc"]]
        oc_parts += [f"{n:,} times in the {'v2' if k == 'v2' else 'Prithvi'} benchmark {t}"
                     for (k, t), n in sorted(BENCH_OC.items()) if n]
        others = sum(1 for s in SOAKS if not s["oc"]) + sum(1 for n in BENCH_OC.values() if not n)
        oc_txt = ("the module's over-current protection fired " + "; ".join(oc_parts)
                  + f"; it fired in none of the other {others} runs") if oc_parts else ""
        tw = [(m, c, x["mean_w"]) for m, c, x in tiles]
        top = max((x for x in v2_rows if x["mean_w"]), key=lambda x: x["mean_w"])
        sustained = max((s["loaded_w"] for s in SOAKS if s["loaded_w"]), default=None)
        out.append(["Onboard compute fits 10-25 W",
                    "Module input power (VDD_IN, the module's own INA3221 sensor): idle, running the paper model on "
                    "tiles, and the highest draw seen",
                    f"Idle {r(lo_i, 2)}-{r(hi_i, 2)} W across modes and clock settings (at rest where measured); "
                    f"paper model on 100-window tiles "
                    + ", ".join(f"{m} {'pinned' if c == 'pinned' else 'DVFS'} {r(w, 2)} W" for m, c, w in tw)
                    + f"; highest mean over any benchmark row {r(top['mean_w'], 2)} W ({top['mode']}, {top['label']})"
                    + (f"; highest sustained (soak) {r(sustained, 2)} W" if sustained else "")
                    + (f"; {oc_txt}" if oc_txt else ""),
                    (f"{'Within' if top['mean_w'] < 25 else 'NOT within'} 25 W in every mode tested; the paper model "
                     f"on tiles drew {r(min(x['mean_w'] - idle_for(x) for _, _, x in tiles), 2)}-"
                     f"{r(max(x['mean_w'] - idle_for(x) for _, _, x in tiles), 2)} W above the idle for the budget"),
                    "Module power only: the carrier board, SSD, fan, radio, camera and power conversion are not "
                    "included. The 10-25 W envelope was written for the 12U design; the 6U budget needs its own "
                    "number."])
    # speed per tile
    if tiles:
        slow = max(tiles, key=lambda t: t[2]["p50"])
        out.append(["Inference under 10 ms per tile (NPU target)",
                    "Paper model, one tile = 100 windows, TensorRT FP16 engine, trtexec latency including copies",
                    ", ".join(f"{m} {'pinned' if c == 'pinned' else 'DVFS'} {r(x['p50'])} ms (p99 {r(x['p99'])})"
                              for m, c, x in tiles),
                    f"{'Met in every mode' if slow[2]['p50'] < 10 else 'NOT met in every mode'}: slowest "
                    f"{r(slow[2]['p50'])} ms p50 ({slow[0]}, {'pinned' if slow[1] == 'pinned' else 'DVFS'})",
                    "The Orin Nano has no DLA (NVIDIA's NPU-style engine); these ran on its GPU. A tile here is "
                    "about 100 cells' 12-month windows, the kit's definition; the image preprocessing that makes "
                    "the windows is not timed."])
    # duty cycling
    duty = [s for s in SOAKS if s["duty"]]
    if duty or boots:
        d = duty[0] if duty else None
        bt = [float(b[1]) for b in boots if b[1]]
        st = [float(b[2]) for b in boots if b[2]]
        out.append(["Accelerators are duty-cycled",
                    "F11: 60 s of tiles, 240 s idle, for 60 min at 15W; and the time to boot from a reboot",
                    (f"On {r(d['on_w'], 2)} W, off {r(d['off_w'], 2)} W, average {r(d['mean_w'], 2)} W over the test; "
                     if d else "") +
                    (f"reboot command to SSH {min(bt):.0f}-{max(bt):.0f} s over {len(bt)} reboots"
                     + (f", systemd boot {min(st):.1f}-{max(st):.1f} s" if st else "") if bt else ""),
                    "Measured: idle between bursts costs the idle power above; powering the board off between "
                    "passes trades that for about a minute of boot",
                    "Off-phase power is idle module power with the model loaded, not power-gated; the board was never "
                    "switched off, so off-state power was not measured."])
    # one copy vs ten
    one_ten = [(m, c, tile(t), tile(t, "seed42_fp16_dyn")) for m, c, t in runs if tile(t) and tile(t, "seed42_fp16_dyn")]
    if one_ten:
        out.append(["One copy or the ten-copy average (left open in the ML draft)",
                    "Same tile (100 windows), TensorRT, trtexec: the ten-seed ensemble (fused, the paper model) vs "
                    "one seed",
                    "; ".join(f"{m} {'pinned' if c == 'pinned' else 'DVFS'}: ten {r(a['p50'])} ms / "
                              f"{sig(a['mj'] * 100)} mJ, one {r(b['p50'])} ms / {sig(b['mj'] * 100)} mJ"
                              for m, c, a, b in one_ten),
                    (f"Ten copies took {min(a['p50'] / b['p50'] for _, _, a, b in one_ten):.1f}-"
                     f"{max(a['p50'] / b['p50'] for _, _, a, b in one_ten):.1f}x the time and "
                     f"{min(a['mj'] / b['mj'] for _, _, a, b in one_ten):.1f}-"
                     f"{max(a['mj'] / b['mj'] for _, _, a, b in one_ten):.1f}x the energy of one copy per tile"),
                    "Only the ten-seed ensemble's accuracy is reported (AUC 0.663); a single seed's accuracy was not "
                    "computed. The one-seed engine ran in FP32 (TensorRT chose no FP16 layers) while the ten-copy "
                    "engine ran in FP16."])
    # the Prithvi model against the same per-tile target, for comparison (it is not the paper's model)
    ft_runs = sorted({(x["mode"], x["clocks"], x["tag"]) for x in ft_rows}, key=lambda t: mode_key(t[0], t[1]))
    reg = []
    for m, c, t in ft_runs:
        e = find(ft_rows, tag=t, runtime="TensorRT (trtexec)", variant="default", engine="encoder_fp16_dyn", batch=100)
        h = find(ft_rows, tag=t, runtime="TensorRT (trtexec)", variant="default", engine="gru_o3_fp16_dyn", batch=25)
        ex = find(ft_rows, tag=t, runtime="ONNX Runtime CPU", family="encoder", batch=25)
        hx = find(ft_rows, tag=t, runtime="ONNX Runtime CPU", family="gru_o3", batch=25)
        if (e and h) or (ex and hx):
            reg.append((m, c, e, h, ex, hx))
    if reg:
        nums, cpu, ratio = [], [], []
        for m, c, e, h, ex, hx in reg:
            parts = []
            if ex and hx:
                parts.append(f"CPU {r((4 * ex['p50'] + hx['p50']) / 1000, 2)} s (outputs match)")
                cpu.append(4 * ex["p50"] + hx["p50"])
                if e and h:
                    ratio.append((4 * ex["p50"] + hx["p50"]) / (e["p50"] + h["p50"]))
            elif m == "7W":
                parts.append("CPU not possible (ONNX Runtime aborts in 7W)")
            if e and h:
                parts.append(f"TensorRT FP16 {r(e['p50'] + h['p50'], 1)} ms (outputs did NOT match)")
            nums.append(f"{m} {'pinned' if c == 'pinned' else 'DVFS'}: " + ", ".join(parts))
        out.append(["(Prithvi fresh-test model, for comparison) the same per-tile target",
                    "One region-month: the encoder on one region pass (100 chips = 25 cells x 4 chips) plus the GRU "
                    "head on the region's 25 windows; CPU (ONNX Runtime FP32) and TensorRT FP16",
                    "; ".join(nums),
                    (f"Only the CPU path gave the reference outputs, at {r(min(cpu) / 1000, 2)}-{r(max(cpu) / 1000, 2)} s "
                     f"per region-month; TensorRT FP16 timed {r(min(ratio), 0)}-{r(max(ratio), 0)}x faster in the same "
                     "modes, but its encoder engines gave wrong outputs on this board" if cpu and ratio else
                     "No path with matching outputs was timed"),
                    "Not the paper's model. Its units differ: a cell needs 4 image chips per satellite pass, and the "
                    "encoder runs once per pass, the head once per month. On this board TensorRT 10.3 did not "
                    "reproduce the encoder (any batch, FP32 or FP16) or the GRU head's variable-batch engines (P2); "
                    "ONNX Runtime aborts in the 7W mode."])
    write_csv("J7_paper_claims_check.csv",
              ["Statement in the manuscript", "What was measured", "Numbers", "Verdict", "Limits of the test"], out,
              "The manuscript's power, heat and speed statements against the board")


# ------------------------------------------------------------------------------------------------ budget sheet

V2_KIT = Path.home() / "Desktop/DEWSat_BiLSTM_v2/jetson_kit"
FT_KIT = Path.home() / "Desktop/DEWSat_Prithvi_pilot/jetson_ft"
STUDY_REGIONS = 702            # v2's regions (558 development + 144 fresh test), 25 cells each
FT_META = {}                   # Prithvi bench-level memory, filled by ft_tables


def budget_sheet(v2_rows, boots, ft_rows):
    """One long sheet: every budget line (power, energy, heat, time, memory, storage, data, reliability) with its
    value, unit, conditions and source. Measured values are copied; derived ones say how they were derived."""
    print("budget sheet")
    B = []

    def add(budget, item, model, value, unit, cond, src):
        if value in (None, ""):
            return
        B.append([budget, item, model, value, unit, cond, src])

    PM = "Bi-LSTM ten-seed ensemble (paper model)"
    ONE = "Bi-LSTM one seed"
    FTM = "Prithvi encoder + GRU head (fresh-test model)"
    runs = sorted({(x["mode"], x["clocks"], x["tag"]) for x in v2_rows}, key=lambda t: mode_key(t[0], t[1]))

    def cl(c):
        return "pinned clocks" if c == "pinned" else "dynamic clocks"

    def g(tag, eng, batch, rows=v2_rows):
        return find(rows, tag=tag, runtime="TensorRT (trtexec)", variant="default", engine=eng, batch=batch)

    # power
    both = ("v2", "ft")
    keys = {(s["mode"], s["clocks"]) for s in SOAKS} | {(x["mode"], x["clocks"]) for x in RESTED}
    for m, c in sorted(keys, key=lambda q: mode_key(q[0], q[1])):
        src = rest_sources(m, c, both)
        add("Power", "Idle at rest (nothing running, no load for the 4-10 min before)", "board",
            rest_txt([v for v, _ in src]), "W",
            f"{m}, {cl(c)}; module input power (VDD_IN); {len(src)} reading{'s' if len(src) != 1 else ''}: "
            + "; ".join(f"{w} {r(v, 2)} W" for v, w in src), "J2, J0, P1")
    for m, c, t in runs:
        age = BOOT_AGE.get(("v2", t))
        add("Power", "Idle at the start of the benchmark", "board", r(rows_idle(v2_rows, t), 2), "W",
            f"{m}, {cl(c)}; module input power (VDD_IN); "
            + (f"{age:.0f} min after boot" if age is not None else "time since boot not logged"), "J0")
    for m, c, t in runs:
        x = g(t, "fused10_fp16_dyn", 100)
        if x:
            add("Power", "Running the model on 100-window tiles back to back", PM, r(x["mean_w"], 2), "W",
                f"{m}, {cl(c)}; TensorRT FP16; module input power", "J1")
            add("Power", "Highest single sample while running tiles", PM, r(x["max_w"], 2), "W", f"{m}, {cl(c)}", "J1")
    for s in SOAKS:
        if s["duty"]:
            add("Power", "Duty cycle: 60 s of tiles, 240 s idle (average over the test)", PM, r(s["mean_w"], 2), "W",
                f"{s['mode']}, {cl(s['clocks'])}; {s['name']}; on {r(s['on_w'], 2)} W, off {r(s['off_w'], 2)} W",
                "J2")
        else:
            add("Power", f"Sustained continuous load, batch {s['batch']} ({r(s['minutes'], 0)} min)", PM,
                r(s["loaded_w"], 2), "W", f"{s['mode']}, {cl(s['clocks'])}; {s['name']}", "J2")
    for s in SOAKS:
        if s["oc"]:
            add("Power", "Over-current protection events (the module's own limit)", PM,
                f"{s['oc']} in {r(s['minutes'], 0)} min", "count",
                f"{s['mode']}, {cl(s['clocks'])}; {s['name']}, batch {s['batch']}, continuous; {s['oc_txt']}; "
                f"{s['errors']} errors, p50 drift {r(s['p50_drift_pct'], 2)}%", "J2")
    for (k, t), n in sorted(BENCH_OC.items()):
        if n:
            add("Power", "Over-current protection events during a benchmark", PM if k == "v2" else FTM, str(n),
                "count", f"benchmark {t}", "bench json")
    zero = sum(1 for s in SOAKS if not s["oc"]) + sum(1 for n in BENCH_OC.values() if not n)
    if zero:
        add("Power", "Over-current protection events, all other runs", "board", "0", "count",
            f"{zero} runs: every other soak and benchmark (v2 and Prithvi)", "J2, bench json")
    top = max((x for x in v2_rows if x["mean_w"]), key=lambda x: x["mean_w"], default=None)
    if top:
        add("Power", "Highest mean over any 5 s benchmark row (any engine, batch, runtime)", "v2 engines",
            r(top["mean_w"], 2), "W", f"{top['mode']}, {cl(top['clocks'])}: {top['label']}", "J5")
    ft_top = max((x for x in ft_rows if x["mean_w"]), key=lambda x: x["mean_w"], default=None)
    if ft_top:
        add("Power", "Highest mean over any Prithvi benchmark row", FTM, r(ft_top["mean_w"], 2), "W",
            f"{ft_top['mode']}, {cl(ft_top['clocks'])}: {ft_top['label']} (power is real; this engine's outputs "
            f"{'matched' if ft_match(ft_top).startswith('yes') else 'did NOT match'})", "P3")
    for m, c, t in sorted({(x["mode"], x["clocks"], x["tag"]) for x in ft_rows}, key=lambda q: mode_key(q[0], q[1])):
        e = g(t, "encoder_fp16_dyn", 100, ft_rows)
        h = g(t, "gru_o3_fp16_dyn", 25, ft_rows)
        ex = find(ft_rows, tag=t, runtime="ONNX Runtime CPU", family="encoder", batch=25)
        hx = find(ft_rows, tag=t, runtime="ONNX Runtime CPU", family="gru_o3", batch=25)
        anyr = next((x for x in ft_rows if x["tag"] == t and x["idle_w"] is not None), None)
        if anyr:
            age = BOOT_AGE.get(("ft", t))
            add("Power", "Idle at the start of the benchmark", "board (Prithvi run)", r(anyr["idle_w"], 2), "W",
                f"{m}, {cl(c)}; " + (f"{age:.0f} min after boot" if age is not None else "time since boot not logged"),
                "P1")
        if ex:
            add("Power", "Encoder on the CPU, 25-chip batches back to back (outputs match)", FTM, r(ex["mean_w"], 2),
                "W", f"{m}, {cl(c)}; ONNX Runtime FP32", "P1")
        if hx:
            add("Power", "GRU head on the CPU, 25-window batches back to back (outputs match)", FTM,
                r(hx["mean_w"], 2), "W", f"{m}, {cl(c)}; ONNX Runtime FP32", "P1")
        if e:
            add("Power", "Encoder on TensorRT, 100-chip batches back to back (outputs did NOT match; timing only)",
                FTM, r(e["mean_w"], 2), "W", f"{m}, {cl(c)}; TensorRT FP16", "P1")
        if h:
            add("Power", "GRU head on TensorRT, 25-window batches (outputs did NOT match; timing only)", FTM,
                r(h["mean_w"], 2), "W", f"{m}, {cl(c)}; TensorRT FP16 variable-batch engine", "P1")

    # energy
    n_tiles = STUDY_REGIONS * 25 / 100
    for m, c, t in runs:
        x = g(t, "fused10_fp16_dyn", 100)
        o = g(t, "seed42_fp16_dyn", 100)
        if x:
            add("Energy", "One tile (100 windows)", PM, sig(x["mj"] * 100), "mJ", f"{m}, {cl(c)}; module energy", "J1")
            add("Energy", "One tile above idle", PM, sig(x["marg_mj"] * 100), "mJ", f"{m}, {cl(c)}", "J1")
            add("Energy", f"All {STUDY_REGIONS} regions once ({STUDY_REGIONS * 25:,} windows)", PM,
                sig(x["mj"] * 100 * n_tiles / 1000), "J",
                f"{m}, {cl(c)}; derived: {n_tiles:g} tiles x the tile energy", "J1 (derived)")
        if o:
            add("Energy", "One tile (100 windows)", ONE, sig(o["mj"] * 100), "mJ", f"{m}, {cl(c)}", "J1")
    idle_by_mode = {}
    for m, c, t in runs:
        if c == "dvfs" or m not in idle_by_mode:
            v = rest_idle(m, c, ("v2", "ft"))
            idle_by_mode[m] = (c, sum(v) / len(v) if v else rows_idle(v2_rows, t),
                               "at rest" if v else "at the start of the benchmark")
    for m, (c, w, how) in sorted(idle_by_mode.items(), key=lambda kv: MODE_ORDER.get(kv[0], 9)):
        if w is not None:
            add("Energy", "Board left on and idle for a day", "board", sig(w * 24, 3), "Wh",
                f"{m}, {cl(c)}; derived: idle {r(w, 2)} W ({how}) x 24 h", "J0/J2 (derived)")
    for m, c, t in sorted({(x["mode"], x["clocks"], x["tag"]) for x in ft_rows}, key=lambda q: mode_key(q[0], q[1])):
        e = g(t, "encoder_fp16_dyn", 100, ft_rows)
        h = g(t, "gru_o3_fp16_dyn", 25, ft_rows)
        ex = find(ft_rows, tag=t, runtime="ONNX Runtime CPU", family="encoder", batch=25)
        hx = find(ft_rows, tag=t, runtime="ONNX Runtime CPU", family="gru_o3", batch=25)
        if ex and hx:
            rm = ex["mj"] * 100 + hx["mj"] * 25
            add("Energy", "One region-month with one pass, CPU path (100 chips + 25 windows; outputs match)", FTM,
                sig(rm), "mJ", f"{m}, {cl(c)}; ONNX Runtime FP32; derived from P1", "P4")
            add("Energy", f"All {STUDY_REGIONS} regions with one pass each, CPU path (outputs match)", FTM,
                sig(rm * STUDY_REGIONS / 1000), "J",
                f"{m}, {cl(c)}; derived: {STUDY_REGIONS} x the region-month energy ({sig(rm * STUDY_REGIONS / 3.6e6)} Wh)",
                "P4 (derived)")
        if e and h:
            add("Energy", "One region-month with one pass, TensorRT FP16 (outputs did NOT match; timing only)", FTM,
                sig(e["mj"] * 100 + h["mj"] * 25), "mJ", f"{m}, {cl(c)}; derived from P1", "P4")
            add("Energy", f"All {STUDY_REGIONS} regions with one pass each, TensorRT FP16 (outputs did NOT match)", FTM,
                sig((e["mj"] * 100 + h["mj"] * 25) * STUDY_REGIONS / 1000), "J",
                f"{m}, {cl(c)}; derived: {STUDY_REGIONS} x the region-month energy", "P4 (derived)")

    # time
    bt = [float(b[1]) for b in boots if b[1]]
    st = [float(b[2]) for b in boots if b[2]]
    if bt:
        add("Time", "Reboot command to SSH login", "board", f"{min(bt):.0f}-{max(bt):.0f}", "s",
            f"{len(bt)} reboots, measured on the Mac", "J4")
    if st:
        add("Time", "systemd boot (kernel + userspace)", "board", f"{min(st):.1f}-{max(st):.1f}", "s",
            f"{len(st)} reboots", "J4")
    cold = {}
    for p in files(V2_BASE, "robust.json"):
        cold = (load(p) or {}).get("cold_start") or cold
    for k, v in cold.items():
        if "fused10" in k or "seed42" in k or "ensemble10_fp32" in k:
            add("Time", f"Cold start to the first result ({k.split('/')[-1]})",
                PM if "fused10" in k or "ensemble10" in k else ONE, r(v.get("process_wall_s"), 2), "s",
                f"new process: import {r(v.get('import_s'), 3)} s, load {r(v.get('load_s'), 3)} s, first inference "
                f"{r(v.get('first_inference_ms'), 1)} ms; MAXN_SUPER", "robust.json")
    for m, c, t in runs:
        x = g(t, "fused10_fp16_dyn", 100)
        w1 = g(t, "fused10_fp16_b1", 1)
        if x:
            add("Time", "One tile (100 windows), p50 / p99", PM, f"{r(x['p50'])} / {r(x['p99'])}", "ms",
                f"{m}, {cl(c)}; trtexec, copies included", "J1")
            add("Time", f"All {STUDY_REGIONS} regions once ({STUDY_REGIONS * 25:,} windows)", PM,
                r(STUDY_REGIONS * 25 / x["ips"], 3), "s", f"{m}, {cl(c)}; derived from tile throughput", "J1 (derived)")
        if w1:
            add("Time", "One window alone, p50", PM, r(w1["p50"]), "ms", f"{m}, {cl(c)}; batch-1 engine", "J1")
    for m, c, t in sorted({(x["mode"], x["clocks"], x["tag"]) for x in ft_rows}, key=lambda q: mode_key(q[0], q[1])):
        e1 = g(t, "encoder_fp16_b1", 1, ft_rows)
        e = g(t, "encoder_fp16_dyn", 100, ft_rows)
        h = g(t, "gru_o3_fp16_dyn", 25, ft_rows)
        hb1 = g(t, "gru_o3_fp16_b1", 1, ft_rows)
        ex1 = find(ft_rows, tag=t, runtime="ONNX Runtime CPU", family="encoder", batch=1)
        ex = find(ft_rows, tag=t, runtime="ONNX Runtime CPU", family="encoder", batch=25)
        hx = find(ft_rows, tag=t, runtime="ONNX Runtime CPU", family="gru_o3", batch=25)
        if ex1:
            add("Time", "Encoder, one chip alone on the CPU, p50 (outputs match)", FTM, r(ex1["p50"]), "ms",
                f"{m}, {cl(c)}; ONNX Runtime FP32", "P1")
        if ex and hx:
            add("Time", "One region-month with one pass, CPU path (4 x 25 chips + 25 windows), p50 (outputs match)",
                FTM, r(4 * ex["p50"] + hx["p50"], 0), "ms", f"{m}, {cl(c)}; ONNX Runtime FP32", "P4")
            add("Time", f"All {STUDY_REGIONS} regions with one pass each, CPU path (outputs match)", FTM,
                r(STUDY_REGIONS * (100 / ex["ips"] + 25 / hx["ips"]), 0), "s", f"{m}, {cl(c)}; derived",
                "P4 (derived)")
        if hb1 and ft_match(hb1).startswith("yes"):
            add("Time", "GRU head, one window, TensorRT batch-1 engine, p50 (outputs match)", FTM, r(hb1["p50"]), "ms",
                f"{m}, {cl(c)}; FP16 asked for; trtexec", "P1")
        if e1:
            add("Time", "Encoder, one chip alone on TensorRT FP16, p50 (outputs did NOT match; timing only)", FTM,
                r(e1["p50"]), "ms", f"{m}, {cl(c)}", "P1")
        if e and h:
            add("Time", "One region-month with one pass, TensorRT FP16, p50 (outputs did NOT match; timing only)", FTM,
                r(e["p50"] + h["p50"]), "ms", f"{m}, {cl(c)}", "P4")
            add("Time", f"All {STUDY_REGIONS} regions with one pass each, TensorRT FP16 (outputs did NOT match)", FTM,
                r(STUDY_REGIONS * (100 / e["ips"] + 25 / h["ips"]), 2), "s", f"{m}, {cl(c)}; derived", "P4 (derived)")

    # heat
    for s in SOAKS:
        add("Heat", f"Hottest on-chip sensor, {s['name']}", PM, r(s["hot_c"], 1), "C",
            f"{s['mode']}, {cl(s['clocks'])}, {r(s['minutes'], 0)} min; {s['hot_zone']}; margin "
            f"{r(TRIP_C - s['hot_c'], 1)} C to the 70 C passive trip; room air, stock fan", "J2")
    hb = max((x for x in ft_rows if x["hot_c"] is not None), key=lambda x: x["hot_c"], default=None)
    if hb:
        add("Heat", "Hottest on-chip sensor in any Prithvi benchmark row", FTM, r(hb["hot_c"], 1), "C",
            f"{hb['mode']}: {hb['label']}; no long soak was run for this model", "P3")
    if top:
        add("Heat", "Heat to remove = electrical power (all of it ends as heat)", "board",
            f"{r(min(w for (_, w, _) in idle_by_mode.values() if w), 2)}-{r(max(s['loaded_w'] for s in SOAKS if s['loaded_w']), 2)}"
            if SOAKS else "", "W", "idle in the lowest mode to the worst sustained soak; module only", "J0, J2")
    add("Heat", "Board thermal limits", "board", "70 / 104.5", "C",
        "passive (throttling) trip / critical shutdown on the CPU, GPU and SoC zones (from the board)", "soak trip_points")

    # memory
    mem_total = None
    for p in files(V2_BASE, "sysinfo.json"):
        mem_total = ((load(p) or {}).get("memory_mb") or {}).get("MemTotal") or mem_total
    add("Memory", "Board RAM (shared by CPU and GPU)", "board", f"{mem_total:,}" if mem_total else "", "MB",
        "MemTotal; the stock desktop was running", "sysinfo.json")
    for p, s in newest_by([(p, d) for p in files(V2_BASE, "soak_*.json") if (d := load(p))], key=lambda d: d.get("tag")):
        iv = s.get("intervals") or []
        rss = [i["rss_mb"] for i in iv if i.get("rss_mb") is not None]
        av = [i["mem_available_mb"] for i in iv if i.get("mem_available_mb") is not None]
        if rss:
            add("Memory", f"Runner process memory (RSS), peak, soak {s.get('tag')}", PM, r(max(rss), 0), "MB",
                f"TensorRT, batch {s.get('batch')}; growth {r((s.get('summary') or {}).get('rss_growth_mb'), 1)} MB "
                f"over {r(s.get('minutes_run'), 0)} min", "soak intervals")
        if av:
            add("Memory", f"Lowest free memory (MemAvailable), soak {s.get('tag')}", "board", f"{min(av):,}", "MB",
                "whole board, with the desktop running", "soak intervals")
    for k, v in FT_META.items():
        add("Memory", f"Runner process memory, peak (Prithvi bench {k})", FTM, r(v.get("peak_rss_mb"), 0), "MB",
            "all graphs loaded in turn", "bench json")
    for x in ft_rows:
        if x["runtime"] == "TensorRT (Python)" and x["engine"] in ("encoder_fp16_dyn", "gru_o3_fp16_dyn") \
                and x["batch"] == 100 and x["device_mem_mb"]:
            add("Memory", f"TensorRT working memory of {x['engine']}", FTM, sig(x["device_mem_mb"]), "MB",
                f"{x['mode']}; this engine's outputs {'matched' if ft_match(x).startswith('yes') else 'did NOT match'}",
                "P3")
            break

    # storage
    def fsize(p):
        return Path(p).stat().st_size / 1e6 if Path(p).exists() else None
    add("Storage", "Model weights, 10 x 11,617 parameters (FP32)", PM, "0.46", "MB", "116,170 floats", "v2 state_dicts")
    for name, label in (("fused10_fp32.onnx", PM), ("seed42_fp32.onnx", ONE), ("ensemble10_int8.onnx", PM + ", INT8")):
        add("Storage", f"ONNX file {name}", label, sig(fsize(V2_KIT / "DEWSat_Jetson_v2/models" / name)), "MB", "", "kit")
    for p in files(V2_BASE, "build.json"):
        for name, e in ((load(p) or {}).get("engines") or {}).items():
            if name in ("fused10_fp16_dyn", "fused10_fp16_b1", "seed42_fp16_dyn"):
                add("Storage", f"TensorRT engine {name}", PM if "fused" in name else ONE,
                    sig(e.get("engine_bytes", 0) / 1e6), "MB", "built on the board", "J6")
        break
    add("Storage", "Whole v2 test kit (zip: models, test data, runner)", "kit",
        sig(fsize(V2_KIT / "DEWSat_Jetson_v2.zip")), "MB", "", "kit")
    for name in ("prithvi_encoder_fp16.onnx", "prithvi_encoder_fp32.onnx", "gru_o3_head_fp16.onnx", "gru_o3_head_fp32.onnx"):
        add("Storage", f"ONNX file {name}", FTM, sig(fsize(FT_KIT / "DEWSat_Jetson_FT/models" / name)), "MB", "", "kit")
    for p in files(FT_BASE, "build.json"):
        for name, e in ((load(p) or {}).get("engines") or {}).items():
            if name in ("encoder_fp16_dyn", "gru_o3_fp16_dyn"):
                add("Storage", f"TensorRT engine {name}", FTM, sig((e.get("engine_bytes") or 0) / 1e6), "MB",
                    "built on the board", "build.json")
        break

    # data
    add("Data", "Input per window (12 months x 9 channels, float32)", PM, "432", "bytes", "", "kit manifest")
    add("Data", "Output per window (one drought probability, float32)", PM, "4", "bytes", "", "kit manifest")
    add("Data", "Output per region-month (25 cells)", PM, "100", "bytes", "float32; 25 bytes if sent as 1 byte each",
        "derived")
    add("Data", "Encoder input per chip (6 bands x 224 x 224, int16 DN)", FTM, f"{6 * 224 * 224 * 2:,}", "bytes",
        "1 chip; 4 chips per cell-pass", "kit manifest")
    add("Data", "Head input per window (12 months x 196 values + 2, float32)", FTM, f"{(12 * 196 + 2) * 4:,}",
        "bytes", "", "kit manifest")
    add("Data", "Output per window (4 standardized percentiles, float32)", FTM, "16", "bytes", "", "kit manifest")

    # reliability
    for s in SOAKS:
        add("Reliability", f"Errors / times the probe output changed, {s['name']}", PM,
            f"{s['errors']} / {s['probe_changes']}", "count",
            f"{s['windows']:,} windows in {r(s['minutes'], 0)} min; p50 drift {r(s['p50_drift_pct'], 2)}%", "J2")
    ok = sorted(k[1] for k, v in FT_OK.items() if k[0] == "trt" and v[0] in ("PASS", "WARN"))
    bad = sorted(k[1] for k, v in FT_OK.items() if k[0] == "trt" and v[0] == "FAIL")
    if ok or bad:
        add("Reliability", "Prithvi TensorRT engines whose outputs matched the reference", FTM, str(len(ok)), "count",
            ", ".join(ok), "P2")
        add("Reliability", "Prithvi TensorRT engines whose outputs did NOT match", FTM, str(len(bad)), "count",
            ", ".join(bad) + "; the variable-batch head engines also gave different outputs on two runs", "P2")
    orts = sorted(f"{k[1]} {k[2]}" for k, v in FT_OK.items() if k[0] == "ort" and v[0] in ("PASS", "WARN"))
    if orts:
        add("Reliability", "Prithvi on the CPU (ONNX Runtime) matched the reference", FTM, str(len(orts)), "count",
            ", ".join(orts) + " (FP16 encoder warned, within its fail limit)", "P2")
    add("Reliability", "ONNX Runtime 1.23.2 (CPU) in the 7W mode", "board", "aborts at import", "",
        "exit 134, a vector index assertion in its CPU detection; 7W leaves CPU cores 0-3 online (4-5 offline); "
        "TensorRT runs normally", "b7.log, s2.log")
    write_csv("B0_budget_sheet.csv",
              ["Budget", "Item", "Model", "Value", "Unit", "Conditions", "From"], B,
              "Budget sheet: power, energy, time, heat, memory, storage, data and reliability (one line each)")


# ------------------------------------------------------------------------------------------------ html

def write_html():
    parts = ["<!doctype html><meta charset='utf-8'><title>DEWSat Jetson tables</title>",
             "<style>body{font:13px/1.4 -apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#111;background:#fff}"
             "table{border-collapse:collapse;margin:8px 0 28px}th,td{border:1px solid #bbb;padding:3px 6px;"
             "text-align:left;vertical-align:top}th{background:#f2f2f2}h2{font-size:15px;margin:18px 0 4px}"
             "p{margin:2px 0;color:#444}</style>",
             "<h1 style='font-size:18px'>DEWSat on the Jetson Orin Nano: tables for the paper</h1>",
             "<p>Select a table, copy it, and paste it into Google Docs or Word. The same numbers are in the CSV files "
             "next to this page; README.md explains every column.</p>"]
    for name, title, header, rows in TABLES:
        parts.append(f"<h2>{html.escape(title)}</h2><p>{html.escape(name)}</p><table><tr>"
                     + "".join(f"<th>{html.escape(str(h))}</th>" for h in header) + "</tr>")
        for row in rows:
            parts.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>")
        parts.append("</table>")
    (OUT / "jetson_tables.html").write_text("\n".join(parts), encoding="utf-8")
    print("  wrote jetson_tables.html")


def main():
    soak_tables()          # first: J0 and the budget use the soaks' rested idle readings
    v2_rows = v2_tables()
    parity_v2()
    boots = boot_table()
    parity_ft()            # first: the Prithvi tables say which engines gave the reference outputs
    ft_rows = ft_tables()
    claims_table(v2_rows, boots, ft_rows)
    budget_sheet(v2_rows, boots, ft_rows)
    TABLES.sort(key=lambda t: t[0])
    write_html()
    (OUT / "BUILD_NOTES.txt").write_text("\n".join(NOTES) + ("\n" if NOTES else "no problems found\n"))
    print(f"{len(NOTES)} note(s)" + ("".join("\n  - " + n for n in NOTES) if NOTES else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
