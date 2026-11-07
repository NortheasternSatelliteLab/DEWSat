"""Pipeline status: logs/status_pipeline.json and STATUS.md (with the pull section for the real run)."""

from __future__ import annotations

import os
import sys
import time

from common import CACHE, LOGS, OUT, V2, log, read_json, write_json, write_text


class PipelineStatus:
    def __init__(self):
        self.path = LOGS / "status_pipeline.json"
        self.s = read_json(self.path, {}) or {}
        self.s.setdefault("stages", {})
        self.s["pid"] = os.getpid()
        self.s["process_started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.s["state"] = "running"
        self._last_render = 0.0
        self.n_windows_estimate = 200_000
        self.refresh_counts()
        self.save(force=True)

    def refresh_counts(self):
        meta = read_json(CACHE / "dataset.json")
        if meta and "dev" in meta:
            self.s["data"] = {"dev_regions": len(meta["dev"]["regions"]), "dev_cells": meta["dev"]["n_cells"],
                              "test_regions": len(meta.get("test", {}).get("regions", [])),
                              "test_cells": meta.get("test", {}).get("n_cells", 0),
                              "dev_cells_with_ndmi": meta["dev"]["cells_with_ndmi"],
                              "dev_cells_with_views": meta["dev"]["cells_with_views"]}
            self.n_windows_estimate = int(meta["dev"]["n_cells"] * 0.5 * 30)

    def stage(self, name, **kw):
        self.s["stages"].setdefault(name, {}).update(kw)
        self.save(force=True)

    def unit_done(self, stage, r, n_done, total, t0):
        st = self.s["stages"].setdefault(stage, {})
        st["done"] = n_done
        st["total"] = total
        st.setdefault("failed_units", 0)
        if not r["ok"]:
            st["failed_units"] += 1
            st.setdefault("failed_names", []).append(r["name"])
        el = time.time() - t0
        st["elapsed_h"] = round(el / 3600, 2)
        ran = st.setdefault("ran", 0) + 1
        st["ran"] = ran
        left = total - n_done
        st["eta"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + el / ran * left)) if ran else None
        self.save()

    def finish(self):
        self.s["state"] = "finished"
        self.s["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save(force=True)

    def save(self, force=False):
        write_json(self.path, self.s)
        if force or time.time() - self._last_render > 300:
            self._last_render = time.time()
            self.render()

    def render(self):
        lines = ["## Phases 3-4 pipeline", ""]
        d = self.s.get("data")
        if d:
            lines.append(f"- Data: {d['dev_regions']} development regions ({d['dev_cells']:,} cells), "
                         f"{d['test_regions']} fresh-test regions ({d['test_cells']:,} cells); development cells with "
                         f"SWIR/NDMI {d['dev_cells_with_ndmi']:,}, with per-scene values {d['dev_cells_with_views']:,}.")
        lines.append(f"- Process {self.s.get('pid')} {self.s.get('state')}; started {self.s.get('process_started')}.")
        lines += ["", "| Stage | State | Done / total | Failed | ETA | Minutes |", "|---|---|---|---|---|---|"]
        for k, v in self.s["stages"].items():
            lines.append(f"| {k} | {v.get('state', '')} | {v.get('done', '')} / {v.get('total', '')} | "
                         f"{v.get('failed_units', v.get('failed', ''))} | {v.get('eta', '')} | {v.get('minutes', '')} |")
        write_json(LOGS / "status_pipeline_lines.json", {"lines": lines})
        if OUT == V2:
            try:
                sys.path.insert(0, str(V2 / "code"))
                import status as top
                top.render()
            except Exception as exc:  # noqa: BLE001
                log(f"STATUS.md not rendered: {exc}")
        else:
            write_text(OUT / "STATUS.md", "# Sandbox status\n\n" + "\n".join(lines) + "\n")
