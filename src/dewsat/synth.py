"""Synthetic monthly data with the failure modes real HLS/FLDAS data has.

v1's generator produced a target that was close to a linear function of the
current features, so ridge regression matched the Bi-LSTM and the sequence model
had nothing to earn. This generator keeps the same CSV schema but makes the
problem genuinely sequential:

* soil water is a nonlinear store with seasonal drainage, so the state has to be
  integrated over months rather than read off one row;
* vegetation lags the store by one to two months and saturates, so features are
  a blurred, clipped view of what the target depends on;
* clouds remove whole months in seasonal clusters, so gaps must be handled;
* cells belong to regions that share climate, so holding out regions is a real
  generalization test rather than a reshuffle;
* the target is a percentile taken against a frozen reference period that ends
  before the validation years, mirroring the discipline real targets need.

It is still a toy. It contains no land-surface physics and its numbers say
nothing about drought accuracy.
"""
import csv
from pathlib import Path

import numpy as np

from .contract import CSV_FEATURES, month_number, month_string, season_features, write_json

GENERATOR_ID = "DEWSat v2 synthetic soil-store generator"


def _stream(seed, *keys):
    return np.random.default_rng([seed, *keys])


def _simulate_cell(months, region_climate, rng, cloud_level):
    """Return per-month soil store, vegetation-derived features and observation mask."""
    n = len(months)
    store = 0.4 + 0.1 * rng.normal()
    vegetation = 0.4
    aridity = float(np.clip(0.55 + 0.18 * rng.normal(), 0.15, 0.95))  # local dryness
    lag = rng.uniform(0.35, 0.75)                                      # vegetation inertia
    soil = np.empty(n)
    ndvi = np.empty(n)
    ndvi_std = np.empty(n)
    vci = np.empty(n)
    vci_p10 = np.empty(n)
    valid = np.empty(n)
    observed = np.ones(n, dtype=bool)
    for t, number in enumerate(months):
        phase = 2 * np.pi * (number % 12) / 12
        wet_season = 0.5 * (1 + np.sin(phase))                 # 1 in the rainy peak
        rain = np.exp(region_climate[t] + 0.8 * wet_season - aridity + 0.35 * rng.normal())
        # Saturating infiltration and season-dependent drainage make the store nonlinear.
        drainage = 0.10 + 0.22 * (1 - wet_season)
        store = np.clip(store * (1 - drainage) + 0.30 * rain / (1 + rain), 0.0, 1.0)
        soil[t] = store
        # Vegetation integrates the store with inertia, then saturates.
        vegetation = lag * vegetation + (1 - lag) * store
        greenness = vegetation ** 0.7
        ndvi[t] = np.clip(0.12 + 0.62 * greenness + 0.07 * np.sin(phase) + 0.020 * rng.normal(), -1, 1)
        ndvi_std[t] = np.clip(0.030 + 0.055 * (1 - greenness) + 0.008 * rng.normal(), 0.005, 0.25)
        vci[t] = 100 * np.clip(0.05 + 0.92 * greenness + 0.030 * rng.normal(), 0, 1)
        vci_p10[t] = max(0.0, vci[t] - rng.uniform(5, 20) * (1.3 - 0.5 * greenness))
        # Cloud cover tracks the wet season, so gaps arrive in seasonal runs.
        clear = np.clip(0.94 - cloud_level * (0.25 + 0.75 * wet_season), 0.02, 1.0)
        valid[t] = np.clip(clear * rng.uniform(0.6, 1.05), 0.0, 1.0)
        observed[t] = rng.uniform() < clear
    return {"soil": soil, "NDVI_mean": ndvi, "NDVI_std": ndvi_std, "VCI_mean": vci,
            "VCI_p10": vci_p10, "valid_frac": valid, "observed": observed}


def _percentile_target(soil, reference_mask):
    """Empirical percentile of each month against a frozen reference window.

    The reference period ends before the validation years on purpose: a target
    percentile fitted on the full record would leak held-out information, which
    is the single easiest way to invent skill that does not exist.
    """
    reference = np.sort(soil[reference_mask])
    # Mid-rank convention so ties land between their bracketing ranks.
    below = np.searchsorted(reference, soil, side="left")
    at_or_below = np.searchsorted(reference, soil, side="right")
    return np.clip((below + at_or_below) / (2 * len(reference)), 0.0, 1.0)


def generate_synthetic(output, n_cells=64, seed=42, start="2015-01", end="2024-12",
                       n_regions=4, cloud=0.35, reference_end="2021-12", overwrite=False):
    if n_cells < 2:
        raise ValueError("Use at least two synthetic cells.")
    if not 0 <= cloud <= 1:
        raise ValueError("cloud must be a fraction in [0,1].")
    output = Path(output)
    manifest_path = output.with_suffix(".json")
    if not overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError(f"{output} or its manifest exists; pass --overwrite or pick a new name.")
    months = list(range(month_number(start), month_number(end) + 1))
    reference_mask = np.array([m <= month_number(reference_end) for m in months])
    if reference_mask.sum() < 36:
        raise ValueError("The percentile reference period needs at least 36 months.")

    # Regional climate anomalies: slow multi-year swings shared by every cell in a region.
    region_climate = []
    for region in range(n_regions):
        rng = _stream(seed, 1, region)
        anomaly, series = 0.0, []
        cycle, phase = rng.uniform(30, 70), rng.uniform(0, 2 * np.pi)
        for t in range(len(months)):
            anomaly = 0.85 * anomaly + rng.normal(0, 0.22)
            series.append(anomaly + 0.45 * np.sin(2 * np.pi * t / cycle + phase))
        region_climate.append(np.array(series))

    output.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "month", *CSV_FEATURES, "target", "region_id"])
        for cell in range(n_cells):
            region = cell % n_regions
            series = _simulate_cell(months, region_climate[region], _stream(seed, 2, cell), cloud)
            target = _percentile_target(series["soil"], reference_mask)
            for t, number in enumerate(months):
                if not series["observed"][t]:
                    continue  # A month with no usable observation gets no row at all.
                sin_month, cos_month = season_features(number)
                values = [series["NDVI_mean"][t], series["NDVI_std"][t], series["VCI_mean"][t],
                          series["VCI_p10"][t], series["valid_frac"][t], sin_month, cos_month,
                          target[t]]
                writer.writerow([f"cell_{cell:04d}", month_string(number),
                                 *[f"{float(v):.8f}" for v in values], f"region_{region:02d}"])
                emitted += 1

    manifest = {
        "schema_version": 2,
        "data_kind": "synthetic",
        "target_name": "synthetic_soil_store_percentile",
        "target_range": [0, 1],
        "target_timing": "value_for_its_own_month",
        "spatial_unit": "synthetic_cell",
        "feature_names": CSV_FEATURES,
        "preprocessing": {
            "generator": GENERATOR_ID,
            "seed": seed,
            "n_cells": n_cells,
            "n_regions": n_regions,
            "months": f"{start}..{end}",
            "cloud_level": cloud,
            "target_reference_period": f"{start}..{reference_end}",
            "target_method": "empirical mid-rank percentile of a simulated soil store "
                             "against a frozen reference period",
            "missing_month_policy": "months with no simulated clear observation are omitted, not filled",
            "note": "Toy dynamics with no land-surface physics. Features are invented summaries; "
                    "VCI is not computed from imagery and the target is not FLDAS soil moisture.",
        },
        "known_limitations": [
            "Scores on this data measure software behaviour and nothing about drought accuracy.",
            "Regions share one synthetic climate signal, so held-out-region results are optimistic "
            "relative to real geographic transfer.",
        ],
    }
    write_json(manifest_path, manifest)
    return {"csv": str(output), "manifest": str(manifest_path), "rows": emitted,
            "cells": n_cells, "regions": n_regions,
            "possible_rows": n_cells * len(months),
            "observed_fraction": round(emitted / (n_cells * len(months)), 4),
            "data_kind": "synthetic"}
