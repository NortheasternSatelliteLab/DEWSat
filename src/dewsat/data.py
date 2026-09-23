"""Reading the monthly table, building windows, splitting, and scaling.

Three deliberate departures from v1:

1. A missing month is representable. v1 dropped every window that crossed a
   gap; with realistic cloud cover that discards most of the data set. Here a
   gap becomes a mean-imputed step carrying an explicit `observed` flag, and a
   window is only rejected when it has too little real signal left.
2. The target is "the value for this row's month", so `lead_months` decides
   what is being predicted. Lead 0 reproduces v1's nowcast; lead 3 is a
   three-month-ahead forecast from the same file.
3. Windows are stored as row indices, not as a materialised [N,12,7] cube, and
   batches are gathered and scaled on demand. Overlapping windows share their
   rows, so memory stays proportional to the table, not to the window count.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .contract import (CSV_FEATURES, MODEL_FEATURES, SCALED_INDICES, month_number,
                       month_string, season_features)

GROUP_COLUMNS = ("region_id", "group_id", "block_id")
SEASON_TOLERANCE = 1e-5


def read_monthly_csv(path, require_target=True, min_valid_frac=0.0):
    """Read the monthly table by column name and reject contract violations.

    Rows below `min_valid_frac` are treated as unobserved and removed, so the
    quality threshold can be swept from config without regenerating the CSV.
    """
    features, targets, cells, months, groups = [], [], [], [], []
    warnings, seen = [], set()
    dropped_low_quality = 0
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError("CSV contains duplicate column names.")
        required = {"cell_id", "month", *CSV_FEATURES} | ({"target"} if require_target else set())
        if required - set(headers):
            raise ValueError(f"CSV missing columns: {sorted(required - set(headers))}")
        group_column = next((name for name in GROUP_COLUMNS if name in headers), None)
        for line, row in enumerate(reader, start=2):
            cell = row["cell_id"].strip()
            if not cell:
                raise ValueError(f"Empty cell_id at line {line}.")
            number = month_number(row["month"])
            if (cell, number) in seen:
                raise ValueError(f"Duplicate cell/month at line {line}: {cell} {row['month']}.")
            seen.add((cell, number))
            try:
                values = np.array([float(row[name]) for name in CSV_FEATURES], dtype=np.float64)
            except (TypeError, ValueError):
                raise ValueError(f"Nonnumeric or blank feature at line {line}; omit unusable months "
                                 "instead of writing placeholders.") from None
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite feature at line {line}; omit unusable months.")
            if not -1 <= values[0] <= 1:
                raise ValueError(f"NDVI_mean must be on the physical -1..1 scale (line {line}).")
            if values[1] < 0:
                raise ValueError(f"NDVI_std must be nonnegative (line {line}).")
            if not 0 <= values[4] <= 1:
                raise ValueError(f"valid_frac must be a fraction, not a percentage (line {line}).")
            sin_month, cos_month = season_features(number)
            if not np.allclose(values[5:7], [sin_month, cos_month], atol=SEASON_TOLERANCE):
                raise ValueError(f"month_sin/month_cos do not match {row['month']} at line {line}; "
                                 "use sin/cos(2*pi*(month-1)/12).")
            if values[4] < min_valid_frac or values[4] == 0:
                dropped_low_quality += 1
                continue
            raw_target = (row.get("target") or "").strip()
            if raw_target == "":
                if require_target:
                    raise ValueError(f"Blank target at line {line}; training rows need a target or "
                                     "the month should be omitted.")
                target = np.nan
            else:
                target = float(raw_target)
                if not np.isfinite(target) or not 0 <= target <= 1:
                    raise ValueError(f"target must be finite and within [0,1] at line {line}.")
            features.append(values)
            targets.append(target)
            cells.append(cell)
            months.append(number)
            groups.append(row[group_column].strip() if group_column else cell)
    if not features:
        raise ValueError("No usable rows; the CSV is empty or every row failed the quality threshold.")

    order = np.lexsort((np.array(months), np.array(cells)))
    features = np.asarray(features)[order]
    table = {
        "features": features.astype(np.float32),
        "targets": np.asarray(targets, dtype=np.float32)[order],
        "cell_ids": [cells[i] for i in order],
        "months": np.asarray(months, dtype=np.int64)[order],
        "groups": [groups[i] for i in order],
        "group_column": group_column,
        "dropped_low_quality_rows": dropped_low_quality,
        "path": str(path),
    }
    vci = features[:, 2]
    if len(vci) > 50 and vci.max() <= 1.5:
        raise ValueError("VCI_mean never exceeds 1.5 across the file, which means it is probably on a "
                         "0-1 scale. This contract expects percentage points (about 0-100).")
    if vci.max() > 200:
        warnings.append(f"VCI_mean reaches {vci.max():.1f}; check the reference period if that is unexpected.")
    coverage = _coverage_report(table)
    warnings.extend(coverage.pop("warnings"))
    table["coverage"] = coverage
    table["warnings"] = warnings
    return table


def _coverage_report(table):
    span, present = 0, len(table["months"])
    per_cell = defaultdict(list)
    for cell, month in zip(table["cell_ids"], table["months"]):
        per_cell[cell].append(int(month))
    gaps = 0
    for values in per_cell.values():
        span += max(values) - min(values) + 1
        gaps += (max(values) - min(values) + 1) - len(values)
    warnings = []
    fraction = present / span if span else 0.0
    if fraction < 0.5:
        warnings.append(f"Only {fraction:.0%} of within-cell months are present; expect heavy imputation.")
    return {"cells": len(per_cell), "rows": int(present), "cell_month_span": int(span),
            "observed_fraction_of_span": round(fraction, 4), "interior_missing_months": int(gaps),
            "first_month": month_string(int(table["months"].min())),
            "last_month": month_string(int(table["months"].max())), "warnings": warnings}


def make_windows(table, sequence_length=12, lead_months=0, min_observed_frac=0.6,
                 require_anchor=True, require_target=True):
    """Pair a fixed month grid of inputs with the target `lead_months` later.

    A window is described by its cell, its first input month, and the row index
    of each month in the grid (-1 where the month is missing).
    """
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    if lead_months < 0:
        raise ValueError("lead_months must be zero (nowcast) or positive (forecast).")
    if not 0 < min_observed_frac <= 1:
        raise ValueError("min_observed_frac must be in (0,1].")
    minimum_observed = max(2, int(np.ceil(min_observed_frac * sequence_length)))
    by_cell = defaultdict(dict)
    for index, (cell, month) in enumerate(zip(table["cell_ids"], table["months"])):
        by_cell[cell][int(month)] = index
    targets = table["targets"]

    rows, metadata = [], []
    rejected = defaultdict(int)
    for cell in sorted(by_cell):
        available = by_cell[cell]
        first, last = min(available), max(available)
        for end in range(first + sequence_length - 1, last + 1):
            grid = [available.get(end - offset, -1) for offset in range(sequence_length - 1, -1, -1)]
            observed = sum(1 for i in grid if i >= 0)
            if observed < minimum_observed:
                rejected["too_few_observed_months"] += 1
                continue
            if require_anchor and grid[-1] < 0:
                rejected["anchor_month_missing"] += 1
                continue
            target_month = end + lead_months
            target_row = available.get(target_month, -1)
            if require_target and (target_row < 0 or not np.isfinite(targets[target_row])):
                rejected["target_month_unavailable"] += 1
                continue
            # Newest known target strictly before the target month, for the persistence
            # baseline. The strictness matters: at lead 0 the last input month *is* the
            # target month, so including it would make persistence a perfect oracle.
            persistence, persistence_age = np.nan, -1
            for offset, i in reversed(list(enumerate(grid))):
                month = end - (sequence_length - 1 - offset)
                if month < target_month and i >= 0 and np.isfinite(targets[i]):
                    persistence = float(targets[i])
                    # How stale the carried-forward value is. Gaps make it older
                    # than the lead alone implies, which changes how hard the
                    # persistence baseline is from one lead to the next.
                    persistence_age = target_month - month
                    break
            rows.append(grid)
            metadata.append({
                "cell_id": cell,
                "group": table["groups"][grid[-1] if grid[-1] >= 0 else next(i for i in grid if i >= 0)],
                "input_start_month": month_string(end - sequence_length + 1),
                "input_end_month": month_string(end),
                "target_month": month_string(target_month),
                "target_calendar_month": target_month % 12 + 1,
                "observed_months": observed,
                "target_row": int(target_row),
                "persistence": persistence,
                "persistence_age_months": persistence_age,
            })
    if not rows:
        raise ValueError(
            "No usable windows. Check sequence_length, lead_months and min_observed_frac against the "
            f"coverage report: {table['coverage']}")
    windows = {
        "row_indices": np.asarray(rows, dtype=np.int64),
        "sequence_length": sequence_length,
        "lead_months": lead_months,
        "metadata": metadata,
        "target_months": np.asarray([month_number(m["target_month"]) for m in metadata], dtype=np.int64),
        "groups": np.asarray([m["group"] for m in metadata]),
        "rejected": dict(rejected),
        "min_observed_frac": min_observed_frac,
    }
    target_rows = np.asarray([m["target_row"] for m in metadata], dtype=np.int64)
    windows["y"] = np.where(target_rows >= 0, targets[target_rows], np.nan).astype(np.float32)
    windows["persistence"] = np.asarray([m["persistence"] for m in metadata], dtype=np.float32)
    windows["persistence_age"] = np.asarray([m["persistence_age_months"] for m in metadata],
                                            dtype=np.int64)
    windows["observed_months"] = np.asarray([m["observed_months"] for m in metadata], dtype=np.int64)
    return windows


def split_windows(windows, config):
    """Split by target month, optionally holding out whole spatial blocks too."""
    mode = config.get("split_mode", "temporal")
    if mode not in {"temporal", "block_holdout"}:
        raise ValueError("split_mode must be 'temporal' or 'block_holdout'.")
    months = windows["target_months"]
    bounds, previous_end = {}, None
    for name in ("train", "val", "test"):
        start = month_number(config[f"{name}_start"])
        end = month_number(config[f"{name}_end"])
        if start > end:
            raise ValueError(f"{name}_start must not be after {name}_end.")
        if previous_end is not None and start <= previous_end:
            raise ValueError("train/val/test month ranges must be ordered and disjoint.")
        previous_end = end
        bounds[name] = (start, end)

    block_of = None
    if mode == "block_holdout":
        blocks = np.unique(windows["groups"])
        if len(blocks) < 3:
            raise ValueError(f"block_holdout needs at least three spatial blocks, found {len(blocks)}. "
                             "Add a region_id/group_id column or use split_mode 'temporal'.")
        rng = np.random.default_rng(config.get("split_seed", config.get("seed", 42)))
        shuffled = list(rng.permutation(blocks))
        n_val = max(1, int(round(config.get("val_block_frac", 0.2) * len(shuffled))))
        n_test = max(1, int(round(config.get("test_block_frac", 0.2) * len(shuffled))))
        if n_val + n_test >= len(shuffled):
            raise ValueError("val_block_frac + test_block_frac leave no blocks for training.")
        assignment = {block: "val" for block in shuffled[:n_val]}
        assignment.update({block: "test" for block in shuffled[n_val:n_val + n_test]})
        assignment.update({block: "train" for block in shuffled[n_val + n_test:]})
        block_of = np.asarray([assignment[block] for block in windows["groups"]])

    labelled = np.isfinite(windows["y"])
    embargo = int(config.get("embargo_months", 0))
    splits = {}
    for name, (start, end) in bounds.items():
        mask = labelled & (months >= start) & (months <= end)
        if block_of is not None:
            mask &= block_of == name
        if name == "train" and embargo > 0:
            # Keep training targets clear of the validation window's input months.
            mask &= months <= bounds["val"][0] - 1 - embargo
        index = np.flatnonzero(mask)
        if len(index) < 2:
            raise ValueError(
                f"Split '{name}' has {len(index)} labelled windows. Check the month ranges, "
                "split_mode block fractions, embargo_months and data coverage.")
        splits[name] = index
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(splits[a].tolist()) & set(splits[b].tolist()):
            raise RuntimeError(f"Splits {a} and {b} overlap; this is a bug, not a config error.")
    splits["_mode"] = mode
    if block_of is not None:
        splits["_blocks"] = {name: sorted(set(windows["groups"][block_of == name].tolist()))
                             for name in ("train", "val", "test")}
    return splits


def fit_scaler(table, windows, train_index, kind="zscore"):
    """Fit standardization on the unique observed months inside training windows."""
    if kind not in {"zscore", "robust"}:
        raise ValueError("scaler_kind must be 'zscore' or 'robust'.")
    used = windows["row_indices"][train_index].ravel()
    used = np.unique(used[used >= 0])
    values = table["features"][used][:, SCALED_INDICES].astype(np.float64)
    if kind == "zscore":
        center, spread = values.mean(axis=0), values.std(axis=0)
    else:
        center = np.median(values, axis=0)
        quartiles = np.percentile(values, [25, 75], axis=0)
        spread = (quartiles[1] - quartiles[0]) / 1.3489795  # matches sigma for a normal sample
    return {"kind": kind, "feature_names": MODEL_FEATURES, "scaled_indices": list(SCALED_INDICES),
            "center": center.tolist(), "spread": np.maximum(spread, 1e-6).tolist(),
            "fit_monthly_rows": int(len(used)),
            "fit_last_month": month_string(int(table["months"][used].max()))}


def build_inputs(table, windows, index, scaler):
    """Gather and standardize a batch of windows into [batch, sequence, 8] inputs.

    Missing months become standardized zeros, which is mean imputation, plus a
    zero in the `observed` channel so the network can discount them. Calendar
    features are always exact because they are computed, not observed.
    """
    index = np.asarray(index, dtype=np.int64)
    rows = windows["row_indices"][index]
    sequence_length = windows["sequence_length"]
    present = rows >= 0
    gathered = table["features"][np.where(present, rows, 0)].astype(np.float32)
    center = np.asarray(scaler["center"], dtype=np.float32)
    spread = np.asarray(scaler["spread"], dtype=np.float32)
    gathered[:, :, SCALED_INDICES] = (gathered[:, :, SCALED_INDICES] - center) / spread
    batch = np.zeros((len(index), sequence_length, len(MODEL_FEATURES)), dtype=np.float32)
    batch[:, :, :len(CSV_FEATURES)] = np.where(present[:, :, None], gathered, 0.0)
    batch[:, :, -1] = present.astype(np.float32)
    # Season is a property of the month, so fill it for imputed steps as well.
    starts = np.asarray([month_number(windows["metadata"][i]["input_start_month"]) for i in index])
    grid = starts[:, None] + np.arange(sequence_length)[None, :]
    angle = 2 * np.pi * (grid % 12) / 12
    batch[:, :, 5] = np.sin(angle)
    batch[:, :, 6] = np.cos(angle)
    if not np.isfinite(batch).all():
        raise RuntimeError("Nonfinite model input after scaling; inspect the scaler statistics.")
    return batch
