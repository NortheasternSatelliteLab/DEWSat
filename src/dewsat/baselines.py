"""Reference predictors every reported result is measured against.

A sequence model is only interesting if it beats the cheap answers. v1 compared
against a constant and a ridge fit; the two that actually threaten a drought
model are persistence (last known percentile) and seasonal climatology (what
this cell usually looks like this month), so both are computed here. All of
them are fitted on training windows only.
"""
from collections import defaultdict

import numpy as np


def train_mean(windows, splits):
    value = float(np.mean(windows["y"][splits["train"]]))
    return lambda index: np.full(len(index), value, dtype=np.float32), {"value": value}


def persistence(windows, splits):
    """Carry the newest known target in the input window forward.

    With lead 0 this is nearly unbeatable and exposes a trivial target; with a
    positive lead it is the honest bar a forecast has to clear.
    """
    values = windows["persistence"]
    fallback = float(np.nanmean(windows["y"][splits["train"]]))

    def predict(index):
        out = values[index].astype(np.float32).copy()
        return np.where(np.isfinite(out), out, fallback)

    ages = windows["persistence_age"][splits["test"]]
    usable = ages[ages > 0]
    return predict, {
        "test_coverage": float(np.isfinite(values[splits["test"]]).mean()),
        "fallback": fallback,
        # Gaps make the carried-forward value older than the lead implies, so the
        # difficulty of this baseline is not comparable across leads without it.
        "mean_age_months": float(usable.mean()) if len(usable) else None,
        "max_age_months": int(usable.max()) if len(usable) else None,
    }


def seasonal_climatology(windows, splits):
    """Mean training target per (cell, calendar month), with coarser fallbacks."""
    per_cell_month, per_cell, per_month, overall = defaultdict(list), defaultdict(list), defaultdict(list), []
    for i in splits["train"]:
        meta = windows["metadata"][i]
        value = float(windows["y"][i])
        per_cell_month[(meta["cell_id"], meta["target_calendar_month"])].append(value)
        per_cell[meta["cell_id"]].append(value)
        per_month[meta["target_calendar_month"]].append(value)
        overall.append(value)
    global_mean = float(np.mean(overall))
    tables = ({k: float(np.mean(v)) for k, v in per_cell_month.items()},
              {k: float(np.mean(v)) for k, v in per_cell.items()},
              {k: float(np.mean(v)) for k, v in per_month.items()})
    cell_month, cell_only, month_only = tables
    levels = defaultdict(int)

    def predict(index):
        out = np.empty(len(index), dtype=np.float32)
        for position, i in enumerate(index):
            meta = windows["metadata"][i]
            key = (meta["cell_id"], meta["target_calendar_month"])
            if key in cell_month:
                out[position], level = cell_month[key], "cell_month"
            elif meta["cell_id"] in cell_only:
                out[position], level = cell_only[meta["cell_id"]], "cell"
            elif meta["target_calendar_month"] in month_only:
                out[position], level = month_only[meta["target_calendar_month"]], "calendar_month"
            else:
                out[position], level = global_mean, "global"
            levels[level] += 1
        return out

    return predict, {"fallback_use": levels, "global_mean": global_mean}


def ridge(table, windows, splits, scaler, penalty=1.0):
    """Ridge regression on the flattened standardized window, fitted on train."""
    from .data import build_inputs
    train_index = splits["train"]
    x = build_inputs(table, windows, train_index, scaler).reshape(len(train_index), -1).astype(np.float64)
    y = windows["y"][train_index].astype(np.float64)
    center, offset = x.mean(axis=0), float(y.mean())
    centered = x - center
    coefficients = np.linalg.solve(centered.T @ centered + penalty * np.eye(centered.shape[1]),
                                   centered.T @ (y - offset))

    def predict(index):
        features = build_inputs(table, windows, index, scaler).reshape(len(index), -1).astype(np.float64)
        return np.clip((features - center) @ coefficients + offset, 0, 1).astype(np.float32)

    return predict, {"penalty": penalty, "n_features": int(x.shape[1])}


def build_all(table, windows, splits, scaler, ridge_penalty=1.0):
    """Return {name: (predict_fn, info)} for every reference predictor."""
    return {
        "train_mean": train_mean(windows, splits),
        "persistence": persistence(windows, splits),
        "seasonal_climatology": seasonal_climatology(windows, splits),
        "ridge": ridge(table, windows, splits, scaler, ridge_penalty),
    }
