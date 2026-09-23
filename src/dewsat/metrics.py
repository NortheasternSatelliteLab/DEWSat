"""Metrics and baselines.

v1 reported RMSE/MAE/R2 against a constant and a ridge fit. For drought work
those leave out the two things reviewers ask first: does it beat persistence,
and does it catch dry months. Both are here, along with rank correlation and a
threshold-free detection score, so a run can be judged without extra tooling.
"""
import numpy as np

DRY_THRESHOLD = 0.2


def _spearman(a, b):
    if len(a) < 3:
        return None
    return _pearson(_ranks(a), _ranks(b))


def _ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    # Average ranks within ties so repeated values do not bias the correlation.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique) < len(values):
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def _pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _auroc(labels, scores):
    """Probability a random positive scores above a random negative (ties count half)."""
    labels = np.asarray(labels, dtype=bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = _ranks(np.asarray(scores, dtype=np.float64))
    return float((ranks[labels].sum() - positives * (positives - 1) / 2) / (positives * negatives))


def regression_metrics(target, prediction, dry_threshold=DRY_THRESHOLD):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have the same shape.")
    error = prediction - target
    variance = float(np.square(target - target.mean()).sum())
    dry = target < dry_threshold
    return {
        "n": int(len(target)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1 - np.square(error).sum() / variance) if variance > 1e-12 else None,
        "pearson_r": _pearson(target, prediction),
        "spearman_r": _spearman(target, prediction),
        "dry_n": int(dry.sum()),
        "dry_rmse": float(np.sqrt(np.mean(error[dry] ** 2))) if dry.any() else None,
        "dry_bias": float(np.mean(error[dry])) if dry.any() else None,
    }


def event_metrics(target, prediction, threshold=DRY_THRESHOLD):
    """Contingency scores for the binary event 'target below threshold'.

    Predicting the dry class is the operational question, and RMSE can look fine
    while every dry month is missed, so these are reported separately.
    """
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    observed, forecast = target < threshold, prediction < threshold
    hits = int((observed & forecast).sum())
    misses = int((observed & ~forecast).sum())
    false_alarms = int((~observed & forecast).sum())
    correct_negatives = int((~observed & ~forecast).sum())
    total = hits + misses + false_alarms + correct_negatives
    expected = ((hits + misses) * (hits + false_alarms)
                + (correct_negatives + misses) * (correct_negatives + false_alarms)) / total
    return {
        "threshold": threshold,
        "base_rate": (hits + misses) / total,
        "hits": hits, "misses": misses, "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "pod": hits / (hits + misses) if hits + misses else None,
        "far": false_alarms / (hits + false_alarms) if hits + false_alarms else None,
        "csi": hits / (hits + misses + false_alarms) if hits + misses + false_alarms else None,
        "frequency_bias": (hits + false_alarms) / (hits + misses) if hits + misses else None,
        # Heidke skill score: agreement above chance. 0 means no better than random.
        "hss": ((hits + correct_negatives - expected) / (total - expected)
                if total - expected > 1e-12 else None),
        "auroc_dry": _auroc(observed, -prediction),
    }


def skill_score(model_rmse, reference_rmse):
    """Fraction of a reference model's error removed. Negative means worse."""
    if reference_rmse is None or reference_rmse <= 1e-12:
        return None
    return float(1 - model_rmse / reference_rmse)


def by_calendar_month(target, prediction, calendar_months):
    """RMSE per calendar month, to expose seasonally concentrated failure."""
    target, prediction = np.asarray(target), np.asarray(prediction)
    calendar_months = np.asarray(calendar_months)
    result = {}
    for month in range(1, 13):
        mask = calendar_months == month
        if mask.any():
            result[f"{month:02d}"] = {
                "n": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean((prediction[mask] - target[mask]) ** 2))),
            }
    return result


def evaluate_all(target, prediction, calendar_months=None, dry_threshold=DRY_THRESHOLD):
    result = {**regression_metrics(target, prediction, dry_threshold),
              "event": event_metrics(target, prediction, dry_threshold)}
    if calendar_months is not None:
        result["by_calendar_month"] = by_calendar_month(target, prediction, calendar_months)
    return result
