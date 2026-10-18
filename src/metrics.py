"""Paper metrics with stable definitions for near-zero timing values."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must have the same non-empty shape")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("metrics require finite values")
    error = y_pred - y_true
    absolute = np.abs(error)
    scale = np.abs(y_true) + np.abs(y_pred)
    smape = np.divide(2 * absolute, scale, out=np.zeros_like(absolute), where=scale > 0)
    nonzero = np.abs(y_true) > np.finfo(float).eps
    relative = absolute[nonzero] / np.abs(y_true[nonzero])
    ss_res = np.sum(error**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "n": int(y_true.size),
        "MAE": float(absolute.mean()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "sMAPE_pct": float(100 * smape.mean()),
        "P50RE_pct": float(100 * np.percentile(relative, 50)) if relative.size else float("nan"),
        "P90RE_pct": float(100 * np.percentile(relative, 90)) if relative.size else float("nan"),
        "P99RE_pct": float(100 * np.percentile(relative, 99)) if relative.size else float("nan"),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def micro_macro_metrics(
    frame: pd.DataFrame, true_col: str = "y_true", pred_col: str = "y_pred",
    group_col: str = "cell",
) -> tuple[dict, dict, pd.DataFrame]:
    """Return point-weighted micro and equal-cell macro metrics."""
    required = {true_col, pred_col, group_col}
    if missing := required - set(frame.columns):
        raise ValueError(f"missing metric columns: {sorted(missing)}")
    micro = regression_metrics(frame[true_col].to_numpy(), frame[pred_col].to_numpy())
    rows = []
    for group, values in frame.groupby(group_col, sort=True, dropna=False):
        row = {group_col: group}
        row.update(regression_metrics(values[true_col].to_numpy(), values[pred_col].to_numpy()))
        rows.append(row)
    per_group = pd.DataFrame(rows)
    metric_names = [name for name in per_group.columns if name not in (group_col, "n")]
    macro = {"n_groups": len(per_group)}
    macro.update({name: float(per_group[name].mean(skipna=True)) for name in metric_names})
    return micro, macro, per_group
