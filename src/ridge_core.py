"""Streaming, unpenalized-intercept Ridge and exact NLDM evaluation.

Only small centered moment matrices are retained while fitting. Test targets
and predictions (not feature matrices) are spooled to disk for exact quantiles.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np
import pandas as pd

from features import build_feature_frame, feature_columns
from global_engine import (
    NormalizationStats, iter_complete_table_batches, partition_labels, validation_mask,
)

TIE_RTOL = 1e-10
TIE_ATOL = 1e-12


class Moments:
    """Chan-merged centered joint moments of [X, y], in float64."""

    def __init__(self, features: int):
        self.n = 0
        self.mean = np.zeros(features + 1, dtype=np.float64)
        self.m2 = np.zeros((features + 1, features + 1), dtype=np.float64)

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(y) == 0:
            return
        values = np.column_stack((np.asarray(x, dtype=np.float64), y))
        if values.shape[1] != len(self.mean) or not np.isfinite(values).all():
            raise ValueError("moments require finite X/y of the expected shape")
        n = len(values)
        mean = values.mean(axis=0)
        centered = values - mean
        m2 = centered.T @ centered
        delta = mean - self.mean
        total = self.n + n
        self.m2 += m2 + np.outer(delta, delta) * (self.n * n / total)
        self.mean += delta * (n / total)
        self.n = total

    def normalization(self, names: list[str]) -> NormalizationStats:
        if self.n < 2 or len(names) != len(self.mean) - 1:
            raise ValueError("normalization requires at least two fit rows")
        variance = np.maximum(np.diag(self.m2) / self.n, 0.0)
        x_std = np.sqrt(variance[:-1])
        x_std[x_std < 1e-6] = 1.0
        return NormalizationStats(
            list(names), self.mean[:-1].copy(), x_std,
            float(self.mean[-1]), float(np.sqrt(max(variance[-1], 1e-12))), self.n,
        )


@dataclass
class RidgeModel:
    weights: np.ndarray
    normalization: NormalizationStats
    selected_lambda: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        s = self.normalization
        return ((np.asarray(x, dtype=np.float64) - s.x_mean) / s.x_std @ self.weights
                * s.y_std + s.y_mean)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        s = self.normalization
        with temp.open("wb") as stream:
            np.savez(stream, weights=self.weights, x_mean=s.x_mean, x_std=s.x_std,
                     y_mean=s.y_mean, y_std=s.y_std, n_fit=s.n_fit,
                     feature_names=np.asarray(s.feature_names), selected_lambda=self.selected_lambda)
        temp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "RidgeModel":
        with np.load(path, allow_pickle=False) as a:
            s = NormalizationStats(a["feature_names"].tolist(), a["x_mean"].copy(),
                                   a["x_std"].copy(), float(a["y_mean"]),
                                   float(a["y_std"]), int(a["n_fit"]))
            return cls(a["weights"].copy(), s, float(a["selected_lambda"]))


def solve_candidates(fit: Moments, validation: Moments, names: list[str],
                     lambdas: list[float]) -> tuple[RidgeModel, list[dict]]:
    if validation.n == 0:
        raise ValueError("validation partition is empty")
    if not lambdas or any(not np.isfinite(v) or v <= 0 for v in lambdas):
        raise ValueError("lambda candidates must be finite and strictly positive")
    if len(set(lambdas)) != len(lambdas):
        raise ValueError("lambda candidates must be unique")
    s = fit.normalization(names)
    gram = fit.m2[:-1, :-1] / np.outer(s.x_std, s.x_std)
    gram = (gram + gram.T) / 2
    rhs = fit.m2[:-1, -1] / (s.x_std * s.y_std)
    candidates, models = [], []
    for lam in lambdas:
        alpha = fit.n * lam
        weights = np.linalg.solve(gram + alpha * np.eye(len(names)), rhs)
        model = RidgeModel(weights, s, float(lam))
        raw_weights = weights * s.y_std / s.x_std
        # Evaluate validation MSE from its moments; no test information enters selection.
        residual_vector = np.r_[-raw_weights, 1.0]
        residual_mean = (validation.mean[-1] - s.y_mean
                         - (validation.mean[:-1] - s.x_mean) @ raw_weights)
        variance = float(residual_vector @ validation.m2 @ residual_vector / validation.n)
        if variance < -1e-9 * max(1.0, float(np.trace(validation.m2) / validation.n)):
            raise ArithmeticError("invalid negative validation residual variance")
        mse = (max(variance, 0.0) + float(residual_mean)**2) / s.y_std**2
        if not np.isfinite(mse) or not np.isfinite(weights).all():
            raise ArithmeticError("non-finite Ridge solution")
        candidates.append({"lambda": float(lam), "alpha": float(alpha),
                           "validation_mse_normalized": float(mse)})
        models.append(model)
    minimum = min(c["validation_mse_normalized"] for c in candidates)
    tied = [i for i, c in enumerate(candidates)
            if np.isclose(c["validation_mse_normalized"], minimum, rtol=TIE_RTOL, atol=TIE_ATOL)]
    best = max(tied, key=lambda i: candidates[i]["lambda"])
    return models[best], candidates


def collect_moments(parquet: Path, contract: dict, seed: int, batch_rows: int,
                    validation_ratio: float = 0.1, progress: Callable | None = None):
    fit = Moments(len(feature_columns()))
    validation = Moments(len(feature_columns()))
    counts = {"source": 0, "fit": 0, "validation": 0, "test": 0, "excluded": 0}
    for frame in iter_complete_table_batches(parquet, batch_rows):
        labels = partition_labels(frame, contract)
        val = validation_mask(frame, seed, validation_ratio)
        train = labels == "train"
        selected = frame.loc[train]
        if len(selected):
            x = build_feature_frame(selected).to_numpy(dtype=np.float64)
            y = selected["value"].to_numpy(dtype=np.float64)
            selected_val = val[train]
            fit.add(x[~selected_val], y[~selected_val])
            validation.add(x[selected_val], y[selected_val])
        counts["source"] += len(frame)
        counts["test"] += int(np.count_nonzero(labels == "test"))
        counts["excluded"] += int(np.count_nonzero(labels == "excluded"))
        counts["fit"], counts["validation"] = fit.n, validation.n
        if progress:
            progress("moments", counts.copy())
    return fit, validation, counts


def exact_metrics_from_spool(values: np.ndarray, codes: np.ndarray,
                             cell_names: list[str]) -> tuple[dict, dict, pd.DataFrame, dict]:
    # Only target/prediction vectors are materialized by the existing metric
    # implementation. At 2.7M test points this is hundreds of MB, not a full X.
    from metrics import regression_metrics
    micro = regression_metrics(values[:, 0], values[:, 1])
    rows = []
    for code, cell in enumerate(cell_names):
        mask = codes == code
        local = values[mask]
        if len(local):
            rows.append({"cell_base": cell, **regression_metrics(local[:, 0], local[:, 1])})
    per_cell = pd.DataFrame(rows).sort_values("cell_base").reset_index(drop=True)
    metric_names = [key for key in micro if key != "n"]
    macro = {"n_groups": len(per_cell)}
    macro.update({key: float(per_cell[key].mean(skipna=True)) for key in metric_names})
    undefined = {key: int(per_cell[key].isna().sum()) for key in metric_names}
    return micro, macro, per_cell, undefined


def evaluate_stream(parquet: Path, contract: dict, model: RidgeModel,
                    expected_rows: int, batch_rows: int, scratch: Path,
                    progress: Callable | None = None):
    if expected_rows <= 0:
        raise ValueError("expected test size must be positive")
    scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="ridge-eval-", dir=scratch) as temp:
        values = np.memmap(Path(temp) / "values.bin", dtype="float64", mode="w+",
                           shape=(expected_rows, 2))
        codes = np.memmap(Path(temp) / "cells.bin", dtype="int32", mode="w+",
                          shape=(expected_rows,))
        try:
            offset, negative = 0, 0
            cell_codes: dict[str, int] = {}
            for frame in iter_complete_table_batches(parquet, batch_rows):
                selected = frame.loc[partition_labels(frame, contract) == "test"]
                if selected.empty:
                    continue
                end = offset + len(selected)
                if end > expected_rows:
                    raise ValueError("test size exceeds frozen ANN test count")
                x = build_feature_frame(selected).to_numpy(dtype=np.float64)
                y = selected["value"].to_numpy(dtype=np.float64)
                pred = model.predict(x)
                if not np.isfinite(pred).all() or not np.isfinite(y).all():
                    raise ValueError("non-finite test target or prediction")
                values[offset:end, 0], values[offset:end, 1] = y, pred
                local_codes = []
                for name in selected["cell_base"].astype(str):
                    if name not in cell_codes:
                        cell_codes[name] = len(cell_codes)
                    local_codes.append(cell_codes[name])
                codes[offset:end] = local_codes
                negative += int((y < 0).sum())
                offset = end
                if progress:
                    progress("predict", {"test": offset, "expected_test": expected_rows})
            if offset != expected_rows:
                raise ValueError(f"test size {offset} differs from frozen ANN count {expected_rows}")
            values.flush()
            codes.flush()
            if progress:
                progress("exact_metrics", {"test": offset})
            micro, macro, per_cell, undefined = exact_metrics_from_spool(
                values, codes, list(cell_codes))
            return {"micro": micro, "macro": macro, "n_negative_targets": negative,
                    "macro_undefined_groups": undefined}, per_cell
        finally:
            # Release Windows mappings before TemporaryDirectory removes its own files.
            values._mmap.close()
            codes._mmap.close()
