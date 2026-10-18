"""Streaming utilities for leakage-free feature-conditioned NLDM experiments."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from experiment import ARC_COLS
from features import build_feature_frame, feature_columns
from generalization_split import PVT_COLS

DATA_COLUMNS = sorted(set(ARC_COLS + PVT_COLS + [
    "cell_base", "logic_function", "logic_class", "drive_strength", "vt_class",
    "related_pin_cap_ff", "n_input_pins", "n_output_pins", "slew", "load",
    "log_slew", "log_load", "value",
]))


def file_sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path, parquet: Path, verify_hash: bool = True) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported generalization manifest schema")
    if verify_hash and file_sha256(parquet) != contract["dataset_sha256"]:
        raise ValueError("dataset SHA-256 does not match split contract")
    return contract


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _same_key(frame: pd.DataFrame, key: tuple) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for column, value in zip(ARC_COLS, key):
        series = frame[column]
        mask &= series.isna().to_numpy() if pd.isna(value) else series.eq(value).to_numpy()
    return mask


def iter_complete_table_batches(parquet: Path, batch_rows: int = 250_000) -> Iterator[pd.DataFrame]:
    """Yield batches without splitting a contiguous Liberty table at boundaries."""
    carry: pd.DataFrame | None = None
    for record_batch in pq.ParquetFile(parquet).iter_batches(
        batch_size=batch_rows, columns=DATA_COLUMNS
    ):
        frame = record_batch.to_pandas()
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        last_key = tuple(frame.iloc[-1][ARC_COLS])
        last_mask = _same_key(frame, last_key)
        complete = frame.loc[~last_mask]
        carry = frame.loc[last_mask].copy()
        if not complete.empty:
            yield complete.reset_index(drop=True)
    if carry is not None and not carry.empty:
        yield carry.reset_index(drop=True)


def _mix64(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint64, copy=True)
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return values


def random_point_test_mask(frame: pd.DataFrame, seed: int, ratio: float) -> np.ndarray:
    """Select the exact rounded test count independently inside every full table."""
    identity_hash = pd.util.hash_pandas_object(
        frame[ARC_COLS + ["slew", "load"]], index=False
    ).to_numpy(dtype=np.uint64)
    hashes = _mix64(identity_hash ^ np.uint64(seed))
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(frame[ARC_COLS]), sort=False)
    test = np.zeros(len(frame), dtype=bool)
    for code in range(codes.max() + 1):
        indices = np.flatnonzero(codes == code)
        count = int(np.clip(round(len(indices) * ratio), 1, len(indices) - 1))
        test[indices[np.argsort(hashes[indices], kind="stable")[:count]]] = True
    return test


def _pvt_mask(frame: pd.DataFrame, heldout: dict) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for column in PVT_COLS:
        value = heldout[column]
        if isinstance(value, (int, float)):
            mask &= np.isclose(frame[column].to_numpy(dtype=float), float(value), rtol=0, atol=1e-12)
        else:
            mask &= frame[column].eq(value).to_numpy()
    return mask


def partition_labels(frame: pd.DataFrame, contract: dict) -> np.ndarray:
    mode = contract["mode"]
    if mode == "random_point":
        is_test = random_point_test_mask(
            frame, int(contract["split_seed"]), float(contract["point_test_ratio"])
        )
        return np.where(is_test, "test", "train")
    held_cells = set(contract.get("heldout_cell_bases", []))
    is_cell = frame["cell_base"].isin(held_cells).to_numpy()
    if mode == "unseen_cell":
        return np.where(is_cell, "test", "train")
    is_pvt = _pvt_mask(frame, contract["heldout_pvt"])
    if mode == "unseen_corner":
        return np.where(is_pvt, "test", "train")
    if mode == "joint_unseen":
        labels = np.full(len(frame), "excluded", dtype=object)
        labels[~is_cell & ~is_pvt] = "train"
        labels[is_cell & is_pvt] = "test"
        return labels
    raise ValueError(f"unknown contract mode: {mode}")


def validation_mask(frame: pd.DataFrame, seed: int, ratio: float = 0.1) -> np.ndarray:
    """Deterministic validation subset drawn only from the contract's train partition."""
    identity_hash = pd.util.hash_pandas_object(
        frame[ARC_COLS + ["slew", "load"]], index=False
    ).to_numpy(dtype=np.uint64)
    hashes = _mix64(identity_hash ^ np.uint64(seed + 0x9E3779B9))
    threshold = np.uint64(round(ratio * np.iinfo(np.uint64).max))
    return hashes <= threshold


@dataclass
class NormalizationStats:
    feature_names: list[str]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    n_fit: int

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(),
            "y_mean": self.y_mean, "y_std": self.y_std, "n_fit": self.n_fit,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "NormalizationStats":
        return cls(
            list(payload["feature_names"]), np.asarray(payload["x_mean"], dtype=np.float64),
            np.asarray(payload["x_std"], dtype=np.float64), float(payload["y_mean"]),
            float(payload["y_std"]), int(payload["n_fit"]),
        )


def fit_normalization(
    parquet: Path, contract: dict, batch_rows: int, max_rows: int | None = None,
    validation_ratio: float = 0.1,
) -> NormalizationStats:
    n = 0
    x_sum = np.zeros(len(feature_columns()), dtype=np.float64)
    x_sumsq = np.zeros_like(x_sum)
    y_sum = y_sumsq = 0.0
    seed = int(contract.get("_run_seed", contract.get("split_seed", contract.get("fold", 0))))
    for frame in iter_complete_table_batches(parquet, batch_rows):
        labels = partition_labels(frame, contract)
        fit = (labels == "train") & ~validation_mask(frame, seed, validation_ratio)
        if max_rows is not None:
            fit_indices = np.flatnonzero(fit)[: max(0, max_rows - n)]
            fit = np.zeros(len(frame), dtype=bool)
            fit[fit_indices] = True
        if fit.any():
            x = build_feature_frame(frame.loc[fit]).to_numpy(dtype=np.float64)
            y = frame.loc[fit, "value"].to_numpy(dtype=np.float64)
            n += len(y)
            x_sum += x.sum(axis=0); x_sumsq += np.square(x).sum(axis=0)
            y_sum += float(y.sum()); y_sumsq += float(np.square(y).sum())
        if max_rows is not None and n >= max_rows:
            break
    if n < 2:
        raise ValueError("normalization fit partition has fewer than two rows")
    x_mean = x_sum / n
    x_variance = np.maximum(x_sumsq / n - np.square(x_mean), 0.0)
    x_std = np.sqrt(x_variance)
    x_std[x_std < 1e-6] = 1.0
    y_mean = y_sum / n
    y_std = float(np.sqrt(max(y_sumsq / n - y_mean**2, 1e-12)))
    return NormalizationStats(feature_columns(), x_mean, x_std, y_mean, y_std, n)


def iter_tensors(
    parquet: Path, contract: dict, stats: NormalizationStats, partition: str,
    batch_rows: int, max_rows: int | None = None, validation_ratio: float = 0.1,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, pd.DataFrame]]:
    emitted = 0
    seed = int(contract.get("_run_seed", contract.get("split_seed", contract.get("fold", 0))))
    for frame in iter_complete_table_batches(parquet, batch_rows):
        labels = partition_labels(frame, contract)
        val = validation_mask(frame, seed, validation_ratio)
        if partition == "fit":
            mask = (labels == "train") & ~val
        elif partition == "validation":
            mask = (labels == "train") & val
        elif partition == "test":
            mask = labels == "test"
        else:
            raise ValueError(f"unknown partition: {partition}")
        selected = frame.loc[mask]
        if max_rows is not None:
            selected = selected.iloc[: max(0, max_rows - emitted)]
        if not selected.empty:
            x = build_feature_frame(selected).to_numpy(dtype=np.float32)
            x = ((x - stats.x_mean) / stats.x_std).astype(np.float32)
            y = ((selected["value"].to_numpy(dtype=np.float32) - stats.y_mean) / stats.y_std)
            emitted += len(selected)
            yield torch.from_numpy(x), torch.from_numpy(y[:, None]), selected
        if max_rows is not None and emitted >= max_rows:
            break
