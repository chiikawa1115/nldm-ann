"""Leakage-safe manifests for the thesis generalization experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiment import ARC_COLS, add_point_ids

PVT_COLS = ["corner", "nom_process", "nom_voltage_v", "nom_temperature_c"]
MODES = ("random_point", "unseen_cell", "unseen_corner", "joint_unseen")


def _eligible_balanced_cells(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    pvt_keys = list(df[PVT_COLS].drop_duplicates().itertuples(index=False, name=None))
    group_col = "cell_base" if "cell_base" in df.columns else "cell"
    coverage = df[[group_col, *PVT_COLS]].drop_duplicates().groupby(group_col, dropna=False).size()
    eligible = coverage[coverage == len(pvt_keys)].index
    filtered = df[df[group_col].isin(eligible)].copy()
    if filtered.empty:
        raise ValueError("no cells are represented in every PVT group")
    return filtered, pvt_keys


def make_generalization_manifest(
    df: pd.DataFrame, mode: str, seed: int,
    test_cell_ratio: float = 0.2, point_test_ratio: float = 0.2,
    heldout_pvt_index: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    if mode not in MODES:
        raise ValueError(f"unknown split mode: {mode}")
    required = set(ARC_COLS + PVT_COLS + ["cell"])
    if missing := required - set(df.columns):
        raise ValueError(f"missing split columns: {sorted(missing)}")
    rng = np.random.default_rng(seed)

    if mode == "random_point":
        work = df.copy()
        labels = pd.Series("train", index=work.index, dtype="string")
        for _, group in work.groupby(ARC_COLS, sort=True, dropna=False):
            n_test = int(np.clip(round(len(group) * point_test_ratio), 1, len(group) - 1))
            labels.loc[rng.choice(group.index.to_numpy(), n_test, replace=False)] = "test"
        held_cells: list[str] = []
        held_pvt = None
    else:
        work, pvt_keys = _eligible_balanced_cells(df)
        group_col = "cell_base" if "cell_base" in work.columns else "cell"
        cells = np.array(sorted(work[group_col].unique()))
        rng.shuffle(cells)
        n_test_cells = int(np.clip(round(len(cells) * test_cell_ratio), 1, len(cells) - 1))
        held_cells = sorted(cells[:n_test_cells].tolist())
        if heldout_pvt_index is None:
            heldout_pvt_index = seed % len(pvt_keys)
        held_pvt = pvt_keys[heldout_pvt_index % len(pvt_keys)]
        is_held_cell = work[group_col].isin(held_cells)
        is_held_pvt = pd.Series(True, index=work.index)
        for column, value in zip(PVT_COLS, held_pvt):
            is_held_pvt &= work[column].eq(value)

        if mode == "unseen_cell":
            labels = pd.Series(np.where(is_held_cell, "test", "train"), index=work.index)
            held_pvt = None
        elif mode == "unseen_corner":
            labels = pd.Series(np.where(is_held_pvt, "test", "train"), index=work.index)
            held_cells = []
        else:
            labels = pd.Series("excluded_seen_cell_heldout_pvt", index=work.index)
            labels.loc[is_held_cell & ~is_held_pvt] = "excluded_heldout_cell_seen_pvt"
            labels.loc[~is_held_cell & ~is_held_pvt] = "train"
            labels.loc[is_held_cell & is_held_pvt] = "test"

    manifest = add_point_ids(work)
    manifest["split"] = labels.loc[work.index].to_numpy()
    metadata = {
        "schema_version": 1, "mode": mode, "seed": seed,
        "test_cell_ratio": test_cell_ratio, "point_test_ratio": point_test_ratio,
        "heldout_cells": held_cells,
        "heldout_pvt": dict(zip(PVT_COLS, held_pvt)) if held_pvt is not None else None,
        "n_source_rows": len(df), "n_eligible_rows": len(work),
        "n_train": int((manifest["split"] == "train").sum()),
        "n_test": int((manifest["split"] == "test").sum()),
    }
    assert_no_leakage(manifest, mode)
    return manifest, metadata


def assert_no_leakage(manifest: pd.DataFrame, mode: str) -> None:
    train = manifest[manifest["split"] == "train"]
    test = manifest[manifest["split"] == "test"]
    if train.empty or test.empty:
        raise ValueError("split must contain non-empty train and test partitions")
    if set(train["point_id"]) & set(test["point_id"]):
        raise ValueError("point leakage between train and test")
    group_col = "cell_base" if "cell_base" in manifest.columns else "cell"
    if mode in ("unseen_cell", "joint_unseen") and set(train[group_col]) & set(test[group_col]):
        raise ValueError("cell leakage between train and test")
    if mode in ("unseen_corner", "joint_unseen"):
        train_pvt = set(train[PVT_COLS].itertuples(index=False, name=None))
        test_pvt = set(test[PVT_COLS].itertuples(index=False, name=None))
        if train_pvt & test_pvt:
            raise ValueError("PVT leakage between train and test")
