"""PLAN v4 W2: E1–E4 split manifests, L0 table sample, and label resolution.

A manifest is a small JSON contract; `resolve_labels` turns it into per-row
train/test labels on the mainline delay rows. Every split is defined on whole
groups (table, VT, cell_base, corner) before expanding to points (PLAN 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plan_v4_data import CORNER_ORDER, TABLE_KEY, VT_ORDER

SEEDS = (11, 23, 37, 53, 71)
E1_RATIOS = (0.2, 0.5)
L0_TARGET_TABLES = 2000
L0_SEED = 20261018


def _mix64(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint64, copy=True)
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return values


def table_codes(df: pd.DataFrame) -> np.ndarray:
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(df[TABLE_KEY]), sort=True)
    return codes


def random_point_test_mask(df: pd.DataFrame, seed: int, ratio: float) -> np.ndarray:
    """Exactly round(n*ratio) test points per table (clipped to [1, n-1]), chosen by hash."""
    identity = pd.util.hash_pandas_object(
        df[TABLE_KEY + ["slew", "load"]], index=False).to_numpy(dtype=np.uint64)
    hashes = _mix64(identity ^ np.uint64(seed))
    codes = table_codes(df)
    order = np.lexsort((hashes, codes))
    sizes = np.bincount(codes)
    counts = np.clip(np.rint(sizes * ratio).astype(int), 1, sizes - 1)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    rank = np.empty(len(df), dtype=np.int64)
    rank[order] = np.arange(len(df)) - np.repeat(starts, sizes)
    return rank < counts[codes]


def e1_manifests(dataset_sha: str) -> list[dict]:
    return [{
        "schema_version": 1, "experiment": "E1", "mode": "random_point",
        "dataset_sha256": dataset_sha, "split_seed": seed, "point_test_ratio": ratio,
        "group_key": TABLE_KEY,
        "assignment_rule": "per table, rank points by mix64(hash(table key, slew, load) ^ seed); "
                           "lowest round(n*ratio) clipped to [1, n-1] are test",
    } for ratio in E1_RATIOS for seed in SEEDS]


def e2_manifests(dataset_sha: str) -> list[dict]:
    return [{
        "schema_version": 1, "experiment": "E2", "mode": "leave_one_vt_out",
        "dataset_sha256": dataset_sha, "heldout_vt": vt, "fold": i,
        "reference_vt": "LVT" if vt == "RVT" else "RVT",
        "pin_cap_feature": "pin_cap_ref_vt_lvt" if vt == "RVT" else "pin_cap_ref_vt",
        "vt_encoding": "vt_ordinal (one-hot only as negative-control ablation)",
        "train_seeds": list(SEEDS),
    } for i, vt in enumerate(VT_ORDER)]


def e3_manifests(catalog: pd.DataFrame, dataset_sha: str) -> list[dict]:
    multi = catalog[catalog["in_mainline"] & (catalog["n_sizes_in_type"] > 1)]
    out = []
    for topology, group in multi.sort_values(["topology_type", "size"]).groupby("topology_type"):
        sizes = group["size"].tolist()
        for cell, size in zip(group["cell_base"], sizes):
            out.append({
                "schema_version": 1, "experiment": "E3", "mode": "leave_one_size_out",
                "dataset_sha256": dataset_sha, "topology_type": topology,
                "family": group["family"].iloc[0], "heldout_cell_base": cell,
                "heldout_size": size, "type_sizes": sizes,
                "regime": "extrapolation" if size in (min(sizes), max(sizes)) else "interpolation",
                "train_seeds": list(SEEDS),
            })
    return out


def e4_manifests(dataset_sha: str) -> list[dict]:
    return [{
        "schema_version": 1, "experiment": "E4", "mode": "leave_one_corner_out",
        "dataset_sha256": dataset_sha, "heldout_corner": corner, "fold": i,
        "exploratory": True, "primary_claim": corner == "TT",
        "pin_cap_feature": None if corner == "TT" else "pin_cap_ref_tt",
        "note": "P/V/T are collinear across the three corners; FF/SS held out is 2-point "
                "extrapolation (PLAN 5)",
        "train_seeds": list(SEEDS),
    } for i, corner in enumerate(CORNER_ORDER)]


def l0_table_sample(tables: pd.DataFrame, target: int = L0_TARGET_TABLES,
                    seed: int = L0_SEED) -> pd.DataFrame:
    """Stratified by family x topology type: equal quota per type, leftover quota spread
    round-robin over types that still have tables. `tables`: one row per table."""
    rng = np.random.default_rng(seed)
    shuffled = tables.sample(frac=1.0, random_state=rng.integers(2**31)).reset_index(drop=True)
    shuffled["_rank"] = shuffled.groupby(["family", "topology_type"]).cumcount()
    n_types = shuffled.groupby(["family", "topology_type"]).ngroups
    quota = target // n_types
    picked = shuffled[shuffled["_rank"] < quota]
    remaining = shuffled[shuffled["_rank"] >= quota].sort_values(["_rank", "family", "topology_type"])
    picked = pd.concat([picked, remaining.head(target - len(picked))])
    return picked.drop(columns="_rank").sort_values(TABLE_KEY).reset_index(drop=True)


def resolve_labels(df: pd.DataFrame, manifest: dict) -> np.ndarray:
    """'train' / 'test' per row of the mainline delay frame."""
    mode = manifest["mode"]
    if mode == "random_point":
        test = random_point_test_mask(df, manifest["split_seed"], manifest["point_test_ratio"])
    elif mode == "leave_one_vt_out":
        test = (df["vt_class"] == manifest["heldout_vt"]).to_numpy()
    elif mode == "leave_one_size_out":
        if "topology_type" not in df:
            raise ValueError("E3 needs catalog column topology_type merged into rows")
        test = (df["cell_base"] == manifest["heldout_cell_base"]).to_numpy()
    elif mode == "leave_one_corner_out":
        test = (df["corner"] == manifest["heldout_corner"]).to_numpy()
    else:
        raise ValueError(f"unknown mode {mode}")
    return np.where(test, "test", "train")


def assert_group_disjoint(df: pd.DataFrame, labels: np.ndarray, manifest: dict) -> None:
    """Every split group must sit wholly in one partition; E1 instead checks per-table counts."""
    mode = manifest["mode"]
    group = {"leave_one_vt_out": ["vt_class"], "leave_one_size_out": ["cell_base"],
             "leave_one_corner_out": ["corner"]}.get(mode)
    is_test = labels == "test"
    if group is not None:
        mixed = pd.Series(is_test).groupby([df[c].to_numpy() for c in group]).nunique()
        if (mixed > 1).any():
            raise AssertionError(f"{mode}: groups split across partitions")
        return
    codes = table_codes(df)
    sizes = np.bincount(codes)
    tests = np.bincount(codes, weights=is_test).astype(int)
    expected = np.clip(np.rint(sizes * manifest["point_test_ratio"]).astype(int), 1, sizes - 1)
    if not np.array_equal(tests, expected):
        raise AssertionError("random_point: per-table test counts differ from rule")
