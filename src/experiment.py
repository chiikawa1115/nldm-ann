"""Reproducible sparse-grid splits and experiment manifests."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ARC_COLS = [
    "lib_file", "library_name", "corner", "nom_process", "nom_voltage_v",
    "nom_temperature_c", "cell", "output_pin", "related_pin",
    "timing_sense", "timing_type", "when", "table_type", "template",
]
POINT_COLS = ARC_COLS + ["slew", "load"]


def add_point_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add exact, portable IDs so CSV float formatting cannot change split joins."""
    missing = set(POINT_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    encoded = []
    for row in df[POINT_COLS].itertuples(index=False, name=None):
        fields = [float(value).hex() if isinstance(value, (float, np.floating)) else str(value)
                  for value in row]
        encoded.append(hashlib.blake2b("\x1f".join(fields).encode(), digest_size=16).hexdigest())
    result = df.copy()
    result["point_id"] = encoded
    if result["point_id"].duplicated().any():
        raise ValueError("point identity is not unique; duplicate Liberty grid points exist")
    return result


def _choose_axis(values: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Keep domain boundaries and choose reproducible interior characterization points."""
    values = np.sort(np.unique(values.astype(float)))
    if count >= len(values):
        return values
    if count < 2:
        raise ValueError("A bilinear grid needs at least two points on each axis")
    interior = values[1:-1].copy()
    rng.shuffle(interior)
    return np.sort(np.concatenate(([values[0]], interior[: count - 2], [values[-1]])))


def _axis_counts(n_slew: int, n_load: int, train_ratio: float) -> tuple[int, int]:
    """Choose the realizable rectangular grid closest to the requested ratio."""
    target = n_slew * n_load * train_ratio
    candidates = []
    for keep_slew in range(2, n_slew + 1):
        for keep_load in range(2, n_load + 1):
            if keep_slew == n_slew and keep_load == n_load:
                continue
            error = abs(keep_slew * keep_load - target)
            imbalance = abs(keep_slew / n_slew - keep_load / n_load)
            candidates.append((error, imbalance, -keep_slew * keep_load, keep_slew, keep_load))
    if not candidates:
        raise ValueError("grid is too small to create a holdout")
    *_, keep_slew, keep_load = min(candidates)
    return keep_slew, keep_load


def split_by_regular_subgrid(
    df: pd.DataFrame, train_ratio: float, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retain a rectangular axis subgrid per LUT and return an explicit manifest.

    Boundary points are retained so every held-out original grid point remains inside
    the interpolation domain. Tables with fewer than 3 points on either axis cannot
    produce an interpolation holdout and are omitted.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    missing = set(POINT_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    assignments: list[pd.DataFrame] = []
    for _, group in df.groupby(ARC_COLS, sort=True, dropna=False):
        slews = np.sort(group["slew"].unique())
        loads = np.sort(group["load"].unique())
        if len(slews) < 3 or len(loads) < 3:
            continue
        n_slew, n_load = _axis_counts(len(slews), len(loads), train_ratio)
        kept_slews = _choose_axis(slews, n_slew, rng)
        kept_loads = _choose_axis(loads, n_load, rng)
        part = group[POINT_COLS].copy()
        part["split"] = np.where(
            part["slew"].isin(kept_slews) & part["load"].isin(kept_loads),
            "train",
            "test",
        )
        assignments.append(part)

    if not assignments:
        raise ValueError("no tables contain a grid large enough for bilinear holdout")
    manifest = add_point_ids(pd.concat(assignments, ignore_index=True))
    source = add_point_ids(df)
    keyed = source.merge(manifest[["point_id", "split"]], on="point_id", how="inner",
                         validate="one_to_one")
    return (
        keyed[keyed["split"] == "train"].drop(columns="split").reset_index(drop=True),
        keyed[keyed["split"] == "test"].drop(columns="split").reset_index(drop=True),
        manifest.sort_values("point_id").reset_index(drop=True),
    )


def dataframe_sha256(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    canonical = df[columns] if columns else df
    payload = canonical.sort_values(list(canonical.columns)).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(manifest: pd.DataFrame, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    meta = dict(metadata)
    meta.update({
        "manifest_sha256": file_sha256(path),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_commit": _git_commit(path.parent),
    })
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )


def _git_commit(start: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=start, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
