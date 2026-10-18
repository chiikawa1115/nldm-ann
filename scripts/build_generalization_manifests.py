"""Build compact, versioned split contracts for ASAP7 generalization runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEEDS = (11, 23, 37, 53, 71)


def family(name: str) -> str:
    match = re.search(r"asap7sc7p5t_(AO|OA|INVBUF|SIMPLE|SEQ)_", name)
    return match.group(1) if match else "UNKNOWN"


def unique_cell_catalog(parquet: Path) -> pd.DataFrame:
    pieces = []
    for batch in pq.ParquetFile(parquet).iter_batches(
        batch_size=500_000, columns=["cell_base", "logic_class", "lib_file"]
    ):
        pieces.append(batch.to_pandas().drop_duplicates())
    return unique_cell_catalog_from_frame(pd.concat(pieces, ignore_index=True))


def unique_cell_catalog_from_frame(catalog: pd.DataFrame) -> pd.DataFrame:
    """Validate and reduce cell metadata; split out for deterministic unit tests."""
    catalog = catalog.drop_duplicates().copy()
    catalog["family"] = catalog["lib_file"].map(family)
    consistency = catalog.groupby("cell_base").agg(
        logic_classes=("logic_class", "nunique"), families=("family", "nunique")
    )
    if (consistency > 1).any().any():
        raise ValueError("cell_base maps to inconsistent family or logic class")
    return catalog.drop_duplicates("cell_base")[["cell_base", "logic_class", "family"]]


def stratified_holdout(catalog: pd.DataFrame, seed: int, ratio: float) -> list[str]:
    rng = np.random.default_rng(seed)
    desired = int(round(len(catalog) * ratio))
    strata = []
    for key, group in catalog.groupby(["family", "logic_class"], sort=True):
        cells = group["cell_base"].to_numpy(copy=True)
        rng.shuffle(cells)
        raw = len(cells) * ratio
        base = min(int(np.floor(raw)), max(0, len(cells) - 1))
        strata.append({"key": key, "cells": cells, "take": base, "remainder": raw - base})
    remaining = desired - sum(item["take"] for item in strata)
    for item in sorted(strata, key=lambda x: (-x["remainder"], str(x["key"]))):
        if remaining <= 0:
            break
        if item["take"] < len(item["cells"]):
            item["take"] += 1
            remaining -= 1
    heldout = sorted(cell for item in strata for cell in item["cells"][: item["take"]])
    if len(heldout) != desired:
        raise ValueError(f"could not allocate requested heldout cells: {len(heldout)} != {desired}")
    return heldout


def write_json(path: Path, payload: dict) -> dict:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_manifests(
    parquet: Path, metadata: dict, audit: dict, out_dir: Path,
    cell_ratio: float = 0.2, point_ratio: float = 0.2,
) -> dict:
    """Build the complete compact contract and return its index payload."""
    catalog = unique_cell_catalog(parquet)
    eligible = set(audit["eligible_cells"])
    catalog = catalog[catalog["cell_base"].isin(eligible)].sort_values("cell_base")
    if len(catalog) != len(eligible):
        raise ValueError("audit eligibility and parquet cell catalog disagree")
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "schema_version": 1, "dataset": str(parquet),
        "dataset_sha256": metadata["parquet_sha256"],
        "source_commit": metadata["source_commit"],
        "eligible_cell_bases": len(eligible),
        "group_key": "cell_base",
    }
    index = []
    pvt_conditions = audit["pvt_conditions"]
    for seed in SEEDS:
        point = {**common, "mode": "random_point", "split_seed": seed,
                 "point_test_ratio": point_ratio,
                 "assignment_rule": "within each full Liberty table, sort points by deterministic uint64 hash(seed,full_table_identity,slew,load) and hold out round(n*ratio)"}
        index.append(write_json(out_dir / f"random_point_seed{seed}.json", point))
        heldout = stratified_holdout(catalog, seed, cell_ratio)
        cell = {**common, "mode": "unseen_cell", "split_seed": seed,
                "test_cell_ratio": cell_ratio, "heldout_cell_bases": heldout,
                "stratification": ["library_family", "logic_class"]}
        index.append(write_json(out_dir / f"unseen_cell_seed{seed}.json", cell))
        for pvt_index, pvt in enumerate(pvt_conditions):
            joint = {**common, "mode": "joint_unseen", "split_seed": seed,
                     "test_cell_ratio": cell_ratio, "heldout_cell_bases": heldout,
                     "heldout_pvt": pvt,
                     "cross_quadrants": "excluded_from_primary_train_and_test"}
            index.append(write_json(
                out_dir / f"joint_seed{seed}_pvt{pvt_index}_{pvt['corner']}.json", joint
            ))
    for pvt_index, pvt in enumerate(pvt_conditions):
        corner = {**common, "mode": "unseen_corner", "heldout_pvt": pvt,
                  "fold": pvt_index, "assignment_rule": "leave-one-PVT-condition-out"}
        index.append(write_json(
            out_dir / f"unseen_corner_pvt{pvt_index}_{pvt['corner']}.json", corner
        ))
    index_payload = {"schema_version": 1, "manifests": index, "count": len(index)}
    write_json(out_dir / "manifest_index.json", index_payload)
    catalog.to_csv(out_dir / "eligible_cell_catalog.csv", index=False)
    return index_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--dataset-metadata", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cell-ratio", type=float, default=0.2)
    parser.add_argument("--point-ratio", type=float, default=0.2)
    args = parser.parse_args()
    metadata = json.loads(args.dataset_metadata.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    index = build_manifests(
        args.parquet, metadata, audit, args.out_dir, args.cell_ratio, args.point_ratio
    )
    print(f"manifests={index['count']} eligible_cell_bases={len(audit['eligible_cells'])} out={args.out_dir}")


if __name__ == "__main__":
    main()
