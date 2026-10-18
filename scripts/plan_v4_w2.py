"""PLAN v4 W2 artifacts: E1–E4 manifests, L0 table sample, feature cache.

Usage:
  .venv/Scripts/python.exe scripts/plan_v4_w2.py              # manifests + cache
  .venv/Scripts/python.exe scripts/plan_v4_w2.py --skip-cache # manifests only
Manifests are small and committed. The feature cache (.npy) is git-ignored and
regenerated on each machine; its name carries the dataset hash and a hash of
the feature code so caches from different machines can be compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from plan_v4_data import (  # noqa: E402
    TABLE_KEY, load_delay_rows, provenance, verify_parquet, write_json,
)
from plan_v4_features import build_v4_feature_frame, column_groups  # noqa: E402
from plan_v4_splits import (  # noqa: E402
    assert_group_disjoint, e1_manifests, e2_manifests, e3_manifests, e4_manifests,
    l0_table_sample, resolve_labels,
)

FEATURE_CODE = ["src/features.py", "src/arc_features.py", "src/plan_v4_features.py",
                "src/plan_v4_data.py"]


def code_hash(files: list[str]) -> str:
    digest = hashlib.sha256()
    for name in files:
        # Normalise line endings so Windows (CRLF checkout) and Linux hash the same.
        digest.update((REPO / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=REPO / "data/processed/asap7_nldm.parquet")
    parser.add_argument("--reference", type=Path, default=REPO / "data/asap7_nldm.reference.json",
                        help="committed dataset reference (file + content hash)")
    parser.add_argument("--catalog", type=Path,
                        default=REPO / "experiments/plan_v4/w1/cell_type_catalog.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO / "experiments/plan_v4/manifests")
    parser.add_argument("--cache-dir", type=Path, default=REPO / "data/cache/plan_v4")
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()
    start = time.time()
    metadata = verify_parquet(args.parquet, args.reference)
    sha = metadata["parquet_sha256"]
    prov = provenance(args.parquet, metadata, REPO)
    catalog = pd.read_csv(args.catalog)

    rows = load_delay_rows(args.parquet)
    rows = rows.merge(catalog[["cell_base", "topology_type", "size"]], on="cell_base",
                      how="left", validate="many_to_one")
    rows = rows.sort_values(TABLE_KEY + ["slew", "load"], kind="mergesort").reset_index(drop=True)
    print(f"[load] {len(rows):,} rows, {time.time() - start:.0f}s")

    manifests = (e1_manifests(sha) + e2_manifests(sha) + e3_manifests(catalog, sha)
                 + e4_manifests(sha))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index, checks = [], []
    for manifest in manifests:
        labels = resolve_labels(rows, manifest)
        assert_group_disjoint(rows, labels, manifest)
        n_test = int((labels == "test").sum())
        if n_test == 0 or n_test == len(rows):
            raise AssertionError(f"degenerate split: {manifest}")
        tag = {"E1": lambda m: f"seed{m['split_seed']}_r{int(m['point_test_ratio'] * 100)}",
               "E2": lambda m: m["heldout_vt"],
               "E3": lambda m: m["heldout_cell_base"].replace("_ASAP7_75t", ""),
               "E4": lambda m: m["heldout_corner"]}[manifest["experiment"]](manifest)
        name = f"{manifest['experiment']}_{tag}.json"
        manifest = {**manifest, "test_rows": n_test, "train_rows": len(rows) - n_test}
        write_json(args.out_dir / name, manifest)
        index.append({"path": name, "sha256": file_hash(args.out_dir / name)})
        checks.append({"manifest": name, "test_rows": n_test})
    print(f"[manifests] {len(index)} written, all group-disjoint, {time.time() - start:.0f}s")

    tables = rows.drop_duplicates(TABLE_KEY)[TABLE_KEY + ["family", "topology_type"]]
    l0 = l0_table_sample(tables)
    l0.to_csv(args.out_dir / "L0_table_sample.csv", index=False, lineterminator="\n")
    l0_summary = l0.groupby("family").size().to_dict()
    write_json(args.out_dir / "manifest_index.json", {
        "schema_version": 1, "provenance": prov, "count": len(index), "manifests": index,
        "L0_table_sample": {"path": "L0_table_sample.csv",
                            "sha256": file_hash(args.out_dir / "L0_table_sample.csv"),
                            "tables": int(len(l0)), "by_family": l0_summary,
                            "types": int(l0.groupby(["family", "topology_type"]).ngroups)},
        "by_experiment": pd.Series([m["experiment"] for m in manifests]).value_counts()
        .sort_index().to_dict(),
    })
    print(f"[L0] {len(l0)} tables {l0_summary}")
    if args.skip_cache:
        return

    features = build_v4_feature_frame(rows)
    ch = code_hash(FEATURE_CODE)
    stem = f"features_{sha[:12]}_{ch[:12]}"
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.ascontiguousarray(features.to_numpy(dtype=np.float32))
    np.save(args.cache_dir / f"{stem}.npy", matrix)
    np.save(args.cache_dir / f"{stem}_target.npy", rows["value"].to_numpy(dtype=np.float32))
    rows[TABLE_KEY + ["slew", "load", "family", "topology_type", "size"]].to_parquet(
        args.cache_dir / f"{stem}_index.parquet", index=False)
    write_json(args.cache_dir / f"{stem}.json", {
        "provenance": prov, "feature_code_sha256": ch, "rows": int(len(rows)),
        "columns": list(features.columns), "groups": column_groups(list(features.columns)),
        "row_order": TABLE_KEY + ["slew", "load"],
        "matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
    })
    print(f"[cache] {stem} {matrix.shape}, {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
