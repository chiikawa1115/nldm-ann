"""Fail-fast readiness gate before starting long ASAP7 training runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import pyarrow.parquet as pq
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--allow-cpu", action="store_true", help="local smoke checks only")
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    actual_hash = sha256(args.parquet)
    if actual_hash != metadata["parquet_sha256"]:
        raise SystemExit("FAIL: dataset SHA-256 mismatch")
    index = json.loads((args.manifest_dir / "manifest_index.json").read_text(encoding="utf-8"))
    if index["count"] != 28:
        raise SystemExit(f"FAIL: expected 28 split contracts, found {index['count']}")
    for item in index["manifests"]:
        path = args.manifest_dir / item["path"]
        if sha256(path) != item["sha256"]:
            raise SystemExit(f"FAIL: manifest hash mismatch: {path}")
    rows = pq.ParquetFile(args.parquet).metadata.num_rows
    if rows != metadata["rows"]:
        raise SystemExit(f"FAIL: parquet row mismatch: {rows} != {metadata['rows']}")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("FAIL: torch.cuda.is_available() is false; do not start formal training")
    payload = {
        "status": "PASS", "python": platform.python_version(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda, "rows": rows,
        "dataset_sha256": actual_hash, "split_contracts": index["count"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

