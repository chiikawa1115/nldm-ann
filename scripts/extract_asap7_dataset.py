"""Resume-safe extraction of the pinned ASAP7 source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).parents[1]
EXTRACTOR_SCHEMA_VERSION = 3
sys.path.insert(0, str(ROOT / "src"))
from data_loader import extract_lib_to_dataframe  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["files"][: args.limit]
    parts_dir = args.out.parent / f"{args.out.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for index, entry in enumerate(entries, 1):
        source = args.repo / entry["path"]
        if sha256(source) != entry["sha256"]:
            raise RuntimeError(f"source hash mismatch: {source}")
        part = parts_dir / f"v{EXTRACTOR_SCHEMA_VERSION}_{index:03d}_{entry['sha256'][:12]}.parquet"
        parts.append(part)
        if part.exists():
            print(f"[{index}/{len(entries)}] cached {source.name}")
            continue
        print(f"[{index}/{len(entries)}] extracting {source.name}")
        frame = extract_lib_to_dataframe([source])
        frame.to_parquet(part, index=False)

    writer = None
    total_rows = 0
    try:
        for part in parts:
            table = pq.read_table(part)
            if writer is None:
                writer = pq.ParquetWriter(args.out, table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if total_rows == 0:
        raise RuntimeError("extraction produced no rows")
    metadata = {
        "schema_version": EXTRACTOR_SCHEMA_VERSION, "source_manifest": str(args.manifest),
        "source_commit": manifest["commit"], "selected_files": len(entries),
        "rows": total_rows, "parquet_sha256": sha256(args.out),
        "columns": pq.ParquetFile(args.out).schema_arrow.names,
    }
    metadata_path = args.out.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"rows={total_rows} out={args.out} metadata={metadata_path}")


if __name__ == "__main__":
    main()
