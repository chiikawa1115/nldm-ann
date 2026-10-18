"""Record or check the content fingerprint of the ASAP7 parquet.

The reference lives in git (`data/asap7_nldm.reference.json`) so a machine that
rebuilds the parquet from the public ORFS libs can prove it has the same data.

  # on the machine that holds the reference parquet (done once, committed)
  python scripts/dataset_fingerprint.py write
  # on any other machine after rebuilding
  python scripts/dataset_fingerprint.py check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from plan_v4_data import content_fingerprint, file_sha256, verify_parquet, write_json  # noqa: E402

REFERENCE = REPO / "data/asap7_nldm.reference.json"
PARQUET = REPO / "data/processed/asap7_nldm.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["write", "check"])
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--metadata", type=Path,
                        default=REPO / "data/processed/asap7_nldm.metadata.json")
    args = parser.parse_args()
    if args.action == "check":
        result = verify_parquet(args.parquet, args.reference)
        print(json.dumps({"status": "PASS", "verified_by": result["verified_by"],
                          "rows": result["rows"]}, indent=2))
        return
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if file_sha256(args.parquet) != metadata["parquet_sha256"]:
        raise SystemExit("FAIL: parquet does not match its metadata; refusing to write reference")
    fingerprint = content_fingerprint(pd.read_parquet(args.parquet))
    if fingerprint["rows"] != metadata["rows"]:
        raise SystemExit("FAIL: row count differs from metadata")
    write_json(args.reference, {
        "schema_version": 1,
        "parquet_sha256": metadata["parquet_sha256"],
        "source_commit": metadata["source_commit"],
        "source_manifest": "data/asap7_sources.json",
        "extractor_schema_version": metadata["schema_version"],
        **fingerprint,
    })
    print(f"wrote {args.reference} content_sha256={fingerprint['content_sha256']}")


if __name__ == "__main__":
    main()
