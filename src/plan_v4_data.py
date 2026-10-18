"""Shared PLAN v4 data access: mainline delay rows, cell-type parsing, provenance."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

DELAY_TABLES = ("cell_rise", "cell_fall")
FAMILIES = ("SIMPLE", "AO", "OA", "INVBUF", "SEQ")
MAINLINE_FAMILIES = ("SIMPLE", "AO", "OA", "INVBUF")
VT_ORDER = ("SLVT", "LVT", "RVT", "SRAM")
CORNER_ORDER = ("FF", "TT", "SS")
# One delay table = one Liberty timing arc direction of one cell at one corner.
ARC_COLUMNS = ["output_pin", "related_pin", "timing_sense", "timing_type", "when", "table_type"]
TABLE_KEY = ["cell_base", "vt_class", "corner"] + ARC_COLUMNS

_SIZE_RE = re.compile(r"^(?P<stem>.+?)x(?P<size>p\d+|\d+(?:p\d+)?)(?P<tail>[A-Za-z]*)$")


def family_from_lib(lib_file: str) -> str:
    match = re.search(r"asap7sc7p5t_(AO|OA|INVBUF|SIMPLE|SEQ)_", str(lib_file))
    return match.group(1) if match else "UNKNOWN"


def size_value(label: str) -> float:
    """ASAP7 drive label to number: '2' -> 2.0, 'p33' -> 0.33, '1p5' -> 1.5."""
    text = label[1:] if label.startswith("p") else label
    whole, _, frac = text.partition("p")
    if label.startswith("p"):
        return float(f"0.{whole}")
    return float(f"{whole}.{frac}") if frac else float(whole)


def parse_cell_base(cell_base: str) -> tuple[str, str | None, float, str]:
    """Return (topology_type, size_label, size_value, name_suffix) per PLAN v4 section 2.

    Only a trailing drive-strength token is stripped. A name that continues
    after the drive token (BUFx4f, ICGx4DC) keeps its full stem as the type, so
    it becomes its own single-size topology type; the catalog flags these for
    manual review instead of guessing that 'f'/'DC' variants are the same type.
    """
    stem = re.sub(r"_ASAP7_75t$", "", cell_base)
    match = _SIZE_RE.match(stem)
    if not match:
        return stem, None, float("nan"), ""
    topology = stem if match["tail"] else match["stem"]
    return topology, match["size"], size_value(match["size"]), match["tail"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_parquet(parquet: Path, metadata_path: Path) -> dict:
    """Fail fast when the copied parquet differs from the recorded dataset."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual = file_sha256(parquet)
    if actual != metadata["parquet_sha256"]:
        raise ValueError(
            f"parquet sha256 mismatch: {actual} != {metadata['parquet_sha256']} ({parquet})"
        )
    return metadata


def load_delay_rows(parquet: Path, columns: list[str] | None = None,
                    mainline_only: bool = True) -> pd.DataFrame:
    """Load cell_rise/cell_fall rows with a `family` column; SEQ dropped by default."""
    wanted = None if columns is None else sorted(set(columns) | {"lib_file"})
    df = pd.read_parquet(
        parquet, columns=wanted, filters=[("table_type", "in", list(DELAY_TABLES))]
    )
    df["family"] = df["lib_file"].map(family_from_lib)
    if (df["family"] == "UNKNOWN").any():
        raise ValueError("lib_file without a recognised ASAP7 family")
    if mainline_only:
        df = df[df["family"].isin(MAINLINE_FAMILIES)].reset_index(drop=True)
    return df


def provenance(parquet: Path, metadata: dict, repo: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "scripts"], cwd=repo,
            check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "dataset": parquet.name,
        "dataset_sha256": metadata["parquet_sha256"],
        "source_commit": metadata.get("source_commit"),
        "git_commit": commit,
        "git_dirty_src": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")
