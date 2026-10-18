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


# Rows are uniquely identified by these columns (one point of one table).
ROW_IDENTITY = ["lib_file", "cell", "output_pin", "related_pin", "timing_type",
                "timing_sense", "when", "table_type", "slew", "load"]
CONTENT_FLOAT_DECIMALS = 9


def _column_bytes(values: pd.Series) -> bytes:
    if pd.api.types.is_float_dtype(values):
        arr = np.round(values.to_numpy(dtype="<f8"), CONTENT_FLOAT_DECIMALS)
        arr = np.where(np.isnan(arr), np.nan, arr) + 0.0   # canonical NaN, -0.0 -> 0.0
        return arr.astype("<f8").tobytes()
    if pd.api.types.is_integer_dtype(values):
        return values.to_numpy(dtype="<i8").tobytes()
    return "\x00".join(values.astype(str).tolist()).encode("utf-8")


def content_fingerprint(df: pd.DataFrame) -> dict:
    """Hash of the table content, independent of row order and parquet encoding.

    The parquet file hash changes with pyarrow version or compression even when
    the data are identical, so a rebuilt dataset is checked by content instead:
    rows are put in ROW_IDENTITY order, floats rounded to CONTENT_FLOAT_DECIMALS.
    """
    if df.duplicated(ROW_IDENTITY).any():
        raise ValueError("ROW_IDENTITY does not uniquely identify rows")
    codes = [pd.factorize(df[c], sort=True)[0] for c in reversed(ROW_IDENTITY)]
    ordered = df.iloc[np.lexsort(codes)]
    columns = {c: hashlib.sha256(_column_bytes(ordered[c])).hexdigest()
               for c in sorted(df.columns)}
    overall = hashlib.sha256(
        "\n".join(f"{c}:{h}" for c, h in columns.items()).encode())
    return {"content_sha256": overall.hexdigest(), "rows": int(len(df)),
            "column_sha256": columns, "float_decimals": CONTENT_FLOAT_DECIMALS}


def verify_parquet(parquet: Path, reference_path: Path) -> dict:
    """Fail fast unless the parquet matches the reference dataset.

    Same file hash: accepted directly. Different file hash (e.g. rebuilt on
    another machine): accepted only if the content fingerprint matches. The
    returned dict keeps `parquet_sha256` as the canonical dataset ID that all
    manifests and caches refer to.
    """
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    actual = file_sha256(parquet)
    if actual == reference["parquet_sha256"]:
        return {**reference, "verified_by": "file_sha256", "local_parquet_sha256": actual}
    if "content_sha256" not in reference:
        raise ValueError(
            f"parquet sha256 mismatch: {actual} != {reference['parquet_sha256']} ({parquet}); "
            "reference has no content hash to fall back on")
    found = content_fingerprint(pd.read_parquet(parquet))
    if found["content_sha256"] != reference["content_sha256"]:
        bad = sorted(c for c, h in reference["column_sha256"].items()
                     if found["column_sha256"].get(c) != h)
        raise ValueError(f"parquet content mismatch in columns {bad} ({parquet})")
    return {**reference, "verified_by": "content_sha256", "local_parquet_sha256": actual}


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
        "dataset_verified_by": metadata.get("verified_by"),
        "local_parquet_sha256": metadata.get("local_parquet_sha256"),
        "source_commit": metadata.get("source_commit"),
        "git_commit": commit,
        "git_dirty_src": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Always LF: write_text emits CRLF on Windows, which would change the hashes
    # that other machines compare against.
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_bytes(text.encode("utf-8"))
