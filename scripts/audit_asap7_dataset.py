"""Streaming coverage and feature-readiness audit for the ASAP7 parquet."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TABLE_COLS = [
    "lib_file", "library_name", "corner", "nom_process", "nom_voltage_v",
    "nom_temperature_c", "cell", "cell_base", "output_pin", "related_pin", "timing_sense",
    "timing_type", "when", "table_type", "template",
]
FEATURE_COLS = [
    "drive_strength", "related_pin_cap_ff", "nom_voltage_v", "nom_temperature_c",
    "logic_class", "logic_function", "vt_class",
]
SCAN_COLS = list(dict.fromkeys(TABLE_COLS + FEATURE_COLS + ["value", "slew", "load"]))


def family(name: str) -> str:
    match = re.search(r"asap7sc7p5t_(AO|OA|INVBUF|SIMPLE|SEQ)_", name)
    return match.group(1) if match else "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parquet = pq.ParquetFile(args.parquet)
    total = 0
    missing = {column: 0 for column in FEATURE_COLS}
    numeric = {column: [float("inf"), float("-inf")] for column in ("value", "slew", "load")}
    negative_values = 0
    table_parts, coverage_parts = [], []
    for batch in parquet.iter_batches(batch_size=250_000, columns=SCAN_COLS):
        frame = batch.to_pandas()
        total += len(frame)
        for column in FEATURE_COLS:
            missing[column] += int(frame[column].isna().sum())
        for column in numeric:
            numeric[column][0] = min(numeric[column][0], float(frame[column].min()))
            numeric[column][1] = max(numeric[column][1], float(frame[column].max()))
        negative_values += int((frame["value"] < 0).sum())
        table_parts.append(frame[TABLE_COLS].drop_duplicates())
        coverage_parts.append(frame[
            ["lib_file", "cell", "cell_base", "corner", "nom_process", "nom_voltage_v",
             "nom_temperature_c", "vt_class", "logic_class"]
        ].drop_duplicates())
    tables = pd.concat(table_parts, ignore_index=True).drop_duplicates()
    coverage = pd.concat(coverage_parts, ignore_index=True).drop_duplicates()
    coverage["family"] = coverage["lib_file"].map(family)
    pvt_cols = ["corner", "nom_process", "nom_voltage_v", "nom_temperature_c"]
    pvt = coverage[pvt_cols].drop_duplicates().sort_values(pvt_cols)
    cell_pvt = coverage[["cell_base", *pvt_cols]].drop_duplicates()
    pvt_per_cell = cell_pvt.groupby("cell_base").size()
    eligible = sorted(pvt_per_cell[pvt_per_cell == len(pvt)].index.tolist())
    source_grid = (
        coverage.groupby(["family", "vt_class", "corner"], dropna=False)["cell"]
        .nunique().rename("unique_cells").reset_index()
    )
    duplicate_table_ids = int(tables.duplicated(TABLE_COLS).sum())
    report = {
        "schema_version": 1, "parquet": str(args.parquet), "rows": total,
        "unique_tables": len(tables), "unique_cells": int(coverage["cell"].nunique()),
        "unique_cell_bases": int(coverage["cell_base"].nunique()),
        "eligible_cells_all_pvt": len(eligible), "eligible_cells": eligible,
        "pvt_conditions": pvt.to_dict(orient="records"),
        "vt_classes": sorted(coverage["vt_class"].unique().tolist()),
        "families": sorted(coverage["family"].unique().tolist()),
        "missing_feature_rows": missing,
        "numeric_ranges": {key: {"min": value[0], "max": value[1]} for key, value in numeric.items()},
        "negative_value_rows": negative_values,
        "duplicate_table_id_rows": duplicate_table_ids,
        "source_cell_coverage": source_grid.to_dict(orient="records"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    source_grid.to_csv(args.out.with_suffix(".coverage.csv"), index=False)
    print(json.dumps({key: report[key] for key in (
        "rows", "unique_tables", "unique_cells", "eligible_cells_all_pvt",
        "pvt_conditions", "vt_classes", "families", "missing_feature_rows",
        "negative_value_rows", "duplicate_table_id_rows",
    )}, indent=2))


if __name__ == "__main__":
    main()
