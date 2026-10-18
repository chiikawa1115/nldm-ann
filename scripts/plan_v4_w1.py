"""PLAN v4 W1 artifacts: cell type catalog, feature-conflict report, constraint table.

Usage:
  .venv/Scripts/python.exe scripts/plan_v4_w1.py --out-dir experiments/plan_v4/w1
Re-running on the same parquet and code must reproduce every number.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cell_type_catalog import build_catalog, spot_check_sample, summarize  # noqa: E402
from conflict_report import conflict_report, top_conflict_types  # noqa: E402
from constraint_check import check_all  # noqa: E402
from plan_v4_data import (  # noqa: E402
    load_delay_rows, provenance, verify_parquet, write_json,
)

FEATURE_INPUTS = [
    "cell", "cell_base", "corner", "vt_class", "nom_voltage_v", "nom_temperature_c",
    "drive_strength", "related_pin_cap_ff", "n_input_pins", "n_output_pins",
    "logic_class", "logic_function", "output_pin", "related_pin", "timing_sense",
    "timing_type", "when", "table_type", "slew", "load", "log_slew", "log_load", "value",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=REPO / "data/processed/asap7_nldm.parquet")
    parser.add_argument("--reference", type=Path, default=REPO / "data/asap7_nldm.reference.json",
                        help="committed dataset reference (file + content hash)")
    parser.add_argument("--out-dir", type=Path, default=REPO / "experiments/plan_v4/w1")
    args = parser.parse_args()
    start = time.time()
    metadata = verify_parquet(args.parquet, args.reference)
    prov = provenance(args.parquet, metadata, REPO)

    everything = load_delay_rows(args.parquet, FEATURE_INPUTS + ["lib_file"], mainline_only=False)
    tables = everything.drop_duplicates(
        ["cell_base", "vt_class", "corner", "output_pin", "related_pin", "timing_sense",
         "timing_type", "when", "table_type"]
    ).groupby("cell_base").size().rename("delay_tables").reset_index()
    cells = everything[["cell_base", "family", "logic_class", "vt_class"]].drop_duplicates()
    catalog = build_catalog(cells, tables)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(args.out_dir / "cell_type_catalog.csv", index=False, lineterminator="\n")
    spot_check_sample(catalog).to_csv(args.out_dir / "cell_type_spot_check.csv",
                                      index=False, lineterminator="\n")
    write_json(args.out_dir / "cell_type_catalog_summary.json",
               {"provenance": prov, **summarize(catalog)})
    print(f"[catalog] {len(catalog)} cell_base, {time.time() - start:.0f}s")

    rows = everything[everything["family"] != "SEQ"].reset_index(drop=True)
    del everything
    rows = rows.merge(catalog[["cell_base", "topology_type", "size"]], on="cell_base",
                      how="left", validate="many_to_one")
    report, conflict = conflict_report(rows)
    top = top_conflict_types(rows, conflict)
    top.to_csv(args.out_dir / "conflict_top_types.csv", index=False, lineterminator="\n")
    write_json(args.out_dir / "conflict_report.json", {"provenance": prov, **report})
    print(f"[conflict] {report['baseline_conflict_rows']}/{report['rows']} "
          f"= {report['baseline_conflict_ratio']:.4%}, {time.time() - start:.0f}s")

    results = check_all(rows)
    write_json(args.out_dir / "constraint_check.json", {"provenance": prov, **results})
    table = pd.DataFrame([
        {"rule": name, "unit": res["unit"], "rate": res["rate"],
         "recommendation": res["recommendation"]}
        for name, res in results.items() if not name.startswith("_")
    ])
    table.to_csv(args.out_dir / "constraint_check.csv", index=False, lineterminator="\n")
    print(table.to_string(index=False))
    print(f"[done] {time.time() - start:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
