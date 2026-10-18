import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_generalization_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_generalization_manifests", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)

RUNNER_SCRIPT = Path(__file__).parents[1] / "src" / "run_generalization.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_generalization", RUNNER_SCRIPT)
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(runner)


def _catalog_frame() -> pd.DataFrame:
    rows = []
    for index in range(20):
        rows.append({
            "cell_base": f"CELL{index:02d}",
            "logic_class": "INV" if index % 2 == 0 else "NAND",
            "lib_file": (
                "asap7sc7p5t_INVBUF_RVT_FF.lib"
                if index % 2 == 0 else "asap7sc7p5t_SIMPLE_RVT_FF.lib"
            ),
        })
    return pd.DataFrame(rows)


def test_stratified_holdout_is_exact_and_deterministic():
    catalog = builder.unique_cell_catalog_from_frame(_catalog_frame())
    first = builder.stratified_holdout(catalog, seed=11, ratio=0.2)
    second = builder.stratified_holdout(catalog, seed=11, ratio=0.2)
    assert first == second
    assert len(first) == 4
    selected = catalog[catalog.cell_base.isin(first)]
    assert set(selected.family) == {"INVBUF", "SIMPLE"}
    assert set(selected.logic_class) == {"INV", "NAND"}


def test_compact_manifest_set_has_expected_count_and_stable_hashes():
    runtime = Path(__file__).parent / "runtime"
    runtime.mkdir(exist_ok=True)
    parquet = runtime / "manifest_test_tiny.parquet"
    _catalog_frame().to_parquet(parquet, index=False)
    metadata = {
        "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "source_commit": "deadbeef",
    }
    audit = {
        "eligible_cells": sorted(_catalog_frame().cell_base.unique()),
        "pvt_conditions": [
            {"corner": "FF", "nom_process": 1.0, "nom_voltage_v": 0.77,
             "nom_temperature_c": 0.0},
            {"corner": "SS", "nom_process": -1.0, "nom_voltage_v": 0.63,
             "nom_temperature_c": 100.0},
            {"corner": "TT", "nom_process": 0.0, "nom_voltage_v": 0.70,
             "nom_temperature_c": 25.0},
        ],
    }
    index_a = builder.build_manifests(parquet, metadata, audit, runtime, 0.2, 0.2)
    index_b = builder.build_manifests(parquet, metadata, audit, runtime, 0.2, 0.2)
    assert index_a["count"] == index_b["count"] == 28
    assert index_a["manifests"] == index_b["manifests"]
    for item in index_a["manifests"]:
        digest = hashlib.sha256((runtime / item["path"]).read_bytes()).hexdigest()
        assert digest == item["sha256"]
    unseen = json.loads(
        (runtime / "unseen_cell_seed11.json").read_text(encoding="utf-8")
    )
    assert len(unseen["heldout_cell_bases"]) == 4
    expanded = list(runner.runs(runtime))
    assert len(expanded) == 40
    assert sum(contract["mode"] == "unseen_corner" for _, contract, _ in expanded) == 15
