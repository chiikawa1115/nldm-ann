import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import json

import numpy as np
import pandas as pd
import pytest

from cell_type_catalog import build_catalog, spot_check_sample, summarize
from conflict_report import conflict_report
from constraint_check import (
    check_all, recommend, rule_corner_order, rule_load_monotonic, rule_slew_monotonic,
    rule_vt_order, tables_as_cube,
)
from plan_v4_data import family_from_lib, parse_cell_base, size_value, verify_parquet

SLEWS, LOADS = [5.0, 10.0, 20.0], [1.0, 2.0, 4.0]
CORNER_SCALE = {"FF": 0.8, "TT": 1.0, "SS": 1.3}
VT_SCALE = {"SLVT": 0.9, "LVT": 1.0, "RVT": 1.1, "SRAM": 1.2}


def golden_rows(cells=("INVx1", "INVx2"), when_values=("",)):
    """Physically consistent synthetic delays: every ordering rule holds exactly."""
    rows = []
    for stem in cells:
        topology, _, size, _ = parse_cell_base(f"{stem}_ASAP7_75t")
        for corner, cs in CORNER_SCALE.items():
            for vt, vs in VT_SCALE.items():
                for when_index, when in enumerate(when_values):
                    for table, tf in (("cell_rise", 1.0), ("cell_fall", 0.9)):
                        for slew in SLEWS:
                            for load in LOADS:
                                value = cs * vs * tf * (10 + slew + load / size) + when_index
                                rows.append({
                                    "lib_file": f"asap7sc7p5t_INVBUF_{vt}_{corner}_nldm.lib",
                                    "cell": f"{stem}_ASAP7_75t_{vt[0]}",
                                    "cell_base": f"{stem}_ASAP7_75t", "corner": corner,
                                    "vt_class": vt, "family": "INVBUF",
                                    "topology_type": topology, "size": size,
                                    "output_pin": "Y", "related_pin": "A",
                                    "timing_sense": "negative_unate",
                                    "timing_type": "combinational", "when": when,
                                    "table_type": table, "slew": slew, "load": load,
                                    "log_slew": np.log(slew), "log_load": np.log(load),
                                    "nom_voltage_v": 0.7, "nom_temperature_c": 25.0,
                                    "drive_strength": size, "related_pin_cap_ff": 0.5 * size,
                                    "n_input_pins": 1, "n_output_pins": 1,
                                    "logic_class": "INV", "logic_function": "!A",
                                    "value": value,
                                })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("name,expected", [
    ("INVx1_ASAP7_75t", ("INV", "1", 1.0, "")),
    ("INVxp33_ASAP7_75t", ("INV", "p33", 0.33, "")),
    ("NAND2x1p5_ASAP7_75t", ("NAND2", "1p5", 1.5, "")),
    ("CKINVDCx9p33_ASAP7_75t", ("CKINVDC", "9p33", 9.33, "")),
    ("BUFx4f_ASAP7_75t", ("BUFx4f", "4", 4.0, "f")),
    ("ICGx4DC_ASAP7_75t", ("ICGx4DC", "4", 4.0, "DC")),
    ("HB3xp67_ASAP7_75t", ("HB3", "p67", 0.67, "")),
    ("AO333x2_ASAP7_75t", ("AO333", "2", 2.0, "")),
])
def test_parse_cell_base(name, expected):
    assert parse_cell_base(name) == expected


def test_size_value_and_family():
    assert size_value("p5") == 0.5 and size_value("2p67") == 2.67 and size_value("12") == 12.0
    assert family_from_lib("asap7sc7p5t_AO_RVT_FF_nldm_211120.lib.gz") == "AO"
    assert family_from_lib("other.lib") == "UNKNOWN"


def catalog_inputs(bases_families):
    cells = pd.DataFrame([
        {"cell_base": f"{b}_ASAP7_75t", "family": f, "logic_class": "X", "vt_class": vt}
        for b, f in bases_families for vt in ("SLVT", "LVT", "RVT", "SRAM")
    ])
    counts = pd.DataFrame({"cell_base": cells["cell_base"].unique(), "delay_tables": 24})
    return cells, counts


def test_catalog_types_sizes_and_review_flags():
    cells, counts = catalog_inputs([
        ("INVx1", "INVBUF"), ("INVx2", "INVBUF"), ("BUFx4", "INVBUF"), ("BUFx4f", "INVBUF"),
        ("HB1xp67", "INVBUF"), ("NAND2xp5", "SIMPLE"), ("DFFHQNx1", "SEQ"),
    ])
    cat = build_catalog(cells, counts).set_index("cell_base")
    assert cat.loc["INVx2_ASAP7_75t", "n_sizes_in_type"] == 2
    assert cat.loc["BUFx4f_ASAP7_75t", "topology_type"] == "BUFx4f"
    assert cat.loc["BUFx4f_ASAP7_75t", "needs_review"]
    assert cat.loc["HB1xp67_ASAP7_75t", "needs_review"]
    assert not cat.loc["INVx1_ASAP7_75t", "needs_review"]
    assert not cat.loc["DFFHQNx1_ASAP7_75t", "in_mainline"]
    summary = summarize(cat.reset_index())
    assert summary["mainline_function_types"] == 5   # INV, BUF, BUFx4f, HB1, NAND2
    assert summary["mainline_multi_size_types"] == 1
    assert summary["excluded_seq_cell_base"] == 1


def test_catalog_flags_missing_vt_and_rejects_inconsistent_family():
    cells, counts = catalog_inputs([("INVx1", "INVBUF")])
    cat = build_catalog(cells[cells["vt_class"] != "SRAM"], counts)
    assert cat.loc[0, "needs_review"] and "VT" in cat.loc[0, "review_reason"]
    bad = pd.concat([cells, cells.assign(family="SIMPLE")])
    with pytest.raises(ValueError):
        build_catalog(bad, counts)


def test_spot_check_is_deterministic_and_covers_families():
    cells, counts = catalog_inputs(
        [(f"INVx{i}", "INVBUF") for i in range(1, 9)]
        + [(f"NAND2x{i}", "SIMPLE") for i in range(1, 9)]
        + [("AO21x1", "AO"), ("OA21x2", "OA")])
    cat = build_catalog(cells, counts)
    first, second = spot_check_sample(cat), spot_check_sample(cat)
    assert first.equals(second) and len(first) == 10
    assert set(first["family"]) == {"INVBUF", "SIMPLE", "AO", "OA"}


def test_conflict_report_detects_missing_when_and_resolves_with_it():
    rows = golden_rows(cells=("INVx1",), when_values=("", "(B)"))
    report, mask = conflict_report(rows)
    assert report["baseline_conflict_ratio"] == 1.0 and mask.all()
    assert report["conflict_rows_if_add_one"]["when"] == 0
    assert report["conflict_rows_if_add_one"]["related_pin"] == len(rows)


def test_conflict_report_zero_when_features_identify_rows():
    report, _ = conflict_report(golden_rows(cells=("INVx1",)))
    assert report["baseline_conflict_rows"] == 0


def test_rules_all_pass_on_consistent_golden():
    results = check_all(golden_rows(cells=("INVx1", "INVx2", "INVx4")))
    for name in ("load_monotonic", "slew_monotonic", "corner_order", "vt_order_tt",
                 "vt_order_all_corners", "size_delay", "size_pin_cap"):
        assert results[name]["rate"] == 1.0, name
        assert results[name]["recommendation"] == "hard"
    assert results["rise_fall_consistency"]["median"] == 1.0
    assert results["_meta"]["grid"] == [3, 3]


def test_rules_count_injected_violations_exactly():
    rows = golden_rows(cells=("INVx1",))
    keys, cube = tables_as_cube(rows)
    n_tables = len(keys)
    # One load-direction violation in one table.
    broken = cube.copy()
    broken[0, 0, 2] = broken[0, 0, 1] - 1
    load = rule_load_monotonic(broken)
    assert load["compared"] == n_tables * 3 * 2 and load["ok"] == load["compared"] - 1
    # One slew-direction violation.
    broken = cube.copy()
    broken[0, 2, 0] = broken[0, 1, 0] - 1
    slew = rule_slew_monotonic(broken)
    assert slew["ok"] == slew["compared"] - 1 and slew["violations_by_slew_step"] == [0, 1]
    # FF slower than TT at one grid point of one triple.
    broken = cube.copy()
    ff = keys.index[(keys["corner"] == "FF")][0]
    broken[ff, 1, 1] = 1e6
    corner = rule_corner_order(keys, broken)
    assert corner["compared"] == corner["table_triples"] * 9
    assert corner["ok"] == corner["compared"] - 1
    # SRAM faster than RVT at TT.
    broken = cube.copy()
    sram = keys.index[(keys["corner"] == "TT") & (keys["vt_class"] == "SRAM")][0]
    broken[sram] = 0.0
    vt = rule_vt_order(keys, broken, "TT")
    assert vt["ok"] == vt["compared"] - 9


def test_tables_as_cube_rejects_ragged_table():
    rows = golden_rows(cells=("INVx1",))
    with pytest.raises(ValueError):
        tables_as_cube(rows.iloc[1:])


def test_size_rules_detect_inverted_sizes():
    rows = golden_rows(cells=("INVx1", "INVx2"))
    rows.loc[rows["size"] == 2.0, "value"] += 1000.0
    rows.loc[rows["size"] == 2.0, "related_pin_cap_ff"] = 0.01
    results = check_all(rows)
    assert results["size_delay"]["rate"] == 0.0
    assert results["size_pin_cap"]["rate"] == 0.0
    assert results["size_delay"]["recommendation"] == "report_only"


def test_recommend_thresholds():
    assert recommend(1.0) == "hard" and recommend(0.999) == "hard"
    assert recommend(0.9836) == "penalty" and recommend(0.79) == "report_only"


def test_verify_parquet_rejects_mismatch(tmp_path):
    data = tmp_path / "d.parquet"
    data.write_bytes(b"abc")
    meta = tmp_path / "m.json"
    meta.write_text(json.dumps({"parquet_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_parquet(data, meta)
    meta.write_text(json.dumps({
        "parquet_sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    }), encoding="utf-8")
    assert verify_parquet(data, meta)["parquet_sha256"].startswith("ba7816")


def _write_reference(tmp_path, frame):
    from plan_v4_data import content_fingerprint, file_sha256
    original = tmp_path / "orig.parquet"
    frame.to_parquet(original, index=False)
    reference = tmp_path / "ref.json"
    reference.write_text(json.dumps({
        "parquet_sha256": file_sha256(original), **content_fingerprint(frame)}), encoding="utf-8")
    return original, reference


def test_content_fingerprint_accepts_reencoded_rejects_changed(tmp_path):
    frame = golden_rows(cells=("INVx1",)).assign(lib_file=lambda d: d["lib_file"] + d["table_type"])
    original, reference = _write_reference(tmp_path, frame)
    assert verify_parquet(original, reference)["verified_by"] == "file_sha256"
    shuffled = tmp_path / "shuffled.parquet"
    frame.sample(frac=1.0, random_state=0).to_parquet(shuffled, compression="snappy", index=False)
    result = verify_parquet(shuffled, reference)
    assert result["verified_by"] == "content_sha256"
    assert result["local_parquet_sha256"] != result["parquet_sha256"]
    tiny = tmp_path / "rounding.parquet"
    frame.assign(value=frame["value"] + 1e-12).to_parquet(tiny, index=False)
    assert verify_parquet(tiny, reference)["verified_by"] == "content_sha256"
    changed = frame.copy()
    changed.loc[0, "value"] += 1e-3
    bad = tmp_path / "bad.parquet"
    changed.to_parquet(bad, index=False)
    with pytest.raises(ValueError, match=r"content mismatch in columns \['value'\]"):
        verify_parquet(bad, reference)
    short = tmp_path / "short.parquet"
    frame.iloc[1:].to_parquet(short, index=False)
    with pytest.raises(ValueError):
        verify_parquet(short, reference)
