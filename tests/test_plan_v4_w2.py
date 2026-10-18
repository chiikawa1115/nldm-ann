import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pytest

from arc_features import (
    MAX_PINS, WHEN_SLOT_DONT_CARE, WHEN_SLOT_SELF, WHEN_SLOT_UNUSED, arc_feature_frame,
    canonical_pins, parse_sop, pin_structure, when_slots,
)
from conflict_report import _conflict_mask, _row_hash
from plan_v4_features import build_v4_feature_frame, reference_pin_caps
from plan_v4_splits import (
    assert_group_disjoint, e1_manifests, e2_manifests, e3_manifests, e4_manifests,
    l0_table_sample, random_point_test_mask, resolve_labels,
)
from test_plan_v4_w1 import golden_rows

AO21 = "(A1 * A2) + (B)"


def test_parse_sop_and_canonical_order():
    assert parse_sop("") == ()
    assert parse_sop("(A * !B) + (C)") == ({"A": 1, "B": 0}, {"C": 1})
    assert parse_sop("!A") == ({"A": 0},)
    assert canonical_pins("(C1 * A2) + (B) + (A1)") == ("A1", "A2", "B", "C1")
    with pytest.raises(ValueError):
        parse_sop("(A ^ B)")


def test_pin_structure_matches_boolean_reasoning():
    # AO21 = A1*A2 + B: B flips Y unless A1*A2 is 1 -> 3/4; A1 flips Y iff A2=1 and B=0 -> 1/4.
    infl_b, n_terms_b, len_b, group_b, func_terms, n_inputs = pin_structure(AO21, "B")
    assert (infl_b, n_terms_b, len_b, group_b, func_terms, n_inputs) == (0.75, 1, 1, 1, 2, 3)
    infl_a1, _, len_a1, group_a1, _, _ = pin_structure(AO21, "A1")
    assert (infl_a1, len_a1, group_a1) == (0.25, 2, 2)
    assert pin_structure("!A", "A")[0] == 1.0
    assert pin_structure("(A * !B) + (!A * B)", "A")[0] == 1.0   # XOR
    with pytest.raises(ValueError):
        pin_structure(AO21, "C")


def test_when_slots_encoding():
    slots = when_slots(AO21, "B", "(A1 * !A2)")
    assert slots[:3] == (1.0, 0.0, WHEN_SLOT_SELF)
    assert slots[3:MAX_PINS] == (WHEN_SLOT_UNUSED,) * (MAX_PINS - 3)
    assert slots[-1] == 1.0
    # Two product terms disagreeing on A2 -> mean 0.5; A1 absent from one -> don't care mixes in.
    multi = when_slots(AO21, "B", "(A1 * A2) + (!A2)")
    assert multi[0] == pytest.approx((1 + WHEN_SLOT_DONT_CARE) / 2) and multi[1] == 0.5
    assert multi[-1] == 2.0
    empty = when_slots(AO21, "A1", "")
    assert empty[:3] == (WHEN_SLOT_SELF, WHEN_SLOT_DONT_CARE, WHEN_SLOT_DONT_CARE)
    assert empty[-1] == 0.0
    with pytest.raises(ValueError):
        when_slots(AO21, "B", "(B * A1)")       # when may not constrain the related pin
    with pytest.raises(ValueError):
        when_slots(AO21, "B", "(D)")             # unknown pin


def ao21_rows():
    """Two arcs of one pin that differ only in `when` and have different delays."""
    base = golden_rows(cells=("INVx1",))
    base = base[(base["vt_class"] == "RVT") & (base["corner"] == "TT")]
    frames = []
    for offset, (pin, when, sense) in enumerate([("B", "(A1 * !A2)", "positive_unate"),
                                                 ("B", "(!A1 * A2)", "positive_unate"),
                                                 ("A1", "", "positive_unate")]):
        frames.append(base.assign(cell_base="AO21x1_ASAP7_75t", cell="AO21x1_ASAP7_75t_R",
                                  logic_function=AO21, related_pin=pin, when=when,
                                  timing_sense=sense, value=base["value"] + offset,
                                  n_input_pins=3, logic_class="AO"))
    return pd.concat(frames, ignore_index=True)


def test_arc_features_remove_when_conflict():
    rows = ao21_rows()
    from features import build_feature_frame
    assert _conflict_mask(_row_hash(build_feature_frame(rows)), rows["value"]).all()
    arcs = arc_feature_frame(rows)
    combined = pd.concat([build_feature_frame(rows), arcs], axis=1)
    assert not _conflict_mask(_row_hash(combined), rows["value"]).any()
    assert arcs.index.equals(rows.index)


def test_arc_features_output_index_for_multi_output_cell():
    rows = ao21_rows().head(18).copy()
    rows["output_pin"] = ["CON"] * 9 + ["SN"] * 9
    assert sorted(arc_feature_frame(rows)["output_index"].unique()) == [0.0, 1.0]


def test_reference_pin_caps_never_use_heldout_values():
    rows = golden_rows(cells=("INVx1",))
    rows["related_pin_cap_ff"] = rows["vt_class"].map(
        {"SLVT": 1.0, "LVT": 2.0, "RVT": 3.0, "SRAM": 4.0}) + rows["corner"].map(
        {"FF": 0.1, "TT": 0.2, "SS": 0.3})
    caps = reference_pin_caps(rows)
    assert np.allclose(caps["pin_cap_ref_vt"], 3.0 + rows["corner"].map({"FF": .1, "TT": .2, "SS": .3}))
    assert np.allclose(caps["pin_cap_ref_tt"], rows["vt_class"].map(
        {"SLVT": 1.2, "LVT": 2.2, "RVT": 3.2, "SRAM": 4.2}))
    # E2 holding out SRAM: the feature must equal for SRAM and RVT rows of the same corner.
    tt = rows["corner"] == "TT"
    assert caps.loc[tt & (rows["vt_class"] == "SRAM"), "pin_cap_ref_vt"].nunique() == 1


def test_v4_feature_frame_shape_and_ordinals():
    rows = golden_rows(cells=("INVx1", "INVx2"))
    features = build_v4_feature_frame(rows)
    assert len(features) == len(rows) and not features.isna().any().any()
    assert features["vt_ordinal"].groupby(rows["vt_class"]).first().to_dict() == {
        "LVT": 1.0, "RVT": 2.0, "SLVT": 0.0, "SRAM": 3.0}
    assert features["corner_ordinal"].groupby(rows["corner"]).first().to_dict() == {
        "FF": 0.0, "SS": 2.0, "TT": 1.0}


def test_random_point_mask_exact_counts_and_seed_dependence():
    rows = golden_rows(cells=("INVx1",))          # tables of 9 points
    mask = random_point_test_mask(rows, 11, 0.2)
    per_table = pd.Series(mask).groupby(rows.groupby(
        ["vt_class", "corner", "table_type"]).ngroup()).sum()
    assert (per_table == 2).all()                  # round(9 * 0.2) = 2
    assert (random_point_test_mask(rows, 11, 0.2) == mask).all()
    assert (random_point_test_mask(rows, 23, 0.2) != mask).any()
    assert (pd.Series(random_point_test_mask(rows, 11, 0.5)).groupby(
        rows.groupby(["vt_class", "corner", "table_type"]).ngroup()).sum() == 4).all()


def catalog_for(rows):
    out = rows.drop_duplicates("cell_base")[["cell_base", "family", "topology_type", "size"]]
    out = out.assign(in_mainline=True)
    out["n_sizes_in_type"] = out.groupby("topology_type")["size"].transform("nunique")
    return out


def test_all_manifests_group_disjoint_on_synthetic_rows():
    rows = golden_rows(cells=("INVx1", "INVx2", "INVx4"))
    manifests = (e1_manifests("x") + e2_manifests("x") + e3_manifests(catalog_for(rows), "x")
                 + e4_manifests("x"))
    assert len(manifests) == 10 + 4 + 3 + 3
    for manifest in manifests:
        labels = resolve_labels(rows, manifest)
        assert_group_disjoint(rows, labels, manifest)
        assert 0 < (labels == "test").sum() < len(rows)


def test_e2_e3_e4_manifest_content():
    e2 = {m["heldout_vt"]: m for m in e2_manifests("x")}
    assert e2["RVT"]["reference_vt"] == "LVT" and e2["SRAM"]["reference_vt"] == "RVT"
    assert e2["RVT"]["pin_cap_feature"] == "pin_cap_ref_vt_lvt"
    rows = golden_rows(cells=("INVx1", "INVx2", "INVx4"))
    regimes = {m["heldout_size"]: m["regime"] for m in e3_manifests(catalog_for(rows), "x")}
    assert regimes == {1.0: "extrapolation", 2.0: "interpolation", 4.0: "extrapolation"}
    e4 = {m["heldout_corner"]: m for m in e4_manifests("x")}
    assert e4["TT"]["primary_claim"] and e4["TT"]["pin_cap_feature"] is None
    assert not e4["FF"]["primary_claim"] and all(m["exploratory"] for m in e4.values())


def test_group_disjoint_check_catches_leak():
    rows = golden_rows(cells=("INVx1",))
    manifest = e2_manifests("x")[0]
    labels = resolve_labels(rows, manifest)
    labels[np.flatnonzero(labels == "test")[0]] = "train"
    with pytest.raises(AssertionError):
        assert_group_disjoint(rows, labels, manifest)
    point = e1_manifests("x")[0]
    labels = resolve_labels(rows, point)
    labels[np.flatnonzero(labels == "train")[0]] = "test"
    with pytest.raises(AssertionError):
        assert_group_disjoint(rows, labels, point)


def test_l0_sample_is_stratified_and_deterministic():
    tables = pd.DataFrame([
        {"cell_base": f"{t}{i}", "vt_class": "RVT", "corner": "TT", "output_pin": "Y",
         "related_pin": "A", "timing_sense": "x", "timing_type": "c", "when": "",
         "table_type": "cell_rise", "family": fam, "topology_type": t}
        for fam, t, n in [("AO", "AO333", 500), ("SIMPLE", "INV", 3), ("SIMPLE", "NAND2", 40)]
        for i in range(n)])
    sample = l0_table_sample(tables, target=60, seed=1)
    assert len(sample) == 60 and sample.equals(l0_table_sample(tables, target=60, seed=1))
    counts = sample.groupby("topology_type").size().to_dict()
    assert counts["INV"] == 3                        # small type fully taken
    assert counts["NAND2"] >= 20 and counts["AO333"] >= 20
