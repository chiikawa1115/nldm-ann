import numpy as np
import pandas as pd

from global_engine import NormalizationStats, partition_labels, random_point_test_mask, validation_mask
from generalization_split import PVT_COLS


def frame():
    rows = []
    for corner, process, voltage, temperature in [
        ("FF", 1.0, 0.77, 0.0), ("SS", -1.0, 0.63, 100.0),
    ]:
        for cell in ["INVx1_R", "NAND2x1_R"]:
            for slew in [1.0, 2.0, 4.0]:
                for load in [1.0, 3.0, 9.0]:
                    rows.append({
                        "lib_file": f"demo_{corner}.lib", "library_name": "demo",
                        "corner": corner, "nom_process": process,
                        "nom_voltage_v": voltage, "nom_temperature_c": temperature,
                        "cell": cell, "cell_base": cell.removesuffix("_R"),
                        "output_pin": "Y", "related_pin": "A", "timing_sense": "positive_unate",
                        "timing_type": "combinational", "when": "", "table_type": "cell_rise",
                        "template": "delay_3x3", "slew": slew, "load": load,
                    })
    return pd.DataFrame(rows)


def test_random_point_assignment_is_exact_per_table_and_deterministic():
    data = frame()
    first = random_point_test_mask(data, seed=11, ratio=0.2)
    second = random_point_test_mask(data, seed=11, ratio=0.2)
    assert np.array_equal(first, second)
    data["test"] = first
    counts = data.groupby(["lib_file", "cell"])["test"].sum()
    assert (counts == 2).all()


def test_compact_contract_assignments_are_leakage_free():
    data = frame()
    cell_contract = {
        "mode": "unseen_cell", "heldout_cell_bases": ["INVx1"],
    }
    labels = partition_labels(data, cell_contract)
    assert set(data.loc[labels == "test", "cell_base"]) == {"INVx1"}
    corner_contract = {
        "mode": "unseen_corner", "heldout_pvt": dict(zip(PVT_COLS, ("FF", 1.0, 0.77, 0.0))),
    }
    labels = partition_labels(data, corner_contract)
    assert set(data.loc[labels == "test", "corner"]) == {"FF"}
    joint = {
        "mode": "joint_unseen", "heldout_cell_bases": ["INVx1"],
        "heldout_pvt": corner_contract["heldout_pvt"],
    }
    labels = partition_labels(data, joint)
    train = data.loc[labels == "train"]
    test = data.loc[labels == "test"]
    assert set(train.cell_base).isdisjoint(test.cell_base)
    assert set(train[PVT_COLS].itertuples(index=False, name=None)).isdisjoint(
        test[PVT_COLS].itertuples(index=False, name=None)
    )


def test_validation_assignment_is_deterministic_and_does_not_mutate_inputs():
    data = frame()
    before = data[["slew", "load"]].copy()
    first = validation_mask(data, seed=11, ratio=0.2)
    second = validation_mask(data, seed=11, ratio=0.2)
    assert np.array_equal(first, second)
    pd.testing.assert_frame_equal(before, data[["slew", "load"]])


def test_normalization_payload_round_trip_preserves_safe_scales():
    stats = NormalizationStats(["constant"], np.array([2.0]), np.array([1.0]), 3.0, 4.0, 5)
    restored = NormalizationStats.from_dict(stats.to_dict())
    assert restored.x_std.tolist() == [1.0]
    assert restored.n_fit == 5
