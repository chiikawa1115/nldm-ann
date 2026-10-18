import numpy as np
import pandas as pd
import torch

from features import build_feature_frame, feature_columns, infer_cell_base, infer_drive_strength, infer_logic_class, infer_vt_class
from generalization_split import PVT_COLS, make_generalization_manifest
from models.mlp_nldm import FeatureConditionedNldm


def dataset():
    rows = []
    process = {"SS": -1.0, "TT": 0.0, "FF": 1.0}
    for corner in process:
        for cell in ["INVx1_ASAP7_75t_R", "NAND2x2_ASAP7_75t_R",
                     "OR2x2_ASAP7_75t_R", "BUFx4_ASAP7_75t_R"]:
            for slew in [1.0, 2.0, 4.0]:
                for load in [1.0, 3.0, 9.0]:
                    rows.append({
                        "lib_file": f"demo_RVT_{corner}.lib", "library_name": "demo_RVT",
                        "corner": corner, "nom_process": process[corner],
                        "nom_voltage_v": 0.7, "nom_temperature_c": 25.0,
                        "cell": cell, "output_pin": "Y", "related_pin": "A",
                        "cell_base": infer_cell_base(cell), "logic_function": "!(A & B)",
                        "timing_sense": "negative_unate", "timing_type": "combinational",
                        "when": "", "table_type": "cell_rise", "template": "delay_3x3",
                        "slew": slew, "load": load, "log_slew": np.log(slew),
                        "log_load": np.log(load), "value": slew + load,
                        "logic_class": infer_logic_class(cell),
                        "drive_strength": infer_drive_strength(cell), "vt_class": "RVT",
                        "related_pin_cap_ff": 0.5, "n_input_pins": 2,
                        "n_output_pins": 1,
                    })
    return pd.DataFrame(rows)


def test_feature_inference_and_feature_conditioned_model_have_no_ids():
    assert infer_logic_class("NAND2x2_ASAP7_75t_R") == "NAND"
    assert infer_drive_strength("INVx1_ASAP7_75t_R") == 1.0
    assert infer_drive_strength("INVxp33_ASAP7_75t_R") == 0.33
    assert infer_vt_class("asap7_RVT_FF.lib") == "RVT"
    assert infer_cell_base("INVx1_ASAP7_75t_SL") == "INVx1_ASAP7_75t"
    frame = build_feature_frame(dataset().iloc[:5])
    assert list(frame.columns) == feature_columns()
    assert not any("cell_id" in name or "corner_id" in name for name in frame.columns)
    model = FeatureConditionedNldm(len(frame.columns), hidden_dims=(8, 4))
    assert model(torch.from_numpy(frame.to_numpy())).shape == (5, 1)


def test_random_point_split_keeps_each_table_in_both_partitions():
    manifest, _ = make_generalization_manifest(dataset(), "random_point", seed=11)
    counts = manifest.groupby([
        "cell", "corner", "table_type", "split"
    ]).size().unstack(fill_value=0)
    assert (counts["train"] > 0).all() and (counts["test"] > 0).all()


def test_unseen_cell_split_has_disjoint_cells_but_shared_pvts():
    manifest, _ = make_generalization_manifest(dataset(), "unseen_cell", seed=11)
    train, test = manifest.query("split == 'train'"), manifest.query("split == 'test'")
    assert set(train.cell_base).isdisjoint(test.cell_base)
    assert set(train[PVT_COLS].itertuples(index=False, name=None)) == set(
        test[PVT_COLS].itertuples(index=False, name=None)
    )


def test_unseen_corner_split_has_disjoint_pvt_but_shared_cells():
    manifest, meta = make_generalization_manifest(
        dataset(), "unseen_corner", seed=11, heldout_pvt_index=1
    )
    train, test = manifest.query("split == 'train'"), manifest.query("split == 'test'")
    assert set(train.cell) == set(test.cell)
    assert set(train[PVT_COLS].itertuples(index=False, name=None)).isdisjoint(
        test[PVT_COLS].itertuples(index=False, name=None)
    )
    assert meta["heldout_pvt"] is not None


def test_joint_split_excludes_cross_quadrants():
    manifest, _ = make_generalization_manifest(
        dataset(), "joint_unseen", seed=23, heldout_pvt_index=2
    )
    assert set(manifest["split"]) == {
        "train", "test", "excluded_seen_cell_heldout_pvt",
        "excluded_heldout_cell_seen_pvt",
    }
    train, test = manifest.query("split == 'train'"), manifest.query("split == 'test'")
    assert set(train.cell_base).isdisjoint(test.cell_base)
    assert set(train[PVT_COLS].itertuples(index=False, name=None)).isdisjoint(
        test[PVT_COLS].itertuples(index=False, name=None)
    )
