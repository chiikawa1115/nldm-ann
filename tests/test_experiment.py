import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baseline import bilinear_predict
from experiment import ARC_COLS, POINT_COLS, add_point_ids, dataframe_sha256, file_sha256, split_by_regular_subgrid, write_manifest
from metrics import micro_macro_metrics, regression_metrics


def make_table(corner="TT", n=7):
    metadata = {
        "lib_file": f"demo_{corner}.lib", "library_name": f"demo_{corner}",
        "corner": corner, "nom_process": 1.0, "nom_voltage_v": 1.0,
        "nom_temperature_c": 25.0, "cell": "INVX1", "output_pin": "Y",
        "related_pin": "A", "timing_sense": "negative_unate",
        "timing_type": "combinational", "when": "",
        "table_type": "cell_rise", "template": "delay_7x7",
    }
    rows = []
    for slew in np.linspace(1.0, 7.0, n):
        for load in np.linspace(2.0, 8.0, n):
            rows.append({**metadata, "slew": slew, "load": load,
                         "log_slew": np.log(slew), "log_load": np.log(load),
                         "value": 1 + 2 * slew + 3 * load + 4 * slew * load})
    return pd.DataFrame(rows)


def test_regular_subgrid_is_reproducible_rectangular_and_keeps_boundaries():
    df = make_table()
    train_a, test_a, manifest_a = split_by_regular_subgrid(df, 0.5, seed=11)
    train_b, test_b, manifest_b = split_by_regular_subgrid(df, 0.5, seed=11)
    pd.testing.assert_frame_equal(manifest_a, manifest_b)
    assert len(train_a) == train_a.slew.nunique() * train_a.load.nunique()
    assert set([df.slew.min(), df.slew.max()]) <= set(train_a.slew)
    assert set([df.load.min(), df.load.max()]) <= set(train_a.load)
    assert set(map(tuple, train_a[POINT_COLS].to_numpy())).isdisjoint(
        set(map(tuple, test_a[POINT_COLS].to_numpy()))
    )
    assert len(train_a) + len(test_a) == len(df)


def test_full_table_identity_prevents_cross_corner_leakage():
    df = pd.concat([make_table("TT"), make_table("SS")], ignore_index=True)
    train, test, _ = split_by_regular_subgrid(df, 0.5, seed=3)
    assert train.groupby(ARC_COLS, dropna=False).ngroups == 2
    assert test.groupby(ARC_COLS, dropna=False).ngroups == 2


def test_common_holdout_requests_map_to_distinct_realizable_grids():
    df = make_table()
    train_sizes = []
    for holdout in [0.2, 0.3, 0.5, 0.7]:
        train, _, _ = split_by_regular_subgrid(df, 1 - holdout, seed=1)
        train_sizes.append(len(train))
    assert train_sizes == [42, 35, 25, 15]


def test_bilinear_predict_is_exact_for_bilinear_surface():
    df = make_table()
    train, test, _ = split_by_regular_subgrid(df, 0.5, seed=4)
    prediction = bilinear_predict(train, test)
    np.testing.assert_allclose(prediction, test.value.to_numpy(), rtol=1e-12)


def test_bilinear_predict_rejects_incomplete_grid_and_does_not_extrapolate():
    df = make_table(n=3)
    with pytest.raises(ValueError, match="complete rectangular"):
        bilinear_predict(df.drop(df.index[0]), df.iloc[[0]])
    outside = pd.DataFrame({"slew": [0.0], "load": [5.0]})
    assert np.isnan(bilinear_predict(df, outside)[0])


def test_metrics_include_rmse_and_stable_smape_at_zero():
    result = regression_metrics(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
    assert result["RMSE"] == pytest.approx(np.sqrt(0.5))
    assert result["sMAPE_pct"] == pytest.approx(100 / 3)
    assert result["P50RE_pct"] == pytest.approx(100.0)


def test_micro_and_cell_macro_weighting_are_distinct():
    frame = pd.DataFrame({
        "cell": ["large"] * 4 + ["small"],
        "y_true": [1.0] * 5,
        "y_pred": [1.0] * 4 + [3.0],
    })
    micro, macro, per_cell = micro_macro_metrics(frame)
    assert micro["MAE"] == pytest.approx(0.4)
    assert macro["MAE"] == pytest.approx(1.0)
    assert set(per_cell.cell) == {"large", "small"}


def test_dataframe_hash_is_row_order_independent():
    df = make_table()
    assert dataframe_sha256(df) == dataframe_sha256(df.sample(frac=1, random_state=7))


def test_point_ids_preserve_exact_float_identity_across_csv_formatting():
    df = make_table()
    ids = add_point_ids(df)
    assert ids.point_id.is_unique
    changed = df.copy()
    changed.loc[0, "slew"] = np.nextafter(changed.loc[0, "slew"], np.inf)
    assert add_point_ids(changed).loc[0, "point_id"] != ids.loc[0, "point_id"]


def test_written_manifest_uses_file_hash():
    # Use the repository-owned runtime directory because the managed Windows
    # environment denies pytest's default temporary directory.
    path = Path(__file__).parent / "runtime" / "manifest.csv"
    _, _, manifest = split_by_regular_subgrid(make_table(), 0.5, seed=2)
    write_manifest(manifest, path, {"test": True})
    metadata = json.loads(path.with_suffix(".csv.json").read_text(encoding="utf-8"))
    assert metadata["manifest_sha256"] == file_sha256(path)
