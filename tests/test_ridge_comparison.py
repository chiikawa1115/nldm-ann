"""Small numerical and workflow checks; never use the production Parquet here."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t, ttest_rel

import run_ridge_comparison as runner
from features import build_feature_frame, feature_columns
from global_engine import file_sha256, fit_normalization, partition_labels, validation_mask
from metrics import micro_macro_metrics
from ridge_audit import (
    CODE_ROOT, ann_provenance, audit_inputs, read_json, semantic_hash, validate_config, write_json,
)
from ridge_compare import holm_adjust, paired_rows, summarize_pairs, write_comparison
from ridge_core import (
    Moments, RidgeModel, collect_moments, evaluate_stream, exact_metrics_from_spool, solve_candidates,
)
from test_generalization import dataset


@pytest.fixture
def tiny_project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    frames = []
    for i, table in enumerate(("cell_rise", "cell_fall", "rise_transition", "fall_transition")):
        frame = dataset()
        frame["table_type"] = table
        frame["value"] = 2 * frame.log_slew - frame.log_load + 0.2 * frame.nom_process + i - 2
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    parquet = root / "data/tiny.parquet"
    parquet.parent.mkdir()
    frame.to_parquet(parquet, index=False)
    config = read_json(CODE_ROOT / "experiments/configs/global_ann_ridge_v1.json")
    config["experiment_id"] = "synthetic_test_only"
    config["dataset"].update(path="data/tiny.parquet", sha256=file_sha256(parquet),
                             rows=len(frame), cell_bases=4)
    config["seeds"] = [11]
    config["expected_runs"] = 2
    config["execution"].update(batch_rows_target=100, output_dir="ridge_output")
    pairs = []
    for mode in config["modes"]:
        run_id = f"{mode}_seed11_modelseed11"
        contract = {"schema_version": 1, "mode": mode, "split_seed": 11,
                    "dataset_sha256": file_sha256(parquet), "point_test_ratio": .2,
                    "heldout_cell_bases": [frame.cell_base.iloc[0]] if mode == "unseen_cell" else []}
        test = frame.loc[partition_labels(frame, contract) == "test"].copy()
        test["y_true"] = test.value
        test["y_pred"] = test.value + .3  # Mock ANN evidence, restricted to this fixture.
        micro, macro, _ = micro_macro_metrics(test, group_col="cell_base")
        ann = {"seed": 11, "mode": mode, "micro": micro, "macro": macro,
               "n_negative_targets": int((test.value < 0).sum())}
        manifest = f"manifests/{mode}.json"
        ann_path = f"ann/{mode}.json"
        write_json(root / manifest, contract)
        write_json(root / ann_path, ann)
        pairs.append({"run_id": run_id, "mode": mode, "seed": 11,
                      "manifest": manifest, "manifest_semantic_sha256": semantic_hash(contract),
                      "ann_metrics": ann_path, "ann_metrics_semantic_sha256": semantic_hash(ann),
                      "expected_test_rows": len(test), "expected_cell_groups": test.cell_base.nunique(),
                      "expected_negative_targets": ann["n_negative_targets"]})
    config["pairs"] = pairs
    write_json(root / "plan.json", config)
    return root, config, frame


def test_moments_match_full_centered_statistics_and_chunking():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(139, 4)) + np.array([1e7, 0, -5, 2])
    y = rng.normal(size=139) - 8
    full, chunked = Moments(4), Moments(4)
    full.add(x, y)
    chunked.add(x[:0], y[:0])
    for a in range(0, len(y), 7):
        chunked.add(x[a:a + 7], y[a:a + 7])
    values = np.column_stack((x, y))
    expected = values - values.mean(axis=0)
    np.testing.assert_allclose(full.m2, expected.T @ expected)
    np.testing.assert_allclose(chunked.m2, full.m2, rtol=1e-8, atol=1e-7)
    np.testing.assert_allclose(chunked.mean, full.mean, rtol=1e-12)
    assert full.n == chunked.n == 139
    with pytest.raises(ValueError, match="finite"):
        full.add([[np.nan, 1, 2, 3]], [1])


@pytest.mark.parametrize("constant_target", [False, True])
def test_ridge_matches_augmented_least_squares_with_unpenalized_intercept(constant_target, tmp_path):
    rng = np.random.default_rng(23)
    x = rng.normal(size=(181, 5))
    x[:, 3] = 9  # Constant and exactly collinear columns.
    x[:, 4] = 2 * x[:, 0]
    y = np.full(181, -8.) if constant_target else -8 + 2 * x[:, 0] - x[:, 1] + rng.normal(0, .2, 181)
    fit, val = Moments(5), Moments(5)
    fit.add(x[:120], y[:120])
    val.add(x[120:], y[120:])
    lambdas = [1e-8, 1e-6, 1e-4, 1e-2, 1, 100]
    names = [f"x{i}" for i in range(5)]
    stats = fit.normalization(names)
    z = (x - stats.x_mean) / stats.x_std
    target = (y[:120] - stats.y_mean) / stats.y_std
    for lam in lambdas:
        model, rows = solve_candidates(fit, val, names, [lam])
        design = np.column_stack((z[:120], np.ones(120)))
        penalty = np.diag([np.sqrt(120 * lam)] * 5 + [0])
        coef = np.linalg.lstsq(np.vstack((design, penalty)), np.r_[target, np.zeros(6)], rcond=None)[0]
        reference = (z @ coef[:-1] + coef[-1]) * stats.y_std + stats.y_mean
        np.testing.assert_allclose(model.predict(x), reference, atol=1e-8, rtol=1e-8)
        assert abs(coef[-1]) < 1e-10
        assert rows[0]["alpha"] == 120 * lam
        mse = np.mean(((reference[120:] - y[120:]) / stats.y_std) ** 2)
        assert rows[0]["validation_mse_normalized"] == pytest.approx(mse, rel=1e-8, abs=1e-10)
    best, rows = solve_candidates(fit, val, names, lambdas)
    assert best.selected_lambda == (100 if constant_target else min(rows, key=lambda r: r["validation_mse_normalized"])["lambda"])
    assert best.normalization.x_std[3] == 1
    assert best.predict(x).min() < 0  # Negative predictions are retained.
    path = tmp_path / "ridge.npz"
    best.save(path)
    np.testing.assert_array_equal(RidgeModel.load(path).predict(x), best.predict(x))


@pytest.mark.parametrize("mode_index", [0, 1])
def test_stream_matches_full_frame_splits_normalization_and_metrics(tiny_project, mode_index):
    root, config, frame = tiny_project
    pair = config["pairs"][mode_index]
    contract = read_json(root / pair["manifest"])
    parquet = root / config["dataset"]["path"]
    labels = partition_labels(frame, contract)
    is_val = validation_mask(frame, pair["seed"])
    fit_mask = (labels == "train") & ~is_val
    x = build_feature_frame(frame).to_numpy(dtype=float)
    direct_fit, direct_val = Moments(42), Moments(42)
    direct_fit.add(x[fit_mask], frame.value.to_numpy()[fit_mask])
    direct_val.add(x[(labels == "train") & is_val], frame.value.to_numpy()[(labels == "train") & is_val])
    reference, _ = solve_candidates(direct_fit, direct_val, feature_columns(), config["ridge"]["lambda_grid"])
    original = fit_normalization(parquet, contract, batch_rows=31)
    for batch_rows in (5, 19, 1000):  # The 9-point tables cross both small boundaries.
        fit, val, counts = collect_moments(parquet, contract, 11, batch_rows)
        assert counts == {"source": len(frame), "fit": int(fit_mask.sum()),
                          "validation": int(((labels == "train") & is_val).sum()),
                          "test": int((labels == "test").sum()), "excluded": 0}
        model, _ = solve_candidates(fit, val, feature_columns(), config["ridge"]["lambda_grid"])
        assert model.selected_lambda == reference.selected_lambda
        np.testing.assert_allclose(model.predict(x), reference.predict(x), atol=1e-8)
        np.testing.assert_allclose(model.normalization.x_std, original.x_std, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(model.normalization.x_mean, original.x_mean, rtol=1e-6, atol=1e-8)
        assert model.normalization.y_std == pytest.approx(original.y_std)
    metrics, cells = evaluate_stream(parquet, contract, reference, pair["expected_test_rows"], 17, root / "scratch")
    expected = frame.loc[labels == "test", ["cell_base"]].copy()
    expected["y_true"] = frame.loc[labels == "test", "value"]
    expected["y_pred"] = reference.predict(x[labels == "test"])
    micro, macro, per_cell = micro_macro_metrics(expected, group_col="cell_base")
    assert metrics["micro"] == pytest.approx(micro)
    assert metrics["macro"] == pytest.approx(macro)
    pd.testing.assert_frame_equal(cells, per_cell, check_dtype=False)
    assert metrics["n_negative_targets"] == pair["expected_negative_targets"]
    assert not list((root / "scratch").iterdir())
    for incorrect_count in (pair["expected_test_rows"] - 1, pair["expected_test_rows"] + 1):
        with pytest.raises(ValueError, match="test size"):
            evaluate_stream(parquet, contract, reference, incorrect_count, 100, root / "scratch")
    assert not list((root / "scratch").iterdir())


def test_validation_and_test_values_cannot_leak_into_fit_normalization(tiny_project):
    root, config, frame = tiny_project
    contract = read_json(root / config["pairs"][0]["manifest"])
    path = root / config["dataset"]["path"]
    fit, val, _ = collect_moments(path, contract, 11, 100)
    held = (partition_labels(frame, contract) == "test") | validation_mask(frame, 11)
    frame.loc[held, "value"] = 1e9
    frame.loc[held, "related_pin_cap_ff"] = 1e7
    frame.to_parquet(path, index=False)
    changed_fit, changed_val, _ = collect_moments(path, contract, 11, 100)
    np.testing.assert_array_equal(fit.mean, changed_fit.mean)
    np.testing.assert_array_equal(fit.m2, changed_fit.m2)
    assert changed_val.mean[-1] != val.mean[-1]


def test_exact_spooled_metrics_keep_near_zero_rules_and_undefined_groups(tmp_path):
    values = np.array([[0, 0], [0, 1], [-2, -1], [1e-17, 2e-17], [3, -1], [4, 4.]])
    codes = np.array([0, 0, 1, 1, 1, 1], dtype=np.int32)
    mapped = np.memmap(tmp_path / "values.bin", dtype="float64", shape=values.shape, mode="w+")
    mapped[:] = values
    try:
        micro, macro, cells, undefined = exact_metrics_from_spool(mapped, codes, ["A", "B"])
    finally:
        mapped._mmap.close()
    frame = pd.DataFrame({"y_true": values[:, 0], "y_pred": values[:, 1],
                          "cell_base": np.array(["A", "B"])[codes]})
    m1, m2, expected_cells = micro_macro_metrics(frame, group_col="cell_base")
    assert micro == pytest.approx(m1)
    assert macro == pytest.approx(m2)
    pd.testing.assert_frame_equal(cells, expected_cells)
    assert undefined["R2"] == undefined["P99RE_pct"] == 1


def metadata_for(root, config, pair):
    contract = read_json(root / pair["manifest"])
    stats = fit_normalization(root / config["dataset"]["path"], contract, 100)
    return {"run_id": pair["run_id"], "seed": pair["seed"], "contract": contract,
            "manifest_sha256": file_sha256(root / pair["manifest"]), "normalization": stats.to_dict(),
            "max_rows": None, "git_commit": "synthetic-test-only", "training_config": {"epochs": 10},
            "training_source_sha256": config["reference_source_git_blob_sha256"]}


def test_provenance_pending_match_mismatch_and_line_endings(tiny_project):
    root, config, _ = tiny_project
    pair = config["pairs"][0]
    out = root / "metadata"
    assert ann_provenance(root, config, pair, out)["status"] == "ann_provenance_pending"
    meta = metadata_for(root, config, pair)
    write_json(out / (pair["run_id"] + ".json"), meta)
    assert ann_provenance(root, config, pair, out)["status"] == "ann_provenance_pending"
    result = ann_provenance(root, config, pair, out, meta["normalization"])
    assert result["status"] == "matched"
    manifest = root / pair["manifest"]
    manifest.write_bytes(manifest.read_bytes().replace(b"\n", b"\r\n"))
    assert ann_provenance(root, config, pair, out, meta["normalization"])["status"] == "matched"
    bad = deepcopy(meta)
    bad["normalization"]["n_fit"] += 1
    write_json(out / (pair["run_id"] + ".json"), bad)
    assert ann_provenance(root, config, pair, out, meta["normalization"])["status"] == "mismatch"
    for key, value in (("seed", 99), ("max_rows", 100), ("training_source_sha256", {"src/features.py": "bad"})):
        bad = {**meta, key: value}
        write_json(out / (pair["run_id"] + ".json"), bad)
        assert ann_provenance(root, config, pair, out, meta["normalization"])["status"] == "mismatch"


@pytest.mark.parametrize("changed", ["dataset", "manifest", "ann_metrics", "source"])
def test_input_audit_rejects_changed_evidence(tiny_project, monkeypatch, changed):
    root, config, _ = tiny_project
    assert audit_inputs(root, config)["status"] == "ready_for_run"
    if changed == "dataset":
        config["dataset"]["sha256"] = "wrong"
    elif changed == "source":
        monkeypatch.setattr("ridge_audit.source_hashes", lambda: {})
    else:
        path = root / config["pairs"][0][changed]
        data = read_json(path)
        data["tampered"] = True
        write_json(path, data)
    with pytest.raises(ValueError, match="differs|changed"):
        audit_inputs(root, config)


@pytest.mark.parametrize("section,key,value", [
    ("split", "max_rows", 500), ("ridge", "penalize_intercept", True),
    ("ridge", "lambda_grid", [1, 1]), ("analysis", "macro_group", "cell"),
    ("analysis", "ci_level", .9), ("execution", "automatic_timeout_seconds", 3600),
])
def test_unsupported_config_is_not_silently_ignored(tiny_project, section, key, value):
    _, config, _ = tiny_project
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_preflight_never_fits_and_cli_requires_explicit_run(tiny_project, monkeypatch):
    root, config, _ = tiny_project
    def forbidden(*args, **kwargs):
        pytest.fail("preflight attempted training")
    monkeypatch.setattr(runner, "run_one", forbidden)
    monkeypatch.setattr(runner, "collect_moments", forbidden)
    args = ["--root", str(root), "--config", "plan.json"]
    assert runner.main(["preflight", *args]) == 0
    out = root / config["execution"]["output_dir"]
    assert read_json(out / "input_audit.json")["training_executed_by_audit"] is False
    assert not (out / "models").exists()
    for options in (["run"], ["preflight", "--batch-rows", "0"],
                    ["run", "--run-id", config["pairs"][1]["run_id"]]):
        with pytest.raises(SystemExit) as exc:
            runner.main([*options, *args])
        assert exc.value.code == 2
    for path in (root, root / "src", root / "data", root / "experiments/asap7_generalization_v1/new"):
        with pytest.raises(ValueError, match="overlap"):
            runner.initialize_output(root, path, config)


def test_tiny_cli_end_to_end_reuse_and_corruption(tiny_project, monkeypatch):
    root, config, _ = tiny_project
    args = ["--root", str(root), "--config", "plan.json", "--no-plots"]
    assert runner.main(["run", "--all", *args]) == 0
    out = root / config["execution"]["output_dir"]
    status = read_json(out / "aggregate/comparison_status.json")
    assert status["completed_runs"] == 2
    assert not status["all_ann_provenance_matched"]
    paired = pd.read_csv(out / "aggregate/paired_metrics.csv")
    assert len(paired) == 2 * 2 * 7
    def forbidden(*args, **kwargs):
        pytest.fail("completed run was refit")
    monkeypatch.setattr(runner, "collect_moments", forbidden)
    assert runner.main(["run", "--all", *args]) == 0
    for pair in config["pairs"]:
        write_json(root / "metadata" / (pair["run_id"] + ".json"), metadata_for(root, config, pair))
    assert runner.main(["aggregate", *args, "--ann-metadata-dir", str(root / "metadata")]) == 0
    assert read_json(out / "aggregate/comparison_status.json")["all_ann_provenance_matched"]
    changed = out / "evaluations" / config["pairs"][0]["run_id"] / "metrics.json"
    payload = read_json(changed)
    payload["micro"]["MAE"] = 1e6
    write_json(changed, payload)
    assert runner.main(["aggregate", *args]) == 2
    assert read_json(out / "aggregate/comparison_status.json")["completed_runs"] == 1
    with pytest.raises(ValueError, match="artifact"):
        runner.main(["run", "--all", *args])


def test_interrupted_evaluation_requires_explicit_retry_and_reuses_fit(tiny_project, monkeypatch):
    root, config, _ = tiny_project
    pair = config["pairs"][0]
    out = root / config["execution"]["output_dir"]
    runner.initialize_output(root, out, config)
    audit = audit_inputs(root, config)
    evaluate = runner.evaluate_stream
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(runner, "evaluate_stream", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_one(root, out, config, pair, audit, 100)
    path = out / "evaluations" / pair["run_id"]
    assert read_json(path / "status.json")["status"] == "interrupted"
    assert (path / "fit_complete.json").exists()
    with pytest.raises(ValueError, match="retry-incomplete"):
        runner.run_one(root, out, config, pair, audit, 100)
    monkeypatch.setattr(runner, "evaluate_stream", evaluate)
    def forbidden(*args, **kwargs):
        pytest.fail("resumed evaluation refit the model")
    monkeypatch.setattr(runner, "collect_moments", forbidden)
    assert runner.run_one(root, out, config, pair, audit, 100, retry_incomplete=True)["status"] == "complete"
    with pytest.raises(ValueError, match="changed"):
        runner.run_one(root, out, config, pair, audit, 101)


def paired_fixture():
    config = read_json(CODE_ROOT / "experiments/configs/global_ann_ridge_v1.json")
    rows = []
    for i, mode in enumerate(config["modes"]):
        for j, seed in enumerate(config["seeds"]):
            ann = {"micro": {"MAE": 2 + j, "R2": .9}, "macro": {"MAE": 3 + j, "R2": .8}}
            ridge = {"micro": {"MAE": 3 + 2 * j, "R2": .7},
                     "macro": {"MAE": 3 + j + (-1)**i * (j + 1), "R2": .6}}
            rows += paired_rows(f"{mode}_{seed}", mode, seed, ann, ridge, "matched")
    return config, pd.DataFrame(rows)


def test_paired_signs_zero_denominator_ci_and_holm():
    config, records = paired_fixture()
    assert records.loc[records.metric == "R2", "delta"].tolist() == pytest.approx([.2] * 20)
    assert records.loc[records.metric == "R2", "improvement_pct"].isna().all()
    zero = paired_rows("zero", "random_point", 11,
                       {"micro": {"MAE": 1}, "macro": {}},
                       {"micro": {"MAE": 0}, "macro": {}}, "matched")[0]
    assert zero["delta"] == -1 and np.isnan(zero["improvement_pct"])
    summary = summarize_pairs(records, config)
    primary = summary[(summary.scope == "macro") & (summary.metric == "MAE")]
    assert primary.n_pairs.tolist() == [5, 5]
    assert primary.delta_mean.tolist() == [3., -3.]
    half = t.ppf(.975, 4) * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5)
    assert primary.delta_ci95_low.tolist() == pytest.approx([3 - half, -3 - half])
    selected = records[(records["mode"] == "random_point") & (records.scope == "macro") & (records.metric == "MAE")]
    assert primary.p_two_sided.iloc[0] == pytest.approx(ttest_rel(selected.ridge, selected.ann).pvalue)
    assert primary.p_holm.tolist() == pytest.approx(holm_adjust(primary.p_two_sided.tolist()))
    assert holm_adjust([.04, .01]) == pytest.approx([.04, .02])
    assert summary.loc[summary.metric != "MAE", "p_holm"].isna().all()
    with pytest.raises(ValueError, match="duplicate"):
        summarize_pairs(pd.concat([records, records.iloc[:1]]), config)


@pytest.mark.parametrize("incomplete", ["missing", "pending", "undefined"])
def test_incomplete_or_pending_pairs_never_get_formal_pvalues(incomplete):
    config, records = paired_fixture()
    first = (records["mode"] == "random_point") & (records.seed == 11)
    if incomplete == "missing":
        records = records.loc[~first]
    elif incomplete == "pending":
        records.loc[first, "ann_provenance_status"] = "ann_provenance_pending"
    else:
        records.loc[first, "delta"] = np.nan
    summary = summarize_pairs(records, config)
    assert summary.p_two_sided.isna().all()
    assert summary.p_holm.isna().all()
    assert not summary.formal_inference_eligible.any()


def test_degenerate_variance_is_descriptive_and_mismatch_is_rejected():
    config, records = paired_fixture()
    records.loc[records.metric == "MAE", "delta"] = 1.
    assert summarize_pairs(records, config).p_holm.isna().all()
    records.loc[records.index[0], "ann_provenance_status"] = "mismatch"
    with pytest.raises(ValueError, match="provenance"):
        summarize_pairs(records, config)


def test_report_plots_and_empty_coverage(tmp_path):
    config, records = paired_fixture()
    states = [{"run_id": pair["run_id"], "status": "not_run"} for pair in config["pairs"]]
    write_comparison(tmp_path, config, [], states, make_plots=False)
    assert not read_json(tmp_path / "aggregate/comparison_status.json")["all_runs_complete"]
    assert pd.read_csv(tmp_path / "aggregate/paired_metrics.csv").empty
    states = [{**state, "status": "complete", "ann_provenance_status": "matched"} for state in states]
    write_comparison(tmp_path, config, records.to_dict("records"), states, make_plots=True)
    assert read_json(tmp_path / "aggregate/comparison_status.json")["all_runs_complete"]
    assert (tmp_path / "figures/macro_mae.png").stat().st_size > 1000
    assert (tmp_path / "figures/macro_tail_error.png").stat().st_size > 1000
