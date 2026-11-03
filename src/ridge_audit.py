"""Frozen input checks, ANN provenance, and atomic Ridge artifact utilities."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from features import feature_columns
from global_engine import DATA_COLUMNS, file_sha256

CODE_ROOT = Path(__file__).resolve().parents[1]
NORM_RTOL = 1e-6
NORM_ATOL = 1e-8


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def semantic_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temp.replace(path)


def repo_path(root: Path, relative: str) -> Path:
    result = (root / relative.replace("\\", "/")).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes project: {relative}")
    return result


def source_hashes(code_root: Path = CODE_ROOT) -> dict:
    # Include imported split/metric helpers as well as all Ridge implementation files.
    return {p.relative_to(code_root).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for p in sorted((code_root / "src").glob("*.py"))}


def versions() -> dict:
    result = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "pyarrow", "torch"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def git_state() -> dict:
    def git(*args):
        r = subprocess.run(["git", *args], cwd=CODE_ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--short")}


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported Ridge plan schema")
    names = feature_columns()
    if (config["features"]["names"] != names or config["features"]["count"] != len(names)
            or config["features"]["names_semantic_sha256"] != semantic_hash(names)):
        raise ValueError("frozen feature contract changed")
    if config["features"]["use_v4_cache"]:
        raise ValueError("this comparison does not use v4 features")
    if (config["split"]["max_rows"] is not None
            or not config["split"]["preserve_complete_tables"]
            or not config["split"]["run_seed_equals_split_seed"]
            or config["split"]["validation_ratio"] != 0.1):
        raise ValueError("unsupported split policy; do not silently change the ANN comparison")
    fixed = {"fit_intercept": True, "penalize_intercept": False,
             "refit_on_fit_plus_validation": False, "clip_predictions": False,
             "fit_weights": "uniform_per_point", "normalization_partition": "fit_only",
             "selection_metric": "validation_mse_normalized", "accumulation_dtype": "float64",
             "objective": "mean_squared_error + lambda * squared_l2_norm_of_coefficients",
             "sum_loss_alpha_conversion": "alpha = n_fit * lambda",
             "tie_break": "larger_lambda_within_recorded_numeric_tolerance"}
    if any(config["ridge"].get(k) != v for k, v in fixed.items()):
        raise ValueError("unsupported Ridge fitting policy")
    lambdas = config["ridge"]["lambda_grid"]
    if not lambdas or len(set(lambdas)) != len(lambdas) or any(
            not np.isfinite(v) or v <= 0 for v in lambdas):
        raise ValueError("lambda grid must contain unique finite positive values")
    pairs = config["pairs"]
    if len(pairs) != config["expected_runs"] or len({p["run_id"] for p in pairs}) != len(pairs):
        raise ValueError("missing or duplicate planned runs")
    if len({(p["mode"], p["seed"]) for p in pairs}) != len(pairs):
        raise ValueError("duplicate paired mode/seed")
    expected = {(m, s) for m in config["modes"] for s in config["seeds"]}
    if (not expected or len(set(config["modes"])) != len(config["modes"])
            or len(set(config["seeds"])) != len(config["seeds"])):
        raise ValueError("empty or duplicate modes/seeds")
    if {(p["mode"], p["seed"]) for p in pairs} != expected:
        raise ValueError("pair matrix does not match planned modes/seeds")
    for p in pairs:
        if p["mode"] not in ("random_point", "unseen_cell"):
            raise ValueError("unsupported comparison mode")
        if p["run_id"] != f"{p['mode']}_seed{p['seed']}_modelseed{p['seed']}":
            raise ValueError("invalid run ID")
    if config["execution"]["device"] != "cpu":
        raise ValueError("Ridge runner supports CPU only")
    if config["execution"]["first_run"] not in {p["run_id"] for p in pairs}:
        raise ValueError("first run is absent from plan")
    if (config["execution"]["percentiles"] != "exact_matching_src_metrics"
            or config["execution"]["automatic_timeout_seconds"] is not None):
        raise ValueError("unsupported percentile or timeout policy")
    analysis = {"primary_metric": "macro.MAE", "macro_group": "cell_base",
                "pairing_keys": ["mode", "seed", "scope", "metric"],
                "error_delta": "ridge - ann", "error_improvement_pct": "100 * (ridge - ann) / ridge",
                "zero_ridge_denominator": "undefined_report_absolute_delta", "r2_delta": "ann - ridge",
                "summary_group": "mode", "ci_method": "student_t_on_five_paired_seed_differences",
                "ci_level": 0.95, "formal_test": "two_sided_paired_t",
                "multiplicity": "holm_over_two_primary_mode_macro_MAE_tests", "family_alpha": 0.05,
                "secondary_inference": "descriptive_exploratory", "compare_runtime_as_primary": False}
    if any(config["analysis"].get(k) != v for k, v in analysis.items()):
        raise ValueError("unsupported paired analysis policy")
    if config["dataset"]["target"] != "value" or config["dataset"]["target_unit"] != "ps":
        raise ValueError("unsupported target")


def ann_provenance(root: Path, config: dict, pair: dict, metadata_dir: Path | None,
                   normalization: dict | None = None) -> dict:
    pending, errors = [], []
    if metadata_dir is None or not (metadata_dir / (pair["run_id"] + ".json")).is_file():
        return {"status": "ann_provenance_pending", "pending": ["checkpoint metadata absent"],
                "errors": []}
    path = metadata_dir / (pair["run_id"] + ".json")
    meta = read_json(path)
    required = {"run_id", "seed", "contract", "manifest_sha256", "normalization",
                "max_rows", "git_commit", "training_config", "training_source_sha256"}
    pending.extend("missing " + k for k in sorted(required - meta.keys()))
    for k, expected in (("run_id", pair["run_id"]), ("seed", pair["seed"]), ("max_rows", None)):
        if k in meta and meta[k] != expected:
            errors.append(f"{k} differs")
    if "contract" in meta and semantic_hash(meta["contract"]) != pair["manifest_semantic_sha256"]:
        errors.append("checkpoint contract differs")
    if "manifest_sha256" in meta:
        raw = repo_path(root, pair["manifest"]).read_bytes().replace(b"\r\n", b"\n")
        acceptable = {hashlib.sha256(raw).hexdigest(),
                      hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest()}
        if meta["manifest_sha256"] not in acceptable:
            errors.append("checkpoint manifest hash differs beyond LF/CRLF")
    if not meta.get("git_commit"):
        pending.append("training git commit unavailable")
    if not meta.get("training_config"):
        pending.append("training config unavailable")
    recorded = meta.get("training_source_sha256") or {}
    for name, digest in config["reference_source_git_blob_sha256"].items():
        if name not in recorded or recorded[name] is None:
            pending.append("training source unavailable: " + name)
        elif recorded[name] != digest:
            errors.append("training source changed: " + name)
    old = meta.get("normalization") or {}
    if old.get("feature_names") != feature_columns():
        (errors if "feature_names" in old else pending).append("normalization feature names")
    if normalization is None:
        pending.append("fit normalization has not yet been compared")
    else:
        for key in ("x_mean", "x_std", "y_mean", "y_std", "n_fit"):
            if key not in old:
                pending.append("normalization missing " + key)
                continue
            a, b = np.asarray(old[key]), np.asarray(normalization[key])
            same = a.shape == b.shape and (
                np.array_equal(a, b) if key == "n_fit"
                else np.allclose(a, b, rtol=NORM_RTOL, atol=NORM_ATOL, equal_nan=False))
            if not same:
                errors.append("normalization differs: " + key)
    return {"status": "mismatch" if errors else "ann_provenance_pending" if pending else "matched",
            "pending": pending, "errors": errors, "metadata_sha256": file_sha256(path),
            "normalization_tolerance": {"rtol": NORM_RTOL, "atol": NORM_ATOL}}


def audit_inputs(root: Path, config: dict, metadata_dir: Path | None = None) -> dict:
    validate_config(config)
    parquet = repo_path(root, config["dataset"]["path"])
    digest = file_sha256(parquet)
    if digest != config["dataset"]["sha256"]:
        raise ValueError("dataset SHA-256 differs from frozen plan")
    parquet_file = pq.ParquetFile(parquet)
    if parquet_file.metadata.num_rows != config["dataset"]["rows"]:
        raise ValueError("dataset row count differs from frozen plan")
    if set(DATA_COLUMNS) - set(parquet_file.schema_arrow.names):
        raise ValueError("dataset lacks required ANN feature/identity columns")
    working_sources = source_hashes()
    for name, expected in config["reference_source_git_blob_sha256"].items():
        if working_sources.get(name) != expected:
            raise ValueError("frozen ANN helper changed: " + name)
    pairs = []
    for pair in config["pairs"]:
        mpath = repo_path(root, pair["manifest"])
        apath = repo_path(root, pair["ann_metrics"])
        contract, ann = read_json(mpath), read_json(apath)
        if semantic_hash(contract) != pair["manifest_semantic_sha256"]:
            raise ValueError("manifest changed: " + pair["run_id"])
        if semantic_hash(ann) != pair["ann_metrics_semantic_sha256"]:
            raise ValueError("ANN metrics changed: " + pair["run_id"])
        if (contract["dataset_sha256"] != digest or contract["mode"] != pair["mode"]
                or contract["split_seed"] != pair["seed"]
                or ann["seed"] != pair["seed"] or ann["mode"] != pair["mode"]
                or ann["micro"]["n"] != pair["expected_test_rows"]
                or ann["macro"]["n_groups"] != pair["expected_cell_groups"]
                or ann["n_negative_targets"] != pair["expected_negative_targets"]):
            raise ValueError("pair contract/count mismatch: " + pair["run_id"])
        pairs.append({"run_id": pair["run_id"], "manifest_file_sha256": file_sha256(mpath),
                      "ann_metrics_file_sha256": file_sha256(apath),
                      "ann_provenance": ann_provenance(root, config, pair, metadata_dir)})
    return {"schema_version": 1, "experiment_id": config["experiment_id"],
            "status": "mismatch" if any(p["ann_provenance"]["status"] == "mismatch" for p in pairs)
            else "ready_for_run", "training_executed_by_audit": False,
            "config_semantic_sha256": semantic_hash(config), "dataset_sha256": digest,
            "dataset_rows": parquet_file.metadata.num_rows, "feature_count": len(feature_columns()),
            "source_sha256": working_sources, "versions": versions(), "git": git_state(),
            "pairs": pairs}


def run_signature(config: dict, pair: dict, audit: dict, batch_rows: int) -> str:
    return semantic_hash({"config": semantic_hash(config), "pair": pair,
                          "sources": audit["source_sha256"], "versions": audit["versions"],
                          "batch_rows": batch_rows})


def artifact_hashes(output: Path, paths: list[Path]) -> dict:
    return {p.relative_to(output).as_posix(): file_sha256(p) for p in paths}


def verify_artifacts(output: Path, marker: dict, signature: str) -> None:
    if marker.get("signature") != signature:
        raise ValueError("saved run signature changed; use a separate output directory")
    if not marker.get("artifacts"):
        raise ValueError("saved run has no artifact checksums")
    for relative, expected in marker["artifacts"].items():
        path = repo_path(output, relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError("missing or changed result artifact: " + relative)
