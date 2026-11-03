"""CPU Ridge comparison CLI. Preflight never fits; long runs require the run subcommand."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

from ridge_audit import (
    CODE_ROOT, ann_provenance, artifact_hashes, audit_inputs, git_state, read_json,
    repo_path, run_signature, semantic_hash, verify_artifacts, versions, write_json,
)
from ridge_compare import paired_rows, write_comparison
from ridge_core import (
    TIE_ATOL, TIE_RTOL, RidgeModel, collect_moments, evaluate_stream, solve_candidates,
)
from features import feature_columns


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def peak_memory_bytes() -> int | None:
    try:
        import os
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in (
                        "peak", "working", "peak_paged", "paged", "peak_nonpaged",
                        "nonpaged", "pagefile", "peak_pagefile")]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return int(counters.peak)
            return None
        import resource
        import sys
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, OSError, AttributeError):
        return None


class Progress:
    def __init__(self, run_dir: Path, run_id: str):
        self.run_dir, self.run_id = run_dir, run_id
        self.started = time.perf_counter()
        self.previous = 0.0
        self.stage = None

    def __call__(self, stage: str, counts: dict) -> None:
        current = time.perf_counter()
        if stage == self.stage and current - self.previous < 10:
            return
        self.previous, self.stage = current, stage
        event = {"time_utc": now(), "run_id": self.run_id, "stage": stage,
                 "elapsed_seconds": current - self.started, "counts": counts,
                 "process_peak_memory_bytes": peak_memory_bytes()}
        write_json(self.run_dir / "progress.json", event)
        print(json.dumps(event), flush=True)


def initialize_output(root: Path, output: Path, config: dict) -> None:
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("output must be a dedicated directory inside the project")
    for protected in (root / "src", root / "data", root / "experiments/asap7_generalization_v1"):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("output would overlap source data or original ANN artifacts")
    marker = output / ".ridge_experiment.json"
    identity = {"experiment_id": config["experiment_id"], "config_semantic_sha256": semantic_hash(config)}
    if marker.exists():
        if read_json(marker) != identity:
            raise ValueError("output belongs to a different plan; choose another directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("refusing to write into a nonempty directory without a Ridge marker")
    else:
        write_json(marker, identity)


def check_counts(config: dict, pair: dict, counts: dict) -> None:
    if counts["source"] != config["dataset"]["rows"] or counts["test"] != pair["expected_test_rows"]:
        raise ValueError("streamed source/test counts differ from frozen inputs")
    if counts["excluded"] or counts["fit"] < 2 or counts["validation"] == 0:
        raise ValueError("invalid fit/validation/excluded counts")
    if sum(counts[k] for k in ("fit", "validation", "test", "excluded")) != counts["source"]:
        raise ValueError("partition counts do not cover source rows exactly")


def current_qualification(root: Path, output: Path, config: dict, pair: dict,
                          metadata_dir: Path | None) -> dict:
    meta = read_json(output / "evaluations" / pair["run_id"] / "training_metadata.json")
    qualification = ann_provenance(root, config, pair, metadata_dir, meta["normalization"])
    if qualification["status"] == "mismatch":
        raise ValueError("ANN metadata mismatch: " + "; ".join(qualification["errors"]))
    return qualification


def run_one(root: Path, output: Path, config: dict, pair: dict, audit: dict,
            batch_rows: int, metadata_dir: Path | None = None,
            retry_incomplete: bool = False) -> dict:
    signature = run_signature(config, pair, audit, batch_rows)
    run_id = pair["run_id"]
    run_dir = output / "evaluations" / run_id
    status_path = run_dir / "status.json"
    fit_marker_path = run_dir / "fit_complete.json"
    meta_path = run_dir / "training_metadata.json"
    model_path = output / "models" / (run_id + ".npz")
    previous = read_json(status_path) if status_path.exists() else None
    if previous:
        if previous.get("signature") != signature:
            raise ValueError("run inputs/code/environment/batch size changed; choose another output directory")
        if previous.get("status") == "complete":
            verify_artifacts(output, previous, signature)
            current_qualification(root, output, config, pair, metadata_dir)
            print(f"reused_completed={run_id}", flush=True)
            return previous
        if not retry_incomplete:
            raise ValueError(f"{run_id} is incomplete; inspect status and use --retry-incomplete explicitly")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("unrecognized existing run artifacts")
    qualification = ann_provenance(root, config, pair, metadata_dir)
    if qualification["status"] == "mismatch":
        raise ValueError("ANN metadata mismatch before fitting")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    state = {"schema_version": 1, "run_id": run_id, "status": "running",
             "signature": signature, "started_at": now(), "stage": "moments"}
    write_json(status_path, state)
    progress = Progress(run_dir, run_id)
    parquet = repo_path(root, config["dataset"]["path"])
    contract = read_json(repo_path(root, pair["manifest"]))
    try:
        if fit_marker_path.exists():
            verify_artifacts(output, read_json(fit_marker_path), signature)
            model = RidgeModel.load(model_path)
            current_qualification(root, output, config, pair, metadata_dir)
            progress("reuse_fitted_model", {})
        else:
            progress("moments", {"source": 0})
            fit, validation, counts = collect_moments(
                parquet, contract, pair["seed"], batch_rows,
                config["split"]["validation_ratio"], progress)
            check_counts(config, pair, counts)
            model, candidates = solve_candidates(fit, validation, feature_columns(),
                                                  config["ridge"]["lambda_grid"])
            qualification = ann_provenance(root, config, pair, metadata_dir,
                                           model.normalization.to_dict())
            if qualification["status"] == "mismatch":
                raise ValueError("ANN normalization mismatch: " + "; ".join(qualification["errors"]))
            model.save(model_path)
            write_json(meta_path, {
                "schema_version": 1, "run_id": run_id, "mode": pair["mode"], "seed": pair["seed"],
                "signature": signature, "counts": counts, "batch_rows": batch_rows,
                "normalization": model.normalization.to_dict(), "candidates": candidates,
                "selected_lambda": model.selected_lambda,
                "selected_alpha": model.selected_lambda * model.normalization.n_fit,
                "tie_tolerance": {"rtol": TIE_RTOL, "atol": TIE_ATOL},
                "ann_provenance": qualification, "config_semantic_sha256": semantic_hash(config),
                "dataset_sha256": audit["dataset_sha256"],
                "manifest_semantic_sha256": pair["manifest_semantic_sha256"],
                "source_sha256": audit["source_sha256"], "versions": versions(), "git": git_state(),
                "fit_seconds": time.perf_counter() - started,
                "process_peak_memory_bytes": peak_memory_bytes(), "max_rows": None,
                "intercept_penalized": False, "refit_on_validation": False,
                "normalization_method": "float64_centered_Chan_moments_fit_only",
            })
            write_json(fit_marker_path, {"signature": signature,
                       "artifacts": artifact_hashes(output, [model_path, meta_path])})
        state["stage"] = "evaluate"
        write_json(status_path, state)
        evaluation_started = time.perf_counter()
        metrics, per_cell = evaluate_stream(parquet, contract, model, pair["expected_test_rows"],
                                            batch_rows, output / ".scratch", progress)
        if (metrics["macro"]["n_groups"] != pair["expected_cell_groups"]
                or metrics["n_negative_targets"] != pair["expected_negative_targets"]):
            raise ValueError("test cell groups/negative targets differ from ANN")
        qualification = current_qualification(root, output, config, pair, metadata_dir)
        metrics.update({"schema_version": 1, "mode": pair["mode"], "seed": pair["seed"],
                        "run_id": run_id, "signature": signature, "ann_provenance": qualification})
        metrics_path = run_dir / "metrics.json"
        cells_path = run_dir / "per_cell_metrics.csv"
        per_cell.to_csv(cells_path, index=False)
        write_json(metrics_path, metrics)
        state.update({"status": "complete", "stage": "complete", "finished_at": now(),
                      "evaluation_seconds": time.perf_counter() - evaluation_started,
                      "invocation_seconds": time.perf_counter() - started,
                      "process_peak_memory_bytes": peak_memory_bytes(),
                      "artifacts": artifact_hashes(output, [model_path, meta_path, fit_marker_path,
                                                             metrics_path, cells_path])})
        write_json(status_path, state)
        progress("complete", {"test": metrics["micro"]["n"]})
        return state
    except BaseException as exc:
        state.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      "error": f"{type(exc).__name__}: {exc}", "finished_at": now()})
        write_json(status_path, state)
        raise


def aggregate_results(root: Path, output: Path, config: dict, audit: dict, batch_rows: int,
                      metadata_dir: Path | None = None, make_plots: bool = True) -> dict:
    rows, states = [], []
    for pair in config["pairs"]:
        run_dir = output / "evaluations" / pair["run_id"]
        path = run_dir / "status.json"
        state = {"run_id": pair["run_id"], "status": "not_run"}
        if path.exists():
            saved = read_json(path)
            state["status"] = saved["status"]
            if saved["status"] == "complete":
                try:
                    verify_artifacts(output, saved, run_signature(config, pair, audit, batch_rows))
                    qualification = current_qualification(root, output, config, pair, metadata_dir)
                    state["ann_provenance_status"] = qualification["status"]
                    rows.extend(paired_rows(pair["run_id"], pair["mode"], pair["seed"],
                                           read_json(repo_path(root, pair["ann_metrics"])),
                                           read_json(run_dir / "metrics.json"), qualification["status"]))
                except (ValueError, FileNotFoundError, KeyError) as exc:
                    state.update(status="mismatch", error=str(exc))
        states.append(state)
    write_comparison(output, config, rows, states, make_plots)
    return read_json(output / "aggregate/comparison_status.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "run", "aggregate"])
    parser.add_argument("--root", type=Path, default=CODE_ROOT)
    parser.add_argument("--config", default="experiments/configs/global_ann_ridge_v1.json")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--ann-metadata-dir", type=Path)
    parser.add_argument("--batch-rows", type=int)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-id", action="append")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run" and not (args.run_id or args.all):
        parser.error("run requires --run-id or --all; preflight does not train")
    if args.command != "run" and (args.run_id or args.all or args.retry_incomplete):
        parser.error("run selection/retry options only apply to run")
    root = args.root.resolve()
    config = read_json(repo_path(root, args.config))
    batch_rows = (args.batch_rows if args.batch_rows is not None
                  else config["execution"]["batch_rows_target"])
    if batch_rows <= 0:
        parser.error("--batch-rows must be positive")
    output = repo_path(root, args.out_dir or config["execution"]["output_dir"])
    audit = audit_inputs(root, config, args.ann_metadata_dir)
    initialize_output(root, output, config)
    write_json(output / "input_audit.json", audit)
    if audit["status"] == "mismatch":
        print("preflight=mismatch; inspect input_audit.json", flush=True)
        return 2
    if args.command == "preflight":
        print(json.dumps({"preflight": "PASS", "planned_runs": len(config["pairs"]),
                          "ann_provenance_pending": sum(p["ann_provenance"]["status"] != "matched"
                                                        for p in audit["pairs"]),
                          "training_executed": False, "output": str(output)}), flush=True)
        return 0
    if args.command == "run":
        planned = {p["run_id"]: p for p in config["pairs"]}
        requested = args.run_id if args.run_id else list(planned)
        if len(set(requested)) != len(requested) or set(requested) - planned.keys():
            parser.error("duplicate or unknown run ID")
        first = config["execution"]["first_run"]
        requested = sorted(requested, key=lambda name: (name != first, list(planned).index(name)))
        if first not in requested:
            marker = output / "evaluations" / first / "status.json"
            if not marker.exists() or read_json(marker).get("status") != "complete":
                parser.error("complete the planned first run before other seeds")
            verify_artifacts(output, read_json(marker),
                             run_signature(config, planned[first], audit, batch_rows))
        for name in requested:
            run_one(root, output, config, planned[name], audit, batch_rows,
                    args.ann_metadata_dir, args.retry_incomplete)
    status = aggregate_results(root, output, config, audit, batch_rows,
                               args.ann_metadata_dir, not args.no_plots)
    print(json.dumps(status), flush=True)
    return 2 if any(r["status"] == "mismatch" for r in status["runs"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
