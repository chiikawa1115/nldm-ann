"""Config-driven, restartable multi-seed Experiment A runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from aggregate import load_run_records, plot_overall, summarize_records, write_aggregate


def run(config_path: Path, aggregate_only: bool = False, reevaluate: bool = False) -> None:
    root = Path(__file__).parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = (root / config["dataset"]).resolve()
    output_dir = (root / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    result_files: list[tuple[int, Path]] = []
    for holdout in config["holdout_ratios"]:
        ratio_name = f"holdout_{holdout:g}"
        for seed in config["seeds"]:
            run_dir = output_dir / ratio_name / f"seed_{seed}"
            checkpoint = run_dir / "model.pt"
            result_prefix = run_dir / "result"
            result_json = result_prefix.with_suffix(".json")
            result_files.append((seed, result_json))
            if aggregate_only or (result_json.exists() and not reevaluate):
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            training = config.get("training", {})
            train_command = [
                sys.executable, str(root / "src" / "train.py"),
                "--parquet", str(dataset), "--holdout", str(holdout),
                "--seed", str(seed), "--out", str(checkpoint),
                "--n-epochs", str(training.get("n_epochs", 200)),
                "--patience", str(training.get("patience", 25)),
                "--lr", str(training.get("lr", 0.003)),
                "--weight-decay", str(training.get("weight_decay", 0.0001)),
                "--chunk-size", str(training.get("chunk_size", 1024)),
                "--hidden-dims", *map(str, training.get("hidden_dims", [64, 64])),
            ]
            if not checkpoint.exists():
                subprocess.run(train_command, cwd=root, check=True)
            subprocess.run([
                sys.executable, str(root / "src" / "evaluate.py"),
                "--parquet", str(dataset), "--ckpt", str(checkpoint),
                "--out", str(result_prefix),
            ], cwd=root, check=True)

    missing = [str(path) for _, path in result_files if not path.exists()]
    if missing:
        raise FileNotFoundError("missing run results:\n" + "\n".join(missing))
    records = pd.DataFrame([
        row for seed, path in result_files for row in load_run_records(path, seed)
    ])
    confidence = float(config.get("confidence", 0.95))
    write_aggregate(records, output_dir, confidence)
    plot_overall(summarize_records(records, confidence), output_dir / "accuracy_ci.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--reevaluate", action="store_true",
                        help="reuse checkpoints but regenerate per-run metrics")
    args = parser.parse_args()
    run(args.config, args.aggregate_only, args.reevaluate)


if __name__ == "__main__":
    main()
